"""Five-stage MoDL and PINQI for the BrainWeb saturation-recovery comparison."""

import math
from functools import partial
from typing import Literal

import mr2
import torch
import torch.nn.functional as functional
from data import DELAYS, Sample, acquisition_op
from torch.utils.checkpoint import checkpoint

Fitter = Literal['dictionary', 'lbfgs']
T1_BOUNDS = (0.05, 6.0)


def t1_grid(device: torch.device | None = None) -> torch.Tensor:
    """Return the 1000 log-spaced dictionary entries, excluding the constraint boundaries."""
    return torch.logspace(math.log10(T1_BOUNDS[0]), math.log10(T1_BOUNDS[1]), 1002, device=device)[1:-1]


def fit_objective(
    signal_model: mr2.operators.Operator,
    images: torch.Tensor,
    prior_m0: torch.Tensor,
    prior_t1_coordinate: torch.Tensor,
    weight: torch.Tensor,
) -> mr2.operators.Operator:
    """Return the data and quadratic-prior objective minimized by L-BFGS."""
    data_term = mr2.operators.functionals.L2NormSquared(images) @ signal_model
    prior = mr2.operators.functionals.L2NormSquared(prior_m0) | mr2.operators.functionals.L2NormSquared(
        prior_t1_coordinate
    )
    return 0.5 * (data_term + weight * prior)


class DictionaryFit(torch.nn.Module):
    """Differentiable dictionary fit: prior-aware dictionary matching and one Gauss-Newton step."""

    def __init__(self, signal_model: mr2.operators.Operator, t1_coordinates: torch.Tensor) -> None:
        """Generate the dictionary from the signal model."""
        super().__init__()
        self.dictionary = mr2.operators.DifferentiableDictionaryMatchOp(signal_model, index_of_scaling_parameter=0)
        self.dictionary.append(torch.tensor(1.0 + 0j), t1_coordinates)

    def forward(
        self, images: torch.Tensor, prior_m0: torch.Tensor, prior_t1_coordinate: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the parameters, see `fit_objective` for the arguments."""
        parameters = self.dictionary(images, prior=(prior_m0, prior_t1_coordinate), prior_precision=weight)
        return (parameters[0], parameters[1])


class OptimizerFit(torch.nn.Module):
    """Original PINQI fit: L-BFGS starting at the prior, with implicit differentiation by MR2's default CG."""

    def __init__(self, signal_model: mr2.operators.Operator, max_iterations: int) -> None:
        """Set up the differentiable optimizer."""
        super().__init__()
        self.optimizer = mr2.operators.OptimizerOp(
            partial(fit_objective, signal_model),
            lambda _images, prior_m0, prior_t1_coordinate, _weight: (prior_m0, prior_t1_coordinate),
            optimize=partial(
                mr2.algorithms.optimizers.lbfgs,
                max_iterations=max_iterations,
                tolerance_grad=1e-06,
                tolerance_change=1e-06,
            ),
        )

    def forward(
        self, images: torch.Tensor, prior_m0: torch.Tensor, prior_t1_coordinate: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit the parameters, see `fit_objective` for the arguments."""
        return self.optimizer(images, prior_m0, prior_t1_coordinate, weight)


def masked_ssim(prediction: torch.Tensor, target: torch.Tensor, brain: torch.Tensor) -> torch.Tensor:
    """Return per-slice SSIM using 7x7 windows wholly inside the brain."""
    flat_target, flat_brain = (target.flatten(1), brain.flatten(1))
    maximum = flat_target.masked_fill(~flat_brain, -torch.inf).amax(1)
    minimum = flat_target.masked_fill(~flat_brain, torch.inf).amin(1)
    valid = functional.avg_pool2d(brain.float().flatten(0, 2)[:, None], 7, stride=1).flatten(1).eq(1).any(1)
    data_range = torch.where(valid, maximum - minimum, 1.0).clamp_min(1e-06)
    (score,) = mr2.operators.functionals.SSIM(
        target, brain, data_range=data_range[:, None, None, None, None], window_size=7, reduction='volume'
    )(prediction)
    return torch.where(valid, score.flatten(), torch.nan)


def metrics(m0: torch.Tensor, t1: torch.Tensor, sample: Sample) -> dict[str, torch.Tensor]:
    """Return per-slice brain T1 nRMSE, MAE (s), SSIM, and complex ``m0`` nRMSE."""
    brain = sample['brain']
    target_t1 = torch.where(brain, sample['t1'], 0).flatten(1)
    target_m0 = torch.where(brain, sample['m0'], 0).flatten(1)
    t1_error = torch.where(brain, t1 - sample['t1'], 0).flatten(1)
    m0_error = torch.where(brain, m0 - sample['m0'], 0).flatten(1)
    return {
        't1_nrmse': t1_error.norm(dim=1) / target_t1.norm(dim=1),
        't1_mae_s': t1_error.abs().sum(dim=1) / brain.flatten(1).sum(dim=1),
        't1_rmse_s': (t1_error.square().sum(dim=1) / brain.flatten(1).sum(dim=1)).sqrt(),
        'm0_nrmse': m0_error.norm(dim=1) / target_m0.norm(dim=1),
        'masked_ssim': masked_ssim(t1, sample['t1'], brain),
    }


def normalize(batch: Sample) -> Sample:
    """Compute the initial CG images and normalize."""
    acquisition = acquisition_op(batch['ky'], batch['csm'])
    (images,) = mr2.algorithms.optimizers.cg(acquisition.gram, acquisition.H(batch['data'])[0], max_iterations=2)
    scale = torch.quantile(images.abs().movedim(1, 0).flatten(1), 0.95, dim=1)[:, None, None, None, None]
    m0 = batch['m0'] / scale
    (target_images,) = mr2.operators.models.SaturationRecovery(DELAYS).to(m0.device)(m0, batch['t1'])
    normalized = {'data': batch['data'] / scale, 'initial_images': images / scale, 'm0': m0}
    return {**batch, **normalized, 'target_images': target_images}


class Reconstruction(torch.nn.Module):
    """Shared normalization, image prior and signal model of the two reconstructions."""

    t1_coordinates: torch.Tensor
    fit: DictionaryFit | OptimizerFit | None

    def __init__(self, fitter: Fitter | None) -> None:
        """Initialize the fixed BrainWeb experiment settings."""
        super().__init__()
        self.cg_iterations = 20
        self.image_net = mr2.nn.ComplexAsChannel(
            mr2.nn.nets.UNet(
                n_dim=2,
                n_channels_in=2,
                n_channels_out=2,
                cond_dim=16,
                n_features=(16, 32, 48, 64),
                attention_depths=(),
                n_heads=8,
                encoder_blocks_per_scale=2,
            )
        )
        self.embedding = torch.nn.Embedding(6, 16)
        self.saturation_recovery = mr2.operators.models.SaturationRecovery(DELAYS).requires_grad_(False)
        self.constraints = mr2.operators.ConstraintsOp(((None, None), T1_BOUNDS))
        self.signal_model = (self.saturation_recovery @ self.constraints).requires_grad_(False)
        self.t1_reference_coordinate = float(self.constraints.inverse(torch.tensor(1.0), torch.tensor(1.0))[1])
        t1 = t1_grid()
        self.register_buffer('t1_coordinates', self.constraints.inverse(torch.tensor(1.0), t1)[1])
        if fitter == 'dictionary':
            self.fit = DictionaryFit(self.signal_model, self.t1_coordinates)
        elif fitter == 'lbfgs':
            self.fit = OptimizerFit(self.signal_model, 20)
        else:
            self.fit = None

    def condition(self, stage: int) -> torch.Tensor:
        """Return the learned embedding of a stage; stage 0 is the initial parameter prediction."""
        return self.embedding.weight[stage : stage + 1]

    def denoise(self, images: torch.Tensor, stage: int) -> torch.Tensor:
        """Apply the residual image network to every delay image with shape `(delay, batch, 1, 1, y, x)`."""
        residual = self.image_net(images.flatten(0, 1)[:, 0], cond=self.condition(stage))
        return images + residual.reshape(images.shape)

    def estimate(self, sample: Sample) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the final ``m0`` and T1 (s), each with shape `(batch, 1, 1, y, x)`."""
        return self(sample)[1][-1]


class MoDL(Reconstruction):
    """MoDL with a stage-conditioned residual image network and CG data consistency."""

    def __init__(self, fitter: Fitter | None) -> None:
        """Initialize the fixed BrainWeb experiment settings."""
        super().__init__(fitter or 'dictionary')
        self.end_to_end = fitter is not None
        self.fit_weight = 0.0001
        self.raw_weights = torch.nn.Parameter(torch.full((5, 1), -4.0))

    def forward(self, sample: Sample) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Reconstruct a prepared batch."""
        acquisition = acquisition_op(sample['ky'], sample['csm'])
        gram = acquisition.gram
        (adjoint,) = acquisition.H(sample['data'])
        data_consistency = mr2.operators.ConjugateGradientOp(
            lambda weight, _prior: gram + weight,
            lambda weight, prior: (adjoint + weight * prior,),
            max_iterations=self.cg_iterations,
        )
        images = sample['initial_images']
        for stage, (weight,) in enumerate(functional.softplus(self.raw_weights), start=1):
            prior = self.denoise(images, stage)
            (images,) = data_consistency(weight, prior, initial_value=(images.detach(),))
        if not self.end_to_end:
            return (images, [])
        return images, [self.fit_images(images)]

    def fit_images(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match the final images against the fixed weak zero-M0/reference-T1 prior."""
        m0 = images.new_zeros(images.shape[1:])
        coordinate = images.real.new_full(images.shape[1:], self.t1_reference_coordinate)
        assert self.fit is not None
        m0, coordinate = self.fit(images, m0, coordinate, images.real.new_tensor(self.fit_weight))
        return m0, self.constraints(m0, coordinate)[1]

    def estimate(self, sample: Sample) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit image-only MoDL at evaluation, outside its training graph."""
        images, estimates = self(sample)
        return estimates[-1] if estimates else self.fit_images(images)


class PINQI(Reconstruction):
    """PINQI with shared stage-conditioned image and parameter networks."""

    def __init__(self, fitter: Fitter) -> None:
        """Initialize the fixed BrainWeb experiment settings."""
        super().__init__(fitter)
        self.parameter_net = mr2.nn.ComplexAsChannel(
            mr2.nn.nets.UNet(
                n_dim=2,
                n_channels_in=10,
                n_channels_out=3,
                cond_dim=16,
                n_features=(32, 64, 96, 128),
                attention_depths=(-1,),
                n_heads=8,
                encoder_blocks_per_scale=2,
            ),
            convert_back=False,
        )
        self.raw_weights = torch.nn.Parameter(torch.full((5, 3), -4.0))
        self.recompute = fitter == 'lbfgs'

    def _predict_parameters(self, images: torch.Tensor, stage: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ``m0`` and the T1 coordinate from images with shape `(delay, batch, 1, 1, y, x)`."""
        parameters = self.parameter_net(images[:, :, 0, 0].transpose(0, 1), cond=self.condition(stage))
        m0 = torch.complex(parameters[:, 0], parameters[:, 1])[:, None, None]
        return (m0, parameters[:, 2, None, None] + self.t1_reference_coordinate)

    def forward(self, sample: Sample) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Reconstruct a prepared batch."""
        assert self.fit is not None
        acquisition = acquisition_op(sample['ky'], sample['csm'])
        gram = acquisition.gram
        (adjoint,) = acquisition.H(sample['data'])
        data_consistency = mr2.operators.ConjugateGradientOp(
            lambda image_weight, signal_weight, *_: gram + image_weight + signal_weight,
            lambda image_weight, signal_weight, prior, signal: (
                adjoint + image_weight * prior + signal_weight * signal,
            ),
            max_iterations=self.cg_iterations,
        )
        images = sample['initial_images']
        m0, t1_coordinate = self.predict_parameters(images, 0)
        estimates = [(m0, self.constraints(m0, t1_coordinate)[1])]
        for stage, (image_weight, signal_weight, parameter_weight) in enumerate(
            functional.softplus(self.raw_weights), start=1
        ):
            prior = self.denoise(images, stage)
            (signal,) = self.signal_model(m0, t1_coordinate)
            (images,) = data_consistency(image_weight, signal_weight, prior, signal, initial_value=(images.detach(),))
            prior_m0, prior_t1_coordinate = self.predict_parameters(images, stage)
            m0, t1_coordinate = self.fit(images, prior_m0, prior_t1_coordinate, parameter_weight)
            estimates.append((m0, self.constraints(m0, t1_coordinate)[1]))
        return (images, estimates)

    def predict_parameters(self, images: torch.Tensor, stage: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute activations in the L-BFGS training variant to retain batch size eight."""
        if self.recompute and self.training and torch.is_grad_enabled():
            return checkpoint(self._predict_parameters, images, stage, use_reentrant=False)
        return self._predict_parameters(images, stage)

    def denoise(self, images: torch.Tensor, stage: int) -> torch.Tensor:
        """Apply the shared image network, with recomputation for L-BFGS training."""
        if self.recompute and self.training and torch.is_grad_enabled():
            return checkpoint(super().denoise, images, stage, use_reentrant=False)
        return super().denoise(images, stage)


def cg_dictionary(sample: Sample) -> tuple[torch.Tensor, torch.Tensor]:
    """Ridge CG (lambda 0.001, cap 100), then 1000-entry matching without refinement."""
    op = acquisition_op(sample['ky'], sample['csm'])
    gram = op.gram
    rhs = op.H(sample['data'])[0]
    (images,) = mr2.algorithms.optimizers.cg(gram + 0.001, rhs, max_iterations=100, tolerance=1e-4)
    signal = mr2.operators.models.SaturationRecovery(DELAYS).to(images.device)
    bounds = mr2.operators.ConstraintsOp(((None, None), T1_BOUNDS))
    t1 = t1_grid(images.device)
    coordinates = bounds.inverse(images.new_tensor(1.0 + 0j), t1)[1]
    dictionary = mr2.operators.DictionaryMatchOp(signal @ bounds, index_of_scaling_parameter=0, batch_size=4096)
    dictionary.append(images.new_tensor(1.0 + 0j), coordinates)
    m0, coordinate = dictionary(images)
    return m0, bounds(m0, coordinate)[1]
