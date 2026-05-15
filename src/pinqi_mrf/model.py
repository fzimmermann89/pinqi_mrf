"""PINQI-MRF model skeleton."""

from collections.abc import Sequence
from typing import Any, cast

import cmap
import einops
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import mr2
import numpy as np
import torch
from matplotlib.colors import Colormap
from mr2.operators.NonUniformFastFourierOp import SubspaceNonUniformFastFourierOpGramOp

from pinqi_mrf.acquisition import create_radial_trajectory
from pinqi_mrf.data import BatchType
from pinqi_mrf.losses import charbonnier_loss, ssim_loss
from pinqi_mrf.parameters import R1_STATS, R2_STATS, standardized_rate_to_relaxation_time
from pinqi_mrf.signal_models import SubspaceLookup
from pinqi_mrf.white_kspace_subspace_rhs_sampler import WhiteKspaceSubspaceRhsNoiseSampler

MIN_SUPERVISED_T1 = 0.5
MIN_SUPERVISED_T2 = 0.03
T1_DISPLAY_RANGE = (0.0, 2.0)
T2_DISPLAY_RANGE = (0.020, 0.150)


class PINQI(torch.nn.Module):
    """PINQI reconstruction model."""

    def __init__(
        self,
        signalmodel: mr2.operators.SignalModel,
        n_subspace_coefficients: int,
        n_iterations: int,
        n_features_parameter_net: Sequence[int],
        n_features_image_net: Sequence[int],
        csm_refinement_iterations: tuple[int, ...] = (),
        csm_refinement_shape: tuple[int, int] = (64, 64),
        csm_refinement_weight: float = 1e-2,
        csm_refinement_cg_iterations: int = 10,
        initial_cg_iterations: int = 5,
        initial_cg_regularization: float = 1e-2,
        image_cg_iterations: int = 20,
        image_cg_tolerance: float = 1e-5,
        parameter_ranges: tuple[tuple[float, float, int], ...] = ((-2.0, 6.5, 200), (-2.0, 9.5, 160)),
        nonlinear_solver_batch_size: int = 8192,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.signalmodel = signalmodel
        self.n_parameter_priors = len(parameter_ranges) + 1
        if any(iteration < 0 or iteration >= n_iterations for iteration in csm_refinement_iterations):
            raise ValueError('csm_refinement_iterations must be between 0 and n_iterations - 1.')
        self.csm_refinement_iterations = tuple(dict.fromkeys(csm_refinement_iterations))
        self.csm_refinement_shape = csm_refinement_shape
        self.csm_refinement_weight = csm_refinement_weight
        self.csm_refinement_cg_iterations = csm_refinement_cg_iterations
        self.initial_cg_iterations = initial_cg_iterations
        self.initial_cg_regularization = initial_cg_regularization
        self.image_cg_iterations = image_cg_iterations
        self.image_cg_tolerance = image_cg_tolerance
        self.parameter_net = mr2.nn.nets.UNet(
            n_dim=2,
            n_channels_in=n_subspace_coefficients * 2,
            n_channels_out=self.n_parameter_priors + 1,
            attention_depths=(-1, -2),
            n_features=n_features_parameter_net,
            cond_dim=128,
        )
        self.image_net = mr2.nn.nets.UNet(
            n_dim=2,
            n_channels_in=2,
            n_channels_out=2,
            attention_depths=(),
            n_features=n_features_image_net,
            cond_dim=128,
        )
        self.parameter_net = torch.compile(self.parameter_net)
        self.image_net = torch.compile(self.image_net)
        lambdas_raw = torch.ones(n_iterations, 2 + self.n_parameter_priors)
        self.lambdas_raw = torch.nn.Parameter(lambdas_raw)
        self.softplus = torch.nn.Softplus(beta=5)
        self.iteration_embedding = torch.nn.Embedding(n_iterations, 128)
        self.nonlinear_solver = mr2.operators.DifferentiableDictionaryMatchOp(
            self.signalmodel,
            index_of_scaling_parameter=0,
            batch_size=nonlinear_solver_batch_size,
        )
        nonlinear_solver_grids = [
            torch.linspace(minimum, maximum, n_steps) for minimum, maximum, n_steps in parameter_ranges
        ]
        broadcasted_grids = []
        for parameter_idx, grid in enumerate(nonlinear_solver_grids):
            grid_shape = [1] * len(nonlinear_solver_grids)
            grid_shape[parameter_idx] = grid.numel()
            broadcasted_grids.append(grid.reshape(grid_shape))
        self.nonlinear_solver.append(torch.tensor(1.0 + 0j), *broadcasted_grids)

    def get_linear_solver(self, normal_op: mr2.operators.LinearOperator) -> mr2.operators.ConjugateGradientOp:
        """Create the image update solver."""

        def operator_factory(
            lambda_image: torch.Tensor,
            lambda_q: torch.Tensor,
            *_: torch.Tensor,
        ) -> mr2.operators.LinearOperator:
            return normal_op + lambda_image + lambda_q

        def rhs_factory(
            lambda_image: torch.Tensor,
            lambda_q: torch.Tensor,
            image_reg: torch.Tensor,
            signal: torch.Tensor,
            rhs: torch.Tensor,
        ) -> tuple[torch.Tensor]:
            return (rhs + lambda_image * image_reg + lambda_q * signal,)

        return mr2.operators.ConjugateGradientOp(
            operator_factory=operator_factory,
            rhs_factory=rhs_factory,
            max_iterations=self.image_cg_iterations,
            tolerance=self.image_cg_tolerance,
        )

    def get_csm_refinement_solver(
        self,
        downsampling: mr2.operators.LinearOperator,
        subspace_gram: mr2.operators.LinearOperator,
        regularization_weight: float | torch.Tensor,
    ) -> mr2.operators.ConjugateGradientOp:
        """Create the low-resolution CSM refinement solver."""

        def operator_factory(alpha: torch.Tensor, *_: torch.Tensor) -> mr2.operators.LinearOperator:
            coil_lift = (
                mr2.operators.EinsumOp(
                    alpha,
                    'batch coeff one z y x, batch coil z y x -> batch coeff coil z y x',
                )
                @ downsampling.H
            )
            return coil_lift.H @ subspace_gram @ coil_lift + regularization_weight

        def rhs_factory(
            alpha: torch.Tensor,
            coil_rhs: torch.Tensor,
            csm_low_initial: torch.Tensor,
        ) -> tuple[torch.Tensor]:
            coil_lift = (
                mr2.operators.EinsumOp(
                    alpha,
                    'batch coeff one z y x, batch coil z y x -> batch coeff coil z y x',
                )
                @ downsampling.H
            )
            (rhs,) = coil_lift.H(coil_rhs)
            return (rhs + regularization_weight * csm_low_initial,)

        return mr2.operators.ConjugateGradientOp(
            operator_factory=operator_factory,
            rhs_factory=rhs_factory,
            max_iterations=self.csm_refinement_cg_iterations,
        )

    @staticmethod
    def get_normal_equations(
        coil_rhs: torch.Tensor,
        csm: torch.Tensor,
        subspace_gram: mr2.operators.LinearOperator,
    ) -> tuple[torch.Tensor, mr2.operators.LinearOperator]:
        """Create normal equations for a given CSM estimate."""
        csm_op = mr2.operators.SensitivityOp(csm[:, None])
        normal_op = csm_op.H @ subspace_gram @ csm_op
        (rhs,) = csm_op.H(coil_rhs)
        return rhs, normal_op

    def get_parameter_reg(self, image: torch.Tensor, iteration: int) -> tuple[torch.Tensor, ...]:
        """Predict parameter-space prior from the current image estimate."""
        image_real = einops.rearrange(
            torch.view_as_real(image),
            'batch time 1 1 y x complex -> batch (time complex) y x',
        )
        cond = self.iteration_embedding(torch.tensor(iteration, device=image.device))[None]
        with torch.autocast(device_type=image.device.type, dtype=torch.bfloat16):
            parameters = self.parameter_net(image_real.contiguous(), cond=cond)
        parameters = parameters.float()
        parameters = einops.rearrange(parameters, 'batch parameters y x -> parameters batch 1 1 y x')
        return (torch.complex(parameters[0], parameters[1]), *parameters[2:].unbind())

    def get_image_reg(self, image: torch.Tensor, iteration: int) -> torch.Tensor:
        """Predict image-space prior from the current image estimate."""
        batch = image.shape[0]
        image_real = einops.rearrange(
            torch.view_as_real(image),
            'batch time 1 1 y x complex -> (batch time) complex y x',
        )
        cond = self.iteration_embedding(torch.tensor(iteration, device=image.device))[None]
        with torch.autocast(device_type=image.device.type, dtype=torch.bfloat16):
            image_update = self.image_net(image_real.contiguous(), cond=cond)
        image_real = image_real + image_update.float()
        image_real = einops.rearrange(
            image_real,
            '(batch time) complex y x -> batch time 1 1 y x complex',
            batch=batch,
        )
        return torch.view_as_complex(image_real.contiguous())

    def forward(
        self,
        coil_rhs: torch.Tensor,
        csm_initial: torch.Tensor,
        subspace_gram: mr2.operators.LinearOperator,
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Apply the unrolled PINQI reconstruction."""
        rhs, normal_op = self.get_normal_equations(coil_rhs, csm_initial, subspace_gram)
        images = list(
            mr2.algorithms.optimizers.cg(
                normal_op + self.initial_cg_regularization,
                rhs,
                max_iterations=self.initial_cg_iterations,
            )
        )
        signal = images[-1]
        parameters: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        linear_solver = self.get_linear_solver(normal_op)

        if self.csm_refinement_iterations:
            downsampling = mr2.operators.DownSamplingOp(
                dim=(-2, -1), domain_shape=tuple(csm_initial.shape[-2:]), range_shape=self.csm_refinement_shape
            )
            (csm_low_initial,) = downsampling(csm_initial)
            csm_refinement_solver = self.get_csm_refinement_solver(
                downsampling=downsampling, subspace_gram=subspace_gram, regularization_weight=self.csm_refinement_weight
            )

        for iteration, (lambda_image, lambda_q, *lambda_parameters) in enumerate(self.softplus(self.lambdas_raw)):
            image_reg = self.get_image_reg(images[-1], iteration)
            images.extend(linear_solver(lambda_image, lambda_q, image_reg, signal, rhs))
            parameter_reg = self.get_parameter_reg(images[-1], iteration)
            if len(lambda_parameters) != self.n_parameter_priors:
                raise RuntimeError('Number of parameter prior weights must match the parameter prior tuple.')
            parameter, signal = self.nonlinear_solver(
                images[-1].swapaxes(0, 1),
                prior=parameter_reg,
                prior_precision=tuple(lambda_parameters),
                return_signal=True,
            )
            signal = signal.swapaxes(0, 1)
            parameters.append(parameter)
            if iteration in self.csm_refinement_iterations:
                (csm_low,) = csm_refinement_solver(
                    images[-1], coil_rhs, csm_low_initial, initial_value=(csm_low_initial,)
                )
                (csm_refined,) = downsampling.H(csm_low)
                csm_refined = csm_refined / csm_refined.abs().square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
                alignment = (csm_refined * csm_initial.conj()).sum(dim=1, keepdim=True)
                phase = alignment / alignment.abs().clamp_min(1e-8)
                csm_refined = csm_refined * phase.conj()
                rhs, normal_op = self.get_normal_equations(coil_rhs, csm_refined, subspace_gram)
                linear_solver = self.get_linear_solver(normal_op)
        return images, parameters


class BaselineReconstruction(torch.nn.Module):
    """Unregularized CG reconstruction followed by dictionary matching."""

    def __init__(
        self,
        signalmodel: mr2.operators.SignalModel,
        parameter_ranges: tuple[tuple[float, float, int], ...] = ((-2.0, 6.5, 200), (-2.0, 9.5, 160)),
        nonlinear_solver_batch_size: int = 8192,
    ) -> None:
        """Initialize the baseline reconstruction."""
        super().__init__()
        self.dictionary_match = mr2.operators.DictionaryMatchOp(
            signalmodel,
            index_of_scaling_parameter=0,
            batch_size=nonlinear_solver_batch_size,
        )
        nonlinear_solver_grids = [
            torch.linspace(minimum, maximum, n_steps) for minimum, maximum, n_steps in parameter_ranges
        ]
        broadcasted_grids = []
        for parameter_idx, grid in enumerate(nonlinear_solver_grids):
            grid_shape = [1] * len(nonlinear_solver_grids)
            grid_shape[parameter_idx] = grid.numel()
            broadcasted_grids.append(grid.reshape(grid_shape))
        self.dictionary_match.append(torch.tensor(1.0 + 0j), *broadcasted_grids)

    def forward(
        self,
        rhs: torch.Tensor,
        normal_op: mr2.operators.LinearOperator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply unregularized CG and dictionary matching."""
        images = list(mr2.algorithms.optimizers.cg(normal_op, rhs, max_iterations=10))
        return self.dictionary_match(images[-1].swapaxes(0, 1))


class PinqiModule(pl.LightningModule):
    """Lightning module for PINQI-MRF training."""

    def __init__(
        self,
        signalmodel: mr2.operators.SignalModel | None = None,
        n_subspace_coefficients: int = 10,
        image_size: int = 256,
        n_readout_samples: int = 256,
        n_coils: int = 8,
        n_compressed_coils: int | None = None,
        n_timepoints: int | None = None,
        parameter_stats: tuple[tuple[float, float], ...] = (R1_STATS, R2_STATS),
        parameter_ranges: tuple[tuple[float, float, int], ...] = ((-2.0, 6.5, 200), (-2.0, 9.5, 160)),
        trajectory: str = 'radial',
        n_iterations: int = 4,
        use_ground_truth_csm: bool = False,
        csm_estimation_iterations: int = 2,
        csm_estimation_smoothing_width: int = 11,
        csm_extrapolate: bool = True,
        csm_refinement_iterations: tuple[int, ...] = (),
        csm_refinement_shape: tuple[int, int] = (64, 64),
        csm_refinement_weight: float = 1e-2,
        csm_refinement_cg_iterations: int = 10,
        initial_cg_iterations: int = 5,
        initial_cg_regularization: float = 1e-2,
        image_cg_iterations: int = 20,
        image_cg_tolerance: float = 1e-5,
        n_features_parameter_net: Sequence[int] = (64, 128, 192, 256),
        n_features_image_net: Sequence[int] = (48, 64, 96, 128),
        nonlinear_solver_batch_size: int = 8192,
        lr: float = 3.0e-4,  # noqa: ARG002
        weight_decay: float = 1e-3,  # noqa: ARG002
        loss_weights: Sequence[float] = (0.2, 0.1, 0.1, 0.8),  # noqa: ARG002
        loss_charbonnier: float = 0.0,  # noqa: ARG002
        loss_ssim: float = 0.0,  # noqa: ARG002
    ) -> None:
        """Initialize the Lightning module."""
        super().__init__()
        self.save_hyperparameters(ignore=['signalmodel'])
        self.signalmodel = SubspaceLookup(
            signalmodel=signalmodel or mr2.operators.models.NeuroMRF(),
            n_subspace_coefficients=n_subspace_coefficients,
            parameter_stats=parameter_stats,
            parameter_ranges=parameter_ranges,
            n_timepoints=n_timepoints,
        )
        self.n_coils = n_coils
        self.n_compressed_coils = n_compressed_coils or n_coils
        if self.n_compressed_coils > self.n_coils:
            raise ValueError('n_compressed_coils must not exceed n_coils.')
        self.use_ground_truth_csm = use_ground_truth_csm
        self.csm_estimation_iterations = csm_estimation_iterations
        self.csm_estimation_smoothing_width = csm_estimation_smoothing_width
        self.csm_extrapolate = csm_extrapolate
        self.matrix_size = mr2.data.SpatialDimension(1, image_size, image_size)
        match trajectory:
            case 'radial':
                self.traj = create_radial_trajectory(
                    n_spokes=self.signalmodel.n_timepoints,
                    n_readout_samples=n_readout_samples,
                )
            case _:
                raise ValueError(f'Unknown trajectory: {trajectory}')
        self.pinqi = PINQI(
            signalmodel=self.signalmodel,
            n_subspace_coefficients=n_subspace_coefficients,
            n_iterations=n_iterations,
            n_features_parameter_net=n_features_parameter_net,
            n_features_image_net=n_features_image_net,
            csm_refinement_iterations=csm_refinement_iterations,
            csm_refinement_shape=csm_refinement_shape,
            csm_refinement_weight=csm_refinement_weight,
            csm_refinement_cg_iterations=csm_refinement_cg_iterations,
            initial_cg_iterations=initial_cg_iterations,
            initial_cg_regularization=initial_cg_regularization,
            image_cg_iterations=image_cg_iterations,
            image_cg_tolerance=image_cg_tolerance,
            parameter_ranges=parameter_ranges,
            nonlinear_solver_batch_size=nonlinear_solver_batch_size,
        )
        self.baseline = BaselineReconstruction(
            signalmodel=self.signalmodel,
            parameter_ranges=parameter_ranges,
            nonlinear_solver_batch_size=nonlinear_solver_batch_size,
        )
        self.validation_step_outputs: dict[str, torch.Tensor] = {}
        self.subspace_gram: SubspaceNonUniformFastFourierOpGramOp | None = None
        self.noise_sampler: WhiteKspaceSubspaceRhsNoiseSampler | None = None

    def on_fit_start(self) -> None:
        """Create fixed acquisition operators on the training device."""
        self.setup_acquisition()

    def on_validation_start(self) -> None:
        """Create fixed acquisition operators for standalone validation."""
        self.setup_acquisition()

    def setup_acquisition(self) -> None:
        """Create fixed acquisition operators on the active device."""
        if self.subspace_gram is not None:
            return
        nufft_op = mr2.operators.NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=self.matrix_size,
            encoding_matrix=self.matrix_size,
            traj=self.traj.to(self.device),
        )
        self.subspace_gram = cast(
            SubspaceNonUniformFastFourierOpGramOp,
            nufft_op.toeplitz(subspace=self.signalmodel.pca, subspace_dim=1),
        )
        self.noise_sampler = WhiteKspaceSubspaceRhsNoiseSampler(
            self.subspace_gram,
            n_coils=self.n_coils,
        )

    def simulate_coil_rhs(self, batch: BatchType) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate compressed coilwise subspace RHS data from synthetic maps."""
        if self.subspace_gram is None or self.noise_sampler is None:
            raise RuntimeError('Acquisition operators have not been initialized.')

        (coefficients,) = self.signalmodel(batch['m0'], batch['r1'], batch['r2'])
        coefficients = einops.rearrange(coefficients, 'coeff batch y x -> batch coeff 1 1 y x')
        csm_op = mr2.operators.SensitivityOp(batch['csm'][:, None])
        (coil_coefficients,) = csm_op(coefficients)
        (coil_rhs,) = self.subspace_gram(coil_coefficients)
        coil_rhs = coil_rhs + self.noise_sampler(batch['noise_std'])
        return self.compress_coils(coil_rhs, batch['csm'])

    def compress_coils(self, coil_rhs: torch.Tensor, csm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress the coil dimension using an SVD of the coilwise RHS."""
        n_coils = coil_rhs.shape[2]
        if self.n_compressed_coils >= n_coils:
            return coil_rhs, csm

        data = einops.rearrange(coil_rhs, 'batch coeff coil z y x -> batch coil (coeff z y x)')
        u, _, _ = torch.linalg.svd(data, full_matrices=False)
        compression = u[:, :, : self.n_compressed_coils].mH
        coil_rhs_compressed = einops.einsum(
            compression,
            coil_rhs,
            'batch compressed_coil coil, batch coeff coil z y x -> batch coeff compressed_coil z y x',
        )
        csm_compressed = einops.einsum(
            compression,
            csm,
            'batch compressed_coil coil, batch coil z y x -> batch compressed_coil z y x',
        )
        return coil_rhs_compressed, csm_compressed

    def estimate_initial_csm(self, coil_rhs: torch.Tensor) -> torch.Tensor:
        """Estimate initial CSMs from the dominant subspace coefficient combination."""
        with torch.no_grad():
            data = einops.rearrange(coil_rhs, 'batch coeff coil z y x -> batch coeff (coil z y x)')
            covariance = data @ data.mH
            _, eigenvectors = torch.linalg.eigh(covariance)
            calibration = einops.einsum(
                eigenvectors[..., -1].conj(), coil_rhs, 'batch coeff, batch coeff coil z y x -> batch coil z y x'
            )
            csm = mr2.algorithms.csm.inati(
                calibration,
                smoothing_width=self.csm_estimation_smoothing_width,
                n_iterations=self.csm_estimation_iterations,
                extrapolate=self.csm_extrapolate,
            )
            csm = torch.where(torch.isfinite(csm), csm, 0.0)
        return csm

    def forward(
        self,
        coil_rhs: torch.Tensor,
        csm_initial: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Apply the PINQI model."""
        if self.subspace_gram is None:
            raise RuntimeError('Acquisition operators have not been initialized.')
        csm_initial = csm_initial if csm_initial is not None else self.estimate_initial_csm(coil_rhs)
        return self.pinqi(coil_rhs, csm_initial, self.subspace_gram)

    def loss(
        self,
        predictions: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch: BatchType,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute parameter loss in standardized parameter space."""
        components = {
            'base': torch.tensor(0.0, device=self.device),
            'charbonnier': torch.tensor(0.0, device=self.device),
            'ssim': torch.tensor(0.0, device=self.device),
        }
        target_m0, target_r1, target_r2, mask = batch['m0'], batch['r1'], batch['r2'], batch['mask']
        mask = mask.bool()
        target_t1 = standardized_rate_to_relaxation_time(target_r1, R1_STATS)
        target_t2 = standardized_rate_to_relaxation_time(target_r2, R2_STATS)
        parameter_mask = self._parameter_mask(mask, target_r1, target_r2, target_t1, target_t2)
        final_iteration = len(predictions) - 1
        for iteration, (prediction, weight) in enumerate(zip(predictions, self.hparams.loss_weights, strict=True)):
            prediction_m0, prediction_r1, prediction_r2 = (parameter.squeeze(1, 2) for parameter in prediction)
            loss_r1 = torch.nn.functional.mse_loss(prediction_r1[parameter_mask], target_r1[parameter_mask])
            loss_r2 = torch.nn.functional.mse_loss(prediction_r2[parameter_mask], target_r2[parameter_mask])
            loss_m0 = torch.nn.functional.mse_loss(
                torch.view_as_real(prediction_m0[parameter_mask]),
                torch.view_as_real(target_m0.to(prediction_m0.dtype)[parameter_mask]),
            )
            loss_outside = prediction_m0[~mask].abs().mean()
            components['base'] = components['base'] + weight * (loss_r1 + loss_r2 + 0.5 * loss_m0 + 0.1 * loss_outside)
            if iteration == final_iteration:
                components['r1'] = loss_r1
                components['r2'] = loss_r2
                components['m0'] = loss_m0
                components['outside'] = loss_outside
        if self.hparams.loss_charbonnier or self.hparams.loss_ssim:
            target_parameters = (target_r1[:, None, None], target_r2[:, None, None])
            # The first parameter is the pure scaling parameter M0; detail losses act on quantitative parameters.
            prediction_parameters = predictions[-1][1:]
            if self.hparams.loss_charbonnier:
                components['charbonnier'] = charbonnier_loss(
                    prediction_parameters,
                    target_parameters,
                    parameter_mask,
                )
            if self.hparams.loss_ssim:
                components['ssim'] = ssim_loss(
                    prediction_parameters,
                    target_parameters,
                    parameter_mask,
                )
        loss = (
            components['base']
            + self.hparams.loss_charbonnier * components['charbonnier']
            + self.hparams.loss_ssim * components['ssim']
        )
        return loss, components

    def training_step(self, batch: BatchType, _batch_idx: int) -> torch.Tensor:
        """Run one training step."""
        coil_rhs, csm_ground_truth = self.simulate_coil_rhs(batch)
        csm_initial = csm_ground_truth if self.use_ground_truth_csm else self.estimate_initial_csm(coil_rhs)
        if self.global_step % 10 == 0:
            self.log_csm_diagnostics(csm_initial, prefix='debug/train_csm')
        _images, parameters = self(coil_rhs, csm_initial=csm_initial)
        loss, components = self.loss(parameters, batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.global_step % 10 == 0:
            self.log_dict(
                {f'debug/train_loss_{key}': value.detach() for key, value in components.items()},
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int) -> None:
        """Run one validation step."""
        coil_rhs, csm_ground_truth = self.simulate_coil_rhs(batch)
        csm_initial = csm_ground_truth if self.use_ground_truth_csm else self.estimate_initial_csm(coil_rhs)
        if batch_idx == 0:
            self.log_csm_diagnostics(csm_initial, prefix='debug/val_csm')
        if self.subspace_gram is None:
            raise RuntimeError('Acquisition operators have not been initialized.')
        rhs, normal_op = self.pinqi.get_normal_equations(coil_rhs, csm_initial, self.subspace_gram)
        _images, parameters = self(coil_rhs, csm_initial=csm_initial)
        loss, components = self.loss(parameters, batch)
        _pred_m0, pred_r1, pred_r2 = parameters[-1]
        _baseline_m0, baseline_r1, baseline_r2 = self.baseline(rhs, normal_op)
        mask = batch['mask'].bool()
        batch_size = len(mask)
        target_t1 = standardized_rate_to_relaxation_time(batch['r1'][:, None, None], R1_STATS)
        target_t2 = standardized_rate_to_relaxation_time(batch['r2'][:, None, None], R2_STATS)
        pred_t1 = standardized_rate_to_relaxation_time(pred_r1, R1_STATS)
        pred_t2 = standardized_rate_to_relaxation_time(pred_r2, R2_STATS)
        baseline_t1 = standardized_rate_to_relaxation_time(baseline_r1, R1_STATS)
        baseline_t2 = standardized_rate_to_relaxation_time(baseline_r2, R2_STATS)
        pred_t2_raw = pred_t2
        baseline_t2_raw = baseline_t2
        target_t1_map = target_t1.squeeze(1, 2)
        target_t2_map = target_t2.squeeze(1, 2)
        metric_mask = mask & torch.isfinite(target_t1_map) & torch.isfinite(target_t2_map)
        target_t1 = target_t1.clamp(*T1_DISPLAY_RANGE)
        target_t2 = target_t2.clamp(*T2_DISPLAY_RANGE)
        pred_t1 = pred_t1.clamp(*T1_DISPLAY_RANGE)
        pred_t2 = pred_t2.clamp(*T2_DISPLAY_RANGE)
        baseline_t1 = baseline_t1.clamp(*T1_DISPLAY_RANGE)
        baseline_t2 = baseline_t2.clamp(*T2_DISPLAY_RANGE)
        target_t1 = torch.where(metric_mask[:, None, None], target_t1, torch.zeros_like(target_t1))
        target_t2 = torch.where(metric_mask[:, None, None], target_t2, torch.zeros_like(target_t2))
        pred_t1 = torch.where(metric_mask[:, None, None], pred_t1, torch.zeros_like(pred_t1))
        pred_t2 = torch.where(metric_mask[:, None, None], pred_t2, torch.zeros_like(pred_t2))
        baseline_t1 = torch.where(metric_mask[:, None, None], baseline_t1, torch.zeros_like(baseline_t1))
        baseline_t2 = torch.where(metric_mask[:, None, None], baseline_t2, torch.zeros_like(baseline_t2))
        (ssim_t1,) = mr2.operators.functionals.SSIM(
            target_t1,
            metric_mask,
            data_range=torch.tensor(T1_DISPLAY_RANGE[1] - T1_DISPLAY_RANGE[0], device=self.device),
        )(pred_t1)
        (ssim_t2,) = mr2.operators.functionals.SSIM(
            target_t2,
            metric_mask,
            data_range=torch.tensor(T2_DISPLAY_RANGE[1] - T2_DISPLAY_RANGE[0], device=self.device),
        )(pred_t2)
        (baseline_ssim_t1,) = mr2.operators.functionals.SSIM(
            target_t1,
            metric_mask,
            data_range=torch.tensor(T1_DISPLAY_RANGE[1] - T1_DISPLAY_RANGE[0], device=self.device),
        )(baseline_t1)
        (baseline_ssim_t2,) = mr2.operators.functionals.SSIM(
            target_t2,
            metric_mask,
            data_range=torch.tensor(T2_DISPLAY_RANGE[1] - T2_DISPLAY_RANGE[0], device=self.device),
        )(baseline_t2)
        pred_t1_map = pred_t1.squeeze(1, 2)
        pred_t2_map = pred_t2.squeeze(1, 2)
        target_t1_map = target_t1.squeeze(1, 2)
        target_t2_map = target_t2.squeeze(1, 2)
        baseline_t1_map = baseline_t1.squeeze(1, 2)
        baseline_t2_map = baseline_t2.squeeze(1, 2)
        pred_r2_map = pred_r2.squeeze(1, 2)
        rmse_t1 = torch.sqrt(torch.nn.functional.mse_loss(pred_t1_map[metric_mask], target_t1_map[metric_mask]))
        rmse_t2 = torch.sqrt(torch.nn.functional.mse_loss(pred_t2_map[metric_mask], target_t2_map[metric_mask]))
        baseline_rmse_t1 = torch.sqrt(
            torch.nn.functional.mse_loss(baseline_t1_map[metric_mask], target_t1_map[metric_mask])
        )
        baseline_rmse_t2 = torch.sqrt(
            torch.nn.functional.mse_loss(baseline_t2_map[metric_mask], target_t2_map[metric_mask])
        )
        pred_t2_raw_masked = pred_t2_raw.squeeze(1, 2)[metric_mask]
        baseline_t2_raw_masked = baseline_t2_raw.squeeze(1, 2)[metric_mask]
        target_t2_masked = target_t2_map[metric_mask]
        pred_r2_masked = pred_r2_map[metric_mask]
        pred_r2_rate_masked = pred_r2_masked * R2_STATS[1] + R2_STATS[0]
        self.log('val/ssim_t1', ssim_t1, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/ssim_t2', ssim_t2, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/rmse_t1', rmse_t1, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/rmse_t2', rmse_t2, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/ssim_t1_baseline', baseline_ssim_t1, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/ssim_t2_baseline', baseline_ssim_t2, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/rmse_t1_baseline', baseline_rmse_t1, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/rmse_t2_baseline', baseline_rmse_t2, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(
            {f'val/loss_{key}': value.detach() for key, value in components.items()},
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log('val/debug/pred_t2_abs_max', pred_t2_raw_masked.abs().amax(), on_epoch=True, sync_dist=True)
        self.log('val/debug/pred_t2_min', pred_t2_raw_masked.amin(), on_epoch=True, sync_dist=True)
        self.log('val/debug/pred_t2_p99', torch.quantile(pred_t2_raw_masked, 0.99), on_epoch=True, sync_dist=True)
        self.log('val/debug/baseline_t2_abs_max', baseline_t2_raw_masked.abs().amax(), on_epoch=True, sync_dist=True)
        self.log('val/debug/target_t2_abs_max', target_t2_masked.abs().amax(), on_epoch=True, sync_dist=True)
        self.log('val/debug/pred_r2_min', pred_r2_masked.amin(), on_epoch=True, sync_dist=True)
        self.log('val/debug/pred_r2_rate_min', pred_r2_rate_masked.amin(), on_epoch=True, sync_dist=True)
        self.log(
            'val/debug/pred_t2_invalid_fraction',
            (~torch.isfinite(pred_t2_raw_masked)).float().mean(),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            'val/debug/pred_t2_gt_display_max_fraction',
            (pred_t2_raw_masked > T2_DISPLAY_RANGE[1]).float().mean(),
            on_epoch=True,
            sync_dist=True,
        )

        if batch_idx == 0 and self.trainer.is_global_zero:
            self.validation_step_outputs['target_r1'] = batch['r1'][:, None, None].detach().cpu()
            self.validation_step_outputs['target_r2'] = batch['r2'][:, None, None].detach().cpu()
            self.validation_step_outputs['baseline_r1'] = baseline_r1.detach().cpu()
            self.validation_step_outputs['baseline_r2'] = baseline_r2.detach().cpu()
            self.validation_step_outputs['pred_r1'] = pred_r1.detach().cpu()
            self.validation_step_outputs['pred_r2'] = pred_r2.detach().cpu()
            self.validation_step_outputs['mask'] = mask.detach().cpu()

    def log_csm_diagnostics(self, csm: torch.Tensor, prefix: str) -> None:
        """Log coil-wise CSM normalization diagnostics."""
        norm = csm.abs().square().sum(dim=1).sqrt()
        self.log_dict(
            {
                f'{prefix}_norm_min': norm.detach().amin(),
                f'{prefix}_norm_mean': norm.detach().mean(),
                f'{prefix}_norm_max': norm.detach().amax(),
                f'{prefix}_low_norm_fraction': (norm < 1e-3).float().mean(),
                f'{prefix}_finite_fraction': torch.isfinite(csm).float().mean(),
            },
            on_step=self.training,
            on_epoch=not self.training,
            sync_dist=True,
        )

    def on_validation_epoch_end(self) -> None:
        """Log validation images."""
        if not self.trainer.is_global_zero or not self.validation_step_outputs:
            return

        self._log_relaxation_time_figure(
            't1', 'r1', R1_STATS, '$T_1$ [s]', T1_DISPLAY_RANGE, cmap.Colormap('lipari').to_mpl()
        )
        self._log_relaxation_time_figure(
            't2', 'r2', R2_STATS, '$T_2$ [s]', T2_DISPLAY_RANGE, cmap.Colormap('navia').to_mpl()
        )
        self.validation_step_outputs.clear()

    @staticmethod
    def _parameter_mask(
        mask: torch.Tensor,
        target_r1: torch.Tensor,
        target_r2: torch.Tensor,
        target_t1: torch.Tensor,
        target_t2: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mask used for supervised parameter metrics."""
        return (
            mask
            & torch.isfinite(target_r1)
            & torch.isfinite(target_r2)
            & torch.isfinite(target_t1)
            & torch.isfinite(target_t2)
            & (target_t1 >= MIN_SUPERVISED_T1)
            & (target_t2 >= MIN_SUPERVISED_T2)
        )

    def _log_relaxation_time_figure(
        self,
        relaxation_name: str,
        rate_name: str,
        stats: tuple[float, float],
        label: str,
        value_range: tuple[float, float],
        colormap: Colormap,
    ) -> None:
        """Log one parameter figure to W&B."""
        outputs = self.validation_step_outputs
        samples = len(outputs['mask'])
        vmin, vmax = value_range
        fig, axes = plt.subplots(4, samples, figsize=(4 * samples, 16), squeeze=False)
        for sample in range(samples):
            self.result_plot(
                standardized_rate_to_relaxation_time(outputs[f'target_{rate_name}'][sample], stats),
                standardized_rate_to_relaxation_time(outputs[f'baseline_{rate_name}'][sample], stats),
                standardized_rate_to_relaxation_time(outputs[f'pred_{rate_name}'][sample], stats),
                outputs['mask'][sample],
                axes[:, sample],
                label,
                vmin=vmin,
                vmax=vmax,
                cmap=colormap,
            )
        fig.suptitle(f'{label} Epoch {self.current_epoch}')
        self.logger.experiment.log({f'val/images/{relaxation_name}': fig})
        plt.close(fig)

    @staticmethod
    def result_plot(
        target: torch.Tensor,
        baseline: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        axes: Sequence[plt.Axes] | np.ndarray,
        label: str,
        vmax: float | None = None,
        vmin: float = 0.0,
        cmap: str | Colormap = 'viridis',
    ) -> None:
        """Plot target, prediction, and error."""
        target = target.squeeze().cpu()
        baseline = baseline.squeeze().detach().cpu()
        pred = pred.squeeze().detach().cpu()
        mask = mask.squeeze().detach().bool().cpu()
        target[~mask] = torch.nan
        baseline[~mask] = torch.nan
        pred[~mask] = torch.nan
        difference = target - pred
        if vmax is None:
            vmax = np.nanpercentile(target.numpy(), 99)

        im0 = axes[0].imshow(target, vmin=vmin, vmax=vmax, cmap=cmap)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=label)

        im1 = axes[1].imshow(baseline, vmin=vmin, vmax=vmax, cmap=cmap)
        axes[1].set_title('Baseline')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=label)

        im2 = axes[2].imshow(pred, vmin=vmin, vmax=vmax, cmap=cmap)
        axes[2].set_title('PINQI')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label=label)

        diff_vmax = 0.1 * vmax
        im3 = axes[3].imshow(difference, cmap='coolwarm', vmin=-diff_vmax, vmax=diff_vmax)
        axes[3].set_title('PINQI Error')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label=r'$\Delta$ [s]')

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning-rate scheduler."""
        scalar_names = ('lambdas_raw', 'rezero')
        params, scalar_params = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(scalar_name in name for scalar_name in scalar_names):
                scalar_params.append(parameter)
            else:
                params.append(parameter)

        optimizer = torch.optim.AdamW(
            [
                {'params': params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.lr},
                {'params': scalar_params, 'weight_decay': 0.0, 'lr': self.hparams.lr * 5},
            ],
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.hparams.lr, 5 * self.hparams.lr],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            div_factor=20,
            final_div_factor=300,
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


class LogLambdasCallback(pl.Callback):
    """Log learned regularization weights."""

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: PinqiModule,
        _outputs: dict[str, Any],
        _batch: BatchType,
        _batch_idx: int,
    ) -> None:
        """Log scalar PINQI weights every few steps."""
        if trainer.global_step % 10 != 0:
            return
        lambdas = pl_module.pinqi.softplus(pl_module.pinqi.lambdas_raw).detach().cpu()
        for iteration, iteration_lambdas in enumerate(lambdas):
            pl_module.log_dict(
                {
                    f'parameter/lambda_{lambda_idx}_{iteration}': value
                    for lambda_idx, value in enumerate(iteration_lambdas)
                },
                on_step=True,
                on_epoch=False,
            )
