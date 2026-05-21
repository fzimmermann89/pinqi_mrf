"""Backend reconstruction modules for PINQI-MRF."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import einops
import mr2
import torch

ParameterTuple = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class PINQIConfig:
    """Static reconstruction configuration for the PINQI network."""

    csm_refinement_iterations: tuple[int, ...]
    csm_refinement_shape: tuple[int, int]
    csm_refinement_weight: float
    csm_refinement_cg_iterations: int
    phase_normalization_shape: tuple[int, int]
    initial_cg_iterations: int
    initial_cg_regularization: float
    image_cg_iterations: int
    image_cg_tolerance: float
    n_parameter_priors: int


class PINQI(torch.nn.Module):
    """PINQI reconstruction model."""

    prior_weight_scale: torch.Tensor

    def __init__(
        self,
        signalmodel: mr2.operators.SignalModel,
        n_subspace_coefficients: int,
        n_iterations: int,
        n_features_parameter_net: Sequence[int],
        n_features_image_net: Sequence[int],
        parameter_ranges: tuple[tuple[float, float, int], ...],
        csm_refinement_iterations: tuple[int, ...] = (),
        csm_refinement_shape: tuple[int, int] = (64, 64),
        csm_refinement_weight: float = 1e-2,
        csm_refinement_cg_iterations: int = 10,
        phase_normalize_inputs: bool = True,
        phase_normalization_shape: tuple[int, int] = (32, 32),
        initial_cg_iterations: int = 5,
        initial_cg_regularization: float = 1e-2,
        image_cg_iterations: int = 20,
        image_cg_tolerance: float = 1e-5,
        nonlinear_solver_batch_size: int = 8192,
        weight_scaling: bool = False,
        normalize_inputs: bool = False,
        use_image_net: bool = True,
        use_parameter_net: bool = True,
        use_dictionary_match: bool = True,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        if not use_dictionary_match and not use_parameter_net:
            raise ValueError('use_dictionary_match=False requires use_parameter_net=True.')
        self.signalmodel = signalmodel
        if any(iteration < 0 or iteration >= n_iterations for iteration in csm_refinement_iterations):
            raise ValueError('csm_refinement_iterations must be between 0 and n_iterations - 1.')
        self.config = PINQIConfig(
            csm_refinement_iterations=tuple(dict.fromkeys(csm_refinement_iterations)),
            csm_refinement_shape=csm_refinement_shape,
            csm_refinement_weight=csm_refinement_weight,
            csm_refinement_cg_iterations=csm_refinement_cg_iterations,
            phase_normalization_shape=phase_normalization_shape,
            initial_cg_iterations=initial_cg_iterations,
            initial_cg_regularization=initial_cg_regularization,
            image_cg_iterations=image_cg_iterations,
            image_cg_tolerance=image_cg_tolerance,
            n_parameter_priors=len(parameter_ranges) + 1,
        )
        self.parameter_net = mr2.nn.nets.UNet(
            n_dim=2,
            n_channels_in=n_subspace_coefficients * 2,
            n_channels_out=self.config.n_parameter_priors + 1,
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
        self.parameter_net = cast(mr2.nn.nets.UNet, torch.compile(self.parameter_net))
        self.image_net = cast(mr2.nn.nets.UNet, torch.compile(self.image_net))
        lambdas_raw = torch.ones(n_iterations, 2 + self.config.n_parameter_priors)
        self.lambdas_raw = torch.nn.Parameter(lambdas_raw)
        self.softplus = torch.nn.Softplus(beta=5)
        self.weight_scaling = weight_scaling
        self.normalize_inputs = normalize_inputs
        self.phase_normalize_inputs = phase_normalize_inputs
        self.use_image_net = use_image_net
        self.use_parameter_net = use_parameter_net
        self.use_dictionary_match = use_dictionary_match
        self.parameter_ranges = parameter_ranges
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
        prior_weight_scale = [self.nonlinear_solver.norm_y.square().mean().real]
        prior_weight_scale.extend(
            partial.abs().square().sum(dim=0).mean().real for partial in self.nonlinear_solver.partials
        )
        self.register_buffer('prior_weight_scale', torch.stack(prior_weight_scale).to(torch.float32))

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
            max_iterations=self.config.image_cg_iterations,
            tolerance=self.config.image_cg_tolerance,
        )

    def get_csm_refinement_solver(
        self,
        downsampling: mr2.operators.LinearOperator,
        subspace_gram: mr2.operators.LinearOperator,
        regularization_weight: float | torch.Tensor,
        weight_scale: torch.Tensor | None = None,
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
            operator = coil_lift.H @ subspace_gram @ coil_lift
            if weight_scale is not None:
                operator = weight_scale.reciprocal() * operator
            return operator + regularization_weight

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
            if weight_scale is not None:
                rhs = rhs / weight_scale
            return (rhs + regularization_weight * csm_low_initial,)

        return mr2.operators.ConjugateGradientOp(
            operator_factory=operator_factory,
            rhs_factory=rhs_factory,
            max_iterations=self.config.csm_refinement_cg_iterations,
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

    @staticmethod
    def scale_linear_system(
        rhs: torch.Tensor,
        normal_op: mr2.operators.LinearOperator,
        weight_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, mr2.operators.LinearOperator]:
        """Apply optional scalar normalization to a linear system."""
        if weight_scale is None:
            return rhs, normal_op
        return rhs / weight_scale, weight_scale.reciprocal() * normal_op

    @staticmethod
    def estimate_input_scale(signal: torch.Tensor) -> torch.Tensor:
        """Estimate a robust per-sample image scale from coefficient images."""
        coefficient_norm = torch.linalg.vector_norm(signal, dim=1)
        flattened_norm = coefficient_norm.flatten(start_dim=1)
        scale = torch.quantile(flattened_norm, 0.95, dim=1)
        return scale.clamp_min(1e-8)

    def estimate_phase_correction(self, signal: torch.Tensor) -> torch.Tensor:
        """Estimate a smooth unit phase that rotates the dominant coefficient image to real-positive."""
        low_shape = tuple(
            min(size, target)
            for size, target in zip(signal.shape[-2:], self.config.phase_normalization_shape, strict=True)
        )

        def dominant_phase_source(lowres_signal: torch.Tensor) -> torch.Tensor:
            covariance = lowres_signal.flatten(start_dim=2) @ lowres_signal.flatten(start_dim=2).mH
            _, eigenvectors = torch.linalg.eigh(covariance)
            return einops.einsum(
                eigenvectors[..., -1].conj(),
                lowres_signal,
                'batch coeff, batch coeff z t y x -> batch z t y x',
            )

        phase = mr2.utils.apply_lowres(dominant_phase_source, size=low_shape, dim=(-2, -1))(signal)
        phase = phase / phase.abs().clamp_min(1e-8)
        return phase.conj()[:, None]

    def get_parameter_reg(self, image: torch.Tensor, iteration: int) -> ParameterTuple:
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
        m0 = torch.complex(5.0 * torch.tanh(parameters[0]), 5.0 * torch.tanh(parameters[1]))
        return (m0, *(torch.tanh(parameter) for parameter in parameters[2:].unbind()))

    def project_parameters(self, parameters: ParameterTuple) -> ParameterTuple:
        """Project reconstructed parameters into the configured model support."""
        if len(parameters) != len(self.parameter_ranges) + 1:
            raise RuntimeError('Parameter tuple length must match the configured parameter ranges.')
        m0 = torch.complex(parameters[0].real.clamp(-5.0, 5.0), parameters[0].imag.clamp(-5.0, 5.0))
        quantitative = tuple(
            parameter.clamp(minimum, maximum)
            for parameter, (minimum, maximum, _n_steps) in zip(parameters[1:], self.parameter_ranges, strict=True)
        )
        return (m0, *quantitative)

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
        weight_scale: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[ParameterTuple], torch.Tensor]:
        """Apply the unrolled PINQI reconstruction."""
        csm_current = csm_initial
        input_scale: torch.Tensor | None = None
        rhs, normal_op = self.get_normal_equations(coil_rhs, csm_current, subspace_gram)
        rhs, normal_op = self.scale_linear_system(rhs, normal_op, weight_scale)
        images = list(
            mr2.algorithms.optimizers.cg(
                normal_op + self.config.initial_cg_regularization,
                rhs,
                max_iterations=self.config.initial_cg_iterations,
            )
        )
        phase = None
        if self.phase_normalize_inputs:
            phase = self.estimate_phase_correction(images[-1])[:, 0]
            csm_current = csm_initial * einops.rearrange(phase, 'batch z 1 y x -> batch 1 z y x').conj()
            rhs, normal_op = self.get_normal_equations(coil_rhs, csm_current, subspace_gram)
            rhs, normal_op = self.scale_linear_system(rhs, normal_op, weight_scale)
            images = [image * phase[:, None] for image in images]
        if self.normalize_inputs:
            input_scale = self.estimate_input_scale(images[-1]).reciprocal()
            coil_rhs = coil_rhs * mr2.utils.unsqueeze_at(input_scale, dim=1, n=coil_rhs.ndim - 1)
            rhs = rhs * mr2.utils.unsqueeze_at(input_scale, dim=1, n=rhs.ndim - 1)
            images = [image * mr2.utils.unsqueeze_at(input_scale, dim=1, n=image.ndim - 1) for image in images]
        signal = images[-1]
        parameters: list[ParameterTuple] = []
        csm_output = csm_current
        linear_solver = self.get_linear_solver(normal_op)

        if self.config.csm_refinement_iterations:
            downsampling = mr2.operators.DownSamplingOp(
                dim=(-2, -1), domain_shape=tuple(csm_current.shape[-2:]), range_shape=self.config.csm_refinement_shape
            )
            (csm_low_initial,) = downsampling(csm_current)
            csm_refinement_solver = self.get_csm_refinement_solver(
                downsampling=downsampling,
                subspace_gram=subspace_gram,
                regularization_weight=self.config.csm_refinement_weight,
                weight_scale=weight_scale,
            )

        for iteration, (lambda_image, lambda_q, *lambda_parameters) in enumerate(self.softplus(self.lambdas_raw)):
            if self.use_image_net:
                image_reg = self.get_image_reg(images[-1], iteration)
            else:
                lambda_image = torch.zeros_like(lambda_image)
                image_reg = images[-1]
            images.extend(linear_solver(lambda_image, lambda_q, image_reg, signal, rhs))
            if len(lambda_parameters) != self.config.n_parameter_priors:
                raise RuntimeError('Number of parameter prior weights must match the parameter prior tuple.')
            parameter_reg = self.get_parameter_reg(images[-1], iteration) if self.use_parameter_net else None
            if self.weight_scaling:
                lambda_parameters = tuple(
                    lambda_parameter * scale.to(lambda_parameter)
                    for lambda_parameter, scale in zip(
                        lambda_parameters,
                        self.prior_weight_scale,
                        strict=True,
                    )
                )
            if not self.use_parameter_net:
                parameter, signal = self.nonlinear_solver(images[-1].swapaxes(0, 1), return_signal=True)
                parameter = self.project_parameters(parameter)
                signal = signal.swapaxes(0, 1)
            elif self.use_dictionary_match:
                if parameter_reg is None:
                    raise RuntimeError('Parameter prior is required when use_parameter_net=True.')
                parameter, signal = self.nonlinear_solver(
                    images[-1].swapaxes(0, 1),
                    prior=parameter_reg,
                    prior_precision=tuple(lambda_parameters),
                    return_signal=True,
                )
                parameter = self.project_parameters(parameter)
                signal = signal.swapaxes(0, 1)
            else:
                if parameter_reg is None:
                    raise RuntimeError('Direct parameter prediction requires use_parameter_net=True.')
                parameter = self.project_parameters(parameter_reg)
                (signal,) = self.signalmodel(*parameter)
                signal = signal.swapaxes(0, 1)
            if phase is not None:
                parameter = (parameter[0] * phase.conj(), *parameter[1:])
            parameters.append(parameter)
            csm_output = csm_current
            if iteration in self.config.csm_refinement_iterations:
                (csm_low,) = csm_refinement_solver(
                    images[-1], coil_rhs, csm_low_initial, initial_value=(csm_low_initial,)
                )
                (csm_refined,) = downsampling.H(csm_low)
                csm_refined = csm_refined / csm_refined.abs().square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
                alignment = (csm_refined * csm_current.conj()).sum(dim=1, keepdim=True)
                alignment_phase = alignment / alignment.abs().clamp_min(1e-8)
                csm_current = csm_refined * alignment_phase.conj()
                rhs, normal_op = self.get_normal_equations(coil_rhs, csm_current, subspace_gram)
                rhs, normal_op = self.scale_linear_system(rhs, normal_op, weight_scale)
                linear_solver = self.get_linear_solver(normal_op)
        if phase is not None:
            images = [image * phase[:, None].conj() for image in images]
            csm_output = csm_output * einops.rearrange(phase, 'batch z 1 y x -> batch 1 z y x')
        if input_scale is not None:
            m0_scale = mr2.utils.unsqueeze_at(input_scale.reciprocal(), dim=1, n=parameters[-1][0].ndim - 1)
            parameters = [(parameter[0] * m0_scale, *parameter[1:]) for parameter in parameters]
        return images, parameters, csm_output


class BaselineReconstruction(torch.nn.Module):
    """Unregularized CG reconstruction followed by dictionary matching."""

    def __init__(
        self,
        signalmodel: mr2.operators.SignalModel,
        parameter_ranges: tuple[tuple[float, float, int], ...],
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
    ) -> ParameterTuple:
        """Apply unregularized CG and dictionary matching."""
        images = list(mr2.algorithms.optimizers.cg(normal_op, rhs, max_iterations=10))
        return self.dictionary_match(images[-1].swapaxes(0, 1))
