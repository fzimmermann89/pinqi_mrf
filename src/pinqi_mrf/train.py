"""Shared training infrastructure for PINQI-MRF Lightning modules."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import cmap
import einops
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import mr2
import numpy as np
import torch
from matplotlib.colors import Colormap
from mr2.operators.NonUniformFastFourierOp import SubspaceNonUniformFastFourierOpGramOp
from mr2.utils.sliding_window import sliding_window

from pinqi_mrf.acquisition import AcquisitionData, create_radial_trajectory
from pinqi_mrf.data import BatchType, DelicsRadialReferenceBatch, ValidationLoaderType
from pinqi_mrf.losses import ssim_loss
from pinqi_mrf.parameters import (
    R1_STATS,
    R2_STATS,
    relaxation_time_to_standardized_rate,
    standardized_rate_to_relaxation_time,
)
from pinqi_mrf.pinqi import PINQI, BaselineReconstruction, ParameterTuple
from pinqi_mrf.signal_models import SubspaceLookup
from pinqi_mrf.white_kspace_subspace_rhs_sampler import WhiteKspaceSubspaceRhsNoiseSampler

MIN_SUPERVISED_T1 = 0.3
MIN_SUPERVISED_T2 = 0.03
T1_DISPLAY_RANGE = (0.0, 2.0)
T2_DISPLAY_RANGE = (0.020, 0.150)


@dataclass
class RelaxationTimeValidationResult:
    """Validation metrics and maps for one relaxation-time map."""

    target: torch.Tensor
    baseline: torch.Tensor
    prediction: torch.Tensor
    rmse: torch.Tensor
    baseline_rmse: torch.Tensor
    ssim: torch.Tensor
    baseline_ssim: torch.Tensor


@dataclass
class ParameterValidationResult:
    """Shared parameter-validation outputs used for logging and figures."""

    t1: RelaxationTimeValidationResult
    t2: RelaxationTimeValidationResult
    metric_mask: torch.Tensor


@dataclass
class ValidationFigureData:
    """Cached validation outputs for epoch-end figure logging."""

    target_t1: torch.Tensor
    baseline_t1: torch.Tensor
    prediction_t1: torch.Tensor
    target_t2: torch.Tensor
    baseline_t2: torch.Tensor
    prediction_t2: torch.Tensor
    mask: torch.Tensor
    metric_mask: torch.Tensor


class PretrainHParams(Protocol):
    """Typed view of Lightning hyperparameters used by the training modules."""

    n_coils: int
    n_compressed_coils: int | None
    use_ground_truth_csm: bool
    csm_estimation_iterations: int
    csm_estimation_smoothing_width: int
    csm_extrapolate: bool
    use_density_compensation: bool
    weight_scaling: bool
    normalize_inputs: bool
    lr: float
    weight_decay: float
    loss_weights: Sequence[float]
    loss_l1: float
    loss_ssim: float


class FineTuneHParams(PretrainHParams, Protocol):
    """Typed view of fine-tuning-specific hyperparameters."""

    pretrained_checkpoint: Path | None
    pretrained_checkpoint_strict: bool
    teacher_loss_weight: float
    teacher_augmentations_per_epoch: int
    teacher_spatial_augmentation: bool
    teacher_mix_probability: float
    teacher_mix_beta: float
    ssdu_loss_fraction: float
    ssdu_loss_weight: float


class BaseModule(pl.LightningModule):
    """Shared reconstruction plumbing for PINQI-MRF training regimes."""

    def __init__(
        self,
        *,
        signalmodel: mr2.operators.SignalModel | None = None,
        n_subspace_coefficients: int = 10,
        image_size: int = 256,
        n_readout_samples: int = 256,
        n_coils: int = 12,  # noqa: ARG002
        n_compressed_coils: int | None = 8,  # noqa: ARG002
        n_timepoints: int | None = None,
        parameter_stats: tuple[tuple[float, float], ...] = (R1_STATS, R2_STATS),
        parameter_ranges: tuple[tuple[float, float, int], ...] = ((-2.0, 6.5, 200), (-2.0, 9.5, 160)),
        trajectory: str = 'radial',
        use_density_compensation: bool = False,  # noqa: ARG002
        weight_scaling: bool = True,
        normalize_inputs: bool = True,
        n_iterations: int = 4,
        use_ground_truth_csm: bool = False,  # noqa: ARG002
        csm_estimation_iterations: int = 10,  # noqa: ARG002
        csm_estimation_smoothing_width: int = 11,  # noqa: ARG002
        csm_extrapolate: bool = True,  # noqa: ARG002
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
        loss_l1: float = 0.1,  # noqa: ARG002
        loss_ssim: float = 0.0,  # noqa: ARG002
    ) -> None:
        """Initialize the shared reconstruction state."""
        super().__init__()
        self.save_hyperparameters(ignore=['signalmodel'])
        self.signalmodel = SubspaceLookup(
            signalmodel=signalmodel or mr2.operators.models.NeuroMRF(),
            n_subspace_coefficients=n_subspace_coefficients,
            parameter_stats=parameter_stats,
            parameter_ranges=parameter_ranges,
            n_timepoints=n_timepoints,
        )
        if self.n_compressed_coils > self.hparams.n_coils:
            raise ValueError('n_compressed_coils must not exceed n_coils.')
        self.matrix_size = mr2.data.SpatialDimension(1, image_size, image_size)
        match trajectory:
            case 'radial':
                self.acquisition = create_radial_trajectory(
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
            weight_scaling=weight_scaling,
            normalize_inputs=normalize_inputs,
        )
        self.baseline = BaselineReconstruction(
            signalmodel=self.signalmodel,
            parameter_ranges=parameter_ranges,
            nonlinear_solver_batch_size=nonlinear_solver_batch_size,
        )
        self.validation_step_outputs: dict[str, ValidationFigureData] = {}
        self.subspace_gram: SubspaceNonUniformFastFourierOpGramOp | None = None
        self.subspace_gram_scale: torch.Tensor | None = None
        self.noise_sampler: WhiteKspaceSubspaceRhsNoiseSampler | None = None

    @property
    def n_compressed_coils(self) -> int:
        """Return the effective number of coils after optional compression."""
        return self.hparams.n_compressed_coils or self.hparams.n_coils

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
            traj=self.acquisition.trajectory.to(self.device),
        )
        if self.hparams.use_density_compensation:
            weight = self.acquisition.dcf.data.to(self.device)
            subspace_gram = nufft_op.toeplitz(weight=weight, subspace=self.signalmodel.pca, subspace_dim=1)
            noise_subspace_gram = nufft_op.toeplitz(
                weight=weight.square(),
                subspace=self.signalmodel.pca,
                subspace_dim=1,
            )
        else:
            subspace_gram = nufft_op.toeplitz(weight=None, subspace=self.signalmodel.pca, subspace_dim=1)
            noise_subspace_gram = subspace_gram

        self.subspace_gram = cast(SubspaceNonUniformFastFourierOpGramOp, subspace_gram)
        self.subspace_gram_scale = (
            self.estimate_subspace_gram_scale(self.subspace_gram) if self.hparams.weight_scaling else None
        )
        self.noise_sampler = WhiteKspaceSubspaceRhsNoiseSampler(
            cast(SubspaceNonUniformFastFourierOpGramOp, noise_subspace_gram),
            n_coils=self.hparams.n_coils,
        )

    def estimate_subspace_gram_scale(self, subspace_gram: SubspaceNonUniformFastFourierOpGramOp) -> torch.Tensor:
        """Estimate a scalar normalization for a subspace Toeplitz Gram operator."""
        initial_value = torch.randn(
            (
                1,
                self.hparams.n_subspace_coefficients,
                self.matrix_size.z,
                self.matrix_size.y,
                self.matrix_size.x,
            ),
            device=self.device,
            dtype=torch.complex64,
        )
        scale = subspace_gram.operator_norm(initial_value, dim=(1, 2, 3, 4)).amax().real
        return scale.clamp_min(torch.finfo(scale.dtype).eps)

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
                eigenvectors[..., -1].conj(),
                coil_rhs,
                'batch coeff, batch coeff coil z y x -> batch coil z y x',
            )
            csm = mr2.algorithms.csm.inati(
                calibration,
                smoothing_width=self.hparams.csm_estimation_smoothing_width,
                n_iterations=self.hparams.csm_estimation_iterations,
                extrapolate=self.hparams.csm_extrapolate,
            )
            csm = torch.where(torch.isfinite(csm), csm, 0.0)
        return csm

    def forward(
        self,
        coil_rhs: torch.Tensor,
        csm_initial: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[ParameterTuple]]:
        """Apply the PINQI model."""
        if self.subspace_gram is None:
            raise RuntimeError('Acquisition operators have not been initialized.')
        csm_initial = csm_initial if csm_initial is not None else self.estimate_initial_csm(coil_rhs)
        return self.pinqi(coil_rhs, csm_initial, self.subspace_gram, weight_scale=self.subspace_gram_scale)

    def simulate_coil_rhs(self, batch: BatchType) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate compressed coilwise subspace RHS data from synthetic maps."""
        if self.subspace_gram is None or self.noise_sampler is None:
            raise RuntimeError('Acquisition operators have not been initialized.')

        (coefficients,) = self.signalmodel(*self._parameters_from_batch(batch))
        coefficients = coefficients.movedim(0, 1)
        csm_op = mr2.operators.SensitivityOp(batch['csm'][:, None])
        (coil_coefficients,) = csm_op(coefficients)
        (coil_rhs,) = self.subspace_gram(coil_coefficients)
        coil_rhs = coil_rhs + self.noise_sampler(batch['noise_std'])
        return self.compress_coils(coil_rhs, batch['csm'])

    def supervised_parameter_loss(
        self,
        predictions: Sequence[ParameterTuple],
        batch: BatchType,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute supervised synthetic parameter loss in standardized parameter space."""
        components = {
            'base': torch.tensor(0.0, device=self.device),
            'l1': torch.tensor(0.0, device=self.device),
            'ssim': torch.tensor(0.0, device=self.device),
        }
        target_parameters = self._parameters_from_batch(batch)
        if len(target_parameters) < 3:
            raise RuntimeError('Synthetic supervision currently expects at least m0, r1, and r2 target parameters.')
        target_r1 = target_parameters[1]
        target_r2 = target_parameters[2]
        mask = batch['mask'].bool()
        target_t1 = standardized_rate_to_relaxation_time(target_r1, R1_STATS)
        target_t2 = standardized_rate_to_relaxation_time(target_r2, R2_STATS)
        parameter_mask = self._parameter_mask(mask, target_r1, target_r2, target_t1, target_t2)
        final_iteration = len(predictions) - 1
        for iteration, (prediction, weight) in enumerate(zip(predictions, self.hparams.loss_weights, strict=True)):
            prediction_parameters = tuple(parameter.squeeze(1, 2) for parameter in prediction)
            parameter_losses = []
            for target_index, (prediction_parameter, target_parameter) in enumerate(
                zip(prediction_parameters, target_parameters, strict=True)
            ):
                if torch.is_complex(target_parameter):
                    parameter_loss = torch.nn.functional.mse_loss(
                        torch.view_as_real(prediction_parameter[parameter_mask]),
                        torch.view_as_real(target_parameter.to(prediction_parameter.dtype)[parameter_mask]),
                    )
                else:
                    parameter_loss = torch.nn.functional.mse_loss(
                        prediction_parameter[parameter_mask],
                        target_parameter[parameter_mask],
                    )
                parameter_losses.append(0.5 * parameter_loss if target_index == 0 else parameter_loss)
            loss_outside = prediction_parameters[0][~mask].abs().mean()
            components['base'] = components['base'] + weight * (
                sum(parameter_losses, start=torch.zeros((), device=self.device)) + 0.1 * loss_outside
            )
            if iteration == final_iteration:
                component_names = ('m0', 'r1', 'r2')
                for parameter_index, parameter_loss in enumerate(parameter_losses):
                    name = (
                        component_names[parameter_index]
                        if parameter_index < len(component_names)
                        else f'param{parameter_index}'
                    )
                    components[name] = parameter_loss * 2 if parameter_index == 0 else parameter_loss
                components['outside'] = loss_outside
        if self.hparams.loss_l1 or self.hparams.loss_ssim:
            target_quantitative_parameters = tuple(target[:, None, None] for target in target_parameters[1:])
            prediction_quantitative_parameters = predictions[-1][1:]
            if self.hparams.loss_l1:
                losses_l1 = (
                    torch.nn.functional.l1_loss(
                        prediction.squeeze(1, 2)[parameter_mask],
                        target.squeeze(1, 2)[parameter_mask],
                    )
                    for prediction, target in zip(
                        prediction_quantitative_parameters,
                        target_quantitative_parameters,
                        strict=True,
                    )
                )
                components['l1'] = sum(losses_l1, start=torch.tensor(0.0, device=self.device))
            if self.hparams.loss_ssim:
                components['ssim'] = ssim_loss(
                    prediction_quantitative_parameters,
                    target_quantitative_parameters,
                    parameter_mask,
                )
        loss = (
            components['base'] + self.hparams.loss_l1 * components['l1'] + self.hparams.loss_ssim * components['ssim']
        )
        return loss, components

    @staticmethod
    def _parameters_from_batch(batch: BatchType) -> ParameterTuple:
        """Extract the ordered parameter tuple from a synthetic batch."""
        return (batch['m0'], batch['r1'], batch['r2'])

    def build_coil_rhs(
        self,
        kspace: torch.Tensor,
        acquisition: AcquisitionData,
        subspace_basis: torch.Tensor,
        nufft: mr2.operators.NonUniformFastFourierOp,
    ) -> torch.Tensor:
        """Build coilwise subspace RHS for a selected acquisition."""
        if self.hparams.use_density_compensation:
            (kspace,) = acquisition.dcf.as_operator()(kspace)
        (coil_time_images,) = nufft.H(kspace)
        return einops.einsum(
            subspace_basis.conj(),
            coil_time_images,
            'time coeff, batch time coil z y x -> batch coeff coil z y x',
        )

    def reconstruct_full_kspace(
        self,
        kspace: torch.Tensor,
    ) -> tuple[ParameterTuple, ParameterTuple]:
        """Reconstruct PINQI and baseline parameter maps from full k-space."""
        if self.subspace_gram is None:
            raise RuntimeError('Acquisition operators have not been initialized.')
        acquisition = self.acquisition.to(self.device)
        subspace_basis = self.signalmodel.pca.compression_matrix.mH
        if subspace_basis.ndim > 2:
            subspace_basis = subspace_basis.squeeze(tuple(range(subspace_basis.ndim - 2)))
        nufft = mr2.operators.NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=self.matrix_size,
            encoding_matrix=self.matrix_size,
            traj=acquisition.trajectory,
        )
        coil_rhs = self.build_coil_rhs(kspace, acquisition, subspace_basis, nufft)
        csm_initial = self.estimate_initial_csm(coil_rhs)
        rhs, normal_op = self.pinqi.get_normal_equations(coil_rhs, csm_initial, self.subspace_gram)
        rhs, normal_op = self.pinqi.scale_linear_system(rhs, normal_op, self.subspace_gram_scale)
        _images, parameters = self(coil_rhs, csm_initial=csm_initial)
        baseline_parameters = self.baseline(rhs, normal_op)
        return parameters[-1], baseline_parameters

    def log_csm_diagnostics(self, csm: torch.Tensor, prefix: str) -> None:
        """Log coil-wise CSM normalization diagnostics."""
        norm = csm.abs().square().sum(dim=1).sqrt()
        self.log_dict(
            {
                f'{prefix}/norm_min': norm.detach().amin(),
                f'{prefix}/norm_mean': norm.detach().mean(),
                f'{prefix}/norm_max': norm.detach().amax(),
                f'{prefix}/low_norm_fraction': (norm < 1e-3).float().mean(),
                f'{prefix}/finite_fraction': torch.isfinite(csm).float().mean(),
            },
            on_step=self.training,
            on_epoch=not self.training,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def on_validation_epoch_end(self) -> None:
        """Log validation images."""
        if not self.trainer.is_global_zero or not self.validation_step_outputs:
            return
        for loader_id, outputs in self.validation_step_outputs.items():
            self._log_relaxation_time_figure(
                loader_id,
                outputs.target_t1,
                outputs.baseline_t1,
                outputs.prediction_t1,
                outputs.mask,
                outputs.metric_mask,
                't1',
                '$T_1$ [s]',
                T1_DISPLAY_RANGE,
                cmap.Colormap('lipari').to_mpl(),
            )
            self._log_relaxation_time_figure(
                loader_id,
                outputs.target_t2,
                outputs.baseline_t2,
                outputs.prediction_t2,
                outputs.mask,
                outputs.metric_mask,
                't2',
                '$T_2$ [s]',
                T2_DISPLAY_RANGE,
                cmap.Colormap('navia').to_mpl(),
            )
        self.validation_step_outputs.clear()

    def _evaluate_relaxation_time_validation(
        self,
        target: torch.Tensor,
        baseline: torch.Tensor,
        prediction: torch.Tensor,
        metric_mask: torch.Tensor,
        display_range: tuple[float, float],
        *,
        clip_to_target_range: bool = False,
    ) -> RelaxationTimeValidationResult:
        """Evaluate one relaxation-time map against a reference."""
        target_map = target.squeeze(1, 2)
        baseline_map = baseline.squeeze(1, 2)
        prediction_map = prediction.squeeze(1, 2)
        if clip_to_target_range:
            target_min = target_map.masked_fill(~metric_mask, torch.inf).flatten(start_dim=1).amin(dim=1)
            target_max = target_map.masked_fill(~metric_mask, -torch.inf).flatten(start_dim=1).amax(dim=1)
            target_min = mr2.utils.unsqueeze_at(target_min, dim=1, n=target_map.ndim - 1)
            target_max = mr2.utils.unsqueeze_at(target_max, dim=1, n=target_map.ndim - 1)
            prediction_map = prediction_map.clamp(min=target_min, max=target_max)
            baseline_map = baseline_map.clamp(min=target_min, max=target_max)
        target_masked = target_map[metric_mask]
        baseline_masked = baseline_map[metric_mask]
        prediction_masked = prediction_map[metric_mask]
        metric_mask_ssim = metric_mask[:, None]
        target_ssim = target_map.clamp(*display_range)[:, None]
        baseline_ssim = baseline_map.clamp(*display_range)[:, None]
        prediction_ssim = prediction_map.clamp(*display_range)[:, None]
        target_ssim = torch.where(metric_mask_ssim, target_ssim, torch.zeros_like(target_ssim))
        baseline_ssim = torch.where(metric_mask_ssim, baseline_ssim, torch.zeros_like(baseline_ssim))
        prediction_ssim = torch.where(metric_mask_ssim, prediction_ssim, torch.zeros_like(prediction_ssim))
        data_range = torch.tensor(display_range[1] - display_range[0], device=self.device)
        (ssim,) = mr2.operators.functionals.SSIM(target_ssim, metric_mask_ssim, data_range=data_range)(prediction_ssim)
        (baseline_ssim_value,) = mr2.operators.functionals.SSIM(
            target_ssim,
            metric_mask_ssim,
            data_range=data_range,
        )(baseline_ssim)
        return RelaxationTimeValidationResult(
            target=target,
            baseline=baseline,
            prediction=prediction,
            rmse=torch.sqrt(torch.nn.functional.mse_loss(prediction_masked, target_masked)),
            baseline_rmse=torch.sqrt(torch.nn.functional.mse_loss(baseline_masked, target_masked)),
            ssim=ssim,
            baseline_ssim=baseline_ssim_value,
        )

    def evaluate_parameter_validation(
        self,
        prediction_parameters: ParameterTuple,
        baseline_parameters: ParameterTuple,
        mask: torch.Tensor,
        reference_t1: torch.Tensor,
        reference_t2: torch.Tensor,
        *,
        clip_to_target_range: bool = False,
    ) -> ParameterValidationResult:
        """Evaluate reconstruction quality against reference T1/T2 maps."""
        if len(prediction_parameters) < 3 or len(baseline_parameters) < 3:
            raise RuntimeError('Validation currently expects at least m0, r1, and r2 reconstruction outputs.')
        target_t1 = reference_t1
        target_t2 = reference_t2
        target_r1 = relaxation_time_to_standardized_rate(target_t1, R1_STATS)
        target_r2 = relaxation_time_to_standardized_rate(target_t2, R2_STATS)

        pred_t1 = standardized_rate_to_relaxation_time(prediction_parameters[1], R1_STATS)
        pred_t2 = standardized_rate_to_relaxation_time(prediction_parameters[2], R2_STATS)
        baseline_t1 = standardized_rate_to_relaxation_time(baseline_parameters[1], R1_STATS)
        baseline_t2 = standardized_rate_to_relaxation_time(baseline_parameters[2], R2_STATS)

        target_r1_map = target_r1.squeeze(1, 2)
        target_r2_map = target_r2.squeeze(1, 2)
        target_t1_map = target_t1.squeeze(1, 2)
        target_t2_map = target_t2.squeeze(1, 2)
        metric_mask = self._parameter_mask(mask, target_r1_map, target_r2_map, target_t1_map, target_t2_map)
        return ParameterValidationResult(
            t1=self._evaluate_relaxation_time_validation(
                target_t1,
                baseline_t1,
                pred_t1,
                metric_mask,
                T1_DISPLAY_RANGE,
                clip_to_target_range=clip_to_target_range,
            ),
            t2=self._evaluate_relaxation_time_validation(
                target_t2,
                baseline_t2,
                pred_t2,
                metric_mask,
                T2_DISPLAY_RANGE,
                clip_to_target_range=clip_to_target_range,
            ),
            metric_mask=metric_mask,
        )

    def parameter_validation_metrics(
        self,
        prefix: str,
        result: ParameterValidationResult,
        *,
        include_ssim: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return common parameter-validation metrics for one validation loader."""
        metrics = {
            f'{prefix}/rmse_t1': result.t1.rmse,
            f'{prefix}/ssim_t1_baseline': result.t1.baseline_ssim,
            f'{prefix}/rmse_t1_baseline': result.t1.baseline_rmse,
            f'{prefix}/rmse_t2': result.t2.rmse,
            f'{prefix}/ssim_t2_baseline': result.t2.baseline_ssim,
            f'{prefix}/rmse_t2_baseline': result.t2.baseline_rmse,
        }
        if include_ssim:
            metrics[f'{prefix}/ssim_t1'] = result.t1.ssim
            metrics[f'{prefix}/ssim_t2'] = result.t2.ssim
        return metrics

    def _store_validation_images(
        self,
        loader_id: str,
        result: ParameterValidationResult,
        mask: torch.Tensor,
    ) -> None:
        """Store one validation batch for epoch-end figure logging."""
        self.validation_step_outputs[loader_id] = ValidationFigureData(
            target_t1=result.t1.target.detach().cpu(),
            baseline_t1=result.t1.baseline.detach().cpu(),
            prediction_t1=result.t1.prediction.detach().cpu(),
            target_t2=result.t2.target.detach().cpu(),
            baseline_t2=result.t2.baseline.detach().cpu(),
            prediction_t2=result.t2.prediction.detach().cpu(),
            mask=mask.detach().cpu(),
            metric_mask=result.metric_mask.detach().cpu(),
        )

    @staticmethod
    def _ssim_valid_window_fraction(mask: torch.Tensor) -> torch.Tensor:
        """Return the fraction of SSIM windows fully inside the metric mask."""
        mask_ssim = mask[:, None].float()
        windows = sliding_window(mask_ssim, window_shape=(1, 7, 7), dim=(-3, -2, -1))
        valid_windows = ~torch.isclose(windows, torch.tensor(0, dtype=windows.dtype, device=windows.device)).any(
            (-3, -2, -1),
        )
        return valid_windows.float().mean()

    def mask_debug_metrics(self, prefix: str, mask: torch.Tensor, metric_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return validation mask diagnostics."""
        return {
            f'debug/{prefix}/mask_fraction': mask.float().mean(),
            f'debug/{prefix}/metric_mask_fraction': metric_mask.float().mean(),
            f'debug/{prefix}/metric_mask_removed_fraction': (mask & ~metric_mask).float().mean(),
            f'debug/{prefix}/ssim_valid_window_fraction': self._ssim_valid_window_fraction(metric_mask),
        }

    def _synthetic_validation_step(self, batch: BatchType, batch_idx: int, dataloader_idx: int) -> None:
        """Run one synthetic validation step."""
        prefix = f'val_{dataloader_idx}'
        coil_rhs, csm_ground_truth = self.simulate_coil_rhs(batch)
        csm_initial = csm_ground_truth if self.hparams.use_ground_truth_csm else self.estimate_initial_csm(coil_rhs)
        if batch_idx == 0:
            self.log_csm_diagnostics(csm_initial, prefix=f'debug/{prefix}/csm')
        if self.subspace_gram is None:
            raise RuntimeError('Acquisition operators have not been initialized.')
        rhs, normal_op = self.pinqi.get_normal_equations(coil_rhs, csm_initial, self.subspace_gram)
        rhs, normal_op = self.pinqi.scale_linear_system(rhs, normal_op, self.subspace_gram_scale)
        _images, parameters = self(coil_rhs, csm_initial=csm_initial)
        loss, components = self.supervised_parameter_loss(parameters, batch)
        prediction_parameters = parameters[-1]
        baseline_parameters = self.baseline(rhs, normal_op)
        mask = batch['mask'].bool()
        batch_size = len(mask)
        target_t1 = standardized_rate_to_relaxation_time(batch['r1'][:, None, None], R1_STATS)
        target_t2 = standardized_rate_to_relaxation_time(batch['r2'][:, None, None], R2_STATS)
        result = self.evaluate_parameter_validation(
            prediction_parameters,
            baseline_parameters,
            mask,
            target_t1,
            target_t2,
        )
        metric_mask = result.metric_mask
        pred_t2_masked = result.t2.prediction.squeeze(1, 2)[metric_mask]
        baseline_t2_masked = result.t2.baseline.squeeze(1, 2)[metric_mask]
        target_t2_masked = result.t2.target.squeeze(1, 2)[metric_mask]
        pred_r2_masked = prediction_parameters[2].squeeze(1, 2)[metric_mask]
        pred_r2_rate_masked = pred_r2_masked * R2_STATS[1] + R2_STATS[0]
        self.log(
            f'{prefix}/ssim_t1',
            result.t1.ssim,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
            add_dataloader_idx=False,
        )
        self.log(
            f'{prefix}/ssim_t2',
            result.t2.ssim,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
            add_dataloader_idx=False,
        )
        self.log_dict(
            {
                **self.parameter_validation_metrics(prefix, result, include_ssim=False),
                f'{prefix}/loss': loss,
                **{f'{prefix}/loss_{key}': value.detach() for key, value in components.items()},
                **self.mask_debug_metrics(prefix, mask, metric_mask),
                f'debug/{prefix}/pred_t2_abs_max': pred_t2_masked.abs().amax(),
                f'debug/{prefix}/pred_t2_min': pred_t2_masked.amin(),
                f'debug/{prefix}/pred_t2_p99': torch.quantile(pred_t2_masked, 0.99),
                f'debug/{prefix}/baseline_t2_abs_max': baseline_t2_masked.abs().amax(),
                f'debug/{prefix}/target_t2_abs_max': target_t2_masked.abs().amax(),
                f'debug/{prefix}/pred_r2_min': pred_r2_masked.amin(),
                f'debug/{prefix}/pred_r2_rate_min': pred_r2_rate_masked.amin(),
                f'debug/{prefix}/pred_t2_invalid_fraction': (~torch.isfinite(pred_t2_masked)).float().mean(),
                f'debug/{prefix}/pred_t2_gt_display_max_fraction': (
                    pred_t2_masked > T2_DISPLAY_RANGE[1]
                ).float().mean(),
            },
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
            add_dataloader_idx=False,
        )

        if batch_idx == 0 and self.trainer.is_global_zero:
            self._store_validation_images(prefix, result, mask)

    def _kspace_validation_step(
        self,
        batch: DelicsRadialReferenceBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Run one full-k-space validation step."""
        prediction_parameters, baseline_parameters = self.reconstruct_full_kspace(batch['kspace'])
        mask = batch['mask'].to(self.device).bool()
        prefix = f'val_{dataloader_idx}'
        result = self.evaluate_parameter_validation(
            prediction_parameters,
            baseline_parameters,
            mask,
            batch['reference_t1'].to(self.device)[:, None, None],
            batch['reference_t2'].to(self.device)[:, None, None],
            clip_to_target_range=True,
        )
        self.log_dict(
            {
                **self.parameter_validation_metrics(prefix, result),
                **self.mask_debug_metrics(prefix, mask, result.metric_mask),
            },
            on_epoch=True,
            sync_dist=True,
            batch_size=len(mask),
            add_dataloader_idx=False,
        )
        if batch_idx == 0 and self.trainer.is_global_zero:
            self._store_validation_images(prefix, result, mask)

    def validation_loader_type(self, dataloader_idx: int) -> ValidationLoaderType:
        """Return the validation loader type for the active datamodule."""
        datamodule = self.trainer.datamodule
        if datamodule is None or not hasattr(datamodule, 'validation_loader_types'):
            raise RuntimeError('Validation requires a datamodule with validation_loader_types.')
        loader_types = tuple(datamodule.validation_loader_types)
        return loader_types[dataloader_idx]

    def validation_step(
        self,
        batch: BatchType | DelicsRadialReferenceBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Run one explicit validation step for the configured validation loaders."""
        match self.validation_loader_type(dataloader_idx):
            case ValidationLoaderType.SYNTHETIC:
                self._synthetic_validation_step(batch, batch_idx, dataloader_idx)
            case ValidationLoaderType.KSPACE:
                self._kspace_validation_step(batch, batch_idx, dataloader_idx)
            case _:
                raise ValueError(f'Unknown validation loader type at index {dataloader_idx}.')

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
        loader_id: str,
        target: torch.Tensor,
        baseline: torch.Tensor,
        prediction: torch.Tensor,
        mask: torch.Tensor,
        metric_mask: torch.Tensor,
        name: str,
        label: str,
        value_range: tuple[float, float],
        colormap: Colormap,
    ) -> None:
        """Log one parameter figure to W&B."""
        samples = len(mask)
        vmin, vmax = value_range
        fig, axes = plt.subplots(6, samples, figsize=(4 * samples, 24), squeeze=False)
        for sample in range(samples):
            self.result_plot(
                target[sample],
                baseline[sample],
                prediction[sample],
                mask[sample],
                metric_mask[sample],
                axes[:, sample],
                label,
                vmin=vmin,
                vmax=vmax,
                cmap=colormap,
            )
        fig.suptitle(f'{label} Epoch {self.current_epoch}')
        self.logger.experiment.log({f'{loader_id}/images/{name}': fig})
        plt.close(fig)

    @staticmethod
    def result_plot(
        target: torch.Tensor,
        baseline: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        metric_mask: torch.Tensor,
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
        metric_mask = metric_mask.squeeze().detach().bool().cpu()
        target[~metric_mask] = torch.nan
        baseline[~metric_mask] = torch.nan
        pred[~metric_mask] = torch.nan
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
        axes[4].imshow(mask, vmin=0, vmax=1, cmap='gray')
        axes[4].set_title('Dataset Mask')
        axes[4].axis('off')
        axes[5].imshow(metric_mask, vmin=0, vmax=1, cmap='gray')
        axes[5].set_title('Metric Mask')
        axes[5].axis('off')

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
        pl_module: BaseModule,
        _outputs: dict[str, Any],
        _batch: object,
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
