"""Supervised DELICS PINQI training and held-out-subject evaluation."""

import copy
import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import cmap
import einops
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import mr2
import numpy as np
import torch
import torch._functorch.config
import wandb
from jsonargparse import Namespace
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from matplotlib.colors import Colormap
from mr2.operators.NonUniformFastFourierOp import SubspaceNonUniformFastFourierOpGramOp
from scipy import ndimage

from pinqi_mrf.acquisition import create_radial_trajectory
from pinqi_mrf.baseline import BaselineReconstruction
from pinqi_mrf.data import ParameterBatch, SupervisedDataModule
from pinqi_mrf.delics_frames import DelicsFrameBatch, DelicsFramePreparation
from pinqi_mrf.parameters import LogRelaxationTransform
from pinqi_mrf.pinqi import PINQI, ParameterTuple
from pinqi_mrf.signal_models import SubspaceLookup

MIN_SUPERVISED_T1 = 0.1
MIN_SUPERVISED_T2 = 0.01
M0_LOSS_WEIGHT = 0.5
T1_DISPLAY_RANGE = (0.0, 2.0)
T2_DISPLAY_RANGE = (0.0, 0.200)


@dataclass
class RelaxationTimeValidationResult:
    """Validation metrics and maps for one relaxation-time map."""

    target: torch.Tensor
    baseline: torch.Tensor
    prediction: torch.Tensor
    mae: torch.Tensor
    baseline_mae: torch.Tensor
    nrmse: torch.Tensor
    baseline_nrmse: torch.Tensor
    hfen: torch.Tensor
    baseline_hfen: torch.Tensor
    rmse: torch.Tensor
    baseline_rmse: torch.Tensor
    ssim: torch.Tensor
    baseline_ssim: torch.Tensor
    ssim_plotrange: torch.Tensor
    baseline_ssim_plotrange: torch.Tensor


@dataclass
class ParameterValidationResult:
    """Shared parameter-validation outputs used for logging and figures."""

    t1: RelaxationTimeValidationResult
    t2: RelaxationTimeValidationResult
    metric_mask: torch.Tensor


@dataclass
class ValidationFigureData:
    """Cached validation outputs for epoch-end figure logging."""

    target_m0: torch.Tensor
    baseline_m0: torch.Tensor
    prediction_m0: torch.Tensor
    target_t1: torch.Tensor
    baseline_t1: torch.Tensor
    prediction_t1: torch.Tensor
    target_t2: torch.Tensor
    baseline_t2: torch.Tensor
    prediction_t2: torch.Tensor
    mask: torch.Tensor


class SupervisedModule(pl.LightningModule):
    """Train PINQI on augmented DELICS frames and evaluate the held-out subject."""

    hparams: Any

    def __init__(
        self,
        *,
        signalmodel: mr2.operators.SignalModel | None = None,
        n_subspace_coefficients: int = 10,
        image_size: int = 256,
        n_readout_samples: int = 256,
        n_coils: int = 8,  # noqa: ARG002
        n_compressed_coils: int | None = 8,  # noqa: ARG002
        n_timepoints: int | None = 352,
        relaxation_bounds: tuple[tuple[float, float], ...] = ((0.1, 4.0), (0.01, 2.5)),
        parameter_ranges: tuple[tuple[float, float, int], ...] = ((-0.995, 0.995, 240), (-0.995, 0.995, 200)),
        weight_scaling: bool = True,
        ema_decay: float = 0.97,
        ema_update_interval: int = 4,
        normalize_inputs: bool = True,
        use_image_net: bool = True,
        use_parameter_net: bool = True,
        use_dictionary_match: bool = True,
        n_iterations: int = 4,
        csm_estimation_iterations: int = 10,  # noqa: ARG002
        csm_estimation_smoothing_width: int = 11,  # noqa: ARG002
        csm_extrapolate: bool = True,  # noqa: ARG002
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
        n_features_parameter_net: Sequence[int] = (64, 128, 192, 256),
        n_features_image_net: Sequence[int] = (48, 64, 96, 128),
        nonlinear_solver_batch_size: int = 2048,
        lr: float = 3.0e-4,  # noqa: ARG002
        scalar_lr_multiplier: float = 2.0,  # noqa: ARG002
        weight_decay: float = 1e-3,  # noqa: ARG002
        loss_weights: Sequence[float] = (0.2, 0.1, 0.1, 0.8),  # noqa: ARG002
    ) -> None:
        """Initialize PINQI, its EMA weights and the CG/dictionary baseline."""
        super().__init__()
        if not 0 <= ema_decay <= 1:
            raise ValueError(f'ema_decay must be in [0, 1], got {ema_decay}.')
        if ema_update_interval < 1:
            raise ValueError(f'ema_update_interval must be positive, got {ema_update_interval}.')
        self.save_hyperparameters(ignore=['signalmodel'])
        self.parameter_transform = LogRelaxationTransform(bounds=relaxation_bounds)
        self.signalmodel = SubspaceLookup(
            signalmodel=signalmodel or mr2.operators.models.NeuroMRF(),
            n_subspace_coefficients=n_subspace_coefficients,
            parameter_transform=self.parameter_transform,
            parameter_ranges=parameter_ranges,
            n_timepoints=n_timepoints,
        )
        if self.n_compressed_coils > self.hparams.n_coils:
            raise ValueError('n_compressed_coils must not exceed n_coils.')
        self.matrix_size = mr2.data.SpatialDimension(1, image_size, image_size)
        self.acquisition = create_radial_trajectory(
            n_spokes=self.signalmodel.n_timepoints,
            n_readout_samples=n_readout_samples,
        )
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
            phase_normalize_inputs=phase_normalize_inputs,
            phase_normalization_shape=phase_normalization_shape,
            initial_cg_iterations=initial_cg_iterations,
            initial_cg_regularization=initial_cg_regularization,
            image_cg_iterations=image_cg_iterations,
            image_cg_tolerance=image_cg_tolerance,
            parameter_ranges=parameter_ranges,
            nonlinear_solver_batch_size=nonlinear_solver_batch_size,
            weight_scaling=weight_scaling,
            normalize_inputs=normalize_inputs,
            use_image_net=use_image_net,
            use_parameter_net=use_parameter_net,
            use_dictionary_match=use_dictionary_match,
        )
        self.pinqi_ema = copy.deepcopy(self.pinqi)
        self.pinqi_ema.requires_grad_(False)
        self.pinqi_ema.eval()
        self.baseline = BaselineReconstruction(
            signalmodel=self.signalmodel,
            parameter_ranges=parameter_ranges,
            nonlinear_solver_batch_size=nonlinear_solver_batch_size,
        )
        self.validation_step_outputs: dict[str, list[ValidationFigureData]] = {}
        self.validation_image_batches: dict[str, set[int]] = {}
        self.subspace_gram: SubspaceNonUniformFastFourierOpGramOp | None = None
        self.subspace_gram_scale: torch.Tensor | None = None
        self._frame_preparation: DelicsFramePreparation | None = None

    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Avoid storing runtime acquisition caches in checkpoints."""
        state_dict = checkpoint.get('state_dict')
        if isinstance(state_dict, dict):
            for key in tuple(state_dict):
                if key.startswith('subspace_gram.'):
                    del state_dict[key]

    def on_load_checkpoint(self, checkpoint: dict[str, object]) -> None:
        """Require recorded fold provenance before restoring model weights."""
        if SupervisedDataModule.__qualname__ not in checkpoint:
            raise ValueError(
                'This checkpoint does not record a DELICS fold; start each leave-one-out fold from scratch.'
            )

    @property
    def n_compressed_coils(self) -> int:
        """Return the effective number of coils after optional compression."""
        return self.hparams.n_compressed_coils or self.hparams.n_coils

    def on_fit_start(self) -> None:
        """Create fixed acquisition operators on the training device."""
        self.setup_acquisition()

    def on_validation_start(self) -> None:
        """Create fixed acquisition operators for standalone validation."""
        gc.collect()
        torch.cuda.empty_cache()
        self.setup_acquisition()

    def on_validation_epoch_start(self) -> None:
        """Select quarter, central and three-quarter slices of the held-out subject."""
        self.validation_step_outputs.clear()
        loader = self.trainer.val_dataloaders
        indices = [round((len(loader.dataset) - 1) * fraction) for fraction in (0.25, 0.5, 0.75)]
        self.validation_image_batches = {'val_0': {index // loader.batch_size for index in indices}}

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None,
    ) -> None:
        """Run an optimizer update and update EMA weights at the configured interval."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if (self.global_step + 1) % self.hparams.ema_update_interval:
            return
        with torch.no_grad():
            student_state = self.pinqi.state_dict()
            ema_state = self.pinqi_ema.state_dict()
            for key, ema_value in ema_state.items():
                student_value = student_state[key].detach().to(ema_value.device)
                if torch.is_floating_point(ema_value) or torch.is_complex(ema_value):
                    ema_value.mul_(self.hparams.ema_decay).add_(student_value, alpha=1.0 - self.hparams.ema_decay)
                else:
                    ema_value.copy_(student_value)
        self.pinqi_ema.eval()

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
        subspace_gram = nufft_op.toeplitz(weight=None, subspace=self.signalmodel.pca, subspace_dim=1)

        self.subspace_gram = cast(SubspaceNonUniformFastFourierOpGramOp, subspace_gram)
        self.subspace_gram_scale = (
            self.estimate_subspace_gram_scale(self.subspace_gram) if self.hparams.weight_scaling else None
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
    ) -> tuple[list[torch.Tensor], list[ParameterTuple], torch.Tensor]:
        """Apply the PINQI model."""
        if self.subspace_gram is None:
            raise RuntimeError('Acquisition operators have not been initialized.')
        csm_initial = csm_initial if csm_initial is not None else self.estimate_initial_csm(coil_rhs)
        if not self.training:
            return self.pinqi_ema(coil_rhs, csm_initial, self.subspace_gram, weight_scale=self.subspace_gram_scale)
        return self.pinqi(coil_rhs, csm_initial, self.subspace_gram, weight_scale=self.subspace_gram_scale)

    def supervised_parameter_loss(
        self,
        predictions: Sequence[ParameterTuple],
        batch: ParameterBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Supervise normalized log maps and P90-normalized M0 inside the tissue mask."""
        components = {
            'base': torch.tensor(0.0, device=self.device),
        }
        target_parameters = self._parameters_from_batch(batch)
        mask = batch['mask'].bool()
        target_t1 = batch['t1']
        target_t2 = batch['t2']
        parameter_mask = self._parameter_mask(mask, target_t1, target_t2)
        # Keep M0 supervision independent of each augmented slice's signal amplitude.
        spatial_dims = tuple(range(1, parameter_mask.ndim))
        target_m0 = target_parameters[0].abs()
        m0_scale = (
            torch.stack(
                [
                    torch.quantile(values[valid], 0.9) if valid.any() else values.new_tensor(1.0)
                    for values, valid in zip(target_m0, mask, strict=True)
                ]
            )
            .clamp_min(1e-8)
            .detach()
        )
        m0_scale = m0_scale.reshape(-1, *([1] * (mask.ndim - 1)))
        final_iteration = len(predictions) - 1
        for iteration, (prediction, weight) in enumerate(zip(predictions, self.hparams.loss_weights, strict=True)):
            prediction_parameters = tuple(parameter.squeeze(1, 2) for parameter in prediction)
            m0_difference = torch.where(mask, (prediction_parameters[0].abs() - target_m0) / m0_scale, 0.0)
            count = mask.flatten(1).sum(1).clamp_min(1)
            component_losses = [
                (m0_difference.square().flatten(1).sum(1) / count).mean(),
                *[
                    torch.nn.functional.l1_loss(prediction[parameter_mask], target[parameter_mask])
                    for prediction, target in zip(prediction_parameters[1:], target_parameters[1:], strict=True)
                ],
            ]
            parameter_loss = M0_LOSS_WEIGHT * component_losses[0]
            parameter_loss = parameter_loss + component_losses[1] + component_losses[2]
            outside_m0 = torch.where(mask, 0.0, prediction_parameters[0].abs() / m0_scale)
            outside_count = (~mask).sum(spatial_dims).clamp_min(1)
            loss_outside = (outside_m0.sum(spatial_dims) / outside_count).mean()
            components['base'] = components['base'] + weight * (parameter_loss + 0.1 * loss_outside)
            if iteration == final_iteration:
                component_names = ('m0', 't1', 't2')
                for name, parameter_loss in zip(component_names, component_losses, strict=True):
                    components[name] = parameter_loss
                components['outside'] = loss_outside
        return components['base'], components

    def _parameters_from_batch(self, batch: ParameterBatch) -> ParameterTuple:
        """Extract the ordered parameter tuple from a synthetic batch."""
        t1, t2 = self.parameter_transform.physical_to_normalized(batch['t1'], batch['t2'])
        return (batch['m0'], t1, t2)

    def build_coil_rhs(
        self,
        kspace: torch.Tensor,
        subspace_basis: torch.Tensor,
        nufft: mr2.operators.NonUniformFastFourierOp,
        coilwise: bool = False,
    ) -> torch.Tensor:
        """Build coilwise RHS, retaining the temporal prefix used by the model."""
        n_timepoints = subspace_basis.shape[0]
        if kspace.shape[1] < n_timepoints:
            raise ValueError('K-space has fewer time points than the signal model.')
        kspace = kspace[:, :n_timepoints]
        if coilwise:
            coil_rhs_chunks = []
            for coil_kspace in kspace.unbind(dim=2):
                (coil_time_images,) = nufft.H(coil_kspace[:, :, None])
                coil_rhs_chunks.append(
                    einops.einsum(
                        subspace_basis.conj(),
                        coil_time_images,
                        'time coeff, batch time coil z y x -> batch coeff coil z y x',
                    )
                )
            return torch.cat(coil_rhs_chunks, dim=2)
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
            traj=acquisition.trajectory.to(self.device),
        )
        coil_rhs = self.build_coil_rhs(kspace, subspace_basis, nufft, coilwise=True)
        csm_initial = self.estimate_initial_csm(coil_rhs)
        rhs, normal_op = PINQI.get_normal_equations(coil_rhs, csm_initial, self.subspace_gram)
        rhs, normal_op = PINQI.scale_linear_system(rhs, normal_op, self.subspace_gram_scale)
        _images, parameters, _csm_final = self(coil_rhs, csm_initial=csm_initial)
        baseline_parameters = self.baseline(rhs, normal_op)
        return parameters[-1], baseline_parameters

    def on_validation_epoch_end(self) -> None:
        """Log validation images."""
        if not self.trainer.is_global_zero or not self.validation_step_outputs:
            return
        if not any(isinstance(logger, WandbLogger) for logger in self.loggers):
            self.validation_step_outputs.clear()
            return
        for loader_id, batches in self.validation_step_outputs.items():
            outputs = ValidationFigureData(
                **{
                    field.name: torch.cat([getattr(batch, field.name) for batch in batches])
                    for field in fields(ValidationFigureData)
                }
            )
            self._log_m0_figure(
                loader_id,
                outputs.target_m0,
                outputs.baseline_m0,
                outputs.prediction_m0,
                outputs.mask,
            )
            self._log_relaxation_time_figure(
                loader_id,
                outputs.target_t1,
                outputs.baseline_t1,
                outputs.prediction_t1,
                outputs.mask,
                't1',
                'T1 [s]',
                T1_DISPLAY_RANGE,
                cmap.Colormap('lipari').to_mpl(),
            )
            self._log_relaxation_time_figure(
                loader_id,
                outputs.target_t2,
                outputs.baseline_t2,
                outputs.prediction_t2,
                outputs.mask,
                't2',
                'T2 [s]',
                T2_DISPLAY_RANGE,
                cmap.Colormap('navia').to_mpl(),
            )
        self.validation_step_outputs.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self) -> None:
        """Clear CUDA cache at the end of the training epoch."""
        gc.collect()
        torch.cuda.empty_cache()

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
        target_min = target_map.masked_fill(~metric_mask, torch.inf).flatten(start_dim=1).amin(dim=1)[:, None, None]
        target_max = target_map.masked_fill(~metric_mask, -torch.inf).flatten(start_dim=1).amax(dim=1)[:, None, None]
        metric_baseline_map = baseline_map
        metric_prediction_map = prediction_map
        if clip_to_target_range:
            metric_prediction_map = prediction_map.clamp(min=target_min, max=target_max)
            metric_baseline_map = baseline_map.clamp(min=target_min, max=target_max)
        metric_mask_ssim = metric_mask[:, None]
        target_plot_range = torch.where(metric_mask_ssim, target_map.clamp(*display_range)[:, None], 0.0)
        baseline_plot_range = torch.where(metric_mask_ssim, metric_baseline_map.clamp(*display_range)[:, None], 0.0)
        prediction_plot_range = torch.where(metric_mask_ssim, metric_prediction_map.clamp(*display_range)[:, None], 0.0)
        plot_data_range = torch.tensor(display_range[1] - display_range[0], dtype=target_map.dtype, device=self.device)
        (ssim_plotrange,) = mr2.operators.functionals.SSIM(
            target_plot_range, metric_mask_ssim, data_range=plot_data_range
        )(prediction_plot_range)
        (baseline_ssim_plotrange,) = mr2.operators.functionals.SSIM(
            target_plot_range, metric_mask_ssim, data_range=plot_data_range
        )(baseline_plot_range)
        target_target_range = torch.where(metric_mask_ssim, target_map[:, None], 0.0)
        baseline_target_range = torch.where(metric_mask_ssim, metric_baseline_map[:, None], 0.0)
        prediction_target_range = torch.where(metric_mask_ssim, metric_prediction_map[:, None], 0.0)
        target_data_range = (target_max - target_min).clamp_min(torch.finfo(target_map.dtype).eps)[:, None]
        (ssim,) = mr2.operators.functionals.SSIM(target_target_range, metric_mask_ssim, data_range=target_data_range)(
            prediction_target_range
        )
        (baseline_ssim,) = mr2.operators.functionals.SSIM(
            target_target_range,
            metric_mask_ssim,
            data_range=target_data_range,
        )(baseline_target_range)
        difference = torch.where(metric_mask, metric_prediction_map - target_map, 0.0).flatten(1)
        baseline_difference = torch.where(metric_mask, metric_baseline_map - target_map, 0.0).flatten(1)
        counts = metric_mask.flatten(1).sum(1).clamp_min(1)
        reference_norm = torch.where(metric_mask, target_map, 0.0).flatten(1).norm(dim=1).clamp_min(1e-8)
        return RelaxationTimeValidationResult(
            mae=(difference.abs().sum(1) / counts).mean(),
            baseline_mae=(baseline_difference.abs().sum(1) / counts).mean(),
            nrmse=(difference.norm(dim=1) / reference_norm).mean(),
            baseline_nrmse=(baseline_difference.norm(dim=1) / reference_norm).mean(),
            hfen=self._hfen(metric_prediction_map, target_map, metric_mask),
            baseline_hfen=self._hfen(metric_baseline_map, target_map, metric_mask),
            target=target,
            baseline=baseline,
            prediction=prediction,
            rmse=(difference.square().sum(1) / counts).sqrt().mean(),
            baseline_rmse=(baseline_difference.square().sum(1) / counts).sqrt().mean(),
            ssim=ssim,
            baseline_ssim=baseline_ssim,
            ssim_plotrange=ssim_plotrange,
            baseline_ssim_plotrange=baseline_ssim_plotrange,
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
            raise RuntimeError('Validation currently expects at least m0, T1, and T2 reconstruction outputs.')
        target_t1 = reference_t1
        target_t2 = reference_t2

        pred_t1, pred_t2 = self.parameter_transform.normalized_to_physical(
            prediction_parameters[1],
            prediction_parameters[2],
        )
        baseline_t1, baseline_t2 = self.parameter_transform.normalized_to_physical(
            baseline_parameters[1],
            baseline_parameters[2],
        )

        target_t1_map = target_t1.squeeze(1, 2)
        target_t2_map = target_t2.squeeze(1, 2)
        metric_mask = self._finite_parameter_mask(mask, target_t1_map, target_t2_map)
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
    ) -> dict[str, torch.Tensor]:
        """Return common parameter-validation metrics for one validation loader."""
        metrics = {
            f'{prefix}/mae_t1': result.t1.mae,
            f'{prefix}/mae_t1_baseline': result.t1.baseline_mae,
            f'{prefix}/nrmse_t1': result.t1.nrmse,
            f'{prefix}/nrmse_t1_baseline': result.t1.baseline_nrmse,
            f'{prefix}/hfen_t1': result.t1.hfen,
            f'{prefix}/hfen_t1_baseline': result.t1.baseline_hfen,
            f'{prefix}/rmse_t1': result.t1.rmse,
            f'{prefix}/rmse_t1_baseline': result.t1.baseline_rmse,
            f'{prefix}/ssim_t1': result.t1.ssim,
            f'{prefix}/ssim_t1_baseline': result.t1.baseline_ssim,
            f'{prefix}/ssim_t1_plotrange': result.t1.ssim_plotrange,
            f'{prefix}/ssim_t1_baseline_plotrange': result.t1.baseline_ssim_plotrange,
            f'{prefix}/mae_t2': result.t2.mae,
            f'{prefix}/mae_t2_baseline': result.t2.baseline_mae,
            f'{prefix}/nrmse_t2': result.t2.nrmse,
            f'{prefix}/nrmse_t2_baseline': result.t2.baseline_nrmse,
            f'{prefix}/hfen_t2': result.t2.hfen,
            f'{prefix}/hfen_t2_baseline': result.t2.baseline_hfen,
            f'{prefix}/rmse_t2': result.t2.rmse,
            f'{prefix}/rmse_t2_baseline': result.t2.baseline_rmse,
            f'{prefix}/ssim_t2': result.t2.ssim,
            f'{prefix}/ssim_t2_baseline': result.t2.baseline_ssim,
            f'{prefix}/ssim_t2_plotrange': result.t2.ssim_plotrange,
            f'{prefix}/ssim_t2_baseline_plotrange': result.t2.baseline_ssim_plotrange,
        }
        # HFEN is undefined when no complete 15x15 foreground footprint exists.
        return {key: value for key, value in metrics.items() if 'hfen' not in key or torch.isfinite(value)}

    @staticmethod
    def _hfen(prediction: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return mean per-slice HFEN with the MRpro 15x15, sigma-1.5 LoG definition.

        Filter full maps before retaining windows wholly inside the supplied mask.
        Slices without valid windows or reference high-frequency energy are omitted.
        """
        if min(reference.shape[-2:]) < 15:
            return reference.new_tensor(float('nan'))
        axis = torch.arange(-7, 8, dtype=torch.float64, device=reference.device)
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        radius_squared = x.square() + y.square()
        sigma = 1.5
        kernel = (
            ((radius_squared - 2 * sigma**2) / sigma**4)
            * torch.exp(-radius_squared / (2 * sigma**2))
            / (2 * torch.pi * sigma**2)
        )
        kernel = (kernel - kernel.mean())[None, None]
        valid = torch.nn.functional.conv2d(mask[:, None].double(), torch.ones_like(kernel)) == 225
        predicted_edges = torch.nn.functional.conv2d(prediction[:, None].double(), kernel)
        reference_edges = torch.nn.functional.conv2d(reference[:, None].double(), kernel)
        error = torch.where(valid, predicted_edges - reference_edges, 0.0).flatten(1).norm(dim=1)
        norm = torch.where(valid, reference_edges, 0.0).flatten(1).norm(dim=1)
        return (error / norm)[norm > 0].mean().to(reference.dtype)

    @staticmethod
    def _head_mask(m0: torch.Tensor) -> torch.Tensor:
        """Return display-only head support from reference M0, retaining skull."""
        magnitude = np.nan_to_num(m0.squeeze().abs().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        smooth = ndimage.gaussian_filter(magnitude, sigma=1.0)
        mask = ndimage.binary_opening(smooth > 0.05 * np.quantile(smooth, 0.99), iterations=2)
        mask = ndimage.binary_closing(mask, iterations=2)
        components, _ = ndimage.label(mask)
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        if not sizes.any():
            return torch.zeros_like(m0, dtype=torch.bool)
        mask = ndimage.binary_fill_holes(components == sizes.argmax())
        return torch.from_numpy(ndimage.binary_dilation(mask, iterations=1)).reshape(m0.shape)

    def _store_validation_images(
        self,
        loader_id: str,
        result: ParameterValidationResult,
        mask: torch.Tensor,
        target_m0: torch.Tensor,
        baseline_m0: torch.Tensor,
        prediction_m0: torch.Tensor,
    ) -> None:
        """Append one selected validation batch for epoch-end figure logging."""
        if not any(isinstance(logger, WandbLogger) for logger in self.loggers):
            return
        target_m0 = target_m0.detach().cpu()
        display_mask = torch.stack([self._head_mask(sample) for sample in target_m0]).reshape(mask.shape)
        self.validation_step_outputs.setdefault(loader_id, []).append(
            ValidationFigureData(
                target_m0=target_m0.detach().cpu(),
                baseline_m0=baseline_m0.detach().cpu(),
                prediction_m0=prediction_m0.detach().cpu(),
                target_t1=result.t1.target.detach().cpu(),
                baseline_t1=result.t1.baseline.detach().cpu(),
                prediction_t1=result.t1.prediction.detach().cpu(),
                target_t2=result.t2.target.detach().cpu(),
                baseline_t2=result.t2.baseline.detach().cpu(),
                prediction_t2=result.t2.prediction.detach().cpu(),
                mask=display_mask,
            )
        )

    @staticmethod
    def _parameter_mask(
        mask: torch.Tensor,
        target_t1: torch.Tensor,
        target_t2: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mask used for supervised parameter metrics."""
        return (
            mask
            & torch.isfinite(target_t1)
            & torch.isfinite(target_t2)
            & (target_t1 >= MIN_SUPERVISED_T1)
            & (target_t2 >= MIN_SUPERVISED_T2)
        )

    @staticmethod
    def _finite_parameter_mask(
        mask: torch.Tensor,
        target_t1: torch.Tensor,
        target_t2: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mask used for validation metrics without parameter-range thresholding."""
        return mask & torch.isfinite(target_t1) & torch.isfinite(target_t2)

    def _log_relaxation_time_figure(
        self,
        loader_id: str,
        target: torch.Tensor,
        baseline: torch.Tensor,
        prediction: torch.Tensor,
        mask: torch.Tensor,
        name: str,
        label: str,
        value_range: tuple[float, float],
        colormap: Colormap,
    ) -> None:
        """Log one parameter figure to W&B."""
        samples = len(mask)
        vmin, vmax = value_range
        fig, axes = plt.subplots(4, samples, figsize=(4 * samples, 16), squeeze=False)
        for sample in range(samples):
            self.result_plot(
                target[sample],
                baseline[sample],
                prediction[sample],
                mask[sample],
                axes[:, sample],
                label,
                vmin=vmin,
                vmax=vmax,
                cmap=colormap,
            )
        fig.suptitle(f'{label} Epoch {self.current_epoch}')
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log({f'{loader_id}/images/{name}': wandb.Image(fig)})
        plt.close(fig)

    def _log_m0_figure(
        self,
        loader_id: str,
        target: torch.Tensor,
        baseline: torch.Tensor,
        prediction: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Log one M0 magnitude figure to W&B."""
        samples = len(mask)
        fig, axes = plt.subplots(4, samples, figsize=(4 * samples, 16), squeeze=False)
        for sample in range(samples):
            self.m0_result_plot(
                target[sample],
                baseline[sample],
                prediction[sample],
                mask[sample],
                axes[:, sample],
            )
        fig.suptitle(f'|M0| Epoch {self.current_epoch}')
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log({f'{loader_id}/images/m0_abs': wandb.Image(fig)})
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
        target = target.squeeze().detach().cpu().clone()
        baseline = baseline.squeeze().detach().cpu().clone()
        pred = pred.squeeze().detach().cpu().clone()
        mask = mask.squeeze().detach().bool().cpu()
        target[~mask] = torch.nan
        baseline[~mask] = torch.nan
        pred[~mask] = torch.nan
        difference = pred - target
        if vmax is None:
            vmax = np.nanpercentile(target.numpy(), 99)

        im0 = axes[0].imshow(target, vmin=vmin, vmax=vmax, cmap=cmap)
        axes[0].set_title('Reference')
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
        im3 = axes[3].imshow(difference, cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax)
        axes[3].set_title('PINQI - reference')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label='Error [s]')

    @staticmethod
    def m0_result_plot(
        target: torch.Tensor,
        baseline: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
        axes: Sequence[plt.Axes] | np.ndarray,
    ) -> None:
        """Plot M0 magnitude target, prediction, and error."""
        target_plot = target.squeeze().detach().cpu().abs()
        baseline_plot = baseline.squeeze().detach().cpu().abs()
        pred_plot = pred.squeeze().detach().cpu().abs()
        mask = mask.squeeze().detach().bool().cpu()
        difference = pred_plot - target_plot
        vmax = float(np.nanpercentile(target_plot[mask].numpy(), 99)) if mask.any() else 1.0
        vmin = 0.0
        diff_vmin, diff_vmax = -0.1 * vmax, 0.1 * vmax

        target_plot[~mask] = torch.nan
        baseline_plot[~mask] = torch.nan
        pred_plot[~mask] = torch.nan
        difference[~mask] = torch.nan

        im0 = axes[0].imshow(target_plot, vmin=vmin, vmax=vmax, cmap='gray')
        axes[0].set_title('Reference')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='|M0|')

        im1 = axes[1].imshow(baseline_plot, vmin=vmin, vmax=vmax, cmap='gray')
        axes[1].set_title('Baseline')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='|M0|')

        im2 = axes[2].imshow(pred_plot, vmin=vmin, vmax=vmax, cmap='gray')
        axes[2].set_title('PINQI')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='|M0|')

        im3 = axes[3].imshow(difference, cmap='RdBu_r', vmin=diff_vmin, vmax=diff_vmax)
        axes[3].set_title('PINQI - reference')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label='Error')

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
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
                {
                    'params': scalar_params,
                    'weight_decay': 0.0,
                    'lr': self.hparams.lr * self.hparams.scalar_lr_multiplier,
                },
            ],
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.hparams.lr, self.hparams.scalar_lr_multiplier * self.hparams.lr],
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.1,
            div_factor=20,
            final_div_factor=300,
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    @torch.no_grad()
    def on_after_batch_transfer(self, batch: dict[str, torch.Tensor], dataloader_idx: int) -> object:
        """Consume full frames on CUDA, retaining only subspace data and supervision maps."""
        _ = dataloader_idx
        if not self.training:
            return batch
        acquisition = self.acquisition.to(self.device)
        basis = self.signalmodel.pca.compression_matrix.mH
        if basis.ndim > 2:
            basis = basis.squeeze(tuple(range(basis.ndim - 2)))
        if self._frame_preparation is None:
            nufft = mr2.operators.NonUniformFastFourierOp(
                direction=(-2, -1),
                recon_matrix=self.matrix_size,
                encoding_matrix=self.matrix_size,
                traj=acquisition.trajectory.to(self.device),
            )
            self._frame_preparation = DelicsFramePreparation(nufft, basis)
        rhs, targets = self._frame_preparation(cast(DelicsFrameBatch, batch))
        return {**targets, 'rhs': rhs, 'csm': self.estimate_initial_csm(rhs)}

    def teardown(self, stage: str) -> None:
        """Release native plans before interpreter and CUDA shutdown."""
        _ = stage
        if self._frame_preparation is not None:
            self._frame_preparation.close()
            self._frame_preparation = None

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        """Supervise all reconstruction stages against the augmented DELICS maps."""
        _images, parameters, _csm = self(batch['rhs'], csm_initial=batch['csm'])
        loss, _components = self.supervised_parameter_loss(parameters, cast(ParameterBatch, batch))
        self.log('train/loss', loss, on_epoch=True, prog_bar=True, batch_size=len(batch['mask']))
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Report unrestricted errors as primary metrics and clipped errors separately."""
        predictions, baseline = self.reconstruct_full_kspace(batch['kspace'])
        mask = batch['mask'].to(self.device).bool()
        prefix = 'val_0'
        for clipped, metric_prefix in [(False, prefix), (True, f'{prefix}_clipped')]:
            result = self.evaluate_parameter_validation(
                predictions,
                baseline,
                mask,
                batch['t1'].to(self.device)[:, None, None],
                batch['t2'].to(self.device)[:, None, None],
                clip_to_target_range=clipped,
            )
            self.log_dict(
                self.parameter_validation_metrics(metric_prefix, result),
                on_epoch=True,
                sync_dist=True,
                batch_size=len(mask),
                add_dataloader_idx=False,
            )
            if (
                not clipped
                and batch_idx in self.validation_image_batches.get(prefix, set())
                and self.trainer.is_global_zero
            ):
                self._store_validation_images(
                    prefix,
                    result,
                    mask,
                    batch['m0'].to(self.device)[:, None, None],
                    baseline[0],
                    predictions[0],
                )


class DelicsCLI(LightningCLI):
    """Run one leave-one-subject-out fold with fixed-epoch checkpoint selection."""

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Expose only the supported fit and validate commands."""
        return {
            name: arguments for name, arguments in LightningCLI.subcommands().items() if name in ('fit', 'validate')
        }

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Link acquisition dimensions and add fold output and tracking options."""
        parser.add_argument('--output', type=Path, default=Path('runs/delics'))
        parser.add_argument('--wandb', action='store_true')
        for source, target in [('size', 'image_size'), ('n_coils', 'n_coils'), ('n_timepoints', 'n_timepoints')]:
            parser.link_arguments(f'data.{source}', f'model.{target}')
        parser.link_arguments('data.n_coils', 'model.n_compressed_coils')

    def before_instantiate_classes(self) -> None:
        """Separate fold artifacts and keep the held-out subject out of fitting."""
        config = getattr(self.config, self.subcommand)
        run_dir = Path(config.output) / config.data.held_out_case
        if self.subcommand == 'validate' and not config.ckpt_path:
            raise ValueError('Validation requires --ckpt_path for this fold.')
        if self.subcommand == 'fit' and not config.ckpt_path and any((run_dir / 'checkpoints').glob('*')):
            raise ValueError('This fold already has checkpoints; resume with --ckpt_path or choose a new --output.')
        loggers = [
            Namespace(
                class_path='lightning.pytorch.loggers.CSVLogger',
                init_args=Namespace(save_dir=str(run_dir), name='csv'),
            )
        ]
        if config.wandb:
            loggers.append(
                Namespace(
                    class_path='lightning.pytorch.loggers.WandbLogger',
                    init_args=Namespace(project='pinqi-mrf', name=config.data.held_out_case, save_dir=str(run_dir)),
                )
            )
        config.trainer.default_root_dir = str(run_dir)
        config.trainer.logger = loggers
        config.trainer.callbacks = [
            Namespace(
                class_path='lightning.pytorch.callbacks.ModelCheckpoint',
                init_args=Namespace(dirpath=str(run_dir / 'checkpoints'), save_last=True, save_top_k=0),
            ),
            Namespace(
                class_path='lightning.pytorch.callbacks.LearningRateMonitor',
                init_args=Namespace(logging_interval='step'),
            ),
        ]
        if self.subcommand == 'fit':
            config.trainer.limit_val_batches = 0
            config.trainer.num_sanity_val_steps = 0


def main() -> None:
    """Train a fold or evaluate its final checkpoint."""
    torch.set_float32_matmul_precision('high')
    torch._functorch.config.activation_memory_budget = 0.7
    DelicsCLI(
        SupervisedModule,
        SupervisedDataModule,
        seed_everything_default=42,
        save_config_kwargs={'overwrite': True},
    )


if __name__ == '__main__':
    main()
