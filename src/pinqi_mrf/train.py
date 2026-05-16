"""Shared training infrastructure for PINQI-MRF Lightning modules."""

from collections.abc import Sequence
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

from pinqi_mrf.acquisition import create_radial_trajectory
from pinqi_mrf.parameters import R1_STATS, R2_STATS, standardized_rate_to_relaxation_time
from pinqi_mrf.pinqi import PINQI, BaselineReconstruction, ParameterTuple
from pinqi_mrf.signal_models import SubspaceLookup
from pinqi_mrf.white_kspace_subspace_rhs_sampler import WhiteKspaceSubspaceRhsNoiseSampler

MIN_SUPERVISED_T1 = 0.5
MIN_SUPERVISED_T2 = 0.03
T1_DISPLAY_RANGE = (0.0, 2.0)
T2_DISPLAY_RANGE = (0.020, 0.150)


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
        n_coils: int = 8,  # noqa: ARG002
        n_compressed_coils: int | None = None,  # noqa: ARG002
        n_timepoints: int | None = None,
        parameter_stats: tuple[tuple[float, float], ...] = (R1_STATS, R2_STATS),
        parameter_ranges: tuple[tuple[float, float, int], ...] = ((-2.0, 6.5, 200), (-2.0, 9.5, 160)),
        trajectory: str = 'radial',
        use_density_compensation: bool = True,  # noqa: ARG002
        weight_scaling: bool = False,
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
        loss_l1: float = 0.0,  # noqa: ARG002
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
        )
        self.baseline = BaselineReconstruction(
            signalmodel=self.signalmodel,
            parameter_ranges=parameter_ranges,
            nonlinear_solver_batch_size=nonlinear_solver_batch_size,
        )
        self.validation_step_outputs: dict[str, torch.Tensor] = {}
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
            't1',
            'r1',
            R1_STATS,
            '$T_1$ [s]',
            T1_DISPLAY_RANGE,
            cmap.Colormap('lipari').to_mpl(),
        )
        self._log_relaxation_time_figure(
            't2',
            'r2',
            R2_STATS,
            '$T_2$ [s]',
            T2_DISPLAY_RANGE,
            cmap.Colormap('navia').to_mpl(),
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
