"""PINQI-MRF model skeleton."""

from collections.abc import Sequence
from typing import Any

import cmap
import einops
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import mr2
import numpy as np
import torch
from matplotlib.colors import Colormap

from pinqi_mrf.acquisition import create_radial_trajectory
from pinqi_mrf.data import BatchType
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
        parameter_ranges: tuple[tuple[float, float, int], ...] = ((-2.0, 6.5, 200), (-2.0, 9.5, 160)),
        nonlinear_solver_batch_size: int = 8192,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.signalmodel = signalmodel
        self.parameter_net = mr2.nn.nets.UNet(
            n_dim=2,
            n_channels_in=n_subspace_coefficients * 2,
            n_channels_out=4,
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
        lambdas_raw = torch.ones(n_iterations, 3)
        lambdas_raw[:, 2] = 0.0
        self.lambdas_raw = torch.nn.Parameter(lambdas_raw)
        self.softplus = torch.nn.Softplus(beta=5)
        self.iteration_embedding = torch.nn.Embedding(n_iterations, 128)
        self.nonlinear_solver = mr2.operators.DifferentiableDictionaryMatchOp(
            self.signalmodel,
            index_of_scaling_parameter=0,
            batch_size=nonlinear_solver_batch_size,
        )
        nonlinear_solver_grid = [
            torch.linspace(minimum, maximum, n_steps) for minimum, maximum, n_steps in parameter_ranges
        ]
        self.nonlinear_solver.append(
            torch.tensor(1.0 + 0j),
            nonlinear_solver_grid[0][:, None],
            nonlinear_solver_grid[1][None, :],
        )

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

        return mr2.operators.ConjugateGradientOp(operator_factory=operator_factory, rhs_factory=rhs_factory)

    def get_parameter_reg(self, image: torch.Tensor, iteration: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return torch.complex(parameters[0], parameters[1]), parameters[2], parameters[3]

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
        rhs: torch.Tensor,
        normal_op: mr2.operators.LinearOperator,
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Apply the unrolled PINQI reconstruction."""
        images = list(mr2.algorithms.optimizers.cg(normal_op, rhs, max_iterations=2))
        signal = images[-1]
        parameters: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        linear_solver = self.get_linear_solver(normal_op)

        for iteration, (lambda_image, lambda_q, lambda_parameter) in enumerate(self.softplus(self.lambdas_raw)):
            image_reg = self.get_image_reg(images[-1], iteration)
            images.extend(linear_solver(lambda_image, lambda_q, image_reg, signal, rhs))
            parameter_reg = self.get_parameter_reg(images[-1], iteration)
            parameter, signal = self.nonlinear_solver(
                images[-1].swapaxes(0, 1),
                prior=parameter_reg,
                prior_precision=lambda_parameter,
                return_signal=True,
            )
            signal = signal.swapaxes(0, 1)
            parameters.append(parameter)
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
        nonlinear_solver_grid = [
            torch.linspace(minimum, maximum, n_steps) for minimum, maximum, n_steps in parameter_ranges
        ]
        self.dictionary_match.append(
            torch.tensor(1.0 + 0j),
            nonlinear_solver_grid[0][:, None],
            nonlinear_solver_grid[1][None, :],
        )

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
        parameter_stats: tuple[tuple[float, float], ...] = (R1_STATS, R2_STATS),
        parameter_ranges: tuple[tuple[float, float, int], ...] = ((-2.0, 6.5, 200), (-2.0, 9.5, 160)),
        trajectory: str = 'radial',
        n_iterations: int = 4,
        n_features_parameter_net: Sequence[int] = (64, 128, 192, 256),
        n_features_image_net: Sequence[int] = (32, 48, 64, 96),
        nonlinear_solver_batch_size: int = 8192,
        lr: float = 3.0e-4,  # noqa: ARG002
        weight_decay: float = 1e-3,  # noqa: ARG002
        loss_weights: Sequence[float] = (0.2, 0.1, 0.1, 0.8),  # noqa: ARG002
    ) -> None:
        """Initialize the Lightning module."""
        super().__init__()
        self.save_hyperparameters(ignore=['signalmodel'])
        self.signalmodel = SubspaceLookup(
            signalmodel=signalmodel or mr2.operators.models.NeuroMRF(),
            n_subspace_coefficients=n_subspace_coefficients,
            parameter_stats=parameter_stats,
            parameter_ranges=parameter_ranges,
        )
        self.n_coils = n_coils
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
            parameter_ranges=parameter_ranges,
            nonlinear_solver_batch_size=nonlinear_solver_batch_size,
        )
        self.baseline = BaselineReconstruction(
            signalmodel=self.signalmodel,
            parameter_ranges=parameter_ranges,
            nonlinear_solver_batch_size=nonlinear_solver_batch_size,
        )
        self.validation_step_outputs: dict[str, torch.Tensor] = {}
        self.subspace_gram: mr2.operators.LinearOperator | None = None
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
        self.subspace_gram = nufft_op.toeplitz(subspace=self.signalmodel.pca, subspace_dim=1)
        self.noise_sampler = WhiteKspaceSubspaceRhsNoiseSampler(
            self.subspace_gram,
            n_coils=self.n_coils,
        )

    def prepare_normal_equations(self, batch: BatchType) -> tuple[torch.Tensor, mr2.operators.LinearOperator]:
        """Create batch-specific normal equations from parameter maps and CSMs."""
        if self.subspace_gram is None or self.noise_sampler is None:
            raise RuntimeError('Acquisition operators have not been initialized.')

        (coefficients,) = self.signalmodel(batch['m0'], batch['r1'], batch['r2'])
        coefficients = einops.rearrange(coefficients, 'coeff batch y x -> batch coeff 1 1 y x')
        csm_op = mr2.operators.SensitivityOp(batch['csm'])
        normal_op = csm_op.H @ self.subspace_gram @ csm_op
        (coil_coefficients,) = csm_op(coefficients)
        (rhs_coilwise,) = self.subspace_gram(coil_coefficients)
        rhs_coilwise = rhs_coilwise + self.noise_sampler(batch['noise_std'])
        (rhs,) = csm_op.H(rhs_coilwise)
        return rhs, normal_op

    def forward(
        self,
        rhs: torch.Tensor,
        normal_op: mr2.operators.LinearOperator,
    ) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Apply the PINQI model."""
        return self.pinqi(rhs, normal_op)

    def loss(
        self,
        predictions: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        batch: BatchType,
    ) -> torch.Tensor:
        """Compute parameter loss in standardized parameter space."""
        loss = torch.tensor(0.0, device=self.device)
        target_m0, target_r1, target_r2, mask = map(
            torch.squeeze, (batch['m0'], batch['r1'], batch['r2'], batch['mask'])
        )
        mask = mask.bool()
        target_t1 = standardized_rate_to_relaxation_time(target_r1, R1_STATS)
        target_t2 = standardized_rate_to_relaxation_time(target_r2, R2_STATS)
        parameter_mask = self._parameter_mask(mask, target_r1, target_r2, target_t1, target_t2)
        log_debug = self.training and self.global_step % 10 == 0
        final_iteration = len(predictions) - 1
        for iteration, (prediction, weight) in enumerate(zip(predictions, self.hparams.loss_weights, strict=True)):
            prediction_m0, prediction_r1, prediction_r2 = map(torch.squeeze, prediction)
            loss_r1 = torch.nn.functional.mse_loss(prediction_r1[parameter_mask], target_r1[parameter_mask])
            loss_r2 = torch.nn.functional.mse_loss(prediction_r2[parameter_mask], target_r2[parameter_mask])
            loss_m0 = torch.nn.functional.mse_loss(
                torch.view_as_real(prediction_m0[parameter_mask]),
                torch.view_as_real(target_m0.to(prediction_m0.dtype)[parameter_mask]),
            )
            loss_outside = prediction_m0[~mask].abs().mean()
            if log_debug and iteration == final_iteration:
                self.log_dict(
                    {
                        'debug/loss_r1': loss_r1.detach(),
                        'debug/loss_r2': loss_r2.detach(),
                        'debug/loss_m0': loss_m0.detach(),
                        'debug/loss_outside': loss_outside.detach(),
                        'debug/parameter_mask_fraction': parameter_mask.float().mean(),
                        'debug/target_t1_min': target_t1[mask].detach().amin(),
                        'debug/target_t2_min': target_t2[mask].detach().amin(),
                        'debug/pred_r1_abs_max': prediction_r1.detach().abs().max(),
                        'debug/pred_r2_abs_max': prediction_r2.detach().abs().max(),
                    },
                    on_step=True,
                    on_epoch=False,
                    rank_zero_only=True,
                )
            loss = loss + weight * (loss_r1 + loss_r2 + 0.5 * loss_m0 + 0.1 * loss_outside)
        return loss

    def training_step(self, batch: BatchType, _batch_idx: int) -> torch.Tensor:
        """Run one training step."""
        rhs, normal_op = self.prepare_normal_equations(batch)
        _images, parameters = self(rhs, normal_op)
        loss = self.loss(parameters, batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int) -> None:
        """Run one validation step."""
        rhs, normal_op = self.prepare_normal_equations(batch)
        _images, parameters = self(rhs, normal_op)
        loss = self.loss(parameters, batch)
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
        target_t1_map = target_t1.squeeze(1, 2)
        target_t2_map = target_t2.squeeze(1, 2)
        metric_mask = mask & torch.isfinite(target_t1_map) & torch.isfinite(target_t2_map)
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
        rmse_t1 = torch.sqrt(torch.nn.functional.mse_loss(pred_t1_map[metric_mask], target_t1_map[metric_mask]))
        rmse_t2 = torch.sqrt(torch.nn.functional.mse_loss(pred_t2_map[metric_mask], target_t2_map[metric_mask]))
        baseline_rmse_t1 = torch.sqrt(
            torch.nn.functional.mse_loss(baseline_t1_map[metric_mask], target_t1_map[metric_mask])
        )
        baseline_rmse_t2 = torch.sqrt(
            torch.nn.functional.mse_loss(baseline_t2_map[metric_mask], target_t2_map[metric_mask])
        )
        self.log('val/ssim_t1', ssim_t1, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/ssim_t2', ssim_t2, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/rmse_t1', rmse_t1, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/rmse_t2', rmse_t2, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/ssim_t1_baseline', baseline_ssim_t1, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/ssim_t2_baseline', baseline_ssim_t2, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/rmse_t1_baseline', baseline_rmse_t1, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/rmse_t2_baseline', baseline_rmse_t2, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True, batch_size=batch_size)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self.validation_step_outputs['target_r1'] = batch['r1'][:, None, None].detach().cpu()
            self.validation_step_outputs['target_r2'] = batch['r2'][:, None, None].detach().cpu()
            self.validation_step_outputs['baseline_r1'] = baseline_r1.detach().cpu()
            self.validation_step_outputs['baseline_r2'] = baseline_r2.detach().cpu()
            self.validation_step_outputs['pred_r1'] = pred_r1.detach().cpu()
            self.validation_step_outputs['pred_r2'] = pred_r2.detach().cpu()
            self.validation_step_outputs['mask'] = mask.detach().cpu()

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

        diff_vmax = np.nanpercentile(difference.abs().numpy(), 90)
        im3 = axes[3].imshow(difference, cmap='coolwarm', vmin=-diff_vmax, vmax=diff_vmax)
        axes[3].set_title('PINQI Error')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

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
        for iteration, (lambda_image, lambda_q, lambda_parameter) in enumerate(lambdas):
            pl_module.log_dict(
                {
                    f'parameter/lambda_image_{iteration}': lambda_image,
                    f'parameter/lambda_q_{iteration}': lambda_q,
                    f'parameter/lambda_parameter_{iteration}': lambda_parameter,
                },
                on_step=True,
                on_epoch=False,
            )
