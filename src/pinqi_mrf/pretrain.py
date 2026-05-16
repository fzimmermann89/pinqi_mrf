"""BrainWeb supervised pretraining module for PINQI-MRF."""

from collections.abc import Sequence

import mr2
import torch

from pinqi_mrf.data import BatchType
from pinqi_mrf.losses import ssim_loss
from pinqi_mrf.parameters import R1_STATS, R2_STATS, standardized_rate_to_relaxation_time
from pinqi_mrf.pinqi import ParameterTuple
from pinqi_mrf.train import T1_DISPLAY_RANGE, T2_DISPLAY_RANGE, BaseModule, PretrainHParams


class PretrainModule(BaseModule):
    """Lightning module for BrainWeb supervised PINQI-MRF pretraining."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the pretraining module."""
        super().__init__(**kwargs)
        self.hparams: PretrainHParams  # type: ignore[assignment]

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

    def loss(
        self,
        predictions: Sequence[ParameterTuple],
        batch: BatchType,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute parameter loss in standardized parameter space."""
        components = {
            'base': torch.tensor(0.0, device=self.device),
            'l1': torch.tensor(0.0, device=self.device),
            'ssim': torch.tensor(0.0, device=self.device),
        }
        target_parameters = self._parameters_from_batch(batch)
        if len(target_parameters) < 3:
            raise RuntimeError('Pretraining currently expects at least m0, r1, and r2 target parameters.')
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
        """Extract the ordered parameter tuple from a pretraining batch."""
        return (batch['m0'], batch['r1'], batch['r2'])

    def training_step(self, batch: BatchType, _batch_idx: int) -> torch.Tensor:
        """Run one training step."""
        coil_rhs, csm_ground_truth = self.simulate_coil_rhs(batch)
        csm_initial = csm_ground_truth if self.hparams.use_ground_truth_csm else self.estimate_initial_csm(coil_rhs)
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
        csm_initial = csm_ground_truth if self.hparams.use_ground_truth_csm else self.estimate_initial_csm(coil_rhs)
        if batch_idx == 0:
            self.log_csm_diagnostics(csm_initial, prefix='debug/val_csm')
        if self.subspace_gram is None:
            raise RuntimeError('Acquisition operators have not been initialized.')
        rhs, normal_op = self.pinqi.get_normal_equations(coil_rhs, csm_initial, self.subspace_gram)
        _images, parameters = self(coil_rhs, csm_initial=csm_initial)
        loss, components = self.loss(parameters, batch)
        prediction_parameters = parameters[-1]
        baseline_parameters = self.baseline(rhs, normal_op)
        if len(prediction_parameters) < 3 or len(baseline_parameters) < 3:
            raise RuntimeError('Validation currently expects at least m0, r1, and r2 reconstruction outputs.')
        pred_r1 = prediction_parameters[1]
        pred_r2 = prediction_parameters[2]
        baseline_r1 = baseline_parameters[1]
        baseline_r2 = baseline_parameters[2]
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
