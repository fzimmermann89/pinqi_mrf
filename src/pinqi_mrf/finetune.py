"""DELICS fine-tuning module for PINQI-MRF."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import einops
import mr2
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from mr2.operators.NonUniformFastFourierOp import SubspaceNonUniformFastFourierOpGramOp

from pinqi_mrf.acquisition import AcquisitionData
from pinqi_mrf.data import BatchType, DelicsRadialBatch, FineTuneBatch
from pinqi_mrf.pinqi import ParameterTuple
from pinqi_mrf.pretrain import PretrainHParams
from pinqi_mrf.train import M0_LOSS_WEIGHT, BaseModule

TEACHER_GAMMA_RANGE = (0.8, 1.2)
TEACHER_MAX_GLOBAL_SHIFT = 0.1


class FineTuneHParams(PretrainHParams, Protocol):
    """Typed view of fine-tuning-specific hyperparameters."""

    pretrained_checkpoint: Path | None
    pretrained_checkpoint_strict: bool
    teacher_generated_loss_weight: float
    teacher_augmentations_per_step: int
    teacher_flip_probability: float
    ssdu_loss_fraction: float
    ssdu_loss_weight: float
    ssdu_normal_loss_weight: float
    ssdu_use_density_compensation: bool
    synthetic_anchor_loss_weight: float
    supervised_delics_loss_weight: float


class FineTuneModule(BaseModule):
    """Lightning module for radial k-space fine-tuning."""

    hparams: FineTuneHParams
    current_nufft_in: mr2.operators.NonUniformFastFourierOp | None
    current_nufft_loss: mr2.operators.NonUniformFastFourierOp | None
    current_subspace_gram_in: SubspaceNonUniformFastFourierOpGramOp | None
    current_subspace_gram_loss: SubspaceNonUniformFastFourierOpGramOp | None
    teacher_nufft: mr2.operators.NonUniformFastFourierOp | None

    def __init__(
        self,
        pretrained_checkpoint: Path | None = None,
        pretrained_checkpoint_strict: bool = True,
        teacher_generated_loss_weight: float = 0.3,  # Similar gradient scale: 0.4 with synthetic anchor weight 1.
        teacher_augmentations_per_step: int = 2,
        teacher_flip_probability: float = 0.2,
        ssdu_loss_fraction: float = 0.2,
        ssdu_loss_weight: float = 0.0,  # Similar gradient scale: 0.2 with synthetic anchor weight 1.
        ssdu_normal_loss_weight: float = 0.2,  # Similar gradient scale: 0.1 with synthetic anchor weight 1.
        ssdu_use_density_compensation: bool = True,
        synthetic_anchor_loss_weight: float = 0.5,  # Gradient-scale reference: 1.0.
        supervised_delics_loss_weight: float = 0.0,
        **kwargs: object,
    ) -> None:
        """Initialize the fine-tuning module."""
        super().__init__(**kwargs)  # type: ignore[arg-type]
        if not 0 < ssdu_loss_fraction < 1:
            raise ValueError('ssdu_loss_fraction must be in (0, 1).')
        if teacher_augmentations_per_step < 0:
            raise ValueError('teacher_augmentations_per_step must be non-negative.')
        if not 0 <= teacher_flip_probability <= 1:
            raise ValueError('teacher_flip_probability must be in [0, 1].')
        if ssdu_loss_weight < 0:
            raise ValueError('ssdu_loss_weight must be non-negative.')
        if ssdu_normal_loss_weight < 0:
            raise ValueError('ssdu_normal_loss_weight must be non-negative.')
        if supervised_delics_loss_weight < 0:
            raise ValueError('supervised_delics_loss_weight must be non-negative.')
        self.save_hyperparameters(
            {
                'pretrained_checkpoint': pretrained_checkpoint,
                'pretrained_checkpoint_strict': pretrained_checkpoint_strict,
                'teacher_generated_loss_weight': teacher_generated_loss_weight,
                'teacher_augmentations_per_step': teacher_augmentations_per_step,
                'teacher_flip_probability': teacher_flip_probability,
                'ssdu_loss_fraction': ssdu_loss_fraction,
                'ssdu_loss_weight': ssdu_loss_weight,
                'ssdu_normal_loss_weight': ssdu_normal_loss_weight,
                'ssdu_use_density_compensation': ssdu_use_density_compensation,
                'synthetic_anchor_loss_weight': synthetic_anchor_loss_weight,
                'supervised_delics_loss_weight': supervised_delics_loss_weight,
            }
        )
        self.current_omega_in: torch.Tensor | None = None
        self.current_omega_loss: torch.Tensor | None = None
        self.current_acquisition_in: AcquisitionData | None = None
        self.current_acquisition_loss: AcquisitionData | None = None
        self.current_subspace_basis_in: torch.Tensor | None = None
        self.current_subspace_basis_loss: torch.Tensor | None = None
        self.current_nufft_in: mr2.operators.NonUniformFastFourierOp | None = None
        self.current_nufft_loss: mr2.operators.NonUniformFastFourierOp | None = None
        self.current_subspace_gram_in: SubspaceNonUniformFastFourierOpGramOp | None = None
        self.current_subspace_gram_in_scale: torch.Tensor | None = None
        self.current_subspace_gram_loss: SubspaceNonUniformFastFourierOpGramOp | None = None
        self.current_subspace_gram_loss_scale: torch.Tensor | None = None
        self.teacher_acquisition: AcquisitionData | None = None
        self.teacher_subspace_basis: torch.Tensor | None = None
        self.teacher_nufft: mr2.operators.NonUniformFastFourierOp | None = None
        if pretrained_checkpoint is not None:
            self.load_pretrained_checkpoint(pretrained_checkpoint, strict=pretrained_checkpoint_strict)

    def set_runtime_state(self, name: str, value: object) -> None:
        """Set runtime-only state without registering dynamic operators as DDP-synced modules."""
        self._modules.pop(name, None)
        self._parameters.pop(name, None)
        self._buffers.pop(name, None)
        object.__setattr__(self, name, value)

    def load_pretrained_checkpoint(self, checkpoint_path: Path, *, strict: bool) -> None:
        """Initialize fine-tuning weights from a pretraining checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
            raise ValueError(f'{checkpoint_path} is not a Lightning checkpoint with a state_dict.')
        state_dict = checkpoint['state_dict']
        if not isinstance(state_dict, dict):
            raise TypeError(f'{checkpoint_path} state_dict must be a dictionary.')
        state_dict = dict(state_dict)
        if not any(key.startswith('pinqi_ema.') for key in state_dict):
            state_dict.update(
                {
                    key.replace('pinqi.', 'pinqi_ema.', 1): value
                    for key, value in state_dict.items()
                    if key.startswith('pinqi.')
                }
            )
        incompatible_keys = self.load_state_dict(cast(dict[str, torch.Tensor], state_dict), strict=strict)
        if not strict and (incompatible_keys.missing_keys or incompatible_keys.unexpected_keys):
            missing = ', '.join(incompatible_keys.missing_keys) or 'none'
            unexpected = ', '.join(incompatible_keys.unexpected_keys) or 'none'
            rank_zero_info(f'Loaded pretrained checkpoint with missing keys: {missing}; unexpected keys: {unexpected}')

    def on_fit_start(self) -> None:
        """Initialize online EMA teacher state."""
        super().on_fit_start()
        if (
            self.hparams.teacher_generated_loss_weight
            and self.hparams.pretrained_checkpoint is None
            and self.trainer.ckpt_path is None
        ):
            raise RuntimeError('Teacher fine-tuning requires pretrained_checkpoint or trainer ckpt_path.')
        if self.hparams.teacher_generated_loss_weight:
            self.setup_teacher_operator()

    def on_validation_start(self) -> None:
        """Defer SSDU operator construction to epoch start."""
        super().on_validation_start()

    def on_train_epoch_start(self) -> None:
        """Update the epoch-wise SSDU split and visible normal operator."""
        self.setup_ssdu_split()

    def setup_teacher_operator(self) -> None:
        """Create the full-data teacher NUFFT runtime state."""
        acquisition = self.acquisition.to(self.device)
        subspace_basis = self._subspace_basis(torch.arange(self.signalmodel.n_timepoints))
        nufft = mr2.operators.NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=self.matrix_size,
            encoding_matrix=self.matrix_size,
            traj=acquisition.trajectory.to(self.device),
        )
        self.teacher_acquisition = acquisition
        self.teacher_subspace_basis = subspace_basis
        self.set_runtime_state('teacher_nufft', nufft)

    def _sample_ssdu_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample visible and held-out time indices for the current epoch."""
        n_timepoints = self.signalmodel.n_timepoints
        n_loss = max(1, round(self.hparams.ssdu_loss_fraction * n_timepoints))
        n_loss = min(n_loss, n_timepoints - 1)
        if self._trainer is None:
            epoch = 0
            rank = 0
        else:
            epoch = int(self.trainer.current_epoch)
            rank = int(self.trainer.global_rank)
        seed = (torch.initial_seed() + 1_000_003 * epoch + rank) % (2**63)
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(n_timepoints, generator=generator)
        omega_loss = permutation[:n_loss].sort().values
        omega_in = permutation[n_loss:].sort().values
        return omega_in, omega_loss

    def _subspace_basis(self, indices: torch.Tensor) -> torch.Tensor:
        """Select temporal subspace rows for the current split."""
        basis = self.signalmodel.pca.compression_matrix.mH
        if basis.ndim > 2:
            basis = basis.squeeze(tuple(range(basis.ndim - 2)))
        return basis[indices].to(self.device)

    def setup_ssdu_split(self) -> None:
        """Build the visible SSDU operator for the current epoch."""
        omega_in, omega_loss = self._sample_ssdu_split()
        self.current_omega_in = omega_in
        self.current_omega_loss = omega_loss
        acquisition_in = self.acquisition[omega_in].to(self.device)
        acquisition_loss = self.acquisition[omega_loss].to(self.device)
        basis_in = self._subspace_basis(omega_in)
        basis_loss = self._subspace_basis(omega_loss)
        nufft_in = mr2.operators.NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=self.matrix_size,
            encoding_matrix=self.matrix_size,
            traj=acquisition_in.trajectory.to(self.device),
        )
        nufft_loss = mr2.operators.NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=self.matrix_size,
            encoding_matrix=self.matrix_size,
            traj=acquisition_loss.trajectory.to(self.device),
        )
        weight = acquisition_in.dcf.data if self.hparams.use_density_compensation else None
        subspace_gram = nufft_in.toeplitz(weight=weight, subspace=basis_in, subspace_dim=1)
        weight_loss = acquisition_loss.dcf.data if self.hparams.ssdu_use_density_compensation else None
        self.current_subspace_gram_loss_scale = None
        if self.hparams.ssdu_normal_loss_weight:
            loss_gram = nufft_loss.toeplitz(weight=weight_loss, subspace=basis_loss, subspace_dim=1)
            self.set_runtime_state('current_subspace_gram_loss', loss_gram)
            if self.hparams.weight_scaling:
                self.current_subspace_gram_loss_scale = self.estimate_subspace_gram_scale(loss_gram)
        else:
            self.set_runtime_state('current_subspace_gram_loss', None)
        self.current_acquisition_in = acquisition_in
        self.current_acquisition_loss = acquisition_loss
        self.current_subspace_basis_in = basis_in
        self.current_subspace_basis_loss = basis_loss
        self.set_runtime_state('current_nufft_in', nufft_in)
        self.set_runtime_state('current_nufft_loss', nufft_loss)
        self.set_runtime_state('current_subspace_gram_in', subspace_gram)
        self.current_subspace_gram_in_scale = (
            self.estimate_subspace_gram_scale(self.current_subspace_gram_in) if self.hparams.weight_scaling else None
        )

    def build_visible_coil_rhs(self, kspace: torch.Tensor) -> torch.Tensor:
        """Build coilwise subspace RHS from the current visible k-space split."""
        if (
            self.current_omega_in is None
            or self.current_acquisition_in is None
            or self.current_subspace_basis_in is None
            or self.current_nufft_in is None
        ):
            raise RuntimeError('SSDU split has not been initialized.')
        kspace_in = kspace[:, self.current_omega_in.to(kspace.device)]
        return self.build_coil_rhs(
            kspace_in, self.current_acquisition_in, self.current_subspace_basis_in, self.current_nufft_in
        )

    def synthesize_kspace(self, parameters: ParameterTuple, csm: torch.Tensor) -> torch.Tensor:
        """Synthesize radial k-space from teacher parameter maps and CSMs."""
        (coefficients,) = self.signalmodel(*parameters)
        coefficients = coefficients.movedim(0, 1)
        csm_op = mr2.operators.SensitivityOp(csm[:, None])
        (coil_coefficients,) = csm_op(coefficients)
        basis = self._subspace_basis(torch.arange(self.signalmodel.n_timepoints))
        coil_time_images = einops.einsum(
            basis, coil_coefficients, 'time coeff, batch coeff coil z y x -> batch time coil z y x'
        )
        nufft = mr2.operators.NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=self.matrix_size,
            encoding_matrix=self.matrix_size,
            traj=self.acquisition.trajectory.to(self.device),
        )
        (kspace,) = nufft(coil_time_images)
        return kspace

    def predict_loss_kspace(self, predictions: Sequence[ParameterTuple], csm_initial: torch.Tensor) -> torch.Tensor:
        """Predict held-out k-space from the final parameter estimate."""
        if self.current_subspace_basis_loss is None or self.current_nufft_loss is None:
            raise RuntimeError('SSDU split has not been initialized.')
        (coefficients,) = self.signalmodel(*predictions[-1])
        coefficients = coefficients.movedim(0, 1)
        csm_op = mr2.operators.SensitivityOp(csm_initial[:, None])
        (coil_coefficients,) = csm_op(coefficients)
        coil_time_images = einops.einsum(
            self.current_subspace_basis_loss,
            coil_coefficients,
            'time coeff, batch coeff coil z y x -> batch time coil z y x',
        )
        (kspace_prediction,) = self.current_nufft_loss(coil_time_images)
        return kspace_prediction

    def ssdu_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted SSDU loss on held-out k-space samples."""
        if self.current_acquisition_loss is None:
            raise RuntimeError('SSDU split has not been initialized.')
        residual = prediction - target
        residual_power = residual.abs().square()
        if not self.hparams.ssdu_use_density_compensation:
            return residual_power.mean()
        weight = self.current_acquisition_loss.dcf.data.to(residual_power.device)
        return (weight * residual_power).sum() / weight.sum().clamp_min(torch.finfo(weight.dtype).eps)

    def _apply_teacher_intensity_augmentation(
        self,
        targets: tuple[torch.Tensor, ...],
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Apply global teacher-map intensity augmentations."""
        gamma = torch.empty((targets[0].shape[0],), device=self.device).uniform_(*TEACHER_GAMMA_RANGE)
        shift = torch.empty((targets[0].shape[0], max(len(targets) - 1, 0)), device=self.device).uniform_(
            -TEACHER_MAX_GLOBAL_SHIFT, TEACHER_MAX_GLOBAL_SHIFT
        )
        gamma = mr2.utils.unsqueeze_at(gamma, dim=-1, n=targets[0].ndim - gamma.ndim)
        augmented_targets = [targets[0] * gamma.to(targets[0].dtype)]
        for target_index, target in enumerate(targets[1:], start=1):
            target_mask = mr2.utils.unsqueeze_at(mask, dim=1, n=target.ndim - mask.ndim)
            target_shift = shift[:, target_index - 1]
            target_shift = mr2.utils.unsqueeze_at(
                target_shift,
                dim=-1,
                n=target.ndim - target_shift.ndim,
            )
            augmented_target = torch.where(target_mask, target + target_shift, target)
            augmented_targets.append(augmented_target.clamp(-0.995, 0.995))
        return tuple(augmented_targets)

    def augment_teacher_batch(
        self,
        targets: tuple[torch.Tensor, ...],
        mask: torch.Tensor,
        csm: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """Apply optional independent left-right flips before teacher-map value augmentation."""
        augmented_targets = targets
        augmented_mask = mask
        augmented_csm = csm
        if self.hparams.teacher_flip_probability:
            batch_size = targets[0].shape[0]
            use_flips = torch.rand((batch_size,), device=self.device) < self.hparams.teacher_flip_probability
            flip_components = torch.rand((batch_size, len(targets) + 1), device=self.device) < 0.5
            target_flip_shape = (batch_size,) + (1,) * (targets[0].ndim - 1)
            mask_flip_shape = (batch_size,) + (1,) * (mask.ndim - 1)
            csm_flip_shape = (batch_size,) + (1,) * (csm.ndim - 1)
            augmented_targets = tuple(
                torch.where(
                    (use_flips & flip_components[:, index]).reshape(target_flip_shape),
                    torch.flip(target, dims=(-1,)),
                    target,
                )
                for index, target in enumerate(targets)
            )
            augmented_csm = torch.where(
                (use_flips & flip_components[:, -1]).reshape(csm_flip_shape), torch.flip(csm, dims=(-1,)), csm
            )
            use_flipped_mask = use_flips & flip_components.any(dim=1)
            augmented_mask = torch.where(
                use_flipped_mask.reshape(mask_flip_shape), mask & torch.flip(mask, dims=(-1,)), mask
            )
        augmented_targets = self._apply_teacher_intensity_augmentation(augmented_targets, augmented_mask)
        return augmented_targets, augmented_mask, augmented_csm

    def teacher_parameter_loss(
        self, predictions: Sequence[tuple[torch.Tensor, ...]], targets: tuple[torch.Tensor, ...], mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked parameter loss against teacher maps."""
        prediction_tensors = tuple(parameter.squeeze(1, 2) for parameter in predictions[-1])
        target_tensors = tuple(target.squeeze(1, 2) for target in targets)
        parameter_mask = mask.bool().clone()
        for target in target_tensors:
            if torch.is_complex(target):
                parameter_mask = parameter_mask & torch.isfinite(target.real) & torch.isfinite(target.imag)
            else:
                parameter_mask = parameter_mask & torch.isfinite(target)
        if not parameter_mask.any():
            return torch.zeros((), device=self.device)
        component_losses = self.parameter_component_losses(prediction_tensors, target_tensors, parameter_mask)
        loss = M0_LOSS_WEIGHT * component_losses[0]  # downweight M0
        return sum(component_losses[1:], start=loss)

    def _reconstruct_batch(
        self, batch: DelicsRadialBatch
    ) -> tuple[ParameterTuple, list[ParameterTuple], dict[str, torch.Tensor]]:
        """Run one SSDU reconstruction and return final predictions with held-out loss."""
        if self.current_subspace_gram_in is None:
            self.setup_ssdu_split()
        if self.current_subspace_gram_in is None or self.current_omega_loss is None:
            raise RuntimeError('SSDU split has not been initialized.')
        coil_rhs = self.build_visible_coil_rhs(batch['kspace'])
        csm_initial = self.estimate_initial_csm(coil_rhs)
        _images, parameters, csm_final = self.pinqi(
            coil_rhs,
            csm_initial,
            self.current_subspace_gram_in,
            weight_scale=self.current_subspace_gram_in_scale,
        )
        use_kspace_loss = self.hparams.ssdu_loss_weight > 0
        use_normal_loss = self.hparams.ssdu_normal_loss_weight > 0
        components = {
            'ssdu_kspace': torch.zeros((), device=self.device),
            'ssdu_normal': torch.zeros((), device=self.device),
        }
        if use_kspace_loss or use_normal_loss:
            kspace_loss = batch['kspace'][:, self.current_omega_loss.to(batch['kspace'].device)]
        if use_kspace_loss:
            kspace_prediction = self.predict_loss_kspace(parameters, csm_final)
            components['ssdu_kspace'] = self.ssdu_loss(kspace_prediction, kspace_loss)
        if use_normal_loss:
            if (
                self.current_acquisition_loss is None
                or self.current_subspace_basis_loss is None
                or self.current_nufft_loss is None
                or self.current_subspace_gram_loss is None
            ):
                raise RuntimeError('Normal-space SSDU split has not been initialized.')
            acquisition_loss = self.current_acquisition_loss
            basis_loss = self.current_subspace_basis_loss
            nufft_loss = self.current_nufft_loss
            subspace_gram_loss = self.current_subspace_gram_loss
            (coefficients,) = self.signalmodel(*parameters[-1])
            coefficients = coefficients.movedim(0, 1)
            csm_op = mr2.operators.SensitivityOp(csm_final[:, None])
            (coil_coefficients,) = csm_op(coefficients)
            (predicted_rhs,) = subspace_gram_loss(coil_coefficients)
            if self.hparams.ssdu_use_density_compensation:
                (kspace_loss,) = acquisition_loss.dcf.as_operator()(kspace_loss)
            (loss_time_images,) = nufft_loss.H(kspace_loss)
            target_rhs = einops.einsum(
                basis_loss.conj(),
                loss_time_images,
                'time coeff, batch time coil z y x -> batch coeff coil z y x',
            )
            if self.current_subspace_gram_loss_scale is not None:
                predicted_rhs = predicted_rhs / self.current_subspace_gram_loss_scale
                target_rhs = target_rhs / self.current_subspace_gram_loss_scale
            residual_power = (predicted_rhs - target_rhs).abs().square()
            target_power = target_rhs.abs().square().mean().clamp_min(torch.finfo(target_rhs.real.dtype).eps)
            components['ssdu_normal'] = residual_power.mean() / target_power
        return parameters[-1], parameters, components

    def reconstruct_parameters_from_kspace(self, kspace: torch.Tensor) -> list[ParameterTuple]:
        """Run the student reconstruction from synthetic k-space."""
        if self.current_subspace_gram_in is None:
            self.setup_ssdu_split()
        if self.current_subspace_gram_in is None:
            raise RuntimeError('SSDU split has not been initialized.')
        coil_rhs = self.build_visible_coil_rhs(kspace)
        csm_initial = self.estimate_initial_csm(coil_rhs)
        _images, parameters, _csm_final = self.pinqi(
            coil_rhs,
            csm_initial,
            self.current_subspace_gram_in,
            weight_scale=self.current_subspace_gram_in_scale,
        )
        return parameters

    def reconstruct_parameters_from_teacher_maps(
        self, targets: ParameterTuple, csm: torch.Tensor
    ) -> list[ParameterTuple]:
        """Run the student reconstruction from teacher maps via the visible Toeplitz RHS."""
        if self.current_subspace_gram_in is None:
            self.setup_ssdu_split()
        if self.current_subspace_gram_in is None:
            raise RuntimeError('SSDU split has not been initialized.')
        (coefficients,) = self.signalmodel(*targets)
        coefficients = coefficients.movedim(0, 1)
        csm_op = mr2.operators.SensitivityOp(csm[:, None])
        (coil_coefficients,) = csm_op(coefficients)
        (coil_rhs,) = self.current_subspace_gram_in(coil_coefficients)
        csm_initial = self.estimate_initial_csm(coil_rhs)
        _images, parameters, _csm_final = self.pinqi(
            coil_rhs,
            csm_initial,
            self.current_subspace_gram_in,
            weight_scale=self.current_subspace_gram_in_scale,
        )
        return parameters

    def teacher_generated_loss(
        self,
        batch: DelicsRadialBatch,
        targets: tuple[torch.Tensor, ...],
        csm: torch.Tensor,
    ) -> torch.Tensor:
        """Average supervised losses over teacher-generated k-space samples."""
        n_augmentations = self.hparams.teacher_augmentations_per_step
        if n_augmentations == 0:
            return torch.zeros((), device=self.device)
        mask = batch['mask'].to(self.device).bool()
        loss_sum = torch.zeros((), device=self.device)
        for _augmentation_index in range(n_augmentations):
            augmented_targets, augmented_mask, augmented_csm = self.augment_teacher_batch(targets, mask, csm)
            synthetic_parameters = self.reconstruct_parameters_from_teacher_maps(augmented_targets, augmented_csm)
            loss_sum = loss_sum + self.teacher_parameter_loss(synthetic_parameters, augmented_targets, augmented_mask)
        return loss_sum / n_augmentations

    def supervised_rhs_loss(self, batch: BatchType) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute supervised map loss from a synthesized full-acquisition RHS."""
        coil_rhs, csm_ground_truth = self.simulate_coil_rhs(batch)
        csm_initial = csm_ground_truth if self.hparams.use_ground_truth_csm else self.estimate_initial_csm(coil_rhs)
        _images, parameters, _csm_final = self(coil_rhs, csm_initial=csm_initial)
        return self.supervised_parameter_loss(parameters, batch)

    def teacher_targets_for_batch(self, batch: DelicsRadialBatch) -> tuple[ParameterTuple, torch.Tensor]:
        """Predict teacher targets for a DELICS batch with the online EMA model."""
        if self.teacher_nufft is None:
            self.setup_teacher_operator()
        if (
            self.teacher_acquisition is None
            or self.teacher_subspace_basis is None
            or self.teacher_nufft is None
            or self.subspace_gram is None
        ):
            raise RuntimeError('Online EMA teacher has not been initialized.')
        self.pinqi_ema.eval()
        with torch.no_grad():
            coil_rhs = self.build_coil_rhs(
                batch['kspace'],
                self.teacher_acquisition,
                self.teacher_subspace_basis,
                self.teacher_nufft,
            )
            csm_initial = self.estimate_initial_csm(coil_rhs)
            _images, parameters, csm_final = self.pinqi_ema(
                coil_rhs,
                csm_initial,
                self.subspace_gram,
                weight_scale=self.subspace_gram_scale,
            )
        return tuple(parameter.detach() for parameter in parameters[-1]), csm_final.detach()

    def delics_loss(self, batch: DelicsRadialBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the DELICS fine-tuning loss components for one batch."""
        _final_parameters, parameters, components = self._reconstruct_batch(batch)
        components['teacher_generated'] = torch.zeros((), device=self.device)
        if self.hparams.teacher_generated_loss_weight:
            teacher_targets, teacher_csm = self.teacher_targets_for_batch(batch)
            if self.hparams.teacher_augmentations_per_step:
                components['teacher_generated'] = self.teacher_generated_loss(batch, teacher_targets, teacher_csm)
            else:
                components['teacher_generated'] = self.teacher_parameter_loss(
                    parameters,
                    teacher_targets,
                    batch['mask'],
                )
        loss = (
            self.hparams.ssdu_loss_weight * components['ssdu_kspace']
            + self.hparams.ssdu_normal_loss_weight * components['ssdu_normal']
            + self.hparams.teacher_generated_loss_weight * components['teacher_generated']
        )
        return loss, components

    def training_step(self, batch: FineTuneBatch, _batch_idx: int) -> torch.Tensor:
        """Run one fine-tuning step."""
        delics_loss, components = self.delics_loss(batch['delics'])
        synthetic_loss, synthetic_components = self.supervised_rhs_loss(batch['synthetic'])
        loss = delics_loss + self.hparams.synthetic_anchor_loss_weight * synthetic_loss
        if self.hparams.supervised_delics_loss_weight:
            if 'supervised_delics' not in batch:
                raise RuntimeError('supervised_delics_loss_weight requires supervised_delics_batch_size > 0.')
            supervised_delics_loss, supervised_delics_components = self.supervised_rhs_loss(batch['supervised_delics'])
            loss = loss + self.hparams.supervised_delics_loss_weight * supervised_delics_loss
        else:
            supervised_delics_loss = torch.zeros((), device=self.device)
            supervised_delics_components = {}
        self.log('train/loss', loss, on_epoch=True, prog_bar=True)  # on_step is true automatically in training_step
        self.log('train/ssdu_kspace_loss', components['ssdu_kspace'], on_epoch=True, sync_dist=False)
        if self.hparams.ssdu_normal_loss_weight:
            self.log('train/ssdu_normal_loss', components['ssdu_normal'], on_epoch=True, sync_dist=False)
        if self.hparams.teacher_generated_loss_weight:
            self.log('train/teacher_generated_loss', components['teacher_generated'], on_epoch=True)
        if self.hparams.synthetic_anchor_loss_weight:
            self.log('train/synthetic_loss', synthetic_loss, on_epoch=True)
            self.log_dict(
                {f'debug/train/synthetic_loss_{key}': value.detach() for key, value in synthetic_components.items()},
                on_epoch=True,
            )
        if self.hparams.supervised_delics_loss_weight:
            self.log('train/supervised_delics_loss', supervised_delics_loss, on_epoch=True)
            self.log_dict(
                {
                    f'debug/train/supervised_delics_loss_{key}': value.detach()
                    for key, value in supervised_delics_components.items()
                },
                on_epoch=True,
            )
        return loss
