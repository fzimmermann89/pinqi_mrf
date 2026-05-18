"""DELICS fine-tuning module for PINQI-MRF."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import einops
import mr2
import torch
import torch.distributed as dist
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from mr2.operators.NonUniformFastFourierOp import SubspaceNonUniformFastFourierOpGramOp

from pinqi_mrf.acquisition import AcquisitionData
from pinqi_mrf.data import BatchType, DelicsRadialBatch, FineTuneBatch
from pinqi_mrf.pinqi import ParameterTuple
from pinqi_mrf.train import BaseModule, FineTuneHParams

TEACHER_GAMMA_RANGE = (0.8, 1.2)
TEACHER_MAX_GLOBAL_SHIFT = 0.1


@dataclass(frozen=True)
class TeacherSlice:
    """Teacher pseudo-reference data for one DELICS slice."""

    targets: tuple[torch.Tensor, ...]
    mask: torch.Tensor
    csm: torch.Tensor


class FineTuneModule(BaseModule):
    """Lightning module for radial k-space fine-tuning."""

    def __init__(
        self,
        pretrained_checkpoint: Path | None = None,
        pretrained_checkpoint_strict: bool = True,
        teacher_generated_loss_weight: float = 1.0,
        teacher_augmentations_per_epoch: int = 1,
        teacher_spatial_augmentation: bool = False,
        teacher_mix_probability: float = 0.25,
        teacher_mix_beta: float = 0.4,
        ssdu_loss_fraction: float = 0.2,
        ssdu_loss_weight: float = 1.0,
        synthetic_anchor_loss_weight: float = 1.0,
        **kwargs: object,
    ) -> None:
        """Initialize the fine-tuning module."""
        super().__init__(**kwargs)
        if not 0 < ssdu_loss_fraction < 1:
            raise ValueError('ssdu_loss_fraction must be in (0, 1).')
        if teacher_augmentations_per_epoch < 0:
            raise ValueError('teacher_augmentations_per_epoch must be non-negative.')
        if not 0 <= teacher_mix_probability <= 1:
            raise ValueError('teacher_mix_probability must be in [0, 1].')
        if teacher_mix_beta <= 0:
            raise ValueError('teacher_mix_beta must be positive.')
        self.save_hyperparameters(
            {
                'pretrained_checkpoint': pretrained_checkpoint,
                'pretrained_checkpoint_strict': pretrained_checkpoint_strict,
                'teacher_generated_loss_weight': teacher_generated_loss_weight,
                'teacher_augmentations_per_epoch': teacher_augmentations_per_epoch,
                'teacher_spatial_augmentation': teacher_spatial_augmentation,
                'teacher_mix_probability': teacher_mix_probability,
                'teacher_mix_beta': teacher_mix_beta,
                'ssdu_loss_fraction': ssdu_loss_fraction,
                'ssdu_loss_weight': ssdu_loss_weight,
                'synthetic_anchor_loss_weight': synthetic_anchor_loss_weight,
            }
        )
        self.hparams: FineTuneHParams
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
        self.teacher_cache: dict[tuple[str, int], TeacherSlice] = {}
        if pretrained_checkpoint is not None:
            self.load_pretrained_checkpoint(pretrained_checkpoint, strict=pretrained_checkpoint_strict)

    def load_pretrained_checkpoint(self, checkpoint_path: Path, *, strict: bool) -> None:
        """Initialize fine-tuning weights from a pretraining checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
            raise ValueError(f'{checkpoint_path} is not a Lightning checkpoint with a state_dict.')
        state_dict = checkpoint['state_dict']
        if not isinstance(state_dict, dict):
            raise TypeError(f'{checkpoint_path} state_dict must be a dictionary.')
        incompatible_keys = self.load_state_dict(cast(dict[str, torch.Tensor], state_dict), strict=strict)
        if not strict and (incompatible_keys.missing_keys or incompatible_keys.unexpected_keys):
            missing = ', '.join(incompatible_keys.missing_keys) or 'none'
            unexpected = ', '.join(incompatible_keys.unexpected_keys) or 'none'
            rank_zero_info(f'Loaded pretrained checkpoint with missing keys: {missing}; unexpected keys: {unexpected}')

    def on_fit_start(self) -> None:
        """Defer SSDU operator construction to epoch start."""
        super().on_fit_start()
        if (
            self.hparams.teacher_generated_loss_weight
            and self.hparams.pretrained_checkpoint is None
            and self.trainer.ckpt_path is None
        ):
            raise RuntimeError('Teacher fine-tuning requires pretrained_checkpoint or trainer ckpt_path.')

    def on_validation_start(self) -> None:
        """Defer SSDU operator construction to epoch start."""
        super().on_validation_start()

    def on_train_epoch_start(self) -> None:
        """Update the epoch-wise SSDU split and visible normal operator."""
        self.setup_ssdu_split()
        if self.hparams.teacher_generated_loss_weight:
            self.refresh_teacher_cache()

    def _sample_ssdu_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample visible and held-out time indices for the current epoch."""
        n_timepoints = self.signalmodel.n_timepoints
        n_loss = max(1, round(self.hparams.ssdu_loss_fraction * n_timepoints))
        n_loss = min(n_loss, n_timepoints - 1)
        permutation = torch.randperm(n_timepoints)
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
            traj=acquisition_in.trajectory,
        )
        nufft_loss = mr2.operators.NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=self.matrix_size,
            encoding_matrix=self.matrix_size,
            traj=acquisition_loss.trajectory,
        )
        weight = acquisition_in.dcf.data if self.hparams.use_density_compensation else None
        subspace_gram = nufft_in.toeplitz(weight=weight, subspace=basis_in, subspace_dim=1)
        self.current_acquisition_in = acquisition_in
        self.current_acquisition_loss = acquisition_loss
        self.current_subspace_basis_in = basis_in
        self.current_subspace_basis_loss = basis_loss
        self.current_nufft_in = nufft_in
        self.current_nufft_loss = nufft_loss
        self.current_subspace_gram_in = cast(SubspaceNonUniformFastFourierOpGramOp, subspace_gram)
        self.current_subspace_gram_in_scale = (
            self.estimate_subspace_gram_scale(self.current_subspace_gram_in) if self.hparams.weight_scaling else None
        )

    def refresh_teacher_cache(self) -> None:
        """Predict epoch-level teacher parameter maps for the DELICS training dataset."""
        datamodule = self.trainer.datamodule
        dataset = getattr(datamodule, 'delics_train_dataset', None)
        if dataset is None:
            raise RuntimeError('Teacher refresh requires a datamodule with a delics_train_dataset.')
        world_size = int(getattr(self.trainer, 'world_size', 1))
        global_rank = int(getattr(self.trainer, 'global_rank', 0))

        acquisition = self.acquisition.to(self.device)
        subspace_basis = self._subspace_basis(torch.arange(self.signalmodel.n_timepoints))
        nufft = mr2.operators.NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=self.matrix_size,
            encoding_matrix=self.matrix_size,
            traj=acquisition.trajectory,
        )
        weight = acquisition.dcf.data if self.hparams.use_density_compensation else None
        teacher_subspace_gram = cast(
            SubspaceNonUniformFastFourierOpGramOp,
            nufft.toeplitz(weight=weight, subspace=subspace_basis, subspace_dim=1),
        )
        teacher_weight_scale = self.subspace_gram_scale if self.hparams.weight_scaling else None

        self.teacher_cache.clear()
        was_training = self.training
        self.eval()
        local_teacher_slices: list[tuple[int, TeacherSlice]] = []
        with torch.no_grad():
            for index in range(global_rank, len(dataset), world_size):
                sample = dataset[index]
                kspace = cast(torch.Tensor, sample['kspace'])[None].to(self.device)
                coil_rhs = self.build_coil_rhs(kspace, acquisition, subspace_basis, nufft)
                csm_initial = self.estimate_initial_csm(coil_rhs)
                _images, parameters = self.pinqi(
                    coil_rhs,
                    csm_initial,
                    teacher_subspace_gram,
                    weight_scale=teacher_weight_scale,
                )
                local_teacher_slices.append(
                    (
                        index,
                        TeacherSlice(
                            targets=tuple(parameter[0].detach().cpu() for parameter in parameters[-1]),
                            mask=cast(torch.Tensor, sample['mask']).cpu().bool(),
                            csm=csm_initial[0].detach().cpu(),
                        ),
                    )
                )
        self.train(was_training)
        world_size = int(getattr(self.trainer, 'world_size', 1))
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            gathered_teacher_slices: list[list[tuple[int, TeacherSlice]]] = [[] for _ in range(world_size)]
            dist.all_gather_object(gathered_teacher_slices, local_teacher_slices)
            teacher_slices = [
                teacher_slice
                for rank_teacher_slices in gathered_teacher_slices
                for teacher_slice in rank_teacher_slices
            ]
        else:
            teacher_slices = local_teacher_slices
        for dataset_index, teacher_slice in teacher_slices:
            file_index, slice_index = dataset.samples[dataset_index]
            _path, case_id = dataset.files[file_index]
            self.teacher_cache[(case_id, slice_index)] = teacher_slice
        rank_zero_info(f'Refreshed {len(self.teacher_cache)} DELICS teacher pseudo-reference slices.')

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
        if not self.hparams.use_density_compensation:
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
            augmented_targets.append(torch.where(target_mask, target + target_shift, target))
        return tuple(augmented_targets)

    def augment_teacher_batch(
        self,
        targets: tuple[torch.Tensor, ...],
        mask: torch.Tensor,
        csm: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """Apply fixed CSM flip, optional BrainWeb spatial augmentation, and intensity jitter."""
        csm = torch.flip(csm, dims=(-1,))
        if not self.hparams.teacher_spatial_augmentation:
            return (
                self._apply_teacher_intensity_augmentation(targets, mask),
                mask,
                csm,
            )
        shape = targets[0].shape
        csm_shape = csm.shape
        csm_channels = csm.shape[1] * csm.shape[2]
        target_specs: list[tuple[tuple[int, ...], bool, int]] = []
        channel_parts = []
        for target in targets:
            flattened = target.reshape(shape[0], -1, *shape[-2:])
            if torch.is_complex(target):
                channel_parts.append(flattened.real)
                channel_parts.append(flattened.imag)
            else:
                channel_parts.append(flattened)
            target_specs.append((tuple(target.shape), torch.is_complex(target), flattened.shape[1]))
        channels = torch.cat(
            (
                *channel_parts,
                mask[:, None].float(),
                csm.real.reshape(shape[0], -1, *shape[-2:]),
                csm.imag.reshape(shape[0], -1, *shape[-2:]),
            ),
            dim=1,
        )
        augment = mr2.phantoms.brainweb.augment(size=shape[-1], trim=False)
        augmented = torch.stack([augment(sample) for sample in channels])
        augmented_targets = []
        channel_offset = 0
        for target_shape, is_complex, channel_count in target_specs:
            if is_complex:
                real_part = augmented[:, channel_offset : channel_offset + channel_count]
                imag_part = augmented[:, channel_offset + channel_count : channel_offset + 2 * channel_count]
                augmented_targets.append(torch.complex(real_part, imag_part).reshape(target_shape))
                channel_offset += 2 * channel_count
            else:
                augmented_targets.append(
                    augmented[:, channel_offset : channel_offset + channel_count].reshape(target_shape)
                )
                channel_offset += channel_count
        mask_augmented = augmented[:, channel_offset]
        channel_offset += 1
        csm_real = augmented[:, channel_offset : channel_offset + csm_channels]
        csm_imag = augmented[:, channel_offset + csm_channels : channel_offset + 2 * csm_channels]
        augmented_mask = mask_augmented > 0.5
        augmented_targets = self._apply_teacher_intensity_augmentation(tuple(augmented_targets), augmented_mask)
        augmented_csm = torch.complex(csm_real, csm_imag).reshape(csm_shape)
        return augmented_targets, augmented_mask, augmented_csm

    def maybe_mix_teacher_batch(
        self,
        targets: tuple[torch.Tensor, ...],
        mask: torch.Tensor,
        csm: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """Optionally mix teacher maps with another cached teacher slice."""
        if self.hparams.teacher_mix_probability == 0 or not self.teacher_cache:
            return targets, mask, csm
        if torch.rand((), device=self.device) >= self.hparams.teacher_mix_probability:
            return targets, mask, csm
        keys = sorted(self.teacher_cache)
        key_index = int(torch.randint(len(keys), (), device=self.device).item())
        other_slice = self.teacher_cache[keys[key_index]]
        other_targets = tuple(target[None].to(self.device) for target in other_slice.targets)
        other_mask = other_slice.mask[None].to(self.device).bool()
        concentration = torch.tensor(self.hparams.teacher_mix_beta, device=self.device)
        lam = torch.distributions.Beta(concentration, concentration).sample((1,)).to(self.device)
        lam = mr2.utils.unsqueeze_at(lam, dim=-1, n=targets[0].ndim - lam.ndim)
        mixed_targets = tuple(
            lam * target + (1 - lam) * other for target, other in zip(targets, other_targets, strict=True)
        )
        return mixed_targets, mask & other_mask, csm

    def teacher_parameter_loss(
        self, predictions: Sequence[tuple[torch.Tensor, ...]], targets: tuple[torch.Tensor, ...], mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked parameter loss against cached teacher maps."""
        prediction_tensors = tuple(parameter.squeeze(1, 2) for parameter in predictions[-1])
        target_tensors = tuple(target.squeeze(1, 2) for target in targets)
        parameter_mask = mask.bool()
        for target in target_tensors:
            if torch.is_complex(target):
                parameter_mask &= torch.isfinite(target.real) & torch.isfinite(target.imag)
            else:
                parameter_mask &= torch.isfinite(target)
        if not parameter_mask.any():
            return torch.zeros((), device=self.device)
        loss = torch.zeros((), device=self.device)
        for target_index, (prediction, target) in enumerate(zip(prediction_tensors, target_tensors, strict=True)):
            if torch.is_complex(target):
                component_loss = torch.nn.functional.mse_loss(
                    torch.view_as_real(prediction[parameter_mask]),
                    torch.view_as_real(target.to(prediction.dtype)[parameter_mask]),
                )
            else:
                component_loss = torch.nn.functional.mse_loss(prediction[parameter_mask], target[parameter_mask])
            loss = loss + (0.5 * component_loss if target_index == 0 else component_loss)
        return loss

    def _reconstruct_batch(self, batch: DelicsRadialBatch) -> tuple[ParameterTuple, list[ParameterTuple], torch.Tensor]:
        """Run one SSDU reconstruction and return final predictions with held-out loss."""
        if self.current_subspace_gram_in is None:
            self.setup_ssdu_split()
        if self.current_subspace_gram_in is None or self.current_omega_loss is None:
            raise RuntimeError('SSDU split has not been initialized.')
        coil_rhs = self.build_visible_coil_rhs(batch['kspace'])
        csm_initial = self.estimate_initial_csm(coil_rhs)
        _images, parameters = self.pinqi(
            coil_rhs,
            csm_initial,
            self.current_subspace_gram_in,
            weight_scale=self.current_subspace_gram_in_scale,
        )
        kspace_loss = batch['kspace'][:, self.current_omega_loss.to(batch['kspace'].device)]
        kspace_prediction = self.predict_loss_kspace(parameters, csm_initial)
        return parameters[-1], parameters, self.ssdu_loss(kspace_prediction, kspace_loss)

    def reconstruct_parameters_from_kspace(self, kspace: torch.Tensor) -> list[ParameterTuple]:
        """Run the student reconstruction from synthetic k-space."""
        if self.current_subspace_gram_in is None:
            self.setup_ssdu_split()
        if self.current_subspace_gram_in is None:
            raise RuntimeError('SSDU split has not been initialized.')
        coil_rhs = self.build_visible_coil_rhs(kspace)
        csm_initial = self.estimate_initial_csm(coil_rhs)
        _images, parameters = self.pinqi(
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
        n_augmentations = self.hparams.teacher_augmentations_per_epoch
        if n_augmentations == 0:
            return torch.zeros((), device=self.device)
        losses = []
        mask = batch['mask'].to(self.device).bool()
        for _augmentation_index in range(n_augmentations):
            mixed_targets, mixed_mask, mixed_csm = self.maybe_mix_teacher_batch(targets, mask, csm)
            augmented_targets, augmented_mask, augmented_csm = self.augment_teacher_batch(
                mixed_targets,
                mixed_mask,
                mixed_csm,
            )
            synthetic_kspace = self.synthesize_kspace(augmented_targets, augmented_csm)
            synthetic_parameters = self.reconstruct_parameters_from_kspace(synthetic_kspace)
            losses.append(self.teacher_parameter_loss(synthetic_parameters, augmented_targets, augmented_mask))
        return torch.stack(losses).mean()

    def synthetic_anchor_loss(self, batch: BatchType) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the BrainWeb supervised anchor loss used during fine-tuning."""
        coil_rhs, csm_ground_truth = self.simulate_coil_rhs(batch)
        csm_initial = csm_ground_truth if self.hparams.use_ground_truth_csm else self.estimate_initial_csm(coil_rhs)
        _images, parameters = self(coil_rhs, csm_initial=csm_initial)
        return self.supervised_parameter_loss(parameters, batch)

    def _fine_tune_loss(
        self,
        batch: DelicsRadialBatch,
        *,
        use_teacher: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute SSDU and optional teacher-generated pseudo-reference losses."""
        _final_parameters, parameters, ssdu_loss = self._reconstruct_batch(batch)
        components = {'ssdu': ssdu_loss, 'teacher_generated': torch.zeros((), device=self.device)}
        if use_teacher and self.hparams.teacher_generated_loss_weight:
            case_ids = cast(Sequence[str], batch['case_id'])
            teacher_slices = []
            for case_id, slice_index in zip(case_ids, batch['slice_index'], strict=True):
                key = (case_id, int(slice_index.item()))
                if key not in self.teacher_cache:
                    raise RuntimeError(
                        f'Missing teacher pseudo-reference for {case_id} slice {int(slice_index.item())}.'
                    )
                teacher_slices.append(self.teacher_cache[key])
            teacher_targets = tuple(
                torch.stack(items).to(self.device)
                for items in zip(*(teacher_slice.targets for teacher_slice in teacher_slices), strict=True)
            )
            teacher_csm = torch.stack([teacher_slice.csm for teacher_slice in teacher_slices]).to(self.device)
            if self.hparams.teacher_augmentations_per_epoch:
                components['teacher_generated'] = self.teacher_generated_loss(batch, teacher_targets, teacher_csm)
            else:
                components['teacher_generated'] = self.teacher_parameter_loss(parameters, teacher_targets, batch['mask'])
        loss = (
            self.hparams.ssdu_loss_weight * components['ssdu']
            + self.hparams.teacher_generated_loss_weight * components['teacher_generated']
        )
        return loss, components

    def training_step(self, batch: FineTuneBatch, _batch_idx: int) -> torch.Tensor:
        """Run one fine-tuning step."""
        delics_loss, components = self._fine_tune_loss(batch['delics'], use_teacher=True)
        synthetic_loss, synthetic_components = self.synthetic_anchor_loss(batch['synthetic'])
        loss = delics_loss + self.hparams.synthetic_anchor_loss_weight * synthetic_loss
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/ssdu_loss', components['ssdu'].detach(), on_step=True, on_epoch=True, sync_dist=True)
        if self.hparams.teacher_generated_loss_weight:
            self.log(
                'train/teacher_generated_loss',
                components['teacher_generated'].detach(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        if self.hparams.synthetic_anchor_loss_weight:
            self.log('train/synthetic_loss', synthetic_loss.detach(), on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(
                {
                    f'debug/train/synthetic_loss_{key}': value.detach()
                    for key, value in synthetic_components.items()
                },
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss
