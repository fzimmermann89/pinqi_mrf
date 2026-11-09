"""Supervised DELICS training from scratch, with optional BrainWeb examples."""

from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

import h5py
import mr2
import numpy as np
import torch
import torch._functorch.config
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from pinqi_mrf.data import BatchType, BrainwebDataset, DataModule, DelicsRadialReferenceBatch, ValidationLoaderType
from pinqi_mrf.delics_frames import (
    DelicsFrameBatch,
    DelicsFrameDataset,
    DelicsFramePreparation,
    resize_coil_maps,
    walsh_coil_maps,
)
from pinqi_mrf.train import BaseModule


class SupervisedBatch(TypedDict, total=False):
    """Independent supervised batches combined in one optimizer update."""

    delics: DelicsFrameBatch | BatchType
    delics_rhs: torch.Tensor
    synthetic: BatchType
    synthetic_rhs: torch.Tensor


class SupervisedModule(BaseModule):
    """Train PINQI against DELICS maps and optional BrainWeb maps."""

    def __init__(
        self,
        brainweb_loss_weight: float = 0.0,
        brainweb_delics_csms: bool = False,
        **kwargs: object,
    ) -> None:
        """Initialize supervised training.

        Parameters
        ----------
        brainweb_loss_weight
            Weight of the additional synthetic supervised loss.
        brainweb_delics_csms
            Generate BrainWeb coils from central training-DELICS calibration slices.
        **kwargs
            Arguments forwarded to the shared reconstruction module.
        """
        if brainweb_loss_weight < 0:
            raise ValueError('brainweb_loss_weight must be non-negative.')
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.save_hyperparameters(
            {
                'brainweb_loss_weight': brainweb_loss_weight,
                'brainweb_delics_csms': brainweb_delics_csms,
            }
        )
        self._frame_preparation: DelicsFramePreparation | None = None
        self._brainweb_coils: tuple[torch.Tensor, torch.Tensor] | None = None

    def on_fit_start(self) -> None:
        """Initialize physics and one small CSM pool using training subjects only."""
        super().on_fit_start()
        self._setup_brainweb_coils()

    def on_validation_start(self) -> None:
        """Use the same training-derived CSM pool for standalone validation."""
        super().on_validation_start()
        self._setup_brainweb_coils()

    @torch.no_grad()
    def _setup_brainweb_coils(self) -> None:
        """Estimate Walsh maps once per training subject, one slice at a time."""
        if not self.hparams.brainweb_delics_csms or self._brainweb_coils is not None:
            return
        data = cast(SupervisedDataModule, getattr(self.trainer, 'datamodule', None))
        dataset = data.delics_train_dataset
        if not isinstance(dataset, DelicsFrameDataset):
            raise ValueError('DELICS-derived BrainWeb CSMs require the training frame cache.')
        csms, supports = [], []
        for file_index, (_path, case) in enumerate(dataset.files):
            slices = [z for file, z in dataset.samples if file == file_index]
            slice_index = slices[len(slices) // 2]
            indices = np.linspace(0, self.hparams.n_timepoints - 1, 16).round().astype(int)
            with h5py.File(dataset.cache_path(case), 'r') as cached:
                packed = cached[f'z{slice_index:03d}/frames'][indices]
                frames = torch.view_as_complex(torch.from_numpy(packed.view(np.float16).reshape(*packed.shape, 2)))
            csm, support = walsh_coil_maps(frames.to(device=self.device, dtype=torch.complex64))
            if len(csm) != self.hparams.n_coils:
                raise ValueError('DELICS calibration coil count must equal the BrainWeb coil count.')
            csms.append(csm)
            supports.append(support)
            del frames, packed
        self._brainweb_coils = torch.stack(csms), torch.stack(supports)

    def _set_brainweb_csms(self, batch: BatchType, sample_offset: int | None = None) -> None:
        """Sample training coil fields or select fixed validation fields and fit their coverage."""
        if self._brainweb_coils is None:
            raise RuntimeError('Training-derived BrainWeb CSMs have not been initialized.')
        csms, support = self._brainweb_coils
        batch_size = len(batch['mask'])
        indices = (
            torch.randint(len(csms), (batch_size,), device=csms.device)
            if sample_offset is None
            else (torch.arange(batch_size, device=csms.device) + sample_offset) % len(csms)
        )
        batch['csm'] = resize_coil_maps(csms[indices], support[indices], batch['mask'])

    @torch.no_grad()
    def on_after_batch_transfer(self, batch: object, dataloader_idx: int) -> object:
        """Consume full frames on CUDA, retaining only subspace data and supervision maps."""
        _ = dataloader_idx
        if not self.training or not isinstance(batch, dict) or 'delics' not in batch:
            return batch
        sample = batch.pop('delics')
        acquisition = self.acquisition.to(self.device)
        basis = self.signalmodel.pca.compression_matrix.mH
        if basis.ndim > 2:
            basis = basis.squeeze(tuple(range(basis.ndim - 2)))
        if self._frame_preparation is None:
            if self.hparams.use_density_compensation:
                raise ValueError('Full-frame preparation requires use_density_compensation=False.')
            nufft = mr2.operators.NonUniformFastFourierOp(
                direction=(-2, -1),
                recon_matrix=self.matrix_size,
                encoding_matrix=self.matrix_size,
                traj=acquisition.trajectory.to(self.device),
            )
            self._frame_preparation = DelicsFramePreparation(nufft, basis)
        rhs, targets = self._frame_preparation(sample)
        targets['csm'] = self.estimate_initial_csm(rhs)
        # Neither the returned batch nor the preparation object keeps source frames.
        batch['delics'] = targets
        batch['delics_rhs'] = rhs
        if self.hparams.brainweb_loss_weight:
            synthetic = batch['synthetic']
            if self.hparams.brainweb_delics_csms:
                self._set_brainweb_csms(synthetic)
            rhs_synthetic, csm_true = self.simulate_coil_rhs(synthetic)
            synthetic['csm'] = (
                csm_true if self.hparams.use_ground_truth_csm else self.estimate_initial_csm(rhs_synthetic)
            )
            batch['synthetic_rhs'] = rhs_synthetic
        return batch

    def teardown(self, stage: str) -> None:
        """Release native plans before interpreter and CUDA shutdown."""
        _ = stage
        if self._frame_preparation is not None:
            self._frame_preparation.close()
            self._frame_preparation = None
        self._brainweb_coils = None

    @torch.no_grad()
    def simulate_coil_rhs(self, batch: BatchType) -> tuple[torch.Tensor, torch.Tensor]:
        """Use exact training noise; retain the historical BrainWeb validation sampler."""
        if self._frame_preparation is None or not self.training:
            return super().simulate_coil_rhs(batch)
        if self.subspace_gram is None:
            raise RuntimeError('Acquisition operators have not been initialized.')
        (coefficients,) = self.signalmodel(*self._parameters_from_batch(batch))
        coefficients = coefficients.movedim(0, 1)
        (coil_coefficients,) = mr2.operators.SensitivityOp(batch['csm'][:, None])(coefficients)
        (rhs,) = self.subspace_gram(coil_coefficients)
        rhs.add_(self._frame_preparation.noise_rhs(batch['noise_std'], batch['csm'].shape[1], source='brainweb'))
        return self.compress_coils(rhs, batch['csm'])

    def training_step(self, batch: SupervisedBatch, _batch_idx: int) -> torch.Tensor:
        """Supervise DELICS reconstruction and optional independent BrainWeb reconstruction."""
        reference = cast(BatchType, batch.pop('delics'))
        rhs = batch.pop('delics_rhs')
        csm = reference['csm']
        _images, parameters, _csm = self(rhs, csm_initial=csm)
        delics_loss, _components = self.supervised_parameter_loss(parameters, reference)
        del _images, parameters, _csm, rhs, csm, _components
        loss = delics_loss
        self.log('train/delics_loss', delics_loss, on_epoch=True, batch_size=len(reference['mask']))
        if self.hparams.brainweb_loss_weight:
            synthetic = batch.pop('synthetic')
            rhs_synthetic = batch.pop('synthetic_rhs')
            csm_synthetic = synthetic['csm']
            _images, parameters, _csm = self(rhs_synthetic, csm_initial=csm_synthetic)
            synthetic_loss, _components = self.supervised_parameter_loss(parameters, synthetic)
            loss = loss + self.hparams.brainweb_loss_weight * synthetic_loss
            self.log('train/brainweb_loss', synthetic_loss, on_epoch=True, batch_size=len(synthetic['mask']))
        self.log('train/loss', loss, on_epoch=True, prog_bar=True, batch_size=len(reference['mask']))
        return loss

    def _synthetic_validation_step(self, batch: BatchType, batch_idx: int, dataloader_idx: int) -> None:
        """Validate BrainWeb with a fixed choice of training-derived receive fields."""
        if self.hparams.brainweb_delics_csms:
            self._set_brainweb_csms(batch, sample_offset=batch_idx * len(batch['mask']))
        super()._synthetic_validation_step(batch, batch_idx, dataloader_idx)

    def _kspace_validation_step(self, batch: DelicsRadialReferenceBatch, batch_idx: int, dataloader_idx: int) -> None:
        """Report unrestricted errors as primary metrics and clipped errors separately."""
        predictions, baseline = self.reconstruct_full_kspace(batch['kspace'])
        mask = batch['mask'].to(self.device).bool()
        prefix = f'val_{dataloader_idx}'
        for clipped, metric_prefix in [(False, prefix), (True, f'{prefix}_clipped')]:
            result = self.evaluate_parameter_validation(
                predictions,
                baseline,
                mask,
                batch['reference_t1'].to(self.device)[:, None, None],
                batch['reference_t2'].to(self.device)[:, None, None],
                clip_to_target_range=clipped,
                apply_parameter_thresholds=False,
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
                    batch['reference_m0'].to(self.device)[:, None, None],
                    baseline[0],
                    predictions[0],
                )


class SupervisedDataModule(DataModule):
    """Use explicit disjoint DELICS cases and optional disjoint BrainWeb folders."""

    def __init__(
        self,
        folder: Path,
        frame_folder: Path,
        train_case_ids: Sequence[str],
        val_case_ids: Sequence[str],
        brainweb_loss_weight: float = 0.0,
        brainweb_train_folder: Path | None = None,
        brainweb_val_folder: Path | None = None,
        brainweb_train_subjects: Sequence[str] = (
            's06',
            's18',
            's20',
            's38',
            's41',
            's42',
            's43',
            's44',
            's45',
            's46',
            's47',
            's48',
            's49',
            's50',
            's51',
            's52',
            's53',
            's54',
        ),
        brainweb_val_subjects: Sequence[str] = ('s04', 's05'),
        delics_batch_size: int = 1,
        brainweb_batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 2,
        size: int = 256,
        n_coils: int = 12,
        noise: tuple[float, float] = (0.0, 0.05),
        frame_timepoints: int = 352,
        frame_coils: int = 8,
        frame_noise: tuple[float, float] = (0.002, 0.006),
    ) -> None:
        """Require explicit partitions and augmented full-frame training.

        Parameters
        ----------
        folder
            Radial cache supplying the full-acquisition reference maps and fixed validation inputs.
        train_case_ids
            Explicit DELICS training cases, disjoint from validation.
        val_case_ids
            Explicit DELICS validation cases.
        brainweb_loss_weight
            Weight of the separately averaged BrainWeb supervised loss.
        brainweb_train_folder
            Folder containing the BrainWeb training subjects.
        brainweb_val_folder
            Folder containing disjoint BrainWeb validation subjects.
        brainweb_train_subjects
            Explicit subjects required in the BrainWeb training folder.
        brainweb_val_subjects
            Explicit subjects required in the BrainWeb validation folder.
        delics_batch_size
            Number of DELICS slices per optimizer update.
        brainweb_batch_size
            Number of BrainWeb slices per optimizer update.
        val_batch_size
            Number of slices per validation batch.
        num_workers
            Number of spawned CPU workers per loader.
        size
            In-plane reconstruction size.
        n_coils
            Number of simulated BrainWeb coils before compression.
        noise
            BrainWeb noise-strength range.
        frame_folder
            Prepared coil-frame cache supplying augmented training inputs.
            Validation always uses the fixed radial cache in ``folder``.
        frame_timepoints
            Retained temporal prefix; must equal the model's number of time points.
        frame_coils
            Number of coils in the prepared frame cache.
        frame_noise
            Uniform complex white k-space noise RMS range in scaled signal units.
        """
        if not train_case_ids or not val_case_ids:
            raise ValueError('Both DELICS case partitions must be nonempty.')
        if set(train_case_ids) & set(val_case_ids):
            raise ValueError('DELICS training and validation cases overlap.')
        if len(set(train_case_ids)) != len(train_case_ids) or len(set(val_case_ids)) != len(val_case_ids):
            raise ValueError('DELICS case partitions contain duplicates.')
        if brainweb_loss_weight < 0:
            raise ValueError('brainweb_loss_weight must be non-negative.')
        if brainweb_loss_weight and (brainweb_train_folder is None or brainweb_val_folder is None):
            raise ValueError('Mixed training requires explicit BrainWeb training and validation folders.')
        super().__init__(
            folder=brainweb_train_folder or folder,
            delics_folder=folder,
            delics_val_case_ids=val_case_ids,
            size=size,
            n_coils=n_coils,
            noise=noise,
            batch_size=val_batch_size,
            num_workers=num_workers,
        )
        self.save_hyperparameters()
        for name, value in self.hparams.items():
            if isinstance(value, Path):
                self.hparams[name] = str(value)
        self.train_case_ids = tuple(train_case_ids)
        self.brainweb_loss_weight = brainweb_loss_weight
        self.brainweb_train_folder = brainweb_train_folder
        self.brainweb_val_folder = brainweb_val_folder
        self.brainweb_train_subjects = tuple(brainweb_train_subjects)
        self.brainweb_val_subjects = tuple(brainweb_val_subjects)
        self.delics_batch_size = delics_batch_size
        self.brainweb_batch_size = brainweb_batch_size
        self.frame_folder = frame_folder
        self.frame_timepoints = frame_timepoints
        self.frame_coils = frame_coils
        self.frame_noise = frame_noise
        self.delics_train_dataset: DelicsFrameDataset | None = None
        self.brainweb_train_dataset: BrainwebDataset | None = None
        self.brainweb_validation: BrainwebDataset | None = None

    def prepare_data(self) -> None:
        """Use existing data; never download or generate validation subjects."""

    @property
    def validation_loader_types(self) -> tuple[ValidationLoaderType, ...]:
        """Keep DELICS as val_0 in both experiments."""
        return (ValidationLoaderType.KSPACE,) + ((ValidationLoaderType.SYNTHETIC,) if self.brainweb_loss_weight else ())

    def setup(self, stage: str | None = None) -> None:
        """Verify resolved files are disjoint and construct only enabled datasets."""
        _ = stage
        if self.delics_train_dataset is not None:
            return
        reference = self.build_delics_dataset(case_ids=self.train_case_ids, slice_trim=(0, 0), include_references=True)
        self.delics_val_dataset = self.build_delics_dataset(
            case_ids=self.delics_val_case_ids, slice_trim=(0, 0), include_references=True
        )
        train_files = {p.resolve() for p, _case in reference.files}
        val_files = {p.resolve() for p, _case in self.delics_val_dataset.files}
        if train_files & val_files:
            raise ValueError('DELICS partitions resolve to shared files.')
        for dataset, requested in [
            (reference, self.train_case_ids),
            (self.delics_val_dataset, self.delics_val_case_ids),
        ]:
            if {case for _path, case in dataset.files} != set(requested):
                raise ValueError('A requested DELICS case is missing from the cache.')
        self.delics_train_dataset = DelicsFrameDataset(
            reference,
            self.frame_folder,
            n_timepoints=self.frame_timepoints,
            n_coils=self.frame_coils,
            noise=self.frame_noise,
        )
        if self.brainweb_loss_weight:
            if self.brainweb_train_folder is None or self.brainweb_val_folder is None:
                raise RuntimeError('Missing BrainWeb partitions.')
            train = {p.resolve() for p in self.brainweb_train_folder.glob('s??.h5')}
            val = {p.resolve() for p in self.brainweb_val_folder.glob('s??.h5')}
            if not train or not val or train & val or {p.name for p in train} & {p.name for p in val}:
                raise ValueError('BrainWeb subjects must be nonempty and disjoint.')
            if {p.stem for p in train} != set(self.brainweb_train_subjects):
                raise ValueError('BrainWeb training folder does not match the configured subjects.')
            if {p.stem for p in val} != set(self.brainweb_val_subjects):
                raise ValueError('BrainWeb validation folder does not match the configured subjects.')
            self.brainweb_train_dataset = self.build_brainweb_dataset(
                orientation=('axial', 'coronal', 'sagittal'), random=True
            )
            self.brainweb_validation = BrainwebDataset(
                folder=self.brainweb_val_folder,
                size=self.size,
                n_coils=self.n_coils,
                noise=self.noise,
                t1_bounds=self.t1_bounds,
                t2_bounds=self.t2_bounds,
                orientation=('axial',),
                random=False,
            )

    @staticmethod
    def build_dataloader(
        dataset: torch.utils.data.Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
        sampler: torch.utils.data.Sampler[int] | None = None,
    ) -> torch.utils.data.DataLoader:
        """Spawn workers without inheriting CUDA, logging or HDF5 locks."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            multiprocessing_context='spawn' if num_workers else None,
            persistent_workers=num_workers > 0,
            prefetch_factor=1 if num_workers else None,
            timeout=120 if num_workers else 0,
        )

    def train_dataloader(self) -> CombinedLoader:
        """Pair one DELICS batch with one BrainWeb batch only in mixed mode."""
        if self.delics_train_dataset is None:
            raise RuntimeError('Call setup before requesting loaders.')
        loaders = {
            'delics': self.build_dataloader(
                self.delics_train_dataset,
                batch_size=self.delics_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        }
        if self.brainweb_loss_weight:
            if self.brainweb_train_dataset is None:
                raise RuntimeError('Missing BrainWeb training dataset.')
            sampler = torch.utils.data.RandomSampler(
                self.brainweb_train_dataset,
                replacement=True,
                num_samples=len(loaders['delics']) * self.brainweb_batch_size,
            )
            loaders['synthetic'] = self.build_dataloader(
                self.brainweb_train_dataset,
                batch_size=self.brainweb_batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        return CombinedLoader(loaders, mode='min_size')

    def val_dataloader(self) -> list[torch.utils.data.DataLoader]:
        """Evaluate all held-out DELICS slices and fixed held-out BrainWeb slices."""
        loaders = [self.delics_val_dataloader()]
        if self.brainweb_loss_weight:
            if self.brainweb_validation is None:
                raise RuntimeError('Missing BrainWeb validation dataset.')
            subset = torch.utils.data.Subset(
                self.brainweb_validation, list(range(0, len(self.brainweb_validation), 20))
            )
            loaders.append(
                self.build_dataloader(
                    subset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=False,
                    drop_last=False,
                )
            )
        return loaders


class DelicsCLI(LightningCLI):
    """Keep shared acquisition dimensions and mixture weights consistent."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Link data settings to reconstruction settings."""
        for source, target in [
            ('brainweb_loss_weight', 'brainweb_loss_weight'),
            ('size', 'image_size'),
            ('n_coils', 'n_coils'),
            ('frame_coils', 'n_compressed_coils'),
            ('frame_timepoints', 'n_timepoints'),
        ]:
            parser.link_arguments(f'data.{source}', f'model.{target}', apply_on='instantiate')


def main() -> None:
    """Run Lightning's fit or validate command, including checkpoint resume."""
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
