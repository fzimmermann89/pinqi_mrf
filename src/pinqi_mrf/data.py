"""Data generation for PINQI-MRF experiments."""

import hashlib
import math
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Generic, Literal, TypedDict, TypeVar, cast, overload

import h5py
import lightning.pytorch as pl
import mr2
import torch

from pinqi_mrf.parameters import R1_STATS, R2_STATS, relaxation_time_to_standardized_rate

BRAINWEB_VALUES_3T_RANDOMIZED = {
    'skl': mr2.phantoms.brainweb.BrainwebTissue((0.000, 2.000), (0.020, 0.030), (0.00, 0.05), (-0.2, 0.2)),
    'gry': mr2.phantoms.brainweb.BrainwebTissue((1.200, 2.000), (0.080, 0.120), (0.70, 1.00), (-0.2, 0.2)),
    'wht': mr2.phantoms.brainweb.BrainwebTissue((0.800, 1.500), (0.060, 0.100), (0.50, 0.90), (-0.2, 0.2)),
    'csf': mr2.phantoms.brainweb.BrainwebTissue((2.000, 4.000), (1.300, 2.000), (0.90, 1.00), (-0.2, 0.2)),
    'mrw': mr2.phantoms.brainweb.BrainwebTissue((0.400, 0.600), (0.060, 0.100), (0.70, 1.00), (-0.2, 0.2)),
    'dura': mr2.phantoms.brainweb.BrainwebTissue((2.000, 2.800), (0.200, 0.500), (0.90, 1.00), (-0.2, 0.2)),
    'fat': mr2.phantoms.brainweb.BrainwebTissue((0.300, 0.500), (0.060, 0.100), (0.90, 1.00), (-0.2, 0.2)),
    'fat2': mr2.phantoms.brainweb.BrainwebTissue((0.400, 0.600), (0.060, 0.100), (0.60, 0.90), (-0.2, 0.2)),
    'mus': mr2.phantoms.brainweb.BrainwebTissue((1.200, 1.500), (0.040, 0.060), (0.90, 1.00), (-0.2, 0.2)),
    'm-s': mr2.phantoms.brainweb.BrainwebTissue((0.500, 0.900), (0.300, 0.500), (0.90, 1.00), (-0.2, 0.2)),
    'ves': mr2.phantoms.brainweb.BrainwebTissue((1.700, 2.100), (0.200, 0.400), (0.80, 1.00), (-0.2, 0.2)),
}


class BatchType(TypedDict):
    """Type hint for a batch of data."""

    m0: torch.Tensor
    r1: torch.Tensor
    r2: torch.Tensor
    csm: torch.Tensor
    mask: torch.Tensor
    noise_std: torch.Tensor


class DelicsRadialBatch(TypedDict):
    """Type hint for one DELICS radial k-space sample."""

    case_id: str
    slice_index: torch.Tensor
    kspace: torch.Tensor
    mask: torch.Tensor


class DelicsRadialReferenceBatch(DelicsRadialBatch):
    """Type hint for one DELICS validation sample with reference maps."""

    reference_t1: torch.Tensor
    reference_t2: torch.Tensor


class ValidationLoaderType(StrEnum):
    """Kinds of validation loaders used during training."""

    SYNTHETIC = 'synthetic'
    KSPACE = 'kspace'


IncludeReferences = TypeVar('IncludeReferences', Literal[False], Literal[True])


class DelicsRadialDataset(
    torch.utils.data.Dataset[DelicsRadialBatch | DelicsRadialReferenceBatch],
    Generic[IncludeReferences],
):
    """Dataset for pseudo-acquired DELICS radial k-space with optional reference maps."""

    @overload
    def __init__(
        self,
        folder: Path,
        *,
        case_glob: str = 'delics_radial_case*.h5',
        case_ids: Sequence[str] | None = None,
        delics_scale: float = 250.0,
        delics_noise_std: float = 1e-3,
        include_references: Literal[False] = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        folder: Path,
        *,
        case_glob: str = 'delics_radial_case*.h5',
        case_ids: Sequence[str] | None = None,
        delics_scale: float = 250.0,
        delics_noise_std: float = 1e-3,
        include_references: Literal[True],
    ) -> None: ...

    def __init__(
        self,
        folder: Path,
        *,
        case_glob: str = 'delics_radial_case*.h5',
        case_ids: Sequence[str] | None = None,
        delics_scale: float = 250.0,
        delics_noise_std: float = 1e-3,
        include_references: IncludeReferences = False,
    ) -> None:
        """Initialize the DELICS radial dataset."""
        if delics_scale <= 0:
            raise ValueError('delics_scale must be positive.')
        if delics_noise_std < 0:
            raise ValueError('delics_noise_std must be non-negative.')
        self.folder = folder
        self.case_glob = case_glob
        self.delics_scale = delics_scale
        self.delics_noise_std = delics_noise_std
        self.include_references = include_references
        allowed_case_ids = set(case_ids) if case_ids is not None else None
        self.files: list[tuple[Path, str]] = []
        self.samples: list[tuple[int, int]] = []

        for path in sorted(folder.glob(case_glob)):
            with h5py.File(path, 'r') as handle:
                case_id = str(handle.attrs.get('case', path.stem.removeprefix('delics_radial_')))
                if allowed_case_ids is not None and case_id not in allowed_case_ids:
                    continue
                file_index = len(self.files)
                self.files.append((path, case_id))
                valid = handle['valid'][()].astype(bool)
                for slice_index, is_valid in enumerate(valid):
                    if bool(is_valid):
                        self.samples.append((file_index, slice_index))

        if not self.samples:
            raise ValueError(f'No DELICS radial samples found in {folder} matching {case_glob!r}.')

    def __len__(self) -> int:
        """Return the number of available slices."""
        return len(self.samples)

    @overload
    def __getitem__(self: 'DelicsRadialDataset[Literal[False]]', index: int) -> DelicsRadialBatch: ...

    @overload
    def __getitem__(self: 'DelicsRadialDataset[Literal[True]]', index: int) -> DelicsRadialReferenceBatch: ...

    def __getitem__(self, index: int) -> DelicsRadialBatch | DelicsRadialReferenceBatch:
        """Get one DELICS radial sample with optional reference maps."""
        file_index, slice_index = self.samples[index]
        path, case_id = self.files[file_index]
        with h5py.File(path, 'r') as handle:
            kspace = self.delics_scale * torch.as_tensor(handle['kspace'][slice_index], dtype=torch.complex64)
            if self.delics_noise_std:
                seed = int.from_bytes(
                    hashlib.blake2b(f'{case_id}:{slice_index}'.encode(), digest_size=8).digest(),
                    'little',
                )
                rng = mr2.utils.RandomGenerator(seed)
                noise_real = rng.randn_tensor(kspace.shape, torch.float32)
                noise_imag = rng.randn_tensor(kspace.shape, torch.float32)
                noise = (self.delics_noise_std / math.sqrt(2.0)) * torch.complex(noise_real, noise_imag)
                kspace = kspace + noise.to(torch.complex64)
            sample: DelicsRadialBatch | DelicsRadialReferenceBatch = {
                'case_id': case_id,
                'slice_index': torch.tensor(slice_index, dtype=torch.int64),
                'kspace': kspace,
                'mask': torch.as_tensor(handle['mask'][slice_index], dtype=torch.bool),
            }
            if self.include_references:
                sample['reference_t1'] = torch.as_tensor(handle['reference_t1'][slice_index], dtype=torch.float32)
                sample['reference_t2'] = torch.as_tensor(handle['reference_t2'][slice_index], dtype=torch.float32)
                return cast(DelicsRadialReferenceBatch, sample)
            return cast(DelicsRadialBatch, sample)


class DelicsRadialDataModule(pl.LightningDataModule):
    """Lightning data module for pseudo-acquired DELICS radial k-space."""

    def __init__(
        self,
        folder: Path = Path('/scratch/zimmer08/delics/radial_kspace'),
        *,
        train_case_ids: Sequence[str] | None = None,
        val_case_ids: Sequence[str] = ('case000', 'case007', 'case100'),
        case_glob: str = 'delics_radial_case*.h5',
        batch_size: int = 1,
        num_workers: int = 4,
    ) -> None:
        """Initialize the DELICS radial data module."""
        super().__init__()
        self.save_hyperparameters(ignore=['folder', 'num_workers'])
        self.folder = folder
        self.train_case_ids = train_case_ids
        self.val_case_ids = val_case_ids
        self.case_glob = case_glob
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset: DelicsRadialDataset[Literal[False]] | None = None
        self.val_dataset: DelicsRadialDataset[Literal[True]] | None = None

    @property
    def validation_loader_types(self) -> tuple[ValidationLoaderType, ...]:
        """Return the ordered validation loader types."""
        return (ValidationLoaderType.KSPACE,)

    def setup(self, stage: str | None = None) -> None:
        """Create training and validation datasets."""
        _ = stage
        if self.train_dataset is not None and self.val_dataset is not None:
            return
        if self.train_dataset is None:
            self.train_dataset = DelicsRadialDataset(
                self.folder,
                case_glob=self.case_glob,
                case_ids=self.train_case_ids,
                include_references=False,
            )
        if self.val_dataset is None:
            self.val_dataset = DelicsRadialDataset(
                self.folder,
                case_glob=self.case_glob,
                case_ids=self.val_case_ids,
                include_references=True,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the DELICS training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError('DelicsRadialDataModule.setup() must be called before train_dataloader().')
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=lambda *_: torch.set_num_threads(1),
            drop_last=False,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the DELICS validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError('DelicsRadialDataModule.setup() must be called before val_dataloader().')
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )


class BrainwebDataset(torch.utils.data.Dataset):
    """BrainWeb-based dataset for parameter maps."""

    def __init__(
        self,
        folder: Path,
        size: int = 256,
        n_coils: int = 12,
        noise: tuple[float, float] = (0.0, 0.05),
        r1_stats: tuple[float, float] = R1_STATS,
        r2_stats: tuple[float, float] = R2_STATS,
        orientation: Sequence[Literal['axial', 'coronal', 'sagittal']] = ('axial',),
        bias_field_downsample_factor: int = 16,
        m0_bias_range: tuple[float, float] = (0.8, 1.2),
        t1_bias_range: tuple[float, float] = (0.8, 1.2),
        t2_bias_range: tuple[float, float] = (0.8, 1.2),
        phase_bias_range: tuple[float, float] = (-math.pi / 6, math.pi / 6),
        global_m0_magnitude_range: tuple[float, float] = (0.8, 1.2),
        random: bool = True,
    ) -> None:
        """Initialize the dataset."""
        if bias_field_downsample_factor <= 0:
            raise ValueError('bias_field_downsample_factor must be positive.')
        if random:
            augment = mr2.phantoms.brainweb.augment(size=size)
        else:
            augment = mr2.phantoms.brainweb.augment(
                size=size,
                max_random_shear=0,
                max_random_rotation=0,
                max_random_scaling_factor=0,
                p_horizontal_flip=0,
                p_vertical_flip=1.0,
            )

        self.phantom = mr2.phantoms.brainweb.BrainwebSlices(
            folder=folder,
            what=('m0', 't1', 't2', 'mask'),
            seed='index' if not random else 'random',
            slice_preparation=augment,
            parameters=BRAINWEB_VALUES_3T_RANDOMIZED,
            orientation=orientation,
        )
        self.matrix_size = mr2.data.SpatialDimension(1, size, size)
        self.n_coils = n_coils
        self.noise = noise
        self.r1_stats = r1_stats
        self.r2_stats = r2_stats
        self.bias_field_downsample_factor = bias_field_downsample_factor
        self.m0_bias_range = m0_bias_range
        self.t1_bias_range = t1_bias_range
        self.t2_bias_range = t2_bias_range
        self.phase_bias_range = phase_bias_range
        self.global_m0_magnitude_range = global_m0_magnitude_range
        self.random = random

    def _smooth_random_field(
        self,
        rng: mr2.utils.RandomGenerator,
        shape: tuple[int, int],
        value_range: tuple[float, float],
    ) -> torch.Tensor:
        """Create a low-frequency 2D random field by upsampling a coarse grid."""
        low, high = value_range
        if low > high:
            raise ValueError('value_range lower bound must not exceed the upper bound.')
        coarse_shape = tuple(max(2, math.ceil(length / self.bias_field_downsample_factor)) for length in shape)
        coarse_field = rng.float32_tensor((1, 1, *coarse_shape), low=low, high=high)
        field = torch.nn.functional.interpolate(coarse_field, size=shape, mode='bicubic', align_corners=True).squeeze(
            0, 1
        )
        return field.clamp(min=low, max=high)

    def _augment_parameter_maps(
        self,
        m0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        rng: mr2.utils.RandomGenerator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply smooth multiplicative and phase augmentations to synthetic parameter maps."""
        spatial_shape = tuple(m0.shape[-2:])
        m0_bias = self._smooth_random_field(rng, spatial_shape, self.m0_bias_range).to(m0.dtype)
        t1_bias = self._smooth_random_field(rng, spatial_shape, self.t1_bias_range).to(t1.dtype)
        t2_bias = self._smooth_random_field(rng, spatial_shape, self.t2_bias_range).to(t2.dtype)
        phase_bias = self._smooth_random_field(rng, spatial_shape, self.phase_bias_range)
        global_magnitude = rng.float32(*self.global_m0_magnitude_range)
        global_phase = rng.float32(-math.pi, math.pi)
        global_scale = torch.polar(torch.tensor(global_magnitude, dtype=torch.float32), torch.tensor(global_phase))
        m0 = m0.to(torch.complex64) * m0_bias.to(torch.complex64) * torch.exp(1j * phase_bias).to(torch.complex64)
        m0 = m0 * global_scale.to(torch.complex64)
        t1 = (t1 * t1_bias).clamp_min(torch.finfo(t1.dtype).eps)
        t2 = (t2 * t2_bias).clamp_min(torch.finfo(t2.dtype).eps)
        return m0, t1, t2

    def __len__(self) -> int:
        """Return the number of available slices."""
        return len(self.phantom)

    def __getitem__(self, index: int) -> BatchType:
        """Get one training sample."""
        phantom = self.phantom[index]
        seed = int(torch.randint(0, 1000000, (1,))) if self.random else index
        rng = mr2.utils.RandomGenerator(seed)
        m0 = phantom['m0']
        t1 = phantom['t1']
        t2 = phantom['t2']
        m0, t1, t2 = self._augment_parameter_maps(m0, t1, t2, rng)

        return {
            'm0': m0,
            'r1': relaxation_time_to_standardized_rate(t1, self.r1_stats),
            'r2': relaxation_time_to_standardized_rate(t2, self.r2_stats),
            'csm': mr2.phantoms.coils.birdcage_2d(
                self.n_coils,
                self.matrix_size,
                relative_radius=rng.float32(1.2, 1.8),
            ).squeeze(0)
            if self.n_coils > 1
            else torch.ones(1, *self.matrix_size.zyx, dtype=torch.complex64),
            'mask': phantom['mask'],
            'noise_std': torch.tensor(rng.float32(*self.noise)),
        }


class DataModule(pl.LightningDataModule):
    """Lightning data module for PINQI-MRF training."""

    def __init__(
        self,
        folder: Path = mr2.phantoms.brainweb.CACHE_DIR_BRAINWEB,
        size: int = 256,
        n_coils: int = 12,
        noise: tuple[float, float] = (0.0, 0.05),
        r1_stats: tuple[float, float] = R1_STATS,
        r2_stats: tuple[float, float] = R2_STATS,
        batch_size: int = 6,
        num_workers: int = 4,
    ) -> None:
        """Initialize the data module."""
        super().__init__()
        self.save_hyperparameters(ignore=['folder', 'num_workers'])
        self.folder = folder
        self.size = size
        self.n_coils = n_coils
        self.noise = noise
        self.r1_stats = r1_stats
        self.r2_stats = r2_stats
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset: BrainwebDataset | None = None
        self.val_dataset: torch.utils.data.Subset[BrainwebDataset] | None = None

    @property
    def validation_loader_types(self) -> tuple[ValidationLoaderType, ...]:
        """Return the ordered validation loader types."""
        return (ValidationLoaderType.SYNTHETIC,)

    def prepare_data(self) -> None:
        """Download BrainWeb data if it is not available locally."""
        if not self.folder.exists():
            self.folder.mkdir(parents=True, exist_ok=True)
            mr2.phantoms.brainweb.download_brainweb(output_directory=self.folder, workers=2, progress=True)

    def setup(self, stage: str | None = None) -> None:
        """Create training and validation datasets."""
        _ = stage
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        self.train_dataset = BrainwebDataset(
            folder=self.folder,
            size=self.size,
            n_coils=self.n_coils,
            noise=self.noise,
            r1_stats=self.r1_stats,
            r2_stats=self.r2_stats,
            orientation=('axial', 'coronal', 'sagittal'),
            random=True,
        )
        self.val_dataset = torch.utils.data.Subset(
            BrainwebDataset(
                folder=self.folder,
                size=self.size,
                n_coils=self.n_coils,
                noise=self.noise,
                r1_stats=self.r1_stats,
                r2_stats=self.r2_stats,
                orientation=('axial',),
                random=False,
            ),
            list(range(30, 500, 20)),
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError('DataModule.setup() must be called before train_dataloader().')
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=4 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=lambda *_: torch.set_num_threads(1),
            drop_last=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError('DataModule.setup() must be called before val_dataloader().')
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )


class PretrainDataModule(pl.LightningDataModule):
    """Synthetic pretraining with synthetic and k-space validation loaders."""

    def __init__(
        self,
        folder: Path = mr2.phantoms.brainweb.CACHE_DIR_BRAINWEB,
        *,
        delics_val_folder: Path = Path('/scratch/zimmer08/delics/radial_kspace'),
        delics_val_case_ids: Sequence[str] = ('case000', 'case007', 'case100'),
        size: int = 256,
        n_coils: int = 12,
        noise: tuple[float, float] = (0.0, 0.05),
        r1_stats: tuple[float, float] = R1_STATS,
        r2_stats: tuple[float, float] = R2_STATS,
        batch_size: int = 6,
        num_workers: int = 4,
        delics_val_case_glob: str = 'delics_radial_case*.h5',
        delics_val_batch_size: int = 1,
        delics_val_num_workers: int = 0,
    ) -> None:
        """Initialize the pretraining datamodule."""
        super().__init__()
        self.save_hyperparameters(ignore=['folder', 'num_workers'])
        self.brainweb_datamodule = DataModule(
            folder=folder,
            size=size,
            n_coils=n_coils,
            noise=noise,
            r1_stats=r1_stats,
            r2_stats=r2_stats,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.delics_val_folder = delics_val_folder
        self.delics_val_case_ids = delics_val_case_ids
        self.delics_val_case_glob = delics_val_case_glob
        self.delics_val_batch_size = delics_val_batch_size
        self.delics_val_num_workers = delics_val_num_workers
        self.delics_val_dataset: DelicsRadialDataset[Literal[True]] | None = None

    @property
    def validation_loader_types(self) -> tuple[ValidationLoaderType, ...]:
        """Return the ordered validation loader types."""
        return (ValidationLoaderType.SYNTHETIC, ValidationLoaderType.KSPACE)

    def prepare_data(self) -> None:
        """Prepare synthetic training data."""
        self.brainweb_datamodule.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        """Set up synthetic training and k-space validation."""
        self.brainweb_datamodule.setup(stage)
        if self.delics_val_dataset is None:
            self.delics_val_dataset = DelicsRadialDataset(
                self.delics_val_folder,
                case_glob=self.delics_val_case_glob,
                case_ids=self.delics_val_case_ids,
                include_references=True,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the synthetic training dataloader."""
        return self.brainweb_datamodule.train_dataloader()

    def val_dataloader(self) -> list[torch.utils.data.DataLoader]:
        """Return synthetic validation and k-space validation loaders."""
        if self.delics_val_dataset is None:
            raise RuntimeError('PretrainDataModule.setup() must be called before val_dataloader().')
        return [
            self.brainweb_datamodule.val_dataloader(),
            torch.utils.data.DataLoader(
                self.delics_val_dataset,
                batch_size=self.delics_val_batch_size,
                shuffle=False,
                num_workers=self.delics_val_num_workers,
                pin_memory=False,
                persistent_workers=self.delics_val_num_workers > 0,
                drop_last=False,
            ),
        ]


class FineTuneDataModule(pl.LightningDataModule):
    """K-space fine-tuning with k-space and synthetic validation loaders."""

    def __init__(
        self,
        folder: Path = Path('/scratch/zimmer08/delics/radial_kspace'),
        *,
        val_case_ids: Sequence[str] = ('case000', 'case007', 'case100'),
        train_case_ids: Sequence[str] | None = None,
        case_glob: str = 'delics_radial_case*.h5',
        batch_size: int = 1,
        num_workers: int = 4,
        brainweb_val_folder: Path = mr2.phantoms.brainweb.CACHE_DIR_BRAINWEB,
        brainweb_val_size: int = 256,
        brainweb_val_n_coils: int = 12,
        brainweb_val_noise: tuple[float, float] = (0.0, 0.05),
        brainweb_val_r1_stats: tuple[float, float] = R1_STATS,
        brainweb_val_r2_stats: tuple[float, float] = R2_STATS,
        brainweb_val_batch_size: int = 6,
        brainweb_val_num_workers: int = 0,
    ) -> None:
        """Initialize the fine-tuning datamodule."""
        super().__init__()
        self.save_hyperparameters(ignore=['folder'])
        self.delics_datamodule = DelicsRadialDataModule(
            folder=folder,
            train_case_ids=train_case_ids,
            val_case_ids=val_case_ids,
            case_glob=case_glob,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.brainweb_val_datamodule = DataModule(
            folder=brainweb_val_folder,
            size=brainweb_val_size,
            n_coils=brainweb_val_n_coils,
            noise=brainweb_val_noise,
            r1_stats=brainweb_val_r1_stats,
            r2_stats=brainweb_val_r2_stats,
            batch_size=brainweb_val_batch_size,
            num_workers=brainweb_val_num_workers,
        )

    @property
    def validation_loader_types(self) -> tuple[ValidationLoaderType, ...]:
        """Return the ordered validation loader types."""
        return (ValidationLoaderType.KSPACE, ValidationLoaderType.SYNTHETIC)

    def prepare_data(self) -> None:
        """Prepare synthetic validation data."""
        self.brainweb_val_datamodule.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        """Set up k-space training and synthetic validation."""
        self.delics_datamodule.setup(stage)
        self.brainweb_val_datamodule.setup(stage)

    @property
    def train_dataset(self) -> DelicsRadialDataset[Literal[False]] | None:
        """Expose the k-space training dataset for teacher refresh."""
        return self.delics_datamodule.train_dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the k-space training dataloader."""
        return self.delics_datamodule.train_dataloader()

    def val_dataloader(self) -> list[torch.utils.data.DataLoader]:
        """Return k-space validation and synthetic validation loaders."""
        return [self.delics_datamodule.val_dataloader(), self.brainweb_val_datamodule.val_dataloader()]
