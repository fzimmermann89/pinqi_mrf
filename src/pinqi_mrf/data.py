"""Data generation for PINQI-MRF experiments."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

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


class DelicsRadialDataset(torch.utils.data.Dataset):
    """Dataset for pseudo-acquired DELICS radial k-space H5 files."""

    def __init__(
        self,
        folder: Path,
        *,
        case_glob: str = 'delics_radial_case*.h5',
        case_ids: Sequence[str] | None = None,
    ) -> None:
        """Initialize the dataset."""
        self.folder = folder
        self.case_glob = case_glob
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

    def __getitem__(self, index: int) -> DelicsRadialBatch:
        """Get one DELICS radial k-space sample."""
        file_index, slice_index = self.samples[index]
        path, case_id = self.files[file_index]
        with h5py.File(path, 'r') as handle:
            return {
                'case_id': case_id,
                'slice_index': torch.tensor(slice_index, dtype=torch.int64),
                'kspace': torch.as_tensor(handle['kspace'][slice_index], dtype=torch.complex64),
                'mask': torch.as_tensor(handle['mask'][slice_index], dtype=torch.bool),
            }


class DelicsRadialDataModule(pl.LightningDataModule):
    """Lightning data module for pseudo-acquired DELICS radial k-space."""

    def __init__(
        self,
        folder: Path,
        *,
        train_case_ids: Sequence[str] | None = None,
        val_case_ids: Sequence[str] | None = None,
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
        self.train_dataset: DelicsRadialDataset | None = None
        self.val_dataset: DelicsRadialDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Create training and optional validation datasets."""
        _ = stage
        if self.train_dataset is not None:
            return
        self.train_dataset = DelicsRadialDataset(
            self.folder,
            case_glob=self.case_glob,
            case_ids=self.train_case_ids,
        )
        if self.val_case_ids is not None:
            self.val_dataset = DelicsRadialDataset(
                self.folder,
                case_glob=self.case_glob,
                case_ids=self.val_case_ids,
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

    def val_dataloader(self) -> torch.utils.data.DataLoader | None:
        """Return the optional DELICS validation dataloader."""
        if self.val_case_ids is None:
            return None
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
        n_coils: int = 8,
        noise: tuple[float, float] = (0.0, 0.05),
        r1_stats: tuple[float, float] = R1_STATS,
        r2_stats: tuple[float, float] = R2_STATS,
        orientation: Sequence[Literal['axial', 'coronal', 'sagittal']] = ('axial',),
        random: bool = True,
    ) -> None:
        """Initialize the dataset."""
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
        self.random = random

    def __len__(self) -> int:
        """Return the number of available slices."""
        return len(self.phantom)

    def __getitem__(self, index: int) -> BatchType:
        """Get one training sample."""
        phantom = self.phantom[index]
        seed = int(torch.randint(0, 1000000, (1,))) if self.random else index
        rng = mr2.utils.RandomGenerator(seed)

        return {
            'm0': phantom['m0'],
            'r1': relaxation_time_to_standardized_rate(phantom['t1'], self.r1_stats),
            'r2': relaxation_time_to_standardized_rate(phantom['t2'], self.r2_stats),
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
        n_coils: int = 8,
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
