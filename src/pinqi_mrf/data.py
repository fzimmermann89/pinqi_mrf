"""Data generation for PINQI-MRF experiments."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

import lightning.pytorch as pl
import mr2
import torch

from pinqi_mrf.parameters import R1_STATS, R2_STATS, relaxation_time_to_standardized_rate


class BatchType(TypedDict):
    """Type hint for a batch of data."""

    m0: torch.Tensor
    r1: torch.Tensor
    r2: torch.Tensor
    csm: torch.Tensor
    mask: torch.Tensor
    noise_std: torch.Tensor


class BrainwebDataset(torch.utils.data.Dataset):
    """BrainWeb-based dataset for parameter maps."""

    def __init__(
        self,
        folder: Path,
        size: int = 192,
        n_coils: int = 8,
        noise: tuple[float, float] = (0.0, 0.1),
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
            parameters=mr2.phantoms.brainweb.VALUES_3T_RANDOMIZED,
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
        rng = mr2.utils.RandomGenerator.RandomGenerator(seed)

        return {
            'm0': phantom['m0'],
            'r1': relaxation_time_to_standardized_rate(phantom['t1'], self.r1_stats),
            'r2': relaxation_time_to_standardized_rate(phantom['t2'], self.r2_stats),
            'csm': mr2.phantoms.coils.birdcage_2d(
                self.n_coils,
                self.matrix_size,
                relative_radius=rng.float32(1.2, 1.8),
            )
            if self.n_coils > 1
            else torch.ones(1, 1, *self.matrix_size.zyx, dtype=torch.complex64),
            'mask': phantom['mask'],
            'noise_std': torch.tensor(rng.float32(*self.noise)),
        }


class DataModule(pl.LightningDataModule):
    """Lightning data module for PINQI-MRF training."""

    def __init__(
        self,
        folder: Path,
        size: int = 192,
        n_coils: int = 8,
        noise: tuple[float, float] = (0.0, 0.1),
        r1_stats: tuple[float, float] = R1_STATS,
        r2_stats: tuple[float, float] = R2_STATS,
        batch_size: int = 4,
        num_workers: int = 4,
    ) -> None:
        """Initialize the data module."""
        super().__init__()
        self.save_hyperparameters(ignore=['folder', 'num_workers'])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = BrainwebDataset(
            folder=folder,
            size=size,
            n_coils=n_coils,
            noise=noise,
            r1_stats=r1_stats,
            r2_stats=r2_stats,
            orientation=('axial', 'coronal', 'sagittal'),
            random=True,
        )
        self.val_dataset = torch.utils.data.Subset(
            BrainwebDataset(
                folder=folder,
                size=size,
                n_coils=n_coils,
                noise=noise,
                r1_stats=r1_stats,
                r2_stats=r2_stats,
                orientation=('axial',),
                random=False,
            ),
            list(range(30, 500, 20)),
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the training dataloader."""
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
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
