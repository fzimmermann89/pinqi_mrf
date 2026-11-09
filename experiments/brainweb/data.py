"""BrainWeb saturation-recovery simulation with fixed, subject-specific evaluation draws."""

import hashlib
import tempfile
from bisect import bisect_right
from dataclasses import replace
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import mr2
import torch

DELAYS = (0.5, 1.0, 1.5, 2.0, 8.0)
SUBJECTS = {
    'train': (
        's04',
        's05',
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
    ),
    'validation': ('s51',),
    'test': ('s52', 's53', 's54'),
}
Sample = dict[str, torch.Tensor]


def acquisition_op(ky: torch.Tensor, csm: torch.Tensor) -> mr2.operators.LinearOperator:
    """Return coil sensitivities followed by Cartesian encoding of the acquired lines."""
    shape = csm.shape[-2:]
    kx = torch.arange(-shape[1] // 2, shape[1] // 2, device=ky.device, dtype=ky.dtype)
    trajectory = mr2.data.KTrajectory(ky.new_zeros((1,) * ky.ndim), ky, kx.reshape(*(1,) * (ky.ndim - 1), -1))
    matrix = mr2.data.SpatialDimension(1, *shape)
    return mr2.operators.FourierOp(matrix, matrix, trajectory) @ mr2.operators.SensitivityOp(csm)


def spatial_parameter_multiplier(
    mask: torch.Tensor, smooth_fraction: float, noise_fraction: float, generator: torch.Generator
) -> torch.Tensor:
    """Return bounded smooth and multiscale magnitude variation, unchanged outside tissue."""
    foreground = mask.squeeze()
    multiplier = torch.ones_like(foreground, dtype=torch.float32)
    if not foreground.any():
        return multiplier.reshape_as(mask)
    seed = int(torch.randint(2**31, (), generator=generator))
    coefficients = mr2.phantoms.random_b0_sh_coefficients(max_degree=4, sigma=1.0, seed=seed)
    field = mr2.phantoms.b0map_from_sh_coefficients(
        mr2.data.SpatialDimension(1, *foreground.shape), mr2.data.SpatialDimension(0.0, 1.0, 1.0), coefficients
    )[0].float()
    field -= field[foreground].mean()
    field /= field[foreground].abs().max().clamp_min(1e-06)
    multiplier += smooth_fraction * torch.rand((), generator=generator) * field
    field = torch.zeros_like(multiplier)
    for spacing, weight in ((20.0, 4 / 9), (10.0, 3 / 9), (3.0, 2 / 9)):
        spacing *= 0.75 + 0.5 * torch.rand((), generator=generator).item()
        shape = tuple(max(2, round(size / spacing)) for size in foreground.shape)
        grid = torch.randn((1, 1, *shape), generator=generator)
        smooth = torch.nn.functional.interpolate(grid, size=foreground.shape, mode='bicubic', align_corners=False)[0, 0]
        smooth = (smooth - smooth[foreground].mean()) / smooth[foreground].std(unbiased=False).clamp_min(1e-06)
        field += weight * smooth.tanh()
    multiplier += noise_fraction * torch.rand((), generator=generator) * field
    return torch.where(foreground, multiplier, 1).reshape_as(mask)


class SaturationRecoveryDataset(torch.utils.data.Dataset[Sample]):
    """Simulated undersampled multi-coil saturation-recovery k-space from axial BrainWeb slices."""

    def __init__(self, folder: Path, *, train: bool, step: int = 1) -> None:
        """Initialize the fixed BrainWeb experiment settings."""
        self.folder = folder
        self.image_shape = (192, 192)
        self.acceleration = 8.0
        self.n_center = 8
        self.noise_level = (0.001, 0.04)
        self.train = train
        self.fwhm_ratio = 0.56
        self.spatial_smooth_fraction = 0.2
        self.spatial_noise_fraction = 0.1
        self.spatial_phase_max_rad = 0.6
        self.step = step
        self.signal_model = mr2.operators.models.SaturationRecovery(DELAYS).requires_grad_(False)
        self.csm = mr2.phantoms.coils.birdcage_2d(8, mr2.data.SpatialDimension(1, 192, 192))[0]
        self.phantom = self.create_phantom()

    def create_phantom(self) -> mr2.phantoms.brainweb.BrainwebSlices:
        """Create the BrainWeb slice phantom."""
        parameters = dict(mr2.phantoms.brainweb.VALUES_3T_RANDOMIZED)
        if not self.train:
            parameters['gry'] = replace(parameters['gry'], t1=(1.3, 2.0))
            parameters['wht'] = replace(parameters['wht'], t1=(0.7, 1.3))
        size = max(self.image_shape)
        if self.train:
            preparation = mr2.phantoms.brainweb.augment(size)
        else:
            preparation = mr2.phantoms.brainweb.augment(
                size,
                max_random_shear=0,
                max_random_rotation=0,
                max_random_scaling_factor=0,
                p_horizontal_flip=0,
                p_vertical_flip=1.0,
            )
        return mr2.phantoms.brainweb.BrainwebSlices(
            self.folder,
            what=('m0', 't1', 'mask', 'gry', 'wht', 'csf'),
            parameters=parameters,
            orientation='axial',
            slice_preparation=preparation,
            step=self.step,
            seed='random' if self.train else 'index',
        )

    def __getstate__(self) -> dict[str, Any]:
        """Pickle without the phantom."""
        state = self.__dict__.copy()
        del state['phantom']
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Unpickle and create the phantom in the receiving process."""
        self.__dict__.update(state)
        # BrainwebSlices keeps the default generator; DataLoader seeds it after unpickling.
        self.phantom = self.create_phantom()

    def __len__(self) -> int:
        """Return the number of slices."""
        return len(self.phantom)

    def evaluation_seed(self, index: int, stream: str) -> int:
        """Stable subject + orientation + anatomical slice seed, independent of loader order/step."""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        physical_index = index * self.step
        chunk = bisect_right(self.phantom._ns_slices, physical_index) - 1
        path, axis = self.phantom._files_and_axes[chunk]
        slice_id = int(physical_index - self.phantom._ns_slices[chunk] + self.phantom.skip_slices[axis][0])
        key = f'sr-data-v2|{path.stem}|{axis}|{slice_id}|{stream}'.encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') & (1 << 63) - 1

    def __getitem__(self, index: int) -> Sample:
        """Simulate one slice."""
        if not self.train:
            self.phantom._rng = torch.Generator().manual_seed(self.evaluation_seed(index, 'tissue'))
        phantom = self.phantom[index]
        phantom = {key: value[None, None] for key, value in phantom.items()}
        generator = (
            torch.default_generator
            if self.train
            else torch.Generator().manual_seed(self.evaluation_seed(index, 'spatial'))
        )
        for key in ('m0', 't1'):
            phantom[key] = phantom[key] * spatial_parameter_multiplier(
                phantom['mask'], self.spatial_smooth_fraction, self.spatial_noise_fraction, generator
            )
        if phantom['mask'].any():
            seed = int(torch.randint(2**31, (), generator=generator))
            coefficients = mr2.phantoms.random_b0_sh_coefficients(max_degree=2, sigma=1.0, seed=seed)
            phase = mr2.phantoms.b0map_from_sh_coefficients(
                mr2.data.SpatialDimension(1, *self.image_shape), mr2.data.SpatialDimension(0.0, 1.0, 1.0), coefficients
            ).float()[None]
            foreground = phantom['mask']
            phase -= phase[foreground].mean()
            phase /= torch.quantile(phase[foreground].abs(), 0.9).clamp_min(1e-06)
            strength = self.spatial_phase_max_rad * torch.rand((), generator=generator)
            phantom['m0'] *= torch.polar(torch.ones_like(phase), strength * phase)
        seed = int(torch.randint(2**31, ())) if self.train else self.evaluation_seed(index, 'acquisition')
        trajectory = mr2.data.traj_calculators.KTrajectoryCartesian.gaussian_variable_density(
            mr2.data.SpatialDimension(1, *self.image_shape),
            acceleration=self.acceleration,
            fwhm_ratio=self.fwhm_ratio,
            n_center=self.n_center,
            n_other=(len(self.signal_model.saturation_time),),
            seed=seed,
        )
        (images,) = self.signal_model(phantom['m0'], phantom['t1'])
        (data,) = acquisition_op(trajectory.ky, self.csm)(images)
        generator = torch.Generator().manual_seed(seed)
        noise_level = torch.empty(()).uniform_(*self.noise_level, generator=generator)
        data = data + noise_level * torch.randn(data.shape, dtype=data.dtype, generator=generator)
        return {
            'data': data,
            'ky': trajectory.ky,
            'csm': self.csm,
            'm0': phantom['m0'],
            't1': phantom['t1'],
            'mask': phantom['mask'],
            'brain': phantom['gry'] + phantom['wht'] + phantom['csf'] > 0.5,
            'index': torch.tensor(index),
        }


def collate(samples: list[Sample]) -> Sample:
    """Stack samples; k-space data and lines get the batch as second dimension, after the delays."""
    stacked = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
    stacked['data'] = stacked['data'].movedim(0, 1)
    stacked['ky'] = stacked['ky'].movedim(0, 1)
    return stacked


class BrainWebData(pl.LightningDataModule):
    """Subject-disjoint loaders: 16 training subjects, s51 validation, s52-54 testing."""

    def __init__(self, folder: Path, batch_size: int = 8, workers: int = 4) -> None:
        """Initialize the fixed BrainWeb experiment settings."""
        super().__init__()
        self.folder = folder
        self.batch_size = batch_size
        self.workers = workers
        self.links = tempfile.TemporaryDirectory(prefix='brainweb-')

    def dataset(self, split: str, subjects: tuple[str, ...], step: int) -> SaturationRecoveryDataset:
        """Link only the requested subjects into the phantom's input folder."""
        folder = Path(self.links.name) / split
        folder.mkdir(exist_ok=True)
        for subject in subjects:
            source = (self.folder / f'{subject}.h5').resolve(strict=True)
            target = folder / source.name
            if not target.exists():
                target.symlink_to(source)
        return SaturationRecoveryDataset(folder, train=split == 'train', step=step)

    def setup(self, stage: str) -> None:
        """Create the fixed subject splits."""
        if stage == 'fit':
            self.train_dataset = self.dataset('train', SUBJECTS['train'], 1)
            self.validation_dataset = self.dataset('validation', SUBJECTS['validation'], 10)
        if stage == 'test':
            self.test_datasets = [self.dataset(subject, (subject,), 5) for subject in SUBJECTS['test']]

    def loader(self, dataset: SaturationRecoveryDataset, train: bool = False) -> torch.utils.data.DataLoader:
        """Stack acquisitions after the delay dimension; shuffle training only."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=self.workers,
            collate_fn=collate,
            pin_memory=True,
            multiprocessing_context='spawn' if self.workers else None,
            persistent_workers=self.workers > 0,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return random training draws."""
        return self.loader(self.train_dataset, True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the fixed validation draws."""
        return self.loader(self.validation_dataset)

    def test_dataloader(self) -> list[torch.utils.data.DataLoader]:
        """Return one fixed loader per test subject."""
        return [self.loader(dataset) for dataset in self.test_datasets]
