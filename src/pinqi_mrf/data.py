"""Data generation for PINQI-MRF experiments."""

import hashlib
import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Generic, Literal, NotRequired, TypedDict, TypeVar, overload

import h5py
import lightning.pytorch as pl
import mr2
import torch
import torchvision.transforms.functional
from lightning.pytorch.utilities.combined_loader import CombinedLoader

BrainwebTissueName = Literal['skl', 'gry', 'wht', 'csf', 'mrw', 'dura', 'fat', 'fat2', 'mus', 'm-s', 'ves']

BRAINWEB_VALUES_3T_RANDOMIZED: Mapping[BrainwebTissueName, mr2.phantoms.brainweb.BrainwebTissue] = {
    'skl': mr2.phantoms.brainweb.BrainwebTissue((0.000, 2.000), (0.020, 0.030), (0.00, 0.15), (-0.2, 0.2)),
    'gry': mr2.phantoms.brainweb.BrainwebTissue((0.700, 2.000), (0.025, 0.120), (0.60, 1.10), (-0.2, 0.2)),
    'wht': mr2.phantoms.brainweb.BrainwebTissue((0.550, 1.500), (0.020, 0.100), (0.40, 1.00), (-0.2, 0.2)),
    'csf': mr2.phantoms.brainweb.BrainwebTissue((1.400, 3.000), (0.100, 1.500), (0.20, 1.10), (-0.2, 0.2)),
    'mrw': mr2.phantoms.brainweb.BrainwebTissue((0.400, 0.600), (0.060, 0.100), (0.60, 1.10), (-0.2, 0.2)),
    'dura': mr2.phantoms.brainweb.BrainwebTissue((2.000, 2.800), (0.200, 0.500), (0.80, 1.10), (-0.2, 0.2)),
    'fat': mr2.phantoms.brainweb.BrainwebTissue((0.300, 0.500), (0.060, 0.100), (0.80, 1.10), (-0.2, 0.2)),
    'fat2': mr2.phantoms.brainweb.BrainwebTissue((0.400, 0.600), (0.060, 0.100), (0.50, 1.00), (-0.2, 0.2)),
    'mus': mr2.phantoms.brainweb.BrainwebTissue((1.200, 1.500), (0.040, 0.060), (0.80, 1.10), (-0.2, 0.2)),
    'm-s': mr2.phantoms.brainweb.BrainwebTissue((0.500, 0.900), (0.300, 0.500), (0.80, 1.10), (-0.2, 0.2)),
    'ves': mr2.phantoms.brainweb.BrainwebTissue((1.700, 2.100), (0.200, 0.400), (0.70, 1.10), (-0.2, 0.2)),
}


class BatchType(TypedDict):
    """Type hint for a batch of data."""

    m0: torch.Tensor
    t1: torch.Tensor
    t2: torch.Tensor
    csm: torch.Tensor
    mask: torch.Tensor
    noise_std: torch.Tensor


class DelicsRadialBatch(TypedDict):
    """Type hint for one DELICS radial k-space sample."""

    case_id: torch.Tensor
    slice_index: torch.Tensor
    kspace: torch.Tensor
    mask: torch.Tensor


class DelicsRadialReferenceBatch(BatchType):
    """Type hint for one DELICS validation sample with reference maps."""

    case_id: torch.Tensor
    slice_index: torch.Tensor
    kspace: torch.Tensor
    reference_m0: torch.Tensor
    reference_t1: torch.Tensor
    reference_t2: torch.Tensor


class FineTuneBatch(TypedDict):
    """Type hint for one combined fine-tuning batch."""

    delics: DelicsRadialBatch
    synthetic: BatchType
    supervised_delics: NotRequired[DelicsRadialReferenceBatch]


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
        delics_scale: float = 1.0,
        delics_noise_std: float = 4e-3,
        slice_trim: tuple[int, int] = (0, 0),
        include_references: Literal[False] = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        folder: Path,
        *,
        case_glob: str = 'delics_radial_case*.h5',
        case_ids: Sequence[str] | None = None,
        delics_scale: float = 1.0,
        delics_noise_std: float = 4e-3,
        slice_trim: tuple[int, int] = (0, 0),
        include_references: Literal[True],
    ) -> None: ...

    def __init__(
        self,
        folder: Path,
        *,
        case_glob: str = 'delics_radial_case*.h5',
        case_ids: Sequence[str] | None = None,
        delics_scale: float = 1.0,
        delics_noise_std: float = 4e-3,
        slice_trim: tuple[int, int] = (0, 0),
        include_references: bool = False,
    ) -> None:
        """Initialize the DELICS radial dataset."""
        self.folder = folder
        self.case_glob = case_glob
        self.delics_scale = delics_scale
        self.delics_noise_std = delics_noise_std
        self.slice_trim = slice_trim
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
                first_slice = slice_trim[0]
                last_slice = len(valid) - slice_trim[1]
                for slice_index, is_valid in enumerate(valid):
                    if first_slice <= slice_index < last_slice and bool(is_valid):
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
            sample: DelicsRadialBatch = {
                'case_id': torch.tensor(file_index, dtype=torch.int64),
                'slice_index': torch.tensor(slice_index, dtype=torch.int64),
                'kspace': kspace,
                'mask': torch.as_tensor(handle['mask'][slice_index], dtype=torch.bool),
            }
            if self.include_references:
                csm_initial = torch.as_tensor(handle['csm_initial'][slice_index], dtype=torch.complex64)
                reference_m0 = torch.as_tensor(handle['reference_m0'][slice_index], dtype=torch.complex64)
                reference_t1 = torch.as_tensor(handle['reference_t1'][slice_index], dtype=torch.float32)
                reference_t2 = torch.as_tensor(handle['reference_t2'][slice_index], dtype=torch.float32)
                return {
                    **sample,
                    'm0': reference_m0,
                    't1': reference_t1,
                    't2': reference_t2,
                    'csm': csm_initial,
                    'noise_std': torch.tensor(0.0, dtype=torch.float32),
                    'reference_m0': reference_m0,
                    'reference_t1': reference_t1,
                    'reference_t2': reference_t2,
                }
            return sample


class DelicsRadialDataModule(pl.LightningDataModule):
    """Lightning data module for pseudo-acquired DELICS radial k-space."""

    def __init__(
        self,
        folder: Path = Path('/scratch/zimmer08/delics/radial_kspace_new'),
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


def prepare_brainweb_slice(
    data: torch.Tensor,
    generator: torch.Generator | None = None,
    *,
    size: int,
    random: bool,
) -> torch.Tensor:
    """Apply MR2's affine recipe with scale 0.70-1.00 and fixed front/back orientation.

    MR2's augmentation accepts only symmetric scale jitter. This uses its same trim,
    affine interpolation and crop operations with an asymmetric scale draw instead.
    Validation disables geometric jitter and horizontal flips.
    """
    draws = torch.empty(6).uniform_(-1, 1, generator=generator).tolist()
    scale = size / max(data.shape[-2:])
    if random:
        scale *= 0.85 + 0.15 * draws[3]
    data = data[(slice(None), *mr2.phantoms.brainweb.trim_indices(data.sum(0) > 0.1 * data.amax()))]
    data = torchvision.transforms.functional.affine(
        data,
        angle=10 * draws[2] if random else 0,
        scale=scale,
        shear=[5 * draws[0], 5 * draws[1]] if random else [0, 0],
        translate=draws[4:6],
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        fill=0.0,
    )
    if random and torch.rand((), generator=generator) < 0.5:
        data = torchvision.transforms.functional.hflip(data)
    data = torchvision.transforms.functional.vflip(data)
    return torchvision.transforms.functional.center_crop(data, [size, size])


def spatial_parameter_multiplier(
    mask: torch.Tensor,
    smooth_fraction: float | tuple[float, float],
    noise_fraction: float,
    generator: torch.Generator,
    *,
    degree_two_scale: float = 1.0,
    radial_fraction: float | tuple[float, float] = 0.0,
) -> torch.Tensor:
    """Return bounded smooth and multiscale magnitude variation, unchanged outside tissue.

    The degree-four field follows the MRpro PINQI recipe. Three bicubic Gaussian grids add finer
    variation at approximately 20, 10 and 6 pixels, following its later multiscale recipe.
    Independent calls give independent fields for M0 magnitude, T1 and T2. Degree-two
    coefficients can be amplified before normalizing the combined harmonic field. An optional
    elliptical quadratic field brightens the tissue periphery by a uniform draw from
    ``radial_fraction`` (a scalar means zero to that value); its centre and radii follow
    the tissue bounding box.
    """
    foreground = mask.squeeze()
    multiplier = torch.ones_like(foreground, dtype=torch.float32)
    if not foreground.any():
        return multiplier.reshape_as(mask)
    smooth_min, smooth_max = smooth_fraction if isinstance(smooth_fraction, tuple) else (0.0, smooth_fraction)
    if smooth_max:
        seed = int(torch.randint(2**31, (), generator=generator))
        coefficients = mr2.phantoms.random_b0_sh_coefficients(max_degree=4, sigma=1.0, seed=seed)
        coefficients[3:8] *= degree_two_scale  # Five degree-two terms, following three degree-one terms.
        field = mr2.phantoms.b0map_from_sh_coefficients(
            mr2.data.SpatialDimension(1, *foreground.shape),
            mr2.data.SpatialDimension(0.0, 1.0, 1.0),
            coefficients,
        )[0].float()
        field -= field[foreground].mean()
        field /= field[foreground].abs().max().clamp_min(1e-6)
        strength = smooth_min + (smooth_max - smooth_min) * torch.rand((), generator=generator)
        multiplier += strength * field
    if noise_fraction:
        field = torch.zeros_like(multiplier)
        for spacing, weight in ((20.0, 4 / 9), (10.0, 3 / 9), (6.0, 2 / 9)):
            spacing *= 0.75 + 0.5 * torch.rand((), generator=generator).item()
            shape = tuple(max(2, round(size / spacing)) for size in foreground.shape)
            grid = torch.randn((1, 1, *shape), generator=generator)
            smooth = torch.nn.functional.interpolate(grid, size=foreground.shape, mode='bicubic', align_corners=False)[
                0, 0
            ]
            smooth = (smooth - smooth[foreground].mean()) / smooth[foreground].std(unbiased=False).clamp_min(1e-6)
            field += weight * smooth.tanh()
        multiplier += noise_fraction * torch.rand((), generator=generator) * field
    radial_min, radial_max = radial_fraction if isinstance(radial_fraction, tuple) else (0.0, radial_fraction)
    if radial_max:
        coordinates = torch.nonzero(foreground).float()
        lower, upper = coordinates.amin(dim=0), coordinates.amax(dim=0)
        centre = (lower + upper) / 2
        radii = ((upper - lower) / 2).clamp_min(1)
        y, x = torch.meshgrid(
            torch.arange(foreground.shape[0], dtype=torch.float32),
            torch.arange(foreground.shape[1], dtype=torch.float32),
            indexing='ij',
        )
        radius_squared = ((y - centre[0]) / radii[0]) ** 2 + ((x - centre[1]) / radii[1]) ** 2
        strength = radial_min + (radial_max - radial_min) * torch.rand((), generator=generator)
        multiplier *= 1 + strength * radius_squared.clamp(max=1)
    return torch.where(foreground, multiplier, 1).reshape_as(mask)


class BrainwebDataset(torch.utils.data.Dataset):
    """BrainWeb-based dataset for parameter maps."""

    def __init__(
        self,
        folder: Path,
        size: int = 256,
        n_coils: int = 12,
        noise: tuple[float, float] = (0.0, 0.05),
        t1_bounds: tuple[float, float] = (0.1, 4.0),
        t2_bounds: tuple[float, float] = (0.01, 2.5),
        orientation: Sequence[Literal['axial', 'coronal', 'sagittal']] = ('axial',),
        spatial_smooth_fraction: float = 0.4,
        spatial_m0_smooth_range: tuple[float, float] = (0.2, 0.8),
        spatial_m0_degree_two_scale: float = 2.0,
        spatial_m0_radial_range: tuple[float, float] = (0.2, 0.8),
        spatial_noise_fraction: float = 0.05,
        spatial_phase_max_rad: float = 0.6,
        global_m0_magnitude_range: tuple[float, float] = (0.7, 1.3),
        random: bool = True,
    ) -> None:
        """Initialize MR2 affine and spherical-harmonic/multiscale map augmentation.

        Independent M0, T1 and T2 fields combine smooth degree-four variation with
        finer multiscale variation. M0 strength is drawn uniformly from its range;
        T1 and T2 strengths are drawn from zero to the relaxation-map maximum.
        M0 degree-two coefficients have twice the default spread; a separate quadratic
        centre-to-edge multiplier adds 20 to 80% peripheral brightening by default.
        Spatial phase uses degree two with the specified maximum strength in radians.
        Validation uses the same map augmentation with fixed slice seeds.
        """
        if (
            min(spatial_smooth_fraction, *spatial_m0_smooth_range, spatial_noise_fraction) < 0
            or spatial_m0_smooth_range[0] > spatial_m0_smooth_range[1]
            or max(spatial_smooth_fraction, spatial_m0_smooth_range[1]) + spatial_noise_fraction >= 1
        ):
            raise ValueError('Spatial magnitude fractions must be non-negative and sum to less than one.')
        if (
            spatial_m0_degree_two_scale < 0
            or min(spatial_m0_radial_range) < 0
            or spatial_m0_radial_range[0] > spatial_m0_radial_range[1]
        ):
            raise ValueError('M0 harmonic scaling and radial bounds must be non-negative, with ordered bounds.')
        if spatial_phase_max_rad < 0:
            raise ValueError('Spatial phase strength must be non-negative.')
        self.matrix_size = mr2.data.SpatialDimension(1, size, size)
        self.n_coils = n_coils
        self.noise = noise
        self.t1_bounds = t1_bounds
        self.t2_bounds = t2_bounds
        self.spatial_smooth_fraction = spatial_smooth_fraction
        self.spatial_m0_smooth_range = spatial_m0_smooth_range
        self.spatial_m0_degree_two_scale = spatial_m0_degree_two_scale
        self.spatial_m0_radial_range = spatial_m0_radial_range
        self.spatial_noise_fraction = spatial_noise_fraction
        self.spatial_phase_max_rad = spatial_phase_max_rad
        self.global_m0_magnitude_range = global_m0_magnitude_range
        self.random = random
        self.folder = folder
        self.orientation = tuple(orientation)
        self._build_phantom()

    def _build_phantom(self) -> None:
        """Build augmentation and its random generator in the current process."""
        augment = partial(prepare_brainweb_slice, size=self.matrix_size.x, random=self.random)
        self.phantom = mr2.phantoms.brainweb.BrainwebSlices(
            folder=self.folder,
            what=('m0', 't1', 't2', 'mask'),
            seed='index' if not self.random else 'random',
            slice_preparation=augment,
            parameters=BRAINWEB_VALUES_3T_RANDOMIZED,
            orientation=self.orientation,
        )

    def __getstate__(self) -> dict:
        """Rebuild process-local BrainWeb state when spawning a worker."""
        return {key: value for key, value in self.__dict__.items() if key != 'phantom'}

    def __setstate__(self, state: dict) -> None:
        """Use the worker's default generator and avoid pickling local closures."""
        self.__dict__.update(state)
        self._build_phantom()

    def _augment_parameter_maps(
        self,
        m0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        mask: torch.Tensor,
        rng: mr2.utils.RandomGenerator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply independent spatial magnitude fields and a shared signal phase."""
        m0, t1, t2 = (
            image
            * spatial_parameter_multiplier(
                mask,
                smooth_fraction,
                self.spatial_noise_fraction,
                rng.generator,
                degree_two_scale=degree_two_scale,
                radial_fraction=radial_fraction,
            )
            for image, smooth_fraction, degree_two_scale, radial_fraction in (
                (m0, self.spatial_m0_smooth_range, self.spatial_m0_degree_two_scale, self.spatial_m0_radial_range),
                (t1, self.spatial_smooth_fraction, 1.0, 0.0),
                (t2, self.spatial_smooth_fraction, 1.0, 0.0),
            )
        )
        if self.spatial_phase_max_rad and mask.any():
            seed = int(torch.randint(2**31, (), generator=rng.generator))
            phase = mr2.phantoms.random_b0map(
                self.matrix_size,
                mr2.data.SpatialDimension(0.0, 1.0, 1.0),
                l_max=2,
                sigma_ppm=1.0,
                seed=seed,
            ).float()[0]
            foreground = mask.squeeze().bool()
            phase -= phase[foreground].mean()
            phase /= torch.quantile(phase[foreground].abs(), 0.9).clamp_min(1e-6)
            strength = self.spatial_phase_max_rad * torch.rand((), generator=rng.generator)
            m0 = m0 * torch.polar(torch.ones_like(phase), strength * phase)
        global_magnitude = rng.float32(*self.global_m0_magnitude_range)
        global_phase = rng.float32(-math.pi, math.pi)
        global_scale = torch.polar(torch.tensor(global_magnitude, dtype=torch.float32), torch.tensor(global_phase))
        return m0 * global_scale, t1, t2

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
        m0, t1, t2 = self._augment_parameter_maps(m0, t1, t2, phantom['mask'], rng)

        return {
            'm0': m0,
            't1': t1.clamp(*self.t1_bounds),
            't2': t2.clamp(*self.t2_bounds),
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
    """Shared validation datamodule with BrainWeb and DELICS validation loaders."""

    def __init__(
        self,
        folder: Path = mr2.phantoms.brainweb.CACHE_DIR_BRAINWEB,
        *,
        delics_folder: Path = Path('/scratch/zimmer08/delics/radial_kspace_new'),
        delics_val_case_ids: Sequence[str] = ('case000', 'case007', 'case100'),
        delics_case_glob: str = 'delics_radial_case*.h5',
        size: int = 256,
        n_coils: int = 12,
        noise: tuple[float, float] = (0.0, 0.05),
        t1_bounds: tuple[float, float] = (0.1, 4.0),
        t2_bounds: tuple[float, float] = (0.01, 2.5),
        batch_size: int = 6,
        num_workers: int = 4,
    ) -> None:
        """Initialize the data module."""
        super().__init__()
        self.save_hyperparameters(ignore=['folder', 'num_workers', 'delics_folder'])
        self.folder = folder
        self.delics_folder = delics_folder
        self.delics_val_case_ids = delics_val_case_ids
        self.delics_case_glob = delics_case_glob
        self.size = size
        self.n_coils = n_coils
        self.noise = noise
        self.t1_bounds = t1_bounds
        self.t2_bounds = t2_bounds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.synthetic_val_dataset: torch.utils.data.Subset[BrainwebDataset] | None = None
        self.delics_val_dataset: DelicsRadialDataset[Literal[True]] | None = None

    def build_brainweb_dataset(
        self,
        *,
        orientation: Sequence[Literal['axial', 'coronal', 'sagittal']],
        random: bool,
    ) -> BrainwebDataset:
        """Build a BrainWeb parameter-map dataset with the shared settings."""
        return BrainwebDataset(
            folder=self.folder,
            size=self.size,
            n_coils=self.n_coils,
            noise=self.noise,
            t1_bounds=self.t1_bounds,
            t2_bounds=self.t2_bounds,
            orientation=orientation,
            random=random,
        )

    @overload
    def build_delics_dataset(
        self,
        *,
        case_ids: Sequence[str] | None,
        slice_trim: tuple[int, int],
        include_references: Literal[False],
    ) -> DelicsRadialDataset[Literal[False]]: ...

    @overload
    def build_delics_dataset(
        self,
        *,
        case_ids: Sequence[str] | None,
        slice_trim: tuple[int, int],
        include_references: Literal[True],
    ) -> DelicsRadialDataset[Literal[True]]: ...

    def build_delics_dataset(
        self,
        *,
        case_ids: Sequence[str] | None,
        slice_trim: tuple[int, int],
        include_references: Literal[False] | Literal[True],
    ) -> DelicsRadialDataset[Literal[False]] | DelicsRadialDataset[Literal[True]]:
        """Build a DELICS radial dataset with the shared settings."""
        return DelicsRadialDataset(
            self.delics_folder,
            case_glob=self.delics_case_glob,
            case_ids=case_ids,
            slice_trim=slice_trim,
            include_references=include_references,
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
    ) -> torch.utils.data.DataLoader:
        """Build a dataloader with the project defaults."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            worker_init_fn=lambda *_: torch.set_num_threads(1),
            drop_last=drop_last,
        )

    @property
    def validation_loader_types(self) -> tuple[ValidationLoaderType, ...]:
        """Return the ordered validation loader types."""
        return (ValidationLoaderType.SYNTHETIC, ValidationLoaderType.KSPACE)

    def prepare_data(self) -> None:
        """Download BrainWeb data if it is not available locally."""
        if not self.folder.exists():
            self.folder.mkdir(parents=True, exist_ok=True)
            mr2.phantoms.brainweb.download_brainweb(output_directory=self.folder, workers=2, progress=True)

    def setup(self, stage: str | None = None) -> None:
        """Create the shared validation datasets."""
        _ = stage
        if self.synthetic_val_dataset is None:
            self.synthetic_val_dataset = torch.utils.data.Subset(
                self.build_brainweb_dataset(
                    orientation=('axial',),
                    random=False,
                ),
                list(range(30, 500, 20)),
            )
        if self.delics_val_dataset is None:
            self.delics_val_dataset = self.build_delics_dataset(
                case_ids=self.delics_val_case_ids,
                slice_trim=(0, 0),
                include_references=True,
            )

    def synthetic_val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the BrainWeb validation dataloader."""
        if self.synthetic_val_dataset is None:
            raise RuntimeError('DataModule.setup() must be called before synthetic_val_dataloader().')
        return self.build_dataloader(
            self.synthetic_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
        )

    def delics_val_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the DELICS validation dataloader."""
        if self.delics_val_dataset is None:
            raise RuntimeError('DataModule.setup() must be called before delics_val_dataloader().')
        return self.build_dataloader(
            self.delics_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
        )

    def val_dataloader(self) -> list[torch.utils.data.DataLoader]:
        """Return BrainWeb validation and DELICS validation loaders."""
        return [self.synthetic_val_dataloader(), self.delics_val_dataloader()]


class PretrainDataModule(DataModule):
    """Synthetic pretraining with shared validation and BrainWeb training."""

    def __init__(
        self,
        folder: Path = mr2.phantoms.brainweb.CACHE_DIR_BRAINWEB,
        *,
        delics_val_folder: Path = Path('/scratch/zimmer08/delics/radial_kspace_new'),
        delics_val_case_ids: Sequence[str] = ('case000', 'case007', 'case100'),
        size: int = 256,
        n_coils: int = 12,
        noise: tuple[float, float] = (0.0, 0.05),
        t1_bounds: tuple[float, float] = (0.1, 4.0),
        t2_bounds: tuple[float, float] = (0.01, 2.5),
        batch_size: int = 6,
        num_workers: int = 4,
        delics_val_case_glob: str = 'delics_radial_case*.h5',
    ) -> None:
        """Initialize the pretraining datamodule."""
        super().__init__(
            folder=folder,
            delics_folder=delics_val_folder,
            delics_val_case_ids=delics_val_case_ids,
            delics_case_glob=delics_val_case_glob,
            size=size,
            n_coils=n_coils,
            noise=noise,
            t1_bounds=t1_bounds,
            t2_bounds=t2_bounds,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.train_dataset: BrainwebDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up shared validation and BrainWeb training."""
        super().setup(stage)
        if self.train_dataset is None:
            self.train_dataset = self.build_brainweb_dataset(
                orientation=('axial', 'coronal', 'sagittal'),
                random=True,
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Return the BrainWeb training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError('PretrainDataModule.setup() must be called before train_dataloader().')
        return self.build_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


class FineTuneDataModule(DataModule):
    """DELICS fine-tuning with shared validation and DELICS training."""

    def __init__(
        self,
        folder: Path = Path('/scratch/zimmer08/delics/radial_kspace_new'),
        *,
        val_case_ids: Sequence[str] = ('case000', 'case007', 'case100'),
        train_case_ids: Sequence[str] | None = None,
        case_glob: str = 'delics_radial_case*.h5',
        delics_batch_size: int = 1,
        synthetic_batch_size: int = 2,
        supervised_delics_batch_size: int = 0,
        num_workers: int = 4,
        brainweb_folder: Path = mr2.phantoms.brainweb.CACHE_DIR_BRAINWEB,
        brainweb_size: int = 256,
        brainweb_n_coils: int = 12,
        brainweb_noise: tuple[float, float] = (0.0, 0.05),
        brainweb_t1_bounds: tuple[float, float] = (0.1, 4.0),
        brainweb_t2_bounds: tuple[float, float] = (0.01, 2.5),
        val_batch_size: int = 6,
        val_num_workers: int = 0,
    ) -> None:
        """Initialize the fine-tuning datamodule."""
        super().__init__(
            folder=brainweb_folder,
            delics_folder=folder,
            delics_val_case_ids=val_case_ids,
            delics_case_glob=case_glob,
            size=brainweb_size,
            n_coils=brainweb_n_coils,
            noise=brainweb_noise,
            t1_bounds=brainweb_t1_bounds,
            t2_bounds=brainweb_t2_bounds,
            batch_size=val_batch_size,
            num_workers=val_num_workers,
        )
        self.save_hyperparameters(ignore=['folder'])
        self.train_case_ids = train_case_ids
        self.delics_batch_size = delics_batch_size
        self.synthetic_batch_size = synthetic_batch_size
        self.supervised_delics_batch_size = supervised_delics_batch_size
        self.train_num_workers = num_workers
        self.delics_train_dataset: DelicsRadialDataset[Literal[False]] | None = None
        self.supervised_delics_train_dataset: DelicsRadialDataset[Literal[True]] | None = None
        self.synthetic_train_dataset: BrainwebDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up shared validation and fine-tuning training datasets."""
        super().setup(stage)
        if self.delics_train_dataset is None:
            self.delics_train_dataset = self.build_delics_dataset(
                case_ids=self.train_case_ids,
                slice_trim=(0, 0),
                include_references=False,
            )
        if self.supervised_delics_batch_size > 0 and self.supervised_delics_train_dataset is None:
            self.supervised_delics_train_dataset = self.build_delics_dataset(
                case_ids=self.train_case_ids,
                slice_trim=(0, 0),
                include_references=True,
            )
        if self.synthetic_train_dataset is None:
            self.synthetic_train_dataset = self.build_brainweb_dataset(
                orientation=('axial', 'coronal', 'sagittal'),
                random=True,
            )

    def train_dataloader(self) -> CombinedLoader:
        """Return paired DELICS and synthetic training loaders."""
        if self.delics_train_dataset is None or self.synthetic_train_dataset is None:
            raise RuntimeError('FineTuneDataModule.setup() must be called before train_dataloader().')
        loaders: dict[str, torch.utils.data.DataLoader] = {}
        delics_loader = torch.utils.data.DataLoader(
            self.delics_train_dataset,
            batch_size=self.delics_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=4 if self.train_num_workers > 0 else None,
            persistent_workers=self.train_num_workers > 0,
            worker_init_fn=lambda *_: torch.set_num_threads(1),
            drop_last=False,
        )
        loaders['delics'] = delics_loader
        synthetic_sampler = torch.utils.data.RandomSampler(
            self.synthetic_train_dataset,
            replacement=True,
            num_samples=len(delics_loader) * self.synthetic_batch_size,
        )
        synthetic_loader = torch.utils.data.DataLoader(
            self.synthetic_train_dataset,
            batch_size=self.synthetic_batch_size,
            shuffle=False,
            sampler=synthetic_sampler,
            num_workers=self.train_num_workers,
            pin_memory=True,
            prefetch_factor=4 if self.train_num_workers > 0 else None,
            persistent_workers=self.train_num_workers > 0,
            worker_init_fn=lambda *_: torch.set_num_threads(1),
            drop_last=False,
        )
        loaders['synthetic'] = synthetic_loader
        if self.supervised_delics_batch_size > 0:
            if self.supervised_delics_train_dataset is None:
                raise RuntimeError('FineTuneDataModule.setup() must create supervised DELICS dataset before training.')
            supervised_delics_sampler = torch.utils.data.RandomSampler(
                self.supervised_delics_train_dataset,
                replacement=True,
                num_samples=len(delics_loader) * self.supervised_delics_batch_size,
            )
            loaders['supervised_delics'] = torch.utils.data.DataLoader(
                self.supervised_delics_train_dataset,
                batch_size=self.supervised_delics_batch_size,
                shuffle=False,
                sampler=supervised_delics_sampler,
                num_workers=self.train_num_workers,
                pin_memory=torch.cuda.is_available(),
                prefetch_factor=4 if self.train_num_workers > 0 else None,
                persistent_workers=self.train_num_workers > 0,
                worker_init_fn=lambda *_: torch.set_num_threads(1),
                drop_last=False,
            )
        return CombinedLoader(loaders, mode='min_size')
