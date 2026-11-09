"""Load compressed DELICS frames and prepare augmented training inputs on CUDA."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import h5py
import mr2
import numpy as np
import torch
import torch.nn.functional as functional

from pinqi_mrf.data import DelicsRadialDataset


class DelicsFrameBatch(TypedDict):
    """Unbatched worker output; frames have dimensions time, coil, y, x."""

    frames: NotRequired[torch.Tensor]
    m0: torch.Tensor
    t1: torch.Tensor
    t2: torch.Tensor
    mask: torch.Tensor
    noise_std: torch.Tensor
    bias_field: torch.Tensor
    bias_strength: torch.Tensor


class DelicsFrameDataset(torch.utils.data.Dataset[DelicsFrameBatch]):
    """Load prepared frames and the original full-acquisition reference maps."""

    def __init__(
        self,
        reference: DelicsRadialDataset,
        folder: Path,
        n_timepoints: int = 352,
        n_coils: int = 8,
        noise: tuple[float, float] = (0.002, 0.006),
    ) -> None:
        """Require a complete frame cache for the explicitly selected training slices.

        Parameters
        ----------
        reference
            Radial dataset supplying the training partition, valid slices and reference maps.
        folder
            Directory of per-slice caches written by ``prepare_delics_frames``.
        n_timepoints
            Number of initial RF pulses retained in the simulated acquisition.
        n_coils
            Number of compressed coils. Compression is estimated from the retained prefix.
        noise
            Uniform range for complex k-space noise RMS, after the cached signal scaling.
        """
        if not 0 <= noise[0] <= noise[1]:
            raise ValueError('Noise bounds must be non-negative and ordered.')
        self.files = reference.files
        self.samples = reference.samples
        self.folder = folder
        self.n_timepoints = n_timepoints
        self.n_coils = n_coils
        self.noise = noise
        for file_index, (_path, case) in enumerate(self.files):
            with h5py.File(self.cache_path(case), 'r') as cached:
                for file, slice_index in self.samples:
                    if file == file_index and f'z{slice_index:03d}' not in cached:
                        raise ValueError(f'Missing frame cache for {case}, slice {slice_index}.')

    def cache_path(self, case: str) -> Path:
        """Return the cache path for this temporal prefix and coil count."""
        return self.folder / f'{case}-t{self.n_timepoints}-c{self.n_coils}.h5'

    def __len__(self) -> int:
        """Return the number of valid training slices."""
        return len(self.samples)

    def __getitem__(self, index: int) -> DelicsFrameBatch:
        """Read frames on CPU and draw fresh noise strength and a degree-four bias field."""
        file_index, slice_index = self.samples[index]
        _path, case = self.files[file_index]
        with h5py.File(self.cache_path(case), 'r') as handle:
            cache = handle[f'z{slice_index:03d}']
            packed = cache['frames'][()]
            frames = torch.view_as_complex(torch.from_numpy(packed.view(np.float16).reshape(*packed.shape, 2)))
            m0 = torch.from_numpy(cache['m0'][()])
            t1 = torch.from_numpy(cache['t1'][()])
            t2 = torch.from_numpy(cache['t2'][()])
            mask = torch.from_numpy(cache['mask'][()])
        seed = int(torch.randint(2**31, ()))
        coefficients = mr2.phantoms.random_b0_sh_coefficients(max_degree=4, sigma=1.0, seed=seed)
        field = mr2.phantoms.b0map_from_sh_coefficients(
            mr2.data.SpatialDimension(1, *mask.shape),
            mr2.data.SpatialDimension(0.0, 1.0, 1.0),
            coefficients,
        )[0].float()
        return {
            'frames': frames,
            'm0': m0,
            't1': t1,
            't2': t2,
            'mask': mask,
            'noise_std': self.noise[0] + (self.noise[1] - self.noise[0]) * torch.rand(()),
            'bias_field': field,
            'bias_strength': 0.3 * torch.rand(()),
        }


@dataclass
class _Transform:
    """One shared spatial transform for all coil frames and target maps."""

    y: torch.Tensor
    x: torch.Tensor
    grid: torch.Tensor | None


def _draw_transform(shape: tuple[int, int], device: torch.device) -> _Transform:
    """Draw a left-right flip, integer translations and occasional mild affine transform."""
    height, width = shape
    shift_y, shift_x = torch.randint(-10, 11, (2,)).tolist()
    y = torch.arange(height, device=device) - shift_y
    x = torch.arange(width, device=device) - shift_x
    if torch.rand(()) < 0.5:
        x = width - 1 - x
    # Reflection without a validity mask; index gathering keeps integer shifts exact.
    y = y.remainder(2 * (height - 1))
    x = x.remainder(2 * (width - 1))
    y = torch.where(y >= height, 2 * (height - 1) - y, y)
    x = torch.where(x >= width, 2 * (width - 1) - x, x)
    grid = None
    if torch.rand(()) < 0.3:
        angle = (torch.rand(()) * 20 - 10) * torch.pi / 180
        scale = 0.95 + 0.15 * torch.rand(())
        translation = torch.rand(2) * 2 - 1
        theta = torch.tensor(
            [
                [angle.cos() / scale, -angle.sin() / scale, 2 * translation[0] / width],
                [angle.sin() / scale, angle.cos() / scale, 2 * translation[1] / height],
            ],
            device=device,
        )
        grid = functional.affine_grid(theta[None], [1, 1, height, width], align_corners=False)
    return _Transform(y, x, grid)


def _warp(data: torch.Tensor, transform: _Transform, *, nearest: bool = False) -> torch.Tensor:
    """Apply the same reflection-padded geometry to a stack of real or complex images."""
    data = data.index_select(-2, transform.y).index_select(-1, transform.x)
    if transform.grid is None:
        return data
    is_complex = data.is_complex()
    if is_complex:
        n_images, n_channels, height, width = data.shape
        data = torch.view_as_real(data).permute(0, 1, 4, 2, 3).reshape(n_images, 2 * n_channels, height, width)
    data = functional.grid_sample(
        data,
        transform.grid.expand(len(data), -1, -1, -1),
        mode='nearest' if nearest else 'bilinear',
        padding_mode='reflection',
        align_corners=False,
    )
    if is_complex:
        data = torch.view_as_complex(
            data.reshape(n_images, n_channels, 2, height, width).permute(0, 1, 3, 4, 2).contiguous()
        )
    return data


class DelicsFramePreparation:
    """Build a subspace RHS without retaining time images or synthesizing signal k-space.

    The signal uses each frame's Toeplitz normal operator before subspace projection.
    Added noise is independent complex white k-space noise passed through the actual
    adjoint. Only fixed kernels, trajectory, basis and cuFINUFFT plans persist.
    """

    def __init__(self, nufft: mr2.operators.NonUniformFastFourierOp, basis: torch.Tensor) -> None:
        """Prepare the fixed trajectory and frame-normal kernels on the training device."""
        if basis.ndim != 2 or basis.device.type != 'cuda':
            raise ValueError('Full-frame preparation requires a two-dimensional CUDA subspace basis.')
        self.nufft = nufft
        self.basis = basis
        # MR2 exposes the frame normal through its Gram operator; reuse its scaled kernel.
        gram = nufft.toeplitz()
        self.kernel = cast(torch.Tensor, gram._kernel).squeeze()
        self.omega = nufft._omega.contiguous()
        self.plans: dict[tuple[int, int], Any] = {}

    def close(self) -> None:
        """Destroy native plans while the CUDA context is still active."""
        if self.plans:
            torch.cuda.current_stream(self.basis.device).synchronize()
            self.plans.clear()

    @torch.no_grad()
    def noise_rhs(
        self,
        standard_deviation: torch.Tensor,
        n_coils: int,
    ) -> torch.Tensor:
        """Project exact white k-space noise through a reusable type-one NUFFT plan."""
        import cufinufft

        n_timepoints, n_coefficients = self.basis.shape
        height, width = self.kernel.shape[-2] // 2, self.kernel.shape[-1] // 2
        batch_size = len(standard_deviation)
        n_transforms = batch_size * n_coils
        stream = torch.cuda.current_stream(self.basis.device)
        key = (n_transforms, stream.cuda_stream)
        if key not in self.plans:
            self.plans[key] = cufinufft.Plan(
                1,
                (height, width),
                n_transforms,
                eps=1e-6,
                isign=1,
                dtype='complex64',
                upsampfac=2.0,
                modeord=0,
                gpu_device_id=self.basis.device.index,
                gpu_stream=stream.cuda_stream,
                gpu_method=1,
                gpu_maxbatchsize=n_transforms,
            )
        plan = self.plans[key]
        noise = torch.randn(
            (n_timepoints, batch_size, n_coils, self.omega.shape[-1]),
            dtype=torch.complex64,
            device=self.basis.device,
        )
        # Complex randn already has unit total variance; do not divide by sqrt(2) again.
        noise.mul_(standard_deviation[None, :, None, None])
        noise = noise.reshape(n_timepoints, n_transforms, -1)
        rhs = torch.zeros(
            (n_coefficients, n_transforms, height, width), device=self.basis.device, dtype=torch.complex64
        )
        for start in range(0, n_timepoints, 16):
            end = min(start + 16, n_timepoints)
            images = torch.empty(
                (end - start, n_transforms, height, width), device=self.basis.device, dtype=torch.complex64
            )
            for time in range(start, end):
                plan.setpts(self.omega[time, 0], self.omega[time, 1])
                plan.execute(noise[time], out=images[time - start])
            rhs.add_(torch.einsum('tk,tcyx->kcyx', self.basis[start:end].conj() * self.nufft.scale, images))
            del images
        return rhs.reshape(n_coefficients, batch_size, n_coils, 1, height, width).movedim(1, 0).contiguous()

    @torch.no_grad()
    def __call__(self, batch: DelicsFrameBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Consume frames and return augmented targets plus the compressed subspace RHS."""
        frames = batch.pop('frames').to(torch.complex64)
        batch_size, n_timepoints, n_coils, height, width = frames.shape
        if n_timepoints != len(self.basis) or frames.device != self.basis.device:
            raise ValueError('Frame prefix and device must match the signal-model basis.')
        if (height * 2, width * 2) != self.kernel.shape[-2:]:
            raise ValueError('Frame size must match the reconstruction matrix.')
        transforms = [_draw_transform((height, width), frames.device) for _ in range(batch_size)]
        targets = {
            'm0': batch['m0'].clone(),
            't1': batch['t1'].clone(),
            't2': batch['t2'].clone(),
            'mask': batch['mask'].clone(),
            'noise_std': batch['noise_std'],
        }
        multipliers = []
        for index, transform in enumerate(transforms):
            maps = torch.stack(
                (batch['m0'][index].real, batch['m0'][index].imag, batch['t1'][index], batch['t2'][index])
            )
            maps = _warp(maps[None], transform)[0]
            mask = _warp(batch['mask'][index][None, None].float(), transform, nearest=True)[0, 0].bool()
            field = batch['bias_field'][index]
            if mask.any():
                field = field - field[mask].mean()
                field = (field / field[mask].abs().max().clamp_min(1e-6)).clamp(-1, 1)
            else:
                field = torch.zeros_like(field)
            multiplier = 1 + batch['bias_strength'][index] * field
            multipliers.append(multiplier)
            targets['m0'][index] = torch.complex(maps[0], maps[1]) * multiplier
            targets['t1'][index], targets['t2'][index] = maps[2], maps[3]
            targets['mask'][index] = mask
        rhs = torch.zeros(
            (len(self.basis[0]), batch_size * n_coils, height, width), device=frames.device, dtype=torch.complex64
        )
        # Bound workspaces to eight frames, without allocating a second full frame batch.
        for start in range(0, n_timepoints, 8):
            end = min(start + 8, n_timepoints)
            images = torch.empty(
                (end - start, batch_size, n_coils, height, width), device=frames.device, dtype=torch.complex64
            )
            for index, transform in enumerate(transforms):
                images[:, index] = _warp(frames[index, start:end], transform) * multipliers[index]
            images = images.reshape(end - start, batch_size * n_coils, height, width)
            spectrum = torch.fft.fftn(images, s=self.kernel.shape[-2:], dim=(-2, -1))
            spectrum.mul_(self.kernel[start:end, None])
            torch.fft.ifftn(spectrum, dim=(-2, -1), out=spectrum)
            normal = spectrum[..., :height, :width]
            rhs.add_(torch.einsum('tk,tcyx->kcyx', self.basis[start:end].conj(), normal))
            del images, spectrum, normal
        del frames
        rhs = rhs.reshape(len(self.basis[0]), batch_size, n_coils, 1, height, width).movedim(1, 0).contiguous()
        rhs.add_(self.noise_rhs(targets['noise_std'], n_coils))
        return rhs, targets
