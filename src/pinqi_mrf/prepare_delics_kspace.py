"""Prepare pseudo-acquired radial k-space from DELICS reference frames."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import h5py
import mr2
import numpy as np
import scipy.ndimage as ndimage
import torch
from tqdm import tqdm

from pinqi_mrf.acquisition import RADIAL_GOLDEN_ANGLE, create_radial_trajectory

SOURCE_DELICS_SCRIPT = '/home/zimmer08/thesis/delics.py'
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ConversionConfig:
    """Configuration for DELICS-to-radial-k-space conversion."""

    input_root: Path
    output_root: Path
    case_glob: str = 'reconstructions_rovir_case*.h5'
    n_readout_samples: int | None = None
    device: str = 'auto'
    chunk_slices: int = 1
    overwrite: bool = False
    skip_csm: bool = False


def resolve_device(device: str) -> torch.device:
    """Resolve CLI device string."""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    requested = torch.device(device)
    if requested.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is False.')
    return requested


def create_radial_dcf(n_timepoints: int, n_readout_samples: int) -> np.ndarray:
    """Create a simple mean-normalized radial ramp DCF."""
    center = n_readout_samples // 2
    radius = np.abs(np.arange(n_readout_samples, dtype=np.float32) - center)
    if n_readout_samples > 1:
        nonzero = radius[radius > 0]
        radius[center] = nonzero.min() if nonzero.size else 1.0
    radius /= np.mean(radius, dtype=np.float64)
    dcf = np.broadcast_to(radius.reshape(1, 1, 1, 1, n_readout_samples), (n_timepoints, 1, 1, 1, n_readout_samples))
    return np.asarray(dcf, dtype=np.float32)


def create_mask(
    frames: torch.Tensor,
    *,
    threshold: float = 0.08,
    smooth_sigma: float = 1.0,
    opening_iterations: int = 1,
    closing_iterations: int = 2,
    dilation_iterations: int = 3,
) -> tuple[np.ndarray, bool]:
    """Create a 2D brain mask from coil-resolved time-frame images."""
    energy = frames.abs().square().sum(dim=(0, 1)).sqrt().cpu().numpy().astype(np.float32)
    if not np.isfinite(energy).all() or energy.max() <= 0:
        return np.zeros_like(energy, dtype=bool), False

    energy /= energy.max() + 1e-12
    mask = ndimage.gaussian_filter(energy, sigma=smooth_sigma) > threshold
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=opening_iterations)
    mask = ndimage.binary_closing(mask, iterations=closing_iterations)
    labels, n_labels = ndimage.label(mask)
    if n_labels > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        mask = labels == counts.argmax()
    mask = ndimage.binary_dilation(mask, iterations=dilation_iterations)
    return mask.astype(bool), bool(mask.any())


def estimate_csm(frames: torch.Tensor) -> tuple[np.ndarray, bool]:
    """Estimate initial CSMs from a time-weighted calibration image."""
    energy = frames.abs().square().sum(dim=(1, 2, 3)).sqrt()
    if not torch.isfinite(energy).all() or energy.sum() <= 0:
        csm = torch.zeros(frames.shape[1], 1, *frames.shape[-2:], dtype=torch.complex64)
        csm[0] = 1.0 + 0.0j
        return csm.cpu().numpy().astype(np.complex64), False

    weights = energy / energy.sum().clamp_min(1e-12)
    calibration = (frames * weights[:, None, None, None]).sum(dim=0)
    csm_ok = True
    try:
        csm = mr2.algorithms.csm.inati(calibration[None, :, None], smoothing_width=31).squeeze(0)
    except Exception:  # noqa: BLE001
        csm = calibration[:, None]
        csm_ok = False

    norm = csm.abs().square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-8)
    csm = csm / norm
    csm = torch.where(torch.isfinite(csm), csm, torch.zeros_like(csm))
    return csm.cpu().numpy().astype(np.complex64), csm_ok


def source_case_id(path: Path, source: h5py.File) -> str:
    """Infer a case id from source attrs or filename."""
    if 'case' in source.attrs:
        case = source.attrs['case']
        if isinstance(case, bytes):
            case = case.decode()
        return str(case)
    name = path.stem
    return name.removeprefix('reconstructions_rovir_')


def validate_source(source: h5py.File, path: Path) -> tuple[int, int, int, int, int]:
    """Validate source H5 schema and return basic dimensions."""
    if 'slices' not in source:
        raise ValueError(f'{path} does not contain a /slices dataset.')
    slices = source['slices']
    if slices.ndim != 6:
        raise ValueError(f'{path} /slices must have 6 dimensions [z,time,coil,1,y,x], got {slices.shape}.')
    n_slices, n_timepoints, n_coils, singleton, size_y, size_x = slices.shape
    if singleton != 1:
        raise ValueError(f'{path} /slices singleton dimension must be 1, got shape {slices.shape}.')
    return n_slices, n_timepoints, n_coils, size_y, size_x


def copy_source_attrs(source: h5py.File, target: h5py.File) -> None:
    """Copy source attrs to target with a source_attr_ prefix."""
    for key, value in source.attrs.items():
        target.attrs[f'source_attr_{key}'] = value


def create_output_file(
    target: h5py.File,
    *,
    case: str,
    source_file: Path,
    source: h5py.File,
    n_slices: int,
    n_timepoints: int,
    n_coils: int,
    size_y: int,
    size_x: int,
    n_readout_samples: int,
    skip_csm: bool,
) -> None:
    """Create output datasets and metadata."""
    target.create_dataset(
        'kspace',
        shape=(n_slices, n_timepoints, n_coils, 1, 1, n_readout_samples),
        dtype=np.complex64,
        chunks=(1, n_timepoints, n_coils, 1, 1, n_readout_samples),
        compression='lzf',
    )
    target.create_dataset(
        'mask',
        shape=(n_slices, size_y, size_x),
        dtype=np.bool_,
        chunks=(1, size_y, size_x),
        compression='lzf',
    )
    target.create_dataset('valid', shape=(n_slices,), dtype=np.bool_)
    if not skip_csm:
        target.create_dataset(
            'csm_initial',
            shape=(n_slices, n_coils, 1, size_y, size_x),
            dtype=np.complex64,
            chunks=(1, n_coils, 1, size_y, size_x),
            compression='lzf',
        )
        target.create_dataset('csm_valid', shape=(n_slices,), dtype=np.bool_)

    target.attrs['case'] = case
    target.attrs['source_file'] = str(source_file)
    target.attrs['source_delics_script'] = SOURCE_DELICS_SCRIPT
    target.attrs['n_slices'] = n_slices
    target.attrs['n_timepoints'] = n_timepoints
    target.attrs['n_coils'] = n_coils
    target.attrs['image_size_y'] = size_y
    target.attrs['image_size_x'] = size_x
    target.attrs['n_readout_samples'] = n_readout_samples
    target.attrs['trajectory'] = 'golden_angle_radial_2d'
    target.attrs['radial_golden_angle'] = RADIAL_GOLDEN_ANGLE
    target.attrs['created_by'] = 'pinqi_mrf.prepare_delics_kspace'
    target.attrs['schema_version'] = SCHEMA_VERSION
    copy_source_attrs(source, target)

    if 'coil_compression' in source:
        source_group = target.create_group('source')
        source_group.create_dataset('coil_compression', data=source['coil_compression'][()], dtype=np.complex64)


def write_trajectory_and_dcf(
    target: h5py.File,
    *,
    traj: mr2.data.KTrajectory,
    dcf: np.ndarray,
) -> None:
    """Store trajectory arrays and DCF."""
    trajectory_group = target.create_group('trajectory')
    trajectory_group.create_dataset('kx', data=traj.kx.cpu().numpy().astype(np.float32))
    trajectory_group.create_dataset('ky', data=traj.ky.cpu().numpy().astype(np.float32))
    trajectory_group.create_dataset('kz', data=traj.kz.cpu().numpy().astype(np.float32))
    target.create_dataset('dcf', data=dcf, dtype=np.float32, compression='lzf')


def convert_file(source_path: Path, output_root: Path, *, config: ConversionConfig, device: torch.device) -> Path:
    """Convert one DELICS reconstruction H5 into pseudo-acquired radial k-space."""
    with h5py.File(source_path, 'r') as source:
        n_slices, n_timepoints, n_coils, size_y, size_x = validate_source(source, source_path)
        case = source_case_id(source_path, source)
        n_readout_samples = config.n_readout_samples or size_x
        output_path = output_root / f'delics_radial_{case}.h5'
        tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
        if output_path.exists() and not config.overwrite:
            raise FileExistsError(f'{output_path} already exists. Use --overwrite to replace it.')
        if tmp_path.exists():
            tmp_path.unlink()

        traj = create_radial_trajectory(n_spokes=n_timepoints, n_readout_samples=n_readout_samples).to(device)
        matrix = mr2.data.SpatialDimension(1, size_y, size_x)
        nufft = mr2.operators.NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=matrix,
            encoding_matrix=matrix,
            traj=traj,
        )
        dcf = create_radial_dcf(n_timepoints, n_readout_samples)

        with h5py.File(tmp_path, 'w') as target, torch.inference_mode():
            create_output_file(
                target,
                case=case,
                source_file=source_path,
                source=source,
                n_slices=n_slices,
                n_timepoints=n_timepoints,
                n_coils=n_coils,
                size_y=size_y,
                size_x=size_x,
                n_readout_samples=n_readout_samples,
                skip_csm=config.skip_csm,
            )
            write_trajectory_and_dcf(target, traj=traj.cpu(), dcf=dcf)

            for start in tqdm(range(0, n_slices, config.chunk_slices), desc=f'Converting {case}', leave=False):
                stop = min(start + config.chunk_slices, n_slices)
                for z in range(start, stop):
                    frames = torch.as_tensor(source['slices'][z], dtype=torch.complex64, device=device)
                    if frames.shape != (n_timepoints, n_coils, 1, size_y, size_x):
                        raise ValueError(f'{source_path} /slices[{z}] has unexpected shape: {tuple(frames.shape)}.')
                    frames_2d = frames.squeeze(2)
                    mask, mask_valid = create_mask(frames_2d)
                    (kspace,) = nufft(frames)
                    expected_shape = (n_timepoints, n_coils, 1, 1, n_readout_samples)
                    if kspace.shape != expected_shape:
                        raise RuntimeError(f'NUFFT returned shape {tuple(kspace.shape)}, expected {expected_shape}.')
                    target['kspace'][z] = kspace.cpu().numpy().astype(np.complex64)
                    target['mask'][z] = mask
                    target['valid'][z] = mask_valid and bool(torch.isfinite(frames).all().item())
                    if not config.skip_csm:
                        csm, csm_valid = estimate_csm(frames_2d)
                        target['csm_initial'][z] = csm
                        target['csm_valid'][z] = csm_valid

        tmp_path.replace(output_path)
        return output_path


def find_source_files(input_root: Path, case_glob: str) -> list[Path]:
    """Find source H5 files."""
    return sorted(path for path in input_root.glob(case_glob) if path.is_file())


def convert_all(config: ConversionConfig) -> list[Path]:
    """Convert all matching DELICS H5 files."""
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    if config.chunk_slices < 1:
        raise ValueError('chunk_slices must be at least 1.')
    device = resolve_device(config.device)
    config.output_root.mkdir(parents=True, exist_ok=True)
    source_files = find_source_files(config.input_root, config.case_glob)
    if not source_files:
        raise FileNotFoundError(f'No files matching {config.case_glob!r} found in {config.input_root}.')
    outputs = []
    for source_path in tqdm(source_files, desc='DELICS files'):
        outputs.append(convert_file(source_path, config.output_root, config=config, device=device))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--case-glob', type=str, default='reconstructions_rovir_case*.h5')
    parser.add_argument('--n-readout-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--chunk-slices', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip-csm', action='store_true')
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """Run DELICS k-space conversion."""
    args = build_parser().parse_args(argv)
    config = ConversionConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        case_glob=args.case_glob,
        n_readout_samples=args.n_readout_samples,
        device=args.device,
        chunk_slices=args.chunk_slices,
        overwrite=args.overwrite,
        skip_csm=args.skip_csm,
    )
    outputs = convert_all(config)
    for output in outputs:
        sys.stdout.write(f'{output}\n')


if __name__ == '__main__':
    main()
