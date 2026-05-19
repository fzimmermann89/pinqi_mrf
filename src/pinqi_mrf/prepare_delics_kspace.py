"""Prepare pseudo-acquired radial k-space from DELICS reference frames."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from collections.abc import Callable, Sequence
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
N_COMPRESSED_COILS = 8
DELICS_KSPACE_SCALE = 250.0
DEFAULT_INPUT_ROOT = Path('/scratch/zimmer08/delics')
DEFAULT_OUTPUT_FOLDER = 'radial_kspace_new'
DEFAULT_DICTIONARY_BATCH_SIZE = 1024
DEFAULT_THREADS_PER_WORKER = 16


@dataclass(frozen=True)
class ConversionConfig:
    """Configuration for DELICS-to-radial-k-space conversion."""

    input_root: Path
    output_root: Path
    case_glob: str = 'reconstructions_rovir_case*.h5'
    n_readout_samples: int | None = None
    overwrite: bool = False
    dictionary_batch_size: int = DEFAULT_DICTIONARY_BATCH_SIZE
    threads_per_worker: int = DEFAULT_THREADS_PER_WORKER


@dataclass(frozen=True)
class ReferenceDictionary:
    """Dictionary matcher and signal generator for reference maps."""

    matcher: mr2.operators.DictionaryMatchOp
    signal: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor]]


def configure_worker_threads(n_threads: int) -> None:
    """Keep per-process CPU thread usage bounded for parallel conversion."""
    if n_threads < 1:
        raise ValueError('threads_per_worker must be at least 1.')
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    torch.set_num_threads(n_threads)


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
        csm[0] = torch.ones_like(csm[0])
        return csm.cpu().numpy().astype(np.complex64), False

    weights = energy / energy.sum().clamp_min(1e-12)
    calibration = (frames * weights[:, None, None, None]).sum(dim=0)
    csm_ok = True
    try:
        csm = mr2.algorithms.csm.inati(calibration[None, :, None], smoothing_width=31, extrapolate=True).squeeze(0)
    except Exception:  # noqa: BLE001
        csm = calibration[:, None]
        csm_ok = False

    norm = csm.abs().square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-8)
    csm = csm / norm
    csm = torch.where(torch.isfinite(csm), csm, torch.zeros_like(csm))
    return csm.cpu().numpy().astype(np.complex64), csm_ok


def compress_coils(
    frames: torch.Tensor,
    n_compressed_coils: int = N_COMPRESSED_COILS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress one slice to a fixed number of virtual coils."""
    n_source_coils = frames.shape[1]
    if n_source_coils < n_compressed_coils:
        raise ValueError(f'Cannot compress {n_source_coils} source coils to {n_compressed_coils} coils.')
    data = frames.movedim(1, 0).flatten(1)
    covariance = data @ data.conj().mT
    _, eigenvectors = torch.linalg.eigh(covariance)
    compression = eigenvectors[:, -n_compressed_coils:].mH
    compressed = torch.einsum('oc,tczyx->tozyx', compression, frames)
    return compressed.contiguous(), compression.contiguous()


def create_dictionary_matcher(
    device: torch.device,
    *,
    batch_size: int = DEFAULT_DICTIONARY_BATCH_SIZE,
) -> ReferenceDictionary:
    """Create full-frame NeuroMRF dictionary matcher for reference maps."""
    signal = mr2.operators.models.NeuroMRF().to(device)

    def signal_with_m0(m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor]:
        return signal(m0, t1, t2)

    def signal_without_m0(t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor]:
        m0 = torch.ones((), dtype=torch.complex64, device=device)
        return signal_with_m0(m0, t1, t2)

    dictionary_match = mr2.operators.DictionaryMatchOp(signal_without_m0, batch_size=batch_size)
    t1 = torch.linspace(0.2, 4.0, 400, device=device)[:, None]
    t2 = torch.linspace(0.001, 1.0, 600, device=device)[None, :]
    dictionary_match.append(t1, t2)
    return ReferenceDictionary(matcher=dictionary_match, signal=signal_with_m0)


def dictionary_match_reference(
    frames: torch.Tensor,
    csm: np.ndarray,
    reference_dictionary: ReferenceDictionary,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dictionary-match coil-combined full-frame references."""
    csm_tensor = torch.as_tensor(csm, dtype=frames.dtype, device=frames.device)
    combined = (frames.squeeze(2) * csm_tensor[:, 0].conj()).sum(dim=1)
    t1, t2 = reference_dictionary.matcher(combined.detach())
    (matched_signal,) = reference_dictionary.signal(torch.ones_like(t1, dtype=frames.dtype), t1, t2)
    denominator = matched_signal.abs().square().sum(dim=0).clamp_min(1e-12)
    m0 = (combined * matched_signal.conj()).sum(dim=0) / denominator
    return (
        m0.cpu().numpy().astype(np.complex64),
        t1.cpu().numpy().astype(np.float32),
        t2.cpu().numpy().astype(np.float32),
    )


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
    n_source_coils: int,
    size_y: int,
    size_x: int,
    n_readout_samples: int,
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
    target.attrs['n_slices'] = n_slices
    target.attrs['n_timepoints'] = n_timepoints
    target.attrs['n_coils'] = n_coils
    target.attrs['n_source_coils'] = n_source_coils
    target.attrs['image_size_y'] = size_y
    target.attrs['image_size_x'] = size_x
    target.attrs['n_readout_samples'] = n_readout_samples
    target.attrs['trajectory'] = 'golden_angle_radial_2d'
    target.attrs['radial_golden_angle'] = RADIAL_GOLDEN_ANGLE
    target.attrs['kspace_scale'] = DELICS_KSPACE_SCALE
    target.attrs['created_by'] = 'pinqi_mrf.prepare_delics_kspace'
    copy_source_attrs(source, target)

    target.create_dataset(
        'reference_m0',
        shape=(n_slices, size_y, size_x),
        dtype=np.complex64,
        chunks=(1, size_y, size_x),
        compression='lzf',
    )
    target.create_dataset(
        'reference_t1',
        shape=(n_slices, size_y, size_x),
        dtype=np.float32,
        chunks=(1, size_y, size_x),
        compression='lzf',
    )
    target.create_dataset(
        'reference_t2',
        shape=(n_slices, size_y, size_x),
        dtype=np.float32,
        chunks=(1, size_y, size_x),
        compression='lzf',
    )


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
        n_slices, n_timepoints, n_source_coils, size_y, size_x = validate_source(source, source_path)
        if n_source_coils < N_COMPRESSED_COILS:
            raise ValueError(
                f'{source_path} has {n_source_coils} coils, but {N_COMPRESSED_COILS} compressed coils were requested.'
            )
        case = source_case_id(source_path, source)
        n_readout_samples = config.n_readout_samples or size_x
        output_path = output_root / f'delics_radial_{case}.h5'
        tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
        if output_path.exists() and not config.overwrite:
            raise FileExistsError(f'{output_path} already exists. Use --overwrite to replace it.')
        if tmp_path.exists():
            tmp_path.unlink()

        acquisition = create_radial_trajectory(n_spokes=n_timepoints, n_readout_samples=n_readout_samples)
        traj = acquisition.trajectory.to(device)
        matrix = mr2.data.SpatialDimension(1, size_y, size_x)
        nufft = mr2.operators.NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=matrix,
            encoding_matrix=matrix,
            traj=traj,
        )
        dcf_numpy = acquisition.dcf.data.numpy()
        dictionary_match = create_dictionary_matcher(device, batch_size=config.dictionary_batch_size)

        with h5py.File(tmp_path, 'w') as target, torch.inference_mode():
            create_output_file(
                target,
                case=case,
                source_file=source_path,
                source=source,
                n_slices=n_slices,
                n_timepoints=n_timepoints,
                n_coils=N_COMPRESSED_COILS,
                n_source_coils=n_source_coils,
                size_y=size_y,
                size_x=size_x,
                n_readout_samples=n_readout_samples,
            )
            write_trajectory_and_dcf(target, traj=traj.cpu(), dcf=dcf_numpy)

            for z in tqdm(
                range(n_slices),
                desc=f'{case} slices',
                unit='slice',
                leave=False,
                position=1,
            ):
                frames = torch.as_tensor(source['slices'][z], dtype=torch.complex64, device=device)
                if frames.shape != (n_timepoints, n_source_coils, 1, size_y, size_x):
                    raise ValueError(f'{source_path} /slices[{z}] has unexpected shape: {tuple(frames.shape)}.')
                frames, _compression = compress_coils(frames)
                frames_2d = frames.squeeze(2)
                mask, mask_valid = create_mask(frames_2d)
                (kspace,) = nufft(frames)
                expected_shape = (n_timepoints, N_COMPRESSED_COILS, 1, 1, n_readout_samples)
                if kspace.shape != expected_shape:
                    raise RuntimeError(f'NUFFT returned shape {tuple(kspace.shape)}, expected {expected_shape}.')
                target['kspace'][z] = (DELICS_KSPACE_SCALE * kspace).cpu().numpy().astype(np.complex64)
                target['mask'][z] = mask
                target['valid'][z] = mask_valid and bool(torch.isfinite(frames).all().item())
                csm, csm_valid = estimate_csm(frames_2d)
                target['csm_initial'][z] = csm
                target['csm_valid'][z] = csm_valid
                reference_m0, reference_t1, reference_t2 = dictionary_match_reference(frames, csm, dictionary_match)
                target['reference_m0'][z] = (DELICS_KSPACE_SCALE * reference_m0).astype(np.complex64)
                target['reference_t1'][z] = reference_t1
                target['reference_t2'][z] = reference_t2

            reference_m0_abs = np.abs(target['reference_m0'][()])
            initially_valid = target['valid'][()] & target['csm_valid'][()]
            central_reference_m0 = reference_m0_abs[5:-20]
            if central_reference_m0.size == 0:
                central_reference_m0 = reference_m0_abs
            threshold = np.percentile(central_reference_m0, 90) * 0.05
            good = initially_valid & ((reference_m0_abs > threshold).mean(axis=(-1, -2)) > 0.1)
            if good.any():
                good[: np.argmax(good) + 8] = False
            target['valid'][:] = good

        tmp_path.replace(output_path)
        return output_path


def find_source_files(input_root: Path, case_glob: str) -> list[Path]:
    """Find source H5 files."""
    return sorted(path for path in input_root.glob(case_glob) if path.is_file())


def convert_files_on_gpu(task: tuple[list[Path], ConversionConfig, int, int]) -> list[Path]:
    """Convert a shard of source files in one GPU-pinned worker process."""
    source_paths, config, gpu_index, position = task
    _ = position
    configure_worker_threads(config.threads_per_worker)
    torch.cuda.set_device(gpu_index)
    device = torch.device(f'cuda:{gpu_index}')
    outputs = []
    for source_path in source_paths:
        outputs.append(convert_file(source_path, config.output_root, config=config, device=device))
        torch.cuda.empty_cache()
    return outputs


def convert_all(config: ConversionConfig, *, num_workers: int = 1) -> list[Path]:
    """Convert all matching DELICS H5 files."""
    configure_worker_threads(config.threads_per_worker)
    config.output_root.mkdir(parents=True, exist_ok=True)
    source_files = find_source_files(config.input_root, config.case_glob)
    if not source_files:
        raise FileNotFoundError(f'No files matching {config.case_glob!r} found in {config.input_root}.')
    if num_workers < 1:
        raise ValueError('num_workers must be at least 1.')
    for source_path in source_files:
        with h5py.File(source_path, 'r') as source:
            validate_source(source, source_path)
            case = source_case_id(source_path, source)
        output_path = config.output_root / f'delics_radial_{case}.h5'
        if output_path.exists() and not config.overwrite:
            raise FileExistsError(f'{output_path} already exists. Use --overwrite to replace it.')
    if not torch.cuda.is_available():
        raise RuntimeError('DELICS conversion requires CUDA.')
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError('No CUDA devices are available.')
    num_workers = min(num_workers, len(source_files), n_gpus)
    if num_workers == 1:
        device = torch.device('cuda:0')
        outputs = []
        progress = tqdm(source_files, desc='DELICS files', unit='file', position=0)
        for source_path in progress:
            progress.set_postfix_str(source_path.name)
            outputs.append(convert_file(source_path, config.output_root, config=config, device=device))
        return outputs

    tasks = [
        (source_files[worker_index::num_workers], config, worker_index, worker_index + 1)
        for worker_index in range(num_workers)
        if source_files[worker_index::num_workers]
    ]
    outputs = []
    context = mp.get_context('spawn')
    with context.Pool(processes=num_workers) as pool:
        with tqdm(total=len(source_files), desc='DELICS files', unit='file', position=0) as progress:
            for worker_outputs in pool.imap_unordered(convert_files_on_gpu, tasks):
                outputs.extend(worker_outputs)
                progress.update(len(worker_outputs))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--case-glob', type=str, default='reconstructions_rovir_case*.h5')
    parser.add_argument('--n-readout-samples', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dictionary-batch-size', type=int, default=DEFAULT_DICTIONARY_BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--threads-per-worker', type=int, default=DEFAULT_THREADS_PER_WORKER)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run DELICS k-space conversion."""
    args = build_parser().parse_args(argv)
    output_root = args.output_root or args.input_root / DEFAULT_OUTPUT_FOLDER
    config = ConversionConfig(
        input_root=args.input_root,
        output_root=output_root,
        case_glob=args.case_glob,
        n_readout_samples=args.n_readout_samples,
        overwrite=args.overwrite,
        dictionary_batch_size=args.dictionary_batch_size,
        threads_per_worker=args.threads_per_worker,
    )
    outputs = convert_all(config, num_workers=args.num_workers)
    for output in outputs:
        sys.stdout.write(f'{output}\n')


if __name__ == '__main__':
    main()
