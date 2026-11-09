"""Cache a temporal prefix of reconstructed DELICS coil frames for training."""

import argparse
import logging
from pathlib import Path

import h5py
import mr2
import numpy as np
import torch


def save_slice(
    output: h5py.Group,
    frames: torch.Tensor,
    compression: torch.Tensor,
    reference: h5py.File,
    index: int,
) -> None:
    """Write a slice group with LZF-compressed complex-half frame chunks.

    Reference M0 remains complex64, T1/T2 float32, and the mask boolean.
    Complex-half frames use two IEEE float16 fields, named real and imaginary.
    """
    shape = tuple(frames.shape)
    dtype = np.dtype([('r', '<f2'), ('i', '<f2')])
    error, energy = 0.0, 0.0
    output.attrs.update(
        case=str(reference.attrs['case']),
        slice_index=index,
        signal_scale=float(reference.attrs['kspace_scale']),
        source_file=str(reference.attrs['source_file']),
    )
    dataset = output.create_dataset(
        'frames',
        shape=shape,
        dtype=dtype,
        chunks=(min(8, len(frames)), *shape[1:]),
        compression='lzf',
        shuffle=True,
    )
    for start in range(0, len(frames), 8):
        block = frames[start : start + 8].to(device='cpu', dtype=torch.complex64).contiguous()
        components = torch.view_as_real(block).numpy().astype(np.float16)
        if not np.isfinite(components).all():
            raise ValueError('Frame values cannot be represented as finite complex32.')
        packed = components.view(dtype).reshape(tuple(block.shape))
        dataset[start : start + len(block)] = packed
        restored = torch.view_as_complex(torch.from_numpy(components.astype(np.float32)))
        error += float((restored - block).abs().square().sum())
        energy += float(block.abs().square().sum())
    output.attrs['relative_quantization_error'] = (error / max(energy, 1e-30)) ** 0.5
    output.create_dataset('compression', data=compression.cpu().resolve_conj().numpy())
    for name in ('m0', 't1', 't2'):
        dtype_map = np.complex64 if name == 'm0' else np.float32
        output.create_dataset(
            name, data=reference[f'reference_{name}'][index].astype(dtype_map), compression='lzf', shuffle=True
        )
    output.create_dataset('mask', data=reference['mask'][index].astype(bool), compression='lzf')


def prepare_case(
    reference_path: Path,
    output: Path,
    *,
    n_timepoints: int,
    n_coils: int,
    device: torch.device,
) -> None:
    """Compress each valid slice using only the retained frames and keep the matrix.

    Parameters
    ----------
    reference_path
        Existing radial cache identifying the full-acquisition reconstruction and valid slices.
    output
        Directory receiving one HDF5 file per subject, with groups for valid slices.
    n_timepoints
        Number of initial reconstructed frames to keep.
    n_coils
        Number of compressed output coils.
    device
        Device used for PCA coil compression.
    """
    output.mkdir(parents=True, exist_ok=True)
    with h5py.File(reference_path, 'r') as reference:
        case = str(reference.attrs['case'])
        source_path = Path(str(reference.attrs['source_file']))
        signal_scale = float(reference.attrs['kspace_scale'])
        valid = reference['valid'][()].astype(bool)
        selected = [index for index in range(len(valid)) if valid[index]]
        destination = output / f'{case}-t{n_timepoints}-c{n_coils}.h5'
        if destination.exists():
            with h5py.File(destination, 'r') as cached:
                for index in selected:
                    if f'z{index:03d}' not in cached:
                        raise ValueError(
                            f'Existing cache is incomplete: {destination}; rebuild it in a new output folder.'
                        )
                    group = cached[f'z{index:03d}']
                    if (group.attrs['signal_scale'], group.attrs['source_file']) != (signal_scale, str(source_path)):
                        raise ValueError(f'Existing cache has incompatible provenance: {destination}')
                    if group['frames'].shape[:2] != (n_timepoints, n_coils):
                        raise ValueError(f'Existing cache has incompatible dimensions: {destination}')
            return
        temporary = destination.with_suffix('.h5.tmp')
        with h5py.File(source_path, 'r') as source, h5py.File(temporary, 'w') as cached:
            if not 0 < n_timepoints <= source['slices'].shape[1] or not 0 < n_coils <= source['slices'].shape[2]:
                raise ValueError('Requested prefix or coil count is unavailable in the reconstruction.')
            for index in selected:
                frames = torch.as_tensor(source['slices'][index, :n_timepoints, :, 0]).to(device)
                compression = mr2.operators.PCACompressionOp(
                    frames.movedim(1, -1).reshape(-1, frames.shape[1]),
                    n_components=n_coils,
                    centering=False,
                )
                (compressed,) = compression(frames.movedim(1, -1))
                compressed = compressed.movedim(-1, 1).mul_(signal_scale)
                save_slice(
                    cached.create_group(f'z{index:03d}'),
                    compressed,
                    compression.compression_matrix.squeeze(0),
                    reference,
                    index,
                )
                del frames, compression, compressed
                logging.getLogger(__name__).info('Cached %s slice %d', case, index)
        temporary.replace(destination)


def main() -> None:
    """Prepare only explicitly selected cases; never infer a training/validation partition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-ids', nargs='+', required=True)
    parser.add_argument('--n-timepoints', type=int, default=352)
    parser.add_argument('--n-coils', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    with torch.no_grad():
        for case in arguments.case_ids:
            prepare_case(
                arguments.reference_folder / f'delics_radial_{case}.h5',
                arguments.output,
                n_timepoints=arguments.n_timepoints,
                n_coils=arguments.n_coils,
                device=torch.device(arguments.device),
            )


if __name__ == '__main__':
    main()
