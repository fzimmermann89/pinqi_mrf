"""Benchmark dictionary-matching search variants.

The benchmark isolates the prior-aware discrete search used by
``DifferentiableDictionaryMatchOp``. It uses synthetic tensors with the same
shapes as the PINQI-MRF subspace matching problem.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import torch


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n-voxels', type=int, default=6 * 256 * 256)
    parser.add_argument('--n-atoms', type=int, default=200 * 160)
    parser.add_argument('--signal-dim', type=int, default=10)
    parser.add_argument('--n-parameters', type=int, default=2)
    parser.add_argument('--voxel-batch-size', type=int, default=8192)
    parser.add_argument('--atom-block-size', type=int, default=4096)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--check', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def complex_abs_square(x: torch.Tensor) -> torch.Tensor:
    """Compute squared complex magnitude without calling abs()."""
    return x.real.square() + x.imag.square()


def baseline_abs_square(
    signal: torch.Tensor,
    dictionary: torch.Tensor,
    norm_y: torch.Tensor,
    norm_y_sq: torch.Tensor,
    parameter_grids: tuple[torch.Tensor, ...],
    prior: tuple[torch.Tensor, ...],
    precision: tuple[torch.Tensor, ...],
    voxel_batch_size: int,
) -> torch.Tensor:
    """Baseline matching using abs().square(), matching current mr2 implementation."""
    idx_chunks = []
    for signal_chunk, *chunks in zip(
        signal.split(voxel_batch_size, dim=0),
        *(p.split(voxel_batch_size, dim=0) for p in (*prior, *precision)),
        strict=True,
    ):
        prior_chunk = chunks[: len(prior)]
        precision_chunk = chunks[len(prior) :]
        similarity = signal_chunk @ dictionary.conj()
        scale_precision = precision_chunk[0]
        numerator = similarity * norm_y + scale_precision[:, None] * prior_chunk[0][:, None]
        scores = numerator.abs().square() / (norm_y_sq[None, :] + scale_precision[:, None])
        for grid, p_k, lam_k in zip(parameter_grids, prior_chunk[1:], precision_chunk[1:], strict=True):
            scores = scores - lam_k[:, None] * (grid[None, :] - p_k[:, None]).abs().square()
        idx_chunks.append(scores.argmax(dim=1))
    return torch.cat(idx_chunks)


def real_imag_abs_square(
    signal: torch.Tensor,
    dictionary: torch.Tensor,
    norm_y: torch.Tensor,
    norm_y_sq: torch.Tensor,
    parameter_grids: tuple[torch.Tensor, ...],
    prior: tuple[torch.Tensor, ...],
    precision: tuple[torch.Tensor, ...],
    voxel_batch_size: int,
) -> torch.Tensor:
    """Matching using real.square() + imag.square() instead of abs().square()."""
    idx_chunks = []
    for signal_chunk, *chunks in zip(
        signal.split(voxel_batch_size, dim=0),
        *(p.split(voxel_batch_size, dim=0) for p in (*prior, *precision)),
        strict=True,
    ):
        prior_chunk = chunks[: len(prior)]
        precision_chunk = chunks[len(prior) :]
        similarity = signal_chunk @ dictionary.conj()
        scale_precision = precision_chunk[0]
        numerator = similarity * norm_y + scale_precision[:, None] * prior_chunk[0][:, None]
        scores = complex_abs_square(numerator) / (norm_y_sq[None, :] + scale_precision[:, None])
        for grid, p_k, lam_k in zip(parameter_grids, prior_chunk[1:], precision_chunk[1:], strict=True):
            diff = grid[None, :] - p_k[:, None]
            scores = scores - lam_k[:, None] * diff.square()
        idx_chunks.append(scores.argmax(dim=1))
    return torch.cat(idx_chunks)


def atom_block_streaming(
    signal: torch.Tensor,
    dictionary: torch.Tensor,
    norm_y: torch.Tensor,
    norm_y_sq: torch.Tensor,
    parameter_grids: tuple[torch.Tensor, ...],
    prior: tuple[torch.Tensor, ...],
    precision: tuple[torch.Tensor, ...],
    voxel_batch_size: int,
    atom_block_size: int,
) -> torch.Tensor:
    """Matching with atom-blocked streaming top-1 reduction."""
    idx_chunks = []
    for signal_chunk, *chunks in zip(
        signal.split(voxel_batch_size, dim=0),
        *(p.split(voxel_batch_size, dim=0) for p in (*prior, *precision)),
        strict=True,
    ):
        prior_chunk = chunks[: len(prior)]
        precision_chunk = chunks[len(prior) :]
        best_score = torch.full((signal_chunk.shape[0],), -torch.inf, device=signal.device)
        best_idx = torch.zeros((signal_chunk.shape[0],), dtype=torch.int64, device=signal.device)
        for start in range(0, dictionary.shape[1], atom_block_size):
            stop = min(start + atom_block_size, dictionary.shape[1])
            similarity = signal_chunk @ dictionary[:, start:stop].conj()
            scale_precision = precision_chunk[0]
            numerator = similarity * norm_y[start:stop] + scale_precision[:, None] * prior_chunk[0][:, None]
            scores = complex_abs_square(numerator) / (norm_y_sq[start:stop][None, :] + scale_precision[:, None])
            for grid, p_k, lam_k in zip(parameter_grids, prior_chunk[1:], precision_chunk[1:], strict=True):
                diff = grid[start:stop][None, :] - p_k[:, None]
                scores = scores - lam_k[:, None] * diff.square()
            block_score, block_idx = scores.max(dim=1)
            update = block_score > best_score
            best_score = torch.where(update, block_score, best_score)
            best_idx = torch.where(update, block_idx + start, best_idx)
        idx_chunks.append(best_idx)
    return torch.cat(idx_chunks)


def benchmark(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> torch.Tensor:
    """Benchmark one callable."""
    for _ in range(warmup):
        idx = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        idx = fn()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0.0
    print(f'{name:32s} {sum(times) / len(times):8.4f} s  peak={peak_memory:8.2f} GiB')
    return idx


def main() -> None:
    """Run the benchmark."""
    args = parse_args()
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)
    dtype = torch.complex64
    real_dtype = torch.float32
    generator = torch.Generator(device=device).manual_seed(0)

    signal = torch.randn(args.n_voxels, args.signal_dim, dtype=dtype, device=device, generator=generator)
    dictionary = torch.randn(args.signal_dim, args.n_atoms, dtype=dtype, device=device, generator=generator)
    dictionary = dictionary / dictionary.norm(dim=0, keepdim=True).clamp_min(1e-12)
    norm_y = torch.rand(args.n_atoms, dtype=real_dtype, device=device, generator=generator) + 0.1
    norm_y_sq = norm_y.square()
    parameter_grids = tuple(
        torch.linspace(-2.0, 2.0, args.n_atoms, dtype=real_dtype, device=device) for _ in range(args.n_parameters)
    )
    prior = (
        torch.randn(args.n_voxels, dtype=dtype, device=device, generator=generator),
        *(
            torch.randn(args.n_voxels, dtype=real_dtype, device=device, generator=generator)
            for _ in range(args.n_parameters)
        ),
    )
    precision = tuple(
        torch.rand(args.n_voxels, dtype=real_dtype, device=device, generator=generator) + 0.1
        for _ in range(args.n_parameters + 1)
    )

    print(
        f'n_voxels={args.n_voxels}, n_atoms={args.n_atoms}, signal_dim={args.signal_dim}, '
        f'voxel_batch_size={args.voxel_batch_size}, atom_block_size={args.atom_block_size}'
    )
    with torch.no_grad():
        baseline = benchmark(
            'baseline_abs_square',
            lambda: baseline_abs_square(
                signal,
                dictionary,
                norm_y,
                norm_y_sq,
                parameter_grids,
                prior,
                precision,
                args.voxel_batch_size,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
        )
        benchmark(
            'real_imag_abs_square',
            lambda: real_imag_abs_square(
                signal,
                dictionary,
                norm_y,
                norm_y_sq,
                parameter_grids,
                prior,
                precision,
                args.voxel_batch_size,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
        )
        streaming = benchmark(
            'atom_block_streaming',
            lambda: atom_block_streaming(
                signal,
                dictionary,
                norm_y,
                norm_y_sq,
                parameter_grids,
                prior,
                precision,
                args.voxel_batch_size,
                args.atom_block_size,
            ),
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
        )

        if args.compile:
            compiled_real_imag = torch.compile(real_imag_abs_square)
            benchmark(
                'compiled_real_imag',
                lambda: compiled_real_imag(
                    signal,
                    dictionary,
                    norm_y,
                    norm_y_sq,
                    parameter_grids,
                    prior,
                    precision,
                    args.voxel_batch_size,
                ),
                warmup=max(args.warmup, 1),
                repeat=args.repeat,
                device=device,
            )
            compiled_streaming = torch.compile(atom_block_streaming)
            benchmark(
                'compiled_streaming',
                lambda: compiled_streaming(
                    signal,
                    dictionary,
                    norm_y,
                    norm_y_sq,
                    parameter_grids,
                    prior,
                    precision,
                    args.voxel_batch_size,
                    args.atom_block_size,
                ),
                warmup=max(args.warmup, 1),
                repeat=args.repeat,
                device=device,
            )

        if args.check:
            print(f'streaming_matches_baseline={bool(torch.equal(baseline, streaming))}')


if __name__ == '__main__':
    main()
