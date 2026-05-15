"""Profile one PINQI-MRF training step.

This script runs a small single-process training loop and writes a PyTorch profiler
trace that can be inspected with TensorBoard.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import mr2
import torch

from pinqi_mrf.data import BatchType, BrainwebDataset
from pinqi_mrf.model import MIN_SUPERVISED_T1, MIN_SUPERVISED_T2, PinqiModule
from pinqi_mrf.parameters import R1_STATS, R2_STATS, standardized_rate_to_relaxation_time


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('profiler/pinqi_mrf'))
    parser.add_argument('--brainweb-folder', type=Path, default=mr2.phantoms.brainweb.CACHE_DIR_BRAINWEB)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--n-coils', type=int, default=12)
    parser.add_argument('--n-compressed-coils', type=int, default=8)
    parser.add_argument('--n-iterations', type=int, default=4)
    parser.add_argument('--n-timepoints', type=int, default=None)
    parser.add_argument('--csm-estimation-iterations', type=int, default=1)
    parser.add_argument('--csm-extrapolate', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use-ground-truth-csm', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--initial-cg-iterations', type=int, default=5)
    parser.add_argument('--initial-cg-regularization', type=float, default=1e-2)
    parser.add_argument('--image-cg-iterations', type=int, default=20)
    parser.add_argument('--wait', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--active', type=int, default=3)
    parser.add_argument('--record-shapes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--profile-memory', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def to_device(batch: BatchType, device: torch.device) -> BatchType:
    """Move a batch to device."""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}  # type: ignore[return-value]


def loss_without_logging(
    predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    batch: BatchType,
) -> torch.Tensor:
    """Compute the PINQI-MRF training loss without Lightning logging."""
    target_m0, target_r1, target_r2, mask = map(torch.squeeze, (batch['m0'], batch['r1'], batch['r2'], batch['mask']))
    mask = mask.bool()
    target_t1 = standardized_rate_to_relaxation_time(target_r1, R1_STATS)
    target_t2 = standardized_rate_to_relaxation_time(target_r2, R2_STATS)
    parameter_mask = (
        mask
        & torch.isfinite(target_r1)
        & torch.isfinite(target_r2)
        & torch.isfinite(target_t1)
        & torch.isfinite(target_t2)
        & (target_t1 >= MIN_SUPERVISED_T1)
        & (target_t2 >= MIN_SUPERVISED_T2)
    )
    if not parameter_mask.any():
        raise RuntimeError('Parameter mask is empty.')

    loss = torch.tensor(0.0, device=target_r1.device)
    loss_weights = (0.2, 0.1, 0.1, 0.8)
    for prediction, weight in zip(predictions, loss_weights, strict=False):
        prediction_m0, prediction_r1, prediction_r2 = map(torch.squeeze, prediction)
        loss_r1 = torch.nn.functional.mse_loss(prediction_r1[parameter_mask], target_r1[parameter_mask])
        loss_r2 = torch.nn.functional.mse_loss(prediction_r2[parameter_mask], target_r2[parameter_mask])
        loss_m0 = torch.nn.functional.mse_loss(
            torch.view_as_real(prediction_m0[parameter_mask]),
            torch.view_as_real(target_m0.to(prediction_m0.dtype)[parameter_mask]),
        )
        loss_outside = prediction_m0[~mask].abs().mean()
        loss = loss + weight * (loss_r1 + loss_r2 + 0.5 * loss_m0 + 0.1 * loss_outside)
    return loss


def main() -> None:
    """Run the profiler."""
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.brainweb_folder.exists():
        args.brainweb_folder.mkdir(parents=True, exist_ok=True)
        mr2.phantoms.brainweb.download_brainweb(output_directory=args.brainweb_folder, workers=2, progress=True)

    dataset = BrainwebDataset(
        folder=args.brainweb_folder,
        size=args.size,
        n_coils=args.n_coils,
        random=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        drop_last=True,
    )

    model = PinqiModule(
        image_size=args.size,
        n_coils=args.n_coils,
        n_compressed_coils=args.n_compressed_coils,
        n_timepoints=args.n_timepoints,
        n_iterations=args.n_iterations,
        use_ground_truth_csm=args.use_ground_truth_csm,
        csm_estimation_iterations=args.csm_estimation_iterations,
        csm_extrapolate=args.csm_extrapolate,
        initial_cg_iterations=args.initial_cg_iterations,
        initial_cg_regularization=args.initial_cg_regularization,
        image_cg_iterations=args.image_cg_iterations,
    ).to(device)
    model.train()
    model.setup_acquisition()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    total_steps = args.wait + args.warmup + args.active
    trace_handler = torch.profiler.tensorboard_trace_handler(str(args.output_dir / 'trace'))
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as profiler:
        for step, batch in zip(range(total_steps), loader, strict=False):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function('simulate_coil_rhs'):
                coil_rhs, csm_ground_truth = model.simulate_coil_rhs(batch)
            with torch.profiler.record_function('estimate_or_select_csm'):
                csm_initial = csm_ground_truth if args.use_ground_truth_csm else model.estimate_initial_csm(coil_rhs)
            with torch.profiler.record_function('pinqi_forward'):
                _images, parameters = model(coil_rhs, csm_initial=csm_initial)
            with torch.profiler.record_function('loss'):
                loss = loss_without_logging(parameters, batch)
            with torch.profiler.record_function('backward'):
                loss.backward()
            with torch.profiler.record_function('optimizer_step'):
                optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            print(f'step={step:03d} loss={loss.item():.6g}')
            profiler.step()

    summary = profiler.key_averages().table(
        sort_by='cuda_time_total' if device.type == 'cuda' else 'cpu_time_total',
        row_limit=80,
    )
    summary_file = args.output_dir / 'summary.txt'
    summary_file.write_text(summary)
    print(summary)
    print(f'Wrote TensorBoard trace to: {args.output_dir / "trace"}')
    print(f'Wrote summary to: {summary_file}')


if __name__ == '__main__':
    main()
