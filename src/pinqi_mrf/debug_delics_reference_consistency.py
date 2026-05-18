"""Inspect one DELICS slice against stored references and PINQI reconstructions."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import mr2
import torch

from pinqi_mrf.data import FineTuneDataModule
from pinqi_mrf.finetune import FineTuneModule
from pinqi_mrf.parameters import (
    R1_STATS,
    R2_STATS,
    relaxation_time_to_standardized_rate,
    standardized_rate_to_relaxation_time,
)


def tensor_rms(tensor: torch.Tensor) -> float:
    """Return RMS magnitude for real or complex tensors."""
    return tensor.detach().abs().square().mean().sqrt().item()


def finite_mask(values: torch.Tensor) -> torch.Tensor:
    """Return the finite mask for real or complex tensors."""
    if torch.is_complex(values):
        return torch.isfinite(values.real) & torch.isfinite(values.imag)
    return torch.isfinite(values)


def describe(name: str, values: torch.Tensor) -> str:
    """Return compact finite-value statistics."""
    selected = values[finite_mask(values)]
    if selected.numel() == 0:
        return f'{name}=empty'
    selected = selected.float()
    quantiles = torch.quantile(selected, torch.tensor((0.01, 0.5, 0.99), device=selected.device))
    return (
        f'{name}=mean:{selected.mean().item():.4g} '
        f'std:{selected.std(unbiased=False).item():.4g} '
        f'min:{selected.min().item():.4g} '
        f'p01:{quantiles[0].item():.4g} '
        f'med:{quantiles[1].item():.4g} '
        f'p99:{quantiles[2].item():.4g} '
        f'max:{selected.max().item():.4g}'
    )


def load_model(checkpoint_path: Path, device: torch.device) -> FineTuneModule:
    """Instantiate the fine-tuning module from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = dict(checkpoint.get('hyper_parameters', {}))
    allowed_hparams = {
        name
        for name, parameter in inspect.signature(FineTuneModule).parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    hparams = {key: value for key, value in hparams.items() if key in allowed_hparams}
    model = FineTuneModule(**hparams)
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)
    model.setup_acquisition()
    model.setup_ssdu_split()
    model.eval()
    return model


def find_dataset_index(dataset: Any, case_id: str, slice_index: int) -> int:
    """Find the dataset index for one DELICS case and slice."""
    for dataset_index, (file_index, dataset_slice_index) in enumerate(dataset.samples):
        _path, dataset_case_id = dataset.files[file_index]
        if dataset_case_id == case_id and dataset_slice_index == slice_index:
            return dataset_index
    raise ValueError(f'Could not find {case_id} slice {slice_index} in dataset.')


def plot_panel(
    axes: Any,
    row: int,
    col: int,
    image: torch.Tensor,
    title: str,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot one image panel."""
    ax = axes[row, col]
    im = ax.imshow(image.detach().cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    """Run the DELICS reference consistency debug workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--data-folder', type=Path, default=Path('/scratch/zimmer08/delics/radial_kspace'))
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--slice-index', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('debug_delics_reference_consistency'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    datamodule = FineTuneDataModule(folder=args.data_folder, num_workers=0, val_num_workers=0)
    datamodule.setup('fit')
    dataset = datamodule.build_delics_dataset(case_ids=(args.case_id,), slice_trim=(0, 0), include_references=True)
    dataset_index = find_dataset_index(dataset, args.case_id, args.slice_index)
    sample = dataset[dataset_index]

    file_index, _ = dataset.samples[dataset_index]
    source_path, _case_id = dataset.files[file_index]
    with h5py.File(source_path, 'r') as handle:
        raw_kspace = torch.as_tensor(handle['kspace'][args.slice_index], dtype=torch.complex64)
        reference_m0 = torch.as_tensor(handle['reference_m0'][args.slice_index], dtype=torch.complex64)
        reference_t1 = torch.as_tensor(handle['reference_t1'][args.slice_index], dtype=torch.float32)
        reference_t2 = torch.as_tensor(handle['reference_t2'][args.slice_index], dtype=torch.float32)
        stored_csm = torch.as_tensor(handle['csm_initial'][args.slice_index], dtype=torch.complex64)
        valid = bool(handle['valid'][args.slice_index])

    scaled_kspace = sample['kspace'][None].to(device)
    mask = sample['mask'].to(device).bool()
    reference_t1_batch = sample['reference_t1'][None].to(device)
    reference_t2_batch = sample['reference_t2'][None].to(device)
    reference_m0_batch = reference_m0[None, None, None].to(device)
    stored_csm_batch = stored_csm[None].to(device)

    with torch.no_grad():
        prediction_parameters, baseline_parameters = model.reconstruct_full_kspace(scaled_kspace)

        acquisition = model.acquisition.to(device)
        subspace_basis = model.signalmodel.pca.compression_matrix.mH
        if subspace_basis.ndim > 2:
            subspace_basis = subspace_basis.squeeze(tuple(range(subspace_basis.ndim - 2)))
        nufft = mr2.operators.NonUniformFastFourierOp(
            direction=(-2, -1),
            recon_matrix=model.matrix_size,
            encoding_matrix=model.matrix_size,
            traj=acquisition.trajectory,
        )
        coil_rhs = model.build_coil_rhs(scaled_kspace, acquisition, subspace_basis, nufft)
        csm_estimated = model.estimate_initial_csm(coil_rhs)
        images, _parameters = model(coil_rhs, csm_initial=csm_estimated)
        final_image = images[-1]

        reference_parameters = (
            reference_m0_batch,
            relaxation_time_to_standardized_rate(reference_t1_batch[:, None, None], R1_STATS),
            relaxation_time_to_standardized_rate(reference_t2_batch[:, None, None], R2_STATS),
        )
        reference_kspace_stored_csm = model.synthesize_kspace(reference_parameters, stored_csm_batch)
        reference_kspace_estimated_csm = model.synthesize_kspace(reference_parameters, csm_estimated)
        reference_grid_parameters = model.baseline.dictionary_match(final_image.swapaxes(0, 1))
        grid_t1 = standardized_rate_to_relaxation_time(reference_grid_parameters[1], R1_STATS).squeeze(1, 2)
        grid_t2 = standardized_rate_to_relaxation_time(reference_grid_parameters[2], R2_STATS).squeeze(1, 2)

    pred_t1 = standardized_rate_to_relaxation_time(prediction_parameters[1], R1_STATS).squeeze(0, 1, 2)
    pred_t2 = standardized_rate_to_relaxation_time(prediction_parameters[2], R2_STATS).squeeze(0, 1, 2)
    pred_m0 = prediction_parameters[0].squeeze(0, 1, 2).abs()
    baseline_t1 = standardized_rate_to_relaxation_time(baseline_parameters[1], R1_STATS).squeeze(0, 1, 2)
    baseline_t2 = standardized_rate_to_relaxation_time(baseline_parameters[2], R2_STATS).squeeze(0, 1, 2)
    baseline_m0 = baseline_parameters[0].squeeze(0, 1, 2).abs()

    ref_t1 = reference_t1_batch.squeeze(0, 1, 2)
    ref_t2 = reference_t2_batch.squeeze(0, 1, 2)
    ref_m0 = reference_m0_batch.squeeze(0, 1, 2).abs()
    grid_t1 = grid_t1.squeeze(0)
    grid_t2 = grid_t2.squeeze(0)
    grid_m0 = reference_grid_parameters[0].squeeze(0, 1, 2).abs()
    mask_2d = mask.squeeze(0)

    masked_ref_t1 = ref_t1[mask_2d]
    masked_ref_t2 = ref_t2[mask_2d]
    masked_ref_m0 = ref_m0[mask_2d]
    masked_pred_m0 = pred_m0[mask_2d]
    masked_baseline_m0 = baseline_m0[mask_2d]
    masked_grid_m0 = grid_m0[mask_2d]

    print(f'case={args.case_id} slice={args.slice_index} valid={valid}')
    print(describe('reference_m0_abs', masked_ref_m0))
    print(describe('pred_m0_abs', masked_pred_m0))
    print(describe('baseline_m0_abs', masked_baseline_m0))
    print(describe('grid_m0_abs', masked_grid_m0))
    print(describe('reference_t1_s', masked_ref_t1))
    print(describe('reference_t2_s', masked_ref_t2))
    print(f'raw_kspace_rms={tensor_rms(raw_kspace):.4g}')
    print(f'scaled_kspace_rms={tensor_rms(scaled_kspace):.4g}')
    print(f'reference_kspace_stored_csm_rms={tensor_rms(reference_kspace_stored_csm):.4g}')
    print(f'reference_kspace_estimated_csm_rms={tensor_rms(reference_kspace_estimated_csm):.4g}')
    print(
        f'reference_stored_csm_residual_ratio='
        f'{tensor_rms(reference_kspace_stored_csm - raw_kspace[None].to(device)) / max(tensor_rms(raw_kspace), torch.finfo(torch.float32).eps):.4g}'
    )
    print(
        f'reference_estimated_csm_residual_ratio='
        f'{tensor_rms(reference_kspace_estimated_csm - scaled_kspace) / max(tensor_rms(scaled_kspace), torch.finfo(torch.float32).eps):.4g}'
    )
    print(
        f'pred_t1_rmse={torch.sqrt(torch.nn.functional.mse_loss(pred_t1[mask_2d], ref_t1[mask_2d])).item():.4g} '
        f'baseline_t1_rmse={torch.sqrt(torch.nn.functional.mse_loss(baseline_t1[mask_2d], ref_t1[mask_2d])).item():.4g} '
        f'grid_t1_rmse={torch.sqrt(torch.nn.functional.mse_loss(grid_t1[mask_2d], ref_t1[mask_2d])).item():.4g}'
    )
    print(
        f'pred_t2_rmse={torch.sqrt(torch.nn.functional.mse_loss(pred_t2[mask_2d], ref_t2[mask_2d])).item():.4g} '
        f'baseline_t2_rmse={torch.sqrt(torch.nn.functional.mse_loss(baseline_t2[mask_2d], ref_t2[mask_2d])).item():.4g} '
        f'grid_t2_rmse={torch.sqrt(torch.nn.functional.mse_loss(grid_t2[mask_2d], ref_t2[mask_2d])).item():.4g}'
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / f'{args.case_id}_slice{args.slice_index:03d}.png'

    t1_vmin = float(masked_ref_t1.amin().item())
    t1_vmax = float(masked_ref_t1.amax().item())
    t2_vmin = float(masked_ref_t2.amin().item())
    t2_vmax = float(masked_ref_t2.amax().item())
    m0_vmax = float(torch.quantile(masked_ref_m0.float(), 0.99).item())

    fig, axes = plt.subplots(4, 5, figsize=(18, 14), constrained_layout=True)
    plot_panel(axes, 0, 0, ref_t1, 'Reference T1', cmap='magma', vmin=t1_vmin, vmax=t1_vmax)
    plot_panel(axes, 0, 1, pred_t1, 'PINQI T1', cmap='magma', vmin=t1_vmin, vmax=t1_vmax)
    plot_panel(axes, 0, 2, baseline_t1, 'Baseline T1', cmap='magma', vmin=t1_vmin, vmax=t1_vmax)
    plot_panel(axes, 0, 3, grid_t1, 'Grid-on-final-image T1', cmap='magma', vmin=t1_vmin, vmax=t1_vmax)
    plot_panel(axes, 0, 4, (pred_t1 - ref_t1).abs(), '|PINQI-Ref| T1', cmap='viridis')

    plot_panel(axes, 1, 0, ref_t2, 'Reference T2', cmap='viridis', vmin=t2_vmin, vmax=t2_vmax)
    plot_panel(axes, 1, 1, pred_t2, 'PINQI T2', cmap='viridis', vmin=t2_vmin, vmax=t2_vmax)
    plot_panel(axes, 1, 2, baseline_t2, 'Baseline T2', cmap='viridis', vmin=t2_vmin, vmax=t2_vmax)
    plot_panel(axes, 1, 3, grid_t2, 'Grid-on-final-image T2', cmap='viridis', vmin=t2_vmin, vmax=t2_vmax)
    plot_panel(axes, 1, 4, (pred_t2 - ref_t2).abs(), '|PINQI-Ref| T2', cmap='viridis')

    plot_panel(axes, 2, 0, ref_m0, 'Reference |M0|', cmap='gray', vmin=0.0, vmax=m0_vmax)
    plot_panel(axes, 2, 1, pred_m0, 'PINQI |M0|', cmap='gray', vmin=0.0, vmax=m0_vmax)
    plot_panel(axes, 2, 2, baseline_m0, 'Baseline |M0|', cmap='gray', vmin=0.0, vmax=m0_vmax)
    plot_panel(axes, 2, 3, grid_m0, 'Grid-on-final-image |M0|', cmap='gray', vmin=0.0, vmax=m0_vmax)
    plot_panel(axes, 2, 4, mask_2d.float(), 'Mask', cmap='gray', vmin=0.0, vmax=1.0)

    plot_panel(
        axes,
        3,
        0,
        raw_kspace.abs().square().mean(dim=(0, 1, 2)).sqrt(),
        'Raw radial k-space RMS',
        cmap='gray',
    )
    plot_panel(
        axes,
        3,
        1,
        scaled_kspace.squeeze(0).abs().square().mean(dim=(0, 1, 2)).sqrt(),
        'Scaled radial k-space RMS',
        cmap='gray',
    )
    plot_panel(
        axes,
        3,
        2,
        reference_kspace_stored_csm.squeeze(0).abs().square().mean(dim=(0, 1, 2)).sqrt(),
        'Ref synth k-space RMS\n(stored CSM)',
        cmap='gray',
    )
    plot_panel(
        axes,
        3,
        3,
        reference_kspace_estimated_csm.squeeze(0).abs().square().mean(dim=(0, 1, 2)).sqrt(),
        'Ref synth k-space RMS\n(estimated CSM)',
        cmap='gray',
    )
    plot_panel(
        axes,
        3,
        4,
        final_image.abs().square().sum(dim=1).sqrt().squeeze(0).squeeze(0),
        'Final image coeff norm',
        cmap='gray',
    )
    fig.suptitle(f'{args.case_id} slice {args.slice_index}', fontsize=14)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f'saved figure: {figure_path}')


if __name__ == '__main__':
    main()
