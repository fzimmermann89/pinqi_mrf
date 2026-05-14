"""Debug non-learned CSM estimation and refinement path."""

# ruff: noqa: T201

from __future__ import annotations

import argparse
from pathlib import Path

import mr2
import torch

from pinqi_mrf.data import BRAINWEB_VALUES_3T_RANDOMIZED
from pinqi_mrf.model import PinqiModule
from pinqi_mrf.parameters import R1_STATS, R2_STATS, relaxation_time_to_standardized_rate


def relative_residual(operator: mr2.operators.LinearOperator, x: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Calculate relative normal-equation residual."""
    (operator_x,) = operator(x)
    return (operator_x - rhs).norm() / rhs.norm().clamp_min(1e-12)


def csm_difference(csm: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Calculate phase-anchored relative CSM difference."""
    alignment = (csm * reference.conj()).sum(dim=1, keepdim=True)
    phase = alignment / alignment.abs().clamp_min(1e-8)
    csm = csm * phase.conj()
    return (csm - reference).norm() / reference.norm().clamp_min(1e-12)


def main() -> None:
    """Run the debug script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--brainweb-folder', type=Path, default=mr2.phantoms.brainweb.CACHE_DIR_BRAINWEB)
    parser.add_argument('--index', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--size', type=int, default=192)
    parser.add_argument('--n-coils', type=int, default=12)
    parser.add_argument('--n-compressed-coils', type=int, default=8)
    parser.add_argument('--n-subspace-coefficients', type=int, default=10)
    parser.add_argument('--n-timepoints', type=int, default=300)
    parser.add_argument('--n-readout-samples', type=int, default=192)
    parser.add_argument('--noise-std', type=float, default=0.0)
    parser.add_argument('--initial-cg-iterations', type=int, default=10)
    parser.add_argument('--csm-cg-iterations', type=int, default=10)
    parser.add_argument('--csm-lowres-size', type=int, default=64)
    parser.add_argument('--csm-weight', type=float, default=1e-2)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device)
    print(f'device: {device}')
    print(f'args: {args}')

    if not args.brainweb_folder.exists():
        raise FileNotFoundError(f'BrainWeb folder does not exist: {args.brainweb_folder}')

    phantom_dataset = mr2.phantoms.brainweb.BrainwebSlices(
        folder=args.brainweb_folder,
        what=('m0', 't1', 't2', 'mask'),
        seed='index',
        slice_preparation=mr2.phantoms.brainweb.augment(
            size=args.size,
            max_random_shear=0,
            max_random_rotation=0,
            max_random_scaling_factor=0,
            p_horizontal_flip=0,
            p_vertical_flip=1.0,
        ),
        parameters=BRAINWEB_VALUES_3T_RANDOMIZED,
        orientation=('axial',),
    )
    phantom = phantom_dataset[args.index]
    matrix_size = mr2.data.SpatialDimension(1, args.size, args.size)

    model = PinqiModule(
        n_subspace_coefficients=args.n_subspace_coefficients,
        image_size=args.size,
        n_readout_samples=args.n_readout_samples,
        n_coils=args.n_coils,
        n_compressed_coils=args.n_compressed_coils,
        n_timepoints=args.n_timepoints,
        csm_refinement_iterations=(0,),
        csm_refinement_shape=(args.csm_lowres_size, args.csm_lowres_size),
        csm_refinement_weight=args.csm_weight,
        csm_refinement_cg_iterations=args.csm_cg_iterations,
    ).to(device)
    model.setup_acquisition()

    batch = {
        'm0': phantom['m0'][None].to(device),
        'r1': relaxation_time_to_standardized_rate(phantom['t1'][None].to(device), R1_STATS),
        'r2': relaxation_time_to_standardized_rate(phantom['t2'][None].to(device), R2_STATS),
        'csm': mr2.phantoms.coils.birdcage_2d(args.n_coils, matrix_size).squeeze(0)[None].to(device),
        'mask': phantom['mask'][None].to(device),
        'noise_std': torch.tensor([args.noise_std], device=device),
    }

    print('\nGenerating compressed coilwise RHS...')
    with torch.no_grad():
        coil_rhs = model.simulate_coil_rhs(batch)
    print(f'coil_rhs: shape={tuple(coil_rhs.shape)}, dtype={coil_rhs.dtype}, norm={coil_rhs.norm().item():.6g}')

    print('\nEstimating initial CSMs with rank-1 calibration + Inati...')
    csm_initial = model.estimate_initial_csm(coil_rhs)
    csm_norm = csm_initial.abs().square().sum(dim=1).sqrt()
    print(f'csm_initial: shape={tuple(csm_initial.shape)}, dtype={csm_initial.dtype}')
    print(
        'csm_initial coil norm: '
        f'min={csm_norm.min().item():.6g}, '
        f'mean={csm_norm.mean().item():.6g}, '
        f'max={csm_norm.max().item():.6g}'
    )

    print('\nBuilding initial normal equations and solving for coefficient images...')
    if model.subspace_gram is None:
        raise RuntimeError('Acquisition operators have not been initialized.')
    rhs, normal_op = model.pinqi.get_normal_equations(coil_rhs, csm_initial, model.subspace_gram)
    print(f'rhs: shape={tuple(rhs.shape)}, dtype={rhs.dtype}, norm={rhs.norm().item():.6g}')
    with torch.no_grad():
        alpha_initial, = mr2.algorithms.optimizers.cg(normal_op, rhs, max_iterations=args.initial_cg_iterations)
    residual_initial = relative_residual(normal_op, alpha_initial, rhs)
    print(f'alpha_initial: shape={tuple(alpha_initial.shape)}, norm={alpha_initial.norm().item():.6g}')
    print(f'initial relative residual: {residual_initial.item():.6g}')

    print('\nRunning low-resolution CSM CG update...')
    downsampling = mr2.operators.DownSamplingOp(
        dim=(-2, -1),
        domain_shape=tuple(csm_initial.shape[-2:]),
        range_shape=(args.csm_lowres_size, args.csm_lowres_size),
    )
    (csm_low_initial,) = downsampling(csm_initial)
    csm_refinement_solver = model.pinqi.get_csm_refinement_solver(
        downsampling=downsampling,
        subspace_gram=model.subspace_gram,
        regularization_weight=args.csm_weight,
    )
    csm_low, = csm_refinement_solver(
        alpha_initial,
        coil_rhs,
        csm_low_initial,
        initial_value=(csm_low_initial,),
    )
    csm_refined, = downsampling.H(csm_low)
    csm_refined = csm_refined / csm_refined.abs().square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    alignment = (csm_refined * csm_initial.conj()).sum(dim=1, keepdim=True)
    phase = alignment / alignment.abs().clamp_min(1e-8)
    csm_refined = csm_refined * phase.conj()
    csm_refined_norm = csm_refined.abs().square().sum(dim=1).sqrt()
    print(f'csm_refined: shape={tuple(csm_refined.shape)}, dtype={csm_refined.dtype}')
    print(
        'csm_refined coil norm: '
        f'min={csm_refined_norm.min().item():.6g}, '
        f'mean={csm_refined_norm.mean().item():.6g}, '
        f'max={csm_refined_norm.max().item():.6g}'
    )
    print(f'phase-anchored relative CSM change: {csm_difference(csm_refined, csm_initial).item():.6g}')

    print('\nSolving coefficient images with refined CSMs...')
    rhs_refined, normal_op_refined = model.pinqi.get_normal_equations(coil_rhs, csm_refined, model.subspace_gram)
    with torch.no_grad():
        alpha_refined, = mr2.algorithms.optimizers.cg(
            normal_op_refined,
            rhs_refined,
            max_iterations=args.initial_cg_iterations,
        )
    residual_refined = relative_residual(normal_op_refined, alpha_refined, rhs_refined)
    print(f'rhs_refined: shape={tuple(rhs_refined.shape)}, norm={rhs_refined.norm().item():.6g}')
    print(f'alpha_refined: shape={tuple(alpha_refined.shape)}, norm={alpha_refined.norm().item():.6g}')
    print(f'refined relative residual: {residual_refined.item():.6g}')
    coefficient_change = (alpha_refined - alpha_initial).norm() / alpha_initial.norm().clamp_min(1e-12)
    print(f'coefficient relative change: {coefficient_change.item():.6g}')

    print('\nDone.')


if __name__ == '__main__':
    main()
