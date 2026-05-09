"""Tests for white k-space rhs noise sampling."""

from __future__ import annotations

import torch

from mr2.data import KTrajectory, SpatialDimension
from mr2.operators import NonUniformFastFourierOp
from pinqi_mrf.white_kspace_subspace_rhs_sampler import WhiteKspaceSubspaceRhsNoiseSampler


def create_time_varying_2d_nufft_op(n_timepoints: int = 4, image_shape: tuple[int, int] = (12, 10)) -> torch.nn.Module:
    """Create a small 2D NUFFT with one radial spoke per time point."""
    angles = torch.linspace(0, torch.pi, n_timepoints + 1)[:-1]
    readout = torch.linspace(-image_shape[-1] / 2, image_shape[-1] / 2, image_shape[-1])
    kx = (torch.cos(angles)[:, None] * readout[None, :])[:, None, None, :]
    ky = (torch.sin(angles)[:, None] * readout[None, :])[:, None, None, :]
    kx = kx[:, None, :, :]
    ky = ky[:, None, :, :]
    kz = torch.zeros_like(kx)
    trajectory = KTrajectory(kz=kz, ky=ky, kx=kx)
    return NonUniformFastFourierOp(
        direction=(-2, -1),
        recon_matrix=SpatialDimension(z=1, y=image_shape[0], x=image_shape[1]),
        encoding_matrix=SpatialDimension(z=1, y=image_shape[0], x=image_shape[1]),
        traj=trajectory,
    )


def test_white_kspace_subspace_rhs_noise_sampler() -> None:
    """Sample finite rhs noise with scalar and batched noise levels."""
    n_timepoints = 4
    n_coefficients = 2
    n_coils = 3
    image_shape = (12, 10)
    nufft_op = create_time_varying_2d_nufft_op(n_timepoints=n_timepoints, image_shape=image_shape)
    basis = torch.randn(n_timepoints, n_coefficients, dtype=torch.complex64)
    subspace_gram = nufft_op.toeplitz(subspace=basis)
    sampler = WhiteKspaceSubspaceRhsNoiseSampler(subspace_gram, n_coils=n_coils)

    noise_single = sampler(0.1)
    noise_batch = sampler(torch.tensor([0.1, 0.2]))

    assert noise_single.shape == (n_coefficients, n_coils, 1, *image_shape)
    assert noise_batch.shape == (n_coefficients, 2, n_coils, 1, *image_shape)
    assert torch.isfinite(noise_single).all()
    assert torch.isfinite(noise_batch).all()
