"""Tests for acquisition helpers."""

import torch

from pinqi_mrf.acquisition import create_radial_trajectory


def test_create_radial_trajectory() -> None:
    """Create a radial trajectory with one spoke per time point."""
    n_spokes = 5
    n_readout_samples = 7

    acquisition = create_radial_trajectory(n_spokes=n_spokes, n_readout_samples=n_readout_samples)

    assert acquisition.trajectory.shape == (n_spokes, 1, 1, 1, n_readout_samples)
    assert acquisition.trajectory.kx.shape == (n_spokes, 1, 1, 1, n_readout_samples)
    assert acquisition.trajectory.ky.shape == (n_spokes, 1, 1, 1, n_readout_samples)
    assert acquisition.dcf.data.shape == (1, 1, 1, 1, n_readout_samples)
    assert torch.isfinite(acquisition.dcf.data).all()
    assert (acquisition.dcf.data >= 0).all()
    torch.testing.assert_close(acquisition.dcf.data.mean(), torch.tensor(1.0))
