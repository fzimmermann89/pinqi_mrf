"""Tests for acquisition helpers."""

from pinqi_mrf.acquisition import create_radial_trajectory


def test_create_radial_trajectory() -> None:
    """Create a radial trajectory with one spoke per time point."""
    n_spokes = 5
    n_readout_samples = 7

    trajectory = create_radial_trajectory(n_spokes=n_spokes, n_readout_samples=n_readout_samples)

    assert trajectory.shape == (n_spokes, 1, 1, 1, n_readout_samples)
    assert trajectory.kx.shape == (n_spokes, 1, 1, 1, n_readout_samples)
    assert trajectory.ky.shape == (n_spokes, 1, 1, 1, n_readout_samples)
