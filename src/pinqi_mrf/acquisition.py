"""Acquisition helpers."""

import mr2
import torch

RADIAL_GOLDEN_ANGLE = 1.9416129845950924


def create_radial_trajectory(
    n_spokes: int,
    n_readout_samples: int,
) -> mr2.data.KTrajectory:
    """Create a 2D radial trajectory with one spoke per time frame."""
    return mr2.data.traj_calculators.KTrajectoryRadial2D(RADIAL_GOLDEN_ANGLE)(
        n_k0=n_readout_samples,
        k0_center=n_readout_samples // 2,
        k1_idx=torch.arange(n_spokes)[:, None, None, None, None],
    )
