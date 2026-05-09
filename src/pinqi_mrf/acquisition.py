"""Acquisition helpers."""

from __future__ import annotations

import torch
from mr2.data import KTrajectory
from mr2.data.traj_calculators import KTrajectoryRadial2D

RADIAL_GOLDEN_ANGLE = 1.9416129845950924


def create_radial_trajectory(
    n_spokes: int,
    n_readout_samples: int,
) -> KTrajectory:
    """Create a 2D radial trajectory with one spoke per time frame."""
    return KTrajectoryRadial2D(RADIAL_GOLDEN_ANGLE)(
        n_k0=n_readout_samples,
        k0_center=n_readout_samples // 2,
        k1_idx=torch.arange(n_spokes)[:, None, None, None, None],
    )
