"""Acquisition helpers."""

import mr2
import torch
from mr2.utils import unsqueeze_left

RADIAL_GOLDEN_ANGLE = 1.9416129845950924


def create_radial_trajectory(
    n_spokes: int,
    n_readout_samples: int,
) -> tuple[mr2.data.KTrajectory, mr2.data.DcfData]:
    """Create a 2D radial trajectory and analytical DCF with one spoke per time frame."""
    center = n_readout_samples // 2
    trajectory = mr2.data.traj_calculators.KTrajectoryRadial2D(RADIAL_GOLDEN_ANGLE)(
        n_k0=n_readout_samples,
        k0_center=center,
        k1_idx=torch.arange(n_spokes)[:, None, None, None, None],
    )

    radial = (torch.arange(n_readout_samples) - center).abs().float()
    if n_readout_samples > 1:
        nonzero = radial[radial > 0]
        radial[center] = nonzero.min() if nonzero.numel() else 1.0
    radial = radial / radial.mean()
    dcf = mr2.data.DcfData(unsqueeze_left(radial, 4))

    return trajectory, dcf
