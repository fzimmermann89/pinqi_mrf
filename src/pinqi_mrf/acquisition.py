"""Acquisition helpers."""

import mr2
import torch
from mr2.utils import unsqueeze_left

RADIAL_GOLDEN_ANGLE = 1.9416129845950924


class AcquisitionData(mr2.data.Dataclass):
    """Trajectory and density compensation data."""

    trajectory: mr2.data.KTrajectory
    dcf: mr2.data.DcfData


def create_radial_trajectory(
    n_spokes: int,
    n_readout_samples: int,
) -> AcquisitionData:
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
    radial = radial.float() / radial.float().mean()
    dcf = mr2.data.DcfData(unsqueeze_left(radial, 4))

    return AcquisitionData(trajectory=trajectory, dcf=dcf)
