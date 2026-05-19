"""Loss functions for PINQI-MRF training."""

from collections.abc import Sequence

import mr2
import torch


def charbonnier_loss(
    predictions: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Compute masked Charbonnier loss for a sequence of maps."""
    loss = torch.tensor(0.0)
    mask = mask[:, None, None]
    for prediction, target in zip(predictions, targets, strict=True):
        difference = prediction[mask] - target[mask]
        loss = loss + torch.sqrt(difference.abs().square() + eps**2).mean()
    return loss


def ssim_loss(
    predictions: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    mask: torch.Tensor,
    clamp: tuple[float, float] = (-3.0, 3.0),
) -> torch.Tensor:
    """Compute masked SSIM loss for a sequence of normalized maps."""
    lower, upper = clamp
    loss = torch.tensor(0.0)
    mask = mask[:, None, None]
    for prediction, target in zip(predictions, targets, strict=True):
        prediction = prediction.clamp(lower, upper)
        target = target.clamp(lower, upper)
        data_range = torch.tensor(upper - lower)
        (ssim,) = mr2.operators.functionals.SSIM(target, mask, data_range=data_range)(prediction)
        loss = loss + (1 - ssim)
    return loss
