"""Parameter normalization helpers."""

from __future__ import annotations

import torch


def relaxation_time_to_normalized_rate(relaxation_time: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    """Convert physical relaxation times to normalized relaxation rates."""
    rate = relaxation_time.reciprocal()
    lower, upper = bounds
    return 2 * (rate - lower) / (upper - lower) - 1


def normalized_rate_to_relaxation_time(normalized_rate: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    """Convert normalized relaxation rates to physical relaxation times."""
    lower, upper = bounds
    rate = lower + 0.5 * (normalized_rate + 1) * (upper - lower)
    return rate.reciprocal()
