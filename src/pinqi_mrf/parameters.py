"""Parameter normalization helpers."""

import torch

R1_STATS = (0.6, 0.1)
"""Mean and standard deviation of BrainWeb 3T R1 values inside the brain mask."""

R2_STATS = (6, 2)
"""Mean and standard deviation of BrainWeb 3T R2 values inside the brain mask."""


def relaxation_time_to_standardized_rate(relaxation_time: torch.Tensor, stats: tuple[float, float]) -> torch.Tensor:
    """Convert physical relaxation times to standardized relaxation rates."""
    rate = relaxation_time.reciprocal()
    mean, std = stats
    return (rate - mean) / std


def standardized_rate_to_relaxation_time(standardized_rate: torch.Tensor, stats: tuple[float, float]) -> torch.Tensor:
    """Convert standardized relaxation rates to physical relaxation times."""
    mean, std = stats
    rate = standardized_rate * std + mean
    return rate.reciprocal()
