"""Tests for parameter normalization helpers."""

import torch

from pinqi_mrf.parameters import relaxation_time_to_standardized_rate, standardized_rate_to_relaxation_time


def test_standardized_rate_roundtrip() -> None:
    """Relaxation times can be standardized and converted back."""
    relaxation_time = torch.tensor([0.5, 1.0, 2.0])
    stats = (1.0, 0.5)

    standardized = relaxation_time_to_standardized_rate(relaxation_time, stats)
    result = standardized_rate_to_relaxation_time(standardized, stats)

    torch.testing.assert_close(result, relaxation_time)
