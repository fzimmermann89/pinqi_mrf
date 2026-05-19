"""Tests for parameter normalization helpers."""

import torch

from pinqi_mrf.parameters import LogRelaxationTransform


def test_log_relaxation_transform_roundtrip() -> None:
    """Physical relaxation times can be mapped to normalized coordinates and back."""
    transform = LogRelaxationTransform(bounds=((0.1, 4.0), (0.01, 2.5)))
    t1 = torch.tensor([0.5, 1.0, 2.0])
    t2 = torch.tensor([0.03, 0.08, 0.5])

    q1, q2 = transform.physical_to_normalized(t1, t2)
    result_t1, result_t2 = transform.normalized_to_physical(q1, q2)

    torch.testing.assert_close(result_t1, t1)
    torch.testing.assert_close(result_t2, t2)
