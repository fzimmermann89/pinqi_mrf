"""Tests for signal-model helpers."""

import mr2
import torch

from pinqi_mrf.signal_models import SubspaceLookup


class DummyScalingSignalModel(mr2.operators.SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Small signal model with a pure multiplicative first parameter."""

    def forward(self, m0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> tuple[torch.Tensor]:
        """Evaluate the dummy signal model."""
        signal = torch.stack(
            (
                1 + p1,
                2 + p2,
                p1 + 1j * p2,
            ),
            dim=0,
        )
        return (m0 * signal,)


def test_subspace_lookup_shape_and_scaling() -> None:
    """SubspaceLookup returns complex coefficients and applies scaling outside the LUT."""
    signalmodel = SubspaceLookup(
        DummyScalingSignalModel(),
        n_subspace_coefficients=2,
        parameter_stats=((1.0, 0.5), (5.0, 2.0)),
        parameter_ranges=((-0.5, 0.5, 4), (-0.5, 0.5, 5)),
        n_timepoints=2,
    )

    m0 = torch.tensor([[1.0 + 2.0j, 0.5 - 0.25j]])
    p1 = torch.tensor([[0.0, 0.25]])
    p2 = torch.tensor([[0.0, -0.25]])

    (result,) = signalmodel(m0, p1, p2)
    (unit_scale_result,) = signalmodel(torch.ones_like(m0), p1, p2)

    assert signalmodel.n_timepoints == 2
    assert result.shape == (2, *m0.shape)
    assert result.is_complex()
    torch.testing.assert_close(result, m0.unsqueeze(0) * unit_scale_result)
