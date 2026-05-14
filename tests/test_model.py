"""Tests for model-side operator composition helpers."""

import mr2
import torch

from pinqi_mrf.model import PINQI


def test_csm_refinement_solver_matches_manual_expression() -> None:
    """The low-resolution CSM refinement solver uses the expected normal equations."""
    torch.manual_seed(0)
    alpha = torch.randn(2, 3, 1, 4, 5, dtype=torch.complex64)
    csm_low = torch.randn(2, 4, 1, 2, 3, dtype=torch.complex64)
    coil_rhs = torch.randn(2, 3, 4, 1, 4, 5, dtype=torch.complex64)
    downsampling = mr2.operators.DownSamplingOp(
        dim=(-2, -1),
        domain_shape=(4, 5),
        range_shape=(2, 3),
    )
    regularization_weight = torch.tensor(0.25)
    pinqi = PINQI.__new__(PINQI)
    pinqi.csm_refinement_cg_iterations = 1
    solver = pinqi.get_csm_refinement_solver(
        downsampling=downsampling,
        subspace_gram=mr2.operators.IdentityOp(),
        regularization_weight=regularization_weight,
    )

    coil_lift = mr2.operators.EinsumOp(alpha, 'b t z y x, b c z y x -> b t c z y x') @ downsampling.H
    operator = coil_lift.H @ mr2.operators.IdentityOp() @ coil_lift + regularization_weight
    (expected_operator_result,) = operator(csm_low)
    (expected_rhs,) = coil_lift.H(coil_rhs)
    expected_rhs = expected_rhs + regularization_weight * csm_low

    actual_operator = solver.operator_factory(alpha, coil_rhs, csm_low)
    (actual_operator_result,) = actual_operator(csm_low)
    (actual_rhs,) = solver.rhs_factory(alpha, coil_rhs, csm_low)

    torch.testing.assert_close(actual_operator_result, expected_operator_result)
    torch.testing.assert_close(actual_rhs, expected_rhs)
