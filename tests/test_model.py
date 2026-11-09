"""Tests for model-side operator composition helpers."""

from types import SimpleNamespace
from typing import cast

import mr2
import pytest
import torch

from pinqi_mrf.data import ParameterBatch
from pinqi_mrf.pinqi import PINQI, PINQIConfig
from pinqi_mrf.supervised import SupervisedModule


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


def create_small_module(**kwargs: object) -> SupervisedModule:
    """Create a small supervised module for model and loss checks."""
    return SupervisedModule(
        signalmodel=DummyScalingSignalModel(),
        n_subspace_coefficients=2,
        image_size=8,
        n_readout_samples=8,
        n_coils=2,
        n_compressed_coils=2,
        n_timepoints=2,
        parameter_ranges=((-0.5, 0.5, 4), (-0.5, 0.5, 4)),
        n_iterations=1,
        n_features_parameter_net=(32, 64),
        n_features_image_net=(32, 64),
        nonlinear_solver_batch_size=4,
        **kwargs,  # type: ignore[arg-type]
    )


def test_csm_refinement_solver_matches_manual_expression() -> None:
    """The low-resolution CSM refinement solver uses the expected normal equations."""
    torch.manual_seed(0)
    alpha = torch.randn(2, 3, 1, 1, 4, 5, dtype=torch.complex64)
    csm_low = torch.randn(2, 4, 1, 2, 3, dtype=torch.complex64)
    coil_rhs = torch.randn(2, 3, 4, 1, 4, 5, dtype=torch.complex64)
    downsampling = mr2.operators.DownSamplingOp(
        dim=(-2, -1),
        domain_shape=(4, 5),
        range_shape=(2, 3),
    )
    regularization_weight = torch.tensor(0.25)
    pinqi = PINQI.__new__(PINQI)
    pinqi.config = PINQIConfig(
        csm_refinement_iterations=(),
        csm_refinement_shape=(2, 3),
        csm_refinement_weight=0.0,
        csm_refinement_cg_iterations=1,
        phase_normalization_shape=(2, 3),
        initial_cg_iterations=1,
        initial_cg_regularization=0.0,
        image_cg_iterations=1,
        image_cg_tolerance=0.0,
        n_parameter_priors=3,
    )
    solver = pinqi.get_csm_refinement_solver(
        downsampling=downsampling,
        subspace_gram=mr2.operators.IdentityOp(),
        regularization_weight=regularization_weight,
    )

    coil_lift = (
        mr2.operators.EinsumOp(
            alpha,
            'batch coeff one z y x, batch coil z y x -> batch coeff coil z y x',
        )
        @ downsampling.H
    )
    operator = coil_lift.H @ mr2.operators.IdentityOp() @ coil_lift + regularization_weight
    (expected_operator_result,) = operator(csm_low)
    (expected_rhs,) = coil_lift.H(coil_rhs)
    expected_rhs = expected_rhs + regularization_weight * csm_low

    actual_operator = solver.operator_factory(alpha, coil_rhs, csm_low)
    (actual_operator_result,) = actual_operator(csm_low)
    (actual_rhs,) = solver.rhs_factory(alpha, coil_rhs, csm_low)

    torch.testing.assert_close(actual_operator_result, expected_operator_result)
    torch.testing.assert_close(actual_rhs, expected_rhs)


def test_ema_updates_after_configured_optimizer_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """EMA weights incorporate the optimizer update at the configured step."""
    monkeypatch.setattr(torch, 'compile', lambda module: module)
    module = create_small_module(ema_decay=0.0, ema_update_interval=4)
    parameter = next(module.pinqi.parameters())
    ema_parameter = next(module.pinqi_ema.parameters())
    with torch.no_grad():
        parameter.fill_(2.0)
        ema_parameter.fill_(1.0)
    module._trainer = SimpleNamespace(global_step=3)
    optimizer = torch.optim.SGD(module.pinqi.parameters(), lr=0.0)

    module.optimizer_step(0, 0, optimizer, lambda: None)

    torch.testing.assert_close(ema_parameter, parameter)


def test_supervised_parameter_loss_uses_m0_magnitude(monkeypatch: pytest.MonkeyPatch) -> None:
    """DELICS M0 supervision ignores CSM-gauge phase."""
    monkeypatch.setattr(torch, 'compile', lambda module: module)
    module = create_small_module(loss_weights=(1.0,))
    batch = cast(
        ParameterBatch,
        {
            'm0': torch.ones(1, 2, 2, dtype=torch.complex64),
            't1': torch.ones(1, 2, 2),
            't2': torch.ones(1, 2, 2),
            'mask': torch.tensor([[[True, True], [True, False]]]),
        },
    )
    target_m0, target_t1, target_t2 = module._parameters_from_batch(batch)
    prediction_m0 = -target_m0[:, None, None].clone()
    prediction_m0[..., 1, 1] = 0
    prediction = (prediction_m0, target_t1[:, None, None], target_t2[:, None, None])

    loss, components = module.supervised_parameter_loss([prediction], batch)

    torch.testing.assert_close(components['m0'], torch.tensor(0.0))
    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_validation_reports_primary_target_range_and_additional_plotrange_ssim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primary SSIM uses slice target limits while the extra variant uses plot limits."""
    monkeypatch.setattr(torch, 'compile', lambda module: module)
    module = create_small_module()
    target = torch.linspace(0.03, 0.08, 64).reshape(1, 1, 1, 8, 8)
    baseline = target + 0.02
    prediction = target + 0.04
    mask = torch.ones(1, 8, 8, dtype=torch.bool)
    display_range = (0.02, 0.15)

    result = module._evaluate_relaxation_time_validation(
        target,
        baseline,
        prediction,
        mask,
        display_range,
        clip_to_target_range=True,
    )

    target_map = target.squeeze(1, 2)[:, None]
    prediction_map = prediction.squeeze(1, 2)[:, None]
    plot_data_range = torch.tensor(display_range[1] - display_range[0])
    target_min = target_map.amin()
    target_max = target_map.amax()
    clipped_prediction = prediction_map.clamp(min=target_min, max=target_max)
    (expected_plot_range,) = mr2.operators.functionals.SSIM(
        target_map.clamp(*display_range), mask[:, None], data_range=plot_data_range
    )(clipped_prediction.clamp(*display_range))
    (expected_target_range,) = mr2.operators.functionals.SSIM(
        target_map, mask[:, None], data_range=(target_max - target_min).reshape(1, 1, 1, 1)
    )(clipped_prediction)

    torch.testing.assert_close(result.ssim, expected_target_range)
    torch.testing.assert_close(result.ssim_plotrange, expected_plot_range)
    assert not torch.isclose(result.ssim, result.ssim_plotrange)


def test_input_normalization_uses_coefficient_norm_percentile() -> None:
    """Coefficient-space input normalization uses the robust voxelwise coefficient norm."""
    signal = torch.tensor(
        [
            [
                [[[[1.0 + 0j, 2.0 + 0j]]]],
                [[[[3.0 + 0j, 4.0 + 0j]]]],
            ]
        ],
        dtype=torch.complex64,
    )
    scale = PINQI.estimate_input_scale(signal)
    coefficient_norm = torch.linalg.vector_norm(signal, dim=1).reshape(1, -1)
    expected_scale = torch.quantile(coefficient_norm, 0.95, dim=1)
    torch.testing.assert_close(scale, expected_scale)


def test_project_parameters_clamps_to_model_support() -> None:
    """PINQI projection keeps DDM outputs inside configured support."""
    pinqi = PINQI.__new__(PINQI)
    torch.nn.Module.__init__(pinqi)
    pinqi.parameter_ranges = ((-0.5, 0.5, 4), (-0.25, 0.75, 4))
    parameters = (
        torch.tensor([3.0 - 4.0j, 0.25 + 0.5j]),
        torch.tensor([-1.0, 0.25]),
        torch.tensor([0.5, 1.0]),
    )

    projected = pinqi.project_parameters(parameters)

    torch.testing.assert_close(projected[0].real, torch.tensor([3.0, 0.25]))
    torch.testing.assert_close(projected[0].imag, torch.tensor([-4.0, 0.5]))
    torch.testing.assert_close(projected[1], torch.tensor([-0.5, 0.25]))
    torch.testing.assert_close(projected[2], torch.tensor([0.5, 0.75]))
