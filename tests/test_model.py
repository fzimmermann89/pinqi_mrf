"""Tests for model-side operator composition helpers."""

from pathlib import Path
from typing import cast

import mr2
import pytest
import torch

from pinqi_mrf.data import BatchType
from pinqi_mrf.finetune import FineTuneModule
from pinqi_mrf.pinqi import PINQI, PINQIConfig


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


def create_small_finetune_module(**kwargs: object) -> FineTuneModule:
    """Create a small fine-tuning module for checkpoint-loading tests."""
    return FineTuneModule(
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


def test_finetune_module_loads_pretrained_checkpoint_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fine-tuning checkpoint initialization loads weights without trainer state."""
    monkeypatch.setattr(torch, 'compile', lambda module: module)
    source = create_small_finetune_module()
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters()):
            parameter.fill_(0.01 * (index + 1))

    checkpoint_path = tmp_path / 'pretrain.ckpt'
    torch.save({'state_dict': source.state_dict(), 'epoch': 17, 'global_step': 123}, checkpoint_path)

    target = create_small_finetune_module(pretrained_checkpoint=checkpoint_path)

    for key, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], expected)
    assert target.current_epoch == 0
    assert target.global_step == 0


def test_teacher_parameter_loss_uses_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Teacher parameter loss ignores voxels outside the mask."""
    monkeypatch.setattr(torch, 'compile', lambda module: module)
    module = create_small_finetune_module()
    prediction = (
        torch.zeros(1, 1, 1, 2, 2, dtype=torch.complex64),
        torch.zeros(1, 1, 1, 2, 2),
        torch.zeros(1, 1, 1, 2, 2),
    )
    target = (
        torch.zeros(1, 1, 1, 2, 2, dtype=torch.complex64),
        torch.zeros(1, 1, 1, 2, 2),
        torch.zeros(1, 1, 1, 2, 2),
    )
    target[1][..., 0, 0] = 2.0
    target[1][..., 1, 1] = 10.0
    mask = torch.tensor([[[True, False], [False, False]]])

    loss = module.teacher_parameter_loss([prediction], target, mask)

    torch.testing.assert_close(loss, torch.tensor(4.0))


def test_supervised_parameter_loss_uses_m0_magnitude(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic M0 supervision ignores CSM-gauge phase."""
    monkeypatch.setattr(torch, 'compile', lambda module: module)
    module = create_small_finetune_module(loss_weights=(1.0,))
    batch = cast(
        BatchType,
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


def test_teacher_parameter_loss_uses_m0_magnitude(monkeypatch: pytest.MonkeyPatch) -> None:
    """Teacher M0 supervision ignores CSM-gauge phase."""
    monkeypatch.setattr(torch, 'compile', lambda module: module)
    module = create_small_finetune_module()
    prediction = (
        -torch.ones(1, 1, 1, 2, 2, dtype=torch.complex64),
        torch.zeros(1, 1, 1, 2, 2),
        torch.zeros(1, 1, 1, 2, 2),
    )
    target = (
        torch.ones(1, 1, 1, 2, 2, dtype=torch.complex64),
        torch.zeros(1, 1, 1, 2, 2),
        torch.zeros(1, 1, 1, 2, 2),
    )
    mask = torch.ones(1, 2, 2, dtype=torch.bool)

    loss = module.teacher_parameter_loss([prediction], target, mask)

    torch.testing.assert_close(loss, torch.tensor(0.0))


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
