"""Baseline reconstruction modules for PINQI-MRF."""

import mr2
import torch

from pinqi_mrf.pinqi import ParameterTuple


class BaselineReconstruction(torch.nn.Module):
    """Unregularized CG reconstruction followed by dictionary matching."""

    def __init__(
        self,
        signalmodel: mr2.operators.SignalModel,
        parameter_ranges: tuple[tuple[float, float, int], ...],
        nonlinear_solver_batch_size: int = 8192,
    ) -> None:
        """Initialize the baseline reconstruction."""
        super().__init__()
        self.dictionary_match = mr2.operators.DictionaryMatchOp(
            signalmodel,
            index_of_scaling_parameter=0,
            batch_size=nonlinear_solver_batch_size,
        )
        nonlinear_solver_grids = [
            torch.linspace(minimum, maximum, n_steps) for minimum, maximum, n_steps in parameter_ranges
        ]
        broadcasted_grids = []
        for parameter_idx, grid in enumerate(nonlinear_solver_grids):
            grid_shape = [1] * len(nonlinear_solver_grids)
            grid_shape[parameter_idx] = grid.numel()
            broadcasted_grids.append(grid.reshape(grid_shape))
        self.dictionary_match.append(torch.tensor(1.0 + 0j), *broadcasted_grids)

    def forward(
        self,
        rhs: torch.Tensor,
        normal_op: mr2.operators.LinearOperator,
    ) -> ParameterTuple:
        """Apply unregularized CG and dictionary matching."""
        images = list(mr2.algorithms.optimizers.cg(normal_op, rhs, max_iterations=10))
        return self.dictionary_match(images[-1].swapaxes(0, 1))
