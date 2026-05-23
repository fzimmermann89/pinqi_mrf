"""Parameter-coordinate transforms for PINQI-MRF."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LogRelaxationTransform:
    """Map normalized log-time coordinates to bounded physical relaxation times."""

    bounds: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        """Validate the configured physical relaxation-time bounds."""
        if not self.bounds:
            raise ValueError('bounds must not be empty.')
        for index, (lower, upper) in enumerate(self.bounds):
            if lower <= 0:
                raise ValueError(f'bounds[{index}] lower bound must be positive, got {lower}.')
            if lower >= upper:
                raise ValueError(f'bounds[{index}] must satisfy lower < upper, got ({lower}, {upper}).')

    def _bound_tensors(
        self,
        reference: torch.Tensor,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower, upper = self.bounds[index]
        lower_tensor = torch.as_tensor(lower, dtype=reference.dtype, device=reference.device)
        upper_tensor = torch.as_tensor(upper, dtype=reference.dtype, device=reference.device)
        return lower_tensor, upper_tensor

    def normalized_to_physical(self, *normalized: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Convert normalized log-time coordinates to physical relaxation times."""
        relaxation_times = []
        for index, coordinate in enumerate(normalized):
            lower, upper = self._bound_tensors(coordinate, index)
            log_lower = lower.log()
            log_upper = upper.log()
            log_time = log_lower + 0.5 * (coordinate + 1.0) * (log_upper - log_lower)
            relaxation_times.append(log_time.exp())
        return tuple(relaxation_times)

    def physical_to_normalized(self, *relaxation_times: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Convert physical relaxation times to normalized log-time coordinates."""
        coordinates = []
        for index, relaxation_time in enumerate(relaxation_times):
            lower, upper = self._bound_tensors(relaxation_time, index)
            clipped = relaxation_time.clamp(min=lower, max=upper)
            log_lower = lower.log()
            log_upper = upper.log()
            coordinate = 2.0 * (clipped.log() - log_lower) / (log_upper - log_lower) - 1.0
            coordinates.append(coordinate)
        return tuple(coordinates)
