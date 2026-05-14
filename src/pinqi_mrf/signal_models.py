"""Signal models for PINQI-MRF development."""

import einops
import mr2
import torch

from pinqi_mrf.parameters import R1_STATS, R2_STATS, standardized_rate_to_relaxation_time


class SubspaceLookup(mr2.operators.SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Subspace-compressed signal model evaluated by lookup table.

    The wrapped signal model is assumed to use its first argument as a pure
    multiplicative scaling parameter. The lookup table is built for unit scale,
    and the scale is applied after interpolation.
    """

    def __init__(
        self,
        signalmodel: mr2.operators.SignalModel,
        n_subspace_coefficients: int,
        parameter_stats: tuple[tuple[float, float], ...] = (R1_STATS, R2_STATS),
        parameter_ranges: tuple[tuple[float, float, int], ...] = ((-2.0, 6.5, 200), (-2.0, 9.5, 160)),
        n_timepoints: int | None = None,
    ) -> None:
        """Initialize the subspace-compressed lookup model.

        Parameters
        ----------
        signalmodel
            Signal model mapping ``m0, *relaxation_times`` to time-series signals.
            The first argument must be a pure multiplicative scaling parameter.
        n_subspace_coefficients
            Number of temporal subspace coefficients.
        parameter_stats
            Mean and standard deviation for each standardized relaxation-rate coordinate.
        parameter_ranges
            Standardized LUT support ranges as ``(minimum, maximum, n_steps)`` for each parameter.
        n_timepoints
            Number of leading time points to retain from the signal model. If
            ``None``, all time points are used.
        """
        super().__init__()
        if len(parameter_ranges) != len(parameter_stats):
            raise ValueError('parameter_ranges and parameter_stats must have the same length.')

        def simulate_unit_scale(*parameters: torch.Tensor) -> tuple[torch.Tensor]:
            relaxation_times = tuple(
                standardized_rate_to_relaxation_time(parameter, stats)
                for parameter, stats in zip(parameters, parameter_stats, strict=True)
            )
            scale = torch.tensor(1.0 + 0j, dtype=torch.complex64, device=parameters[0].device)
            return signalmodel(scale, *relaxation_times)

        def create_subspace() -> mr2.operators.PCACompressionOp:
            axes = [
                torch.linspace(minimum, maximum, n_steps, dtype=torch.float32)
                for minimum, maximum, n_steps in parameter_ranges
            ]
            mesh = torch.meshgrid(*axes, indexing='ij')
            with torch.inference_mode():
                (signal,) = simulate_unit_scale(*mesh)
            if n_timepoints is not None:
                signal = signal[:n_timepoints]
            dictionary = einops.rearrange(signal, 'time ... -> (...) time')
            return mr2.operators.PCACompressionOp(dictionary, n_subspace_coefficients, centering=False)

        self.pca = create_subspace()
        self.n_timepoints = self.pca._compression_matrix.shape[-1]

        def simulate_unit_scale_subspace(*parameters: torch.Tensor) -> tuple[torch.Tensor]:
            with torch.inference_mode():
                (signal,) = simulate_unit_scale(*parameters)
            if n_timepoints is not None:
                signal = signal[:n_timepoints]
            batch_shape = signal.shape[1:]
            signal = einops.rearrange(signal, 'time ... -> (...) time')
            (coefficients,) = self.pca(signal)
            coefficients = einops.rearrange(coefficients, 'batch coefficient -> coefficient batch')
            coefficients = coefficients.unflatten(-1, batch_shape)
            return (coefficients,)

        self.lookup = mr2.operators.LookupTableOp(
            simulate_unit_scale_subspace,
            parameter_ranges=parameter_ranges,
        )

    def __call__(self, m0: torch.Tensor, *parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Simulate subspace-compressed coefficients.

        Parameters
        ----------
        m0
            Complex equilibrium signal / proton density.
        parameters
            Standardized relaxation-rate coordinates.

        Returns
        -------
            Subspace coefficients with leading coefficient dimension.
        """
        return super().__call__(m0, *parameters)

    def forward(self, m0: torch.Tensor, *parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the subspace lookup."""
        (coefficients,) = self.lookup(*parameters)
        return (m0 * coefficients,)
