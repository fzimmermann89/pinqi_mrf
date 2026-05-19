"""Signal models for PINQI-MRF development."""

import einops
import mr2
import torch

from pinqi_mrf.parameters import LogRelaxationTransform


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
        parameter_transform: LogRelaxationTransform,
        parameter_ranges: tuple[tuple[float, float, int], ...],
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
        parameter_transform
            Transform mapping normalized log-time coordinates to physical relaxation times.
        parameter_ranges
            Normalized log-time LUT support ranges as ``(minimum, maximum, n_steps)`` for each parameter.
        n_timepoints
            Number of leading time points to retain from the signal model. If
            ``None``, all time points are used.
        """
        super().__init__()
        if len(parameter_ranges) != len(parameter_transform.bounds):
            raise ValueError('parameter_ranges and parameter_transform.bounds must have the same length.')
        for index, (minimum, maximum, n_steps) in enumerate(parameter_ranges):
            if minimum >= maximum:
                raise ValueError(
                    f'parameter_ranges[{index}] must satisfy minimum < maximum, got ({minimum}, {maximum}).'
                )
            if n_steps < 2:
                raise ValueError(f'parameter_ranges[{index}] must use at least two steps, got {n_steps}.')
            if minimum < -1.0 or maximum > 1.0:
                raise ValueError(
                    f'parameter_ranges[{index}] must stay inside normalized log-time support [-1, 1], '
                    f'got ({minimum}, {maximum}).'
                )
        self.parameter_transform = parameter_transform

        def simulate_unit_scale(*parameters: torch.Tensor) -> tuple[torch.Tensor]:
            relaxation_times = self.parameter_transform.normalized_to_physical(*parameters)
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
            Normalized log-time coordinates.

        Returns
        -------
            Subspace coefficients with leading coefficient dimension.
        """
        return super().__call__(m0, *parameters)

    def forward(self, m0: torch.Tensor, *parameters: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the subspace lookup."""
        parameters = mr2.utils.unsqueeze_tensors_at(*parameters, dim=1, ndim=5)
        parameters = torch.broadcast_tensors(*parameters)
        coefficient_samples = []
        for sample_parameters in zip(*parameters, strict=True):
            sample_parameters = tuple(parameter[None] for parameter in sample_parameters)
            (sample_coefficients,) = self.lookup(*sample_parameters)
            coefficient_samples.append(sample_coefficients)
        coefficients = torch.cat(coefficient_samples, dim=1)
        m0 = mr2.utils.unsqueeze_at(m0, dim=min(1, m0.ndim), n=coefficients.ndim - 1 - m0.ndim)
        return (m0 * coefficients,)
