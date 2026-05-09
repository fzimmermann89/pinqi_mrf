"""Signal models for PINQI-MRF development."""

from __future__ import annotations

import einops
import torch

from mr2.operators.SignalModel import SignalModel
from mr2.operators import LookupTableOp, PCACompressionOp
from mr2.operators.models import NeuroMRF

from pinqi_mrf.parameters import normalized_rate_to_relaxation_time


class NeuroMRFSubspaceLookup(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Subspace-compressed NeuroMRF signal model evaluated by lookup table."""

    def __init__(
        self,
        n_subspace_coefficients: int,
        r1_bounds: tuple[float, float],
        r2_bounds: tuple[float, float],
        r1_steps: int = 100,
        r2_steps: int = 100,
        neuro_mrf: NeuroMRF | None = None,
    ) -> None:
        """Initialize the subspace-compressed NeuroMRF lookup model.

        Parameters
        ----------
        n_subspace_coefficients
            Number of temporal subspace coefficients.
        r1_bounds
            Physical R1 bounds used by the normalized R1 coordinate.
        r2_bounds
            Physical R2 bounds used by the normalized R2 coordinate.
        r1_steps
            Number of LUT support points along normalized R1.
        r2_steps
            Number of LUT support points along normalized R2.
        neuro_mrf
            Optional NeuroMRF simulator. If omitted, the default DeliCS
            NeuroMRF sequence is used.
        """
        super().__init__()
        self.r1_bounds = r1_bounds
        self.r2_bounds = r2_bounds
        self.neuro_mrf = NeuroMRF() if neuro_mrf is None else neuro_mrf
        self.pca = self._create_subspace(n_subspace_coefficients, r1_steps, r2_steps)
        self.n_timepoints = self.pca._compression_matrix.shape[-1]
        self.lookup = LookupTableOp(
            self._simulate_unit_scale_subspace_real,
            parameter_ranges=((-1.0, 1.0, r1_steps), (-1.0, 1.0, r2_steps)),
        )

    def __call__(self, m0: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor) -> tuple[torch.Tensor]:
        """Simulate subspace-compressed NeuroMRF coefficients.

        Parameters
        ----------
        m0
            Complex equilibrium signal / proton density.
        r1
            Normalized R1 coordinate in ``[-1, 1]``.
        r2
            Normalized R2 coordinate in ``[-1, 1]``.

        Returns
        -------
            Subspace coefficients with leading coefficient dimension.
        """
        return super().__call__(m0, r1, r2)

    def forward(self, m0: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the NeuroMRF subspace lookup."""
        (coefficients_real,) = self.lookup(r1, r2)
        coefficients = torch.view_as_complex(coefficients_real.moveaxis(1, -1).contiguous())
        return (m0 * coefficients,)

    def _create_subspace(self, n_subspace_coefficients: int, r1_steps: int, r2_steps: int) -> PCACompressionOp:
        """Create a PCA compression operator from a NeuroMRF dictionary."""
        r1_axis = torch.linspace(-1.0, 1.0, r1_steps)[:, None]
        r2_axis = torch.linspace(-1.0, 1.0, r2_steps)[None, :]
        t1 = normalized_rate_to_relaxation_time(r1_axis, self.r1_bounds)
        t2 = normalized_rate_to_relaxation_time(r2_axis, self.r2_bounds)
        with torch.inference_mode():
            (signal,) = self.neuro_mrf(torch.tensor(1.0 + 0j, dtype=torch.complex64), t1, t2)
        dictionary = einops.rearrange(signal, 'time ... -> (...) time')
        return PCACompressionOp(dictionary, n_subspace_coefficients, centering=False)

    def _simulate_unit_scale_subspace_real(self, r1: torch.Tensor, r2: torch.Tensor) -> tuple[torch.Tensor]:
        """Simulate unit-scale NeuroMRF and compress it to the subspace."""
        t1 = normalized_rate_to_relaxation_time(r1, self.r1_bounds)
        t2 = normalized_rate_to_relaxation_time(r2, self.r2_bounds)
        with torch.inference_mode():
            (signal,) = self.neuro_mrf(torch.tensor(1.0 + 0j, dtype=torch.complex64, device=r1.device), t1, t2)
        batch_shape = signal.shape[1:]
        signal = einops.rearrange(signal, 'time ... -> (...) time')
        (coefficients,) = self.pca(signal)
        coefficients = coefficients.mT.reshape(coefficients.shape[-1], *batch_shape)
        return (torch.view_as_real(coefficients).moveaxis(-1, 1).contiguous(),)
