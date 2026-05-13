"""Experimental sampler for subspace rhs noise induced by white k-space noise."""

import math

import mr2
import torch
from mr2.operators.NonUniformFastFourierOp import SubspaceNonUniformFastFourierOpGramOp


class WhiteKspaceSubspaceRhsNoiseSampler(mr2.utils.TensorAttributeMixin):
    r"""Sample coilwise rhs noise under a white k-space noise model.

    The sampler assumes a subspace Toeplitz Gram operator without coil sensitivities.
    If the measurement noise is white complex Gaussian with standard deviation
    ``sigma``, the additive noise term in the normal equations has covariance

    .. math::
        \sigma^2 G,

    where :math:`G` is the subspace Gram operator. The Toeplitz embedding used
    for the fast Gram application is positive semidefinite as a full operator,
    but the cached Fourier-domain blocks are not guaranteed to be positive
    semidefinite at every frequency. The sampler therefore projects each
    frequency block to the positive semidefinite cone before factorization.
    """

    def __init__(self, subspace_gram_op: SubspaceNonUniformFastFourierOpGramOp, n_coils: int) -> None:
        """Initialize the rhs noise sampler.

        Parameters
        ----------
        subspace_gram_op
            Subspace Toeplitz Gram operator with cached Fourier-domain kernel
            ``_kernel`` as in ``SubspaceNonUniformFastFourierOpGramOp`` from the
            ``subspacenufft`` branch.
        n_coils
            Number of receiver coils. Noise is sampled independently across
            coils and correlated across subspace channels according to the Gram
            kernel.
        """
        super().__init__()

        kernel = subspace_gram_op._kernel
        if kernel is None:
            raise ValueError('Subspace Gram operator does not have a cached Toeplitz kernel.')

        self._subspace_dim = subspace_gram_op._subspace_dim
        self._dim = tuple(subspace_gram_op._dim)
        self._recon_shape = tuple(subspace_gram_op._recon_shape)
        self._n_coils = n_coils
        self._n_coefficients = kernel.shape[0]

        spectral_matrix = kernel.movedim((0, 1), (-2, -1))
        spectral_matrix = 0.5 * (spectral_matrix + spectral_matrix.mH)
        spectral_matrix = spectral_matrix.cpu()
        eigenvalues, eigenvectors = torch.linalg.eigh(spectral_matrix)
        diagonal_scale = spectral_matrix.diagonal(dim1=-2, dim2=-1).real.mean().clamp_min(1e-12)
        eigenvalue_floor = 1e-6 * diagonal_scale
        psd_factor = eigenvectors * eigenvalues.clamp_min(eigenvalue_floor).sqrt().unsqueeze(-2)
        self._psd_factor = psd_factor.to(kernel.device)

    def __call__(self, stddev: float | torch.Tensor) -> torch.Tensor:
        """Sample coilwise additive rhs noise.

        Parameters
        ----------
        stddev
            Standard deviation of the white complex k-space noise. The shape of
            ``stddev`` defines the batch dimensions of the sampled noise.

        Returns
        -------
            Coilwise rhs noise with the shape determined by ``stddev``, the
            configured subspace dimension, the coil dimension, and trailing
            spatial axes ``(z, y, x)``.
        """
        return super().__call__(stddev)

    def forward(self, stddev: float | torch.Tensor) -> torch.Tensor:
        """Apply forward sampling."""
        stddev = torch.as_tensor(stddev)
        batch_shape = tuple(stddev.shape)
        output_ndim = stddev.ndim + 5

        white_shape = (self._n_coefficients, *batch_shape, self._n_coils, *self._psd_factor.shape[:-2])
        white = torch.randn(white_shape, dtype=self._psd_factor.dtype, device=self._psd_factor.device) / math.sqrt(2.0)

        noise = torch.einsum('...ab,b...->a...', self._psd_factor, white)
        spatial_dims = tuple(range(noise.ndim - len(self._dim), noise.ndim))
        noise = torch.fft.ifftn(noise, dim=spatial_dims, norm='ortho')

        crop = [slice(None)] * noise.ndim
        for dim, size in zip(spatial_dims, self._recon_shape, strict=True):
            crop[dim] = slice(0, size)
        noise = noise[tuple(crop)]

        noise = noise.movedim(0, mr2.utils.normalize_index(noise.ndim, self._subspace_dim))
        scale = mr2.utils.unsqueeze_at(stddev, mr2.utils.normalize_index(output_ndim, self._subspace_dim), 1)
        scale = mr2.utils.unsqueeze_right(scale, output_ndim - scale.ndim)
        return scale * noise
