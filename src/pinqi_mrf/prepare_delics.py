# %%
import argparse
import sys
from pathlib import Path

import h5py
import mrpro
import numpy as np
import scipy.ndimage as ndimage
import torch
from einops import rearrange
from scipy.io import loadmat
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('case', type=str)
parser.add_argument('--input_path', type=Path, default=Path('/scratch/zimmer08/delics/data/'))
parser.add_argument('--output_path', type=Path, default=Path('/scratch/zimmer08/delics'))
parser.add_argument('--lambda_perp', type=float, default=0.15)
parser.add_argument('--lambda_mask', type=float, default=0.03)
parser.add_argument('--lambda_global', type=float, default=1e-6)
parser.add_argument('--lambda_csm', type=float, default=0.005)
parser.add_argument('--batchsize', type=int, default=10)
parser.add_argument('--n_compressed_coils', type=int, default=12)
parser.add_argument('--n_components', type=int, default=10)
parser.add_argument('--n_iterations', type=int, default=40)

# check if running in vscode code cell/jupyter or in cli
if 'ipykernel' in sys.modules:  # running in vscode code cell/jupyter
    args = parser.parse_args(['case000'])
else:  # running in cli
    args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path
lambda_perp = args.lambda_perp
lambda_mask = args.lambda_mask
lambda_global = args.lambda_global
batchsize = args.batchsize
n_compressed_coils = args.n_compressed_coils
n_components = args.n_components
case = args.case
n_iterations = args.n_iterations
lambda_csm = args.lambda_csm
slice_width = 5
slice_profile = torch.tensor([0, 0.9, 0.97, 1, 0.97, 0.9, 0])


# %%
def load(
    path: Path,
    case: str = 'case000',
):
    matrix = mrpro.data.SpatialDimension(256, 256, 256)

    noise = torch.as_tensor(np.load(path / f'training/{case}/noise.npy'), dtype=torch.complex64)
    noise = rearrange(noise, 'coil k0 -> 1 coil 1 1 k0')
    knoise = mrpro.data.KNoise(noise)

    f = loadmat(path / 'shared/traj_grp48_inacc1.mat')  # %%

    dcf = torch.as_tensor(rearrange(f['DCF'], 'k0 repeat tr ->  1 1  repeat tr k0'))
    dcf_data = mrpro.data.DcfData(dcf).rearrange('... coil k2 k1 k0-> (k1 ...) coil 1 k2 k0 ')

    traj = torch.as_tensor(rearrange(f['k_3d'], 'k0 xyz repeat tr -> xyz  1 1 repeat tr  k0'))
    ktraj = mrpro.data.KTrajectory.from_tensor(traj, scaling_matrix=matrix, axes_order='xyz')

    mrf = np.load(path / f'training/{case}/raw_mrf.npy')
    k = torch.as_tensor(
        rearrange(mrf[: traj.shape[-1]], 'k0 coils (repeat tr ) -> 1 coils repeat tr k0', repeat=traj.shape[-3]),
        dtype=torch.complex64,
    )
    kdata = mrpro.data.KData(
        data=k,
        traj=ktraj,
        header=mrpro.data.KHeader(
            recon_matrix=matrix,
            encoding_matrix=matrix,
            recon_fov=matrix,
            encoding_fov=matrix,
        ),
    )
    kdata = kdata.rearrange('... coil k2 k1 k0-> (k1 ...) coil 1 k2 k0 ')

    return kdata, dcf_data, knoise


print(f'Loading data ({input_path!s}, {case}) ...')
kdata, dcf_data, knoise = load(input_path, case)
# %%
print('Whitening & Compressing coils...')
kdata = kdata.prewhiten(knoise)


# kdata, coilcompression = kdata.compress_coils(n_compressed_coils, return_compression_op=True, rotate=True)
def get_rovir(input_path: Path, case: str, n_compressed_coils: int) -> mrpro.operators.EinsumOp:
    def load_gre(gre_path: Path | str, noise_path: Path | str) -> torch.Tensor:
        noise = torch.as_tensor(np.load(noise_path), dtype=torch.complex64)

        raw = torch.as_tensor(
            rearrange(np.load(gre_path), 'k0 coils (k1 i s) ->coils k0 k1 (s i) ', k1=64, i=2), dtype=torch.complex64
        )
        fft_xy = mrpro.operators.FastFourierOp(dim=(-2, -3))
        (img,) = fft_xy(raw)
        cov = noise @ noise.conj().mT / noise.shape[-1]
        eye = torch.eye(len(raw), dtype=cov.dtype, device=cov.device)
        scale = cov.diagonal().real.mean().clamp_min(1e-12)
        cov = cov + 1e-6 * scale * eye
        L = torch.linalg.cholesky(cov)
        flat = img.flatten(1)
        flat = torch.linalg.solve_triangular(L, flat, upper=False)
        return flat.reshape(img.shape)

    def get_roi(
        img: torch.Tensor, *, target_fov_mm: float = 256.0, gre_fov_mm: float = 440.0, threshold: float = 0.15
    ) -> torch.Tensor:
        rss = torch.linalg.norm(img, dim=0)
        nz, ny, nx = rss.shape
        vol = rss.cpu().numpy().astype(np.float32)
        vol /= vol.max() + 1e-12
        mask = ndimage.gaussian_filter(vol, sigma=1) > threshold
        mask = ndimage.binary_fill_holes(mask)
        area_z = mask.sum(axis=(1, 2)).astype(np.float32)
        z_start = np.argmax(area_z > 0.3 * area_z.max())
        head_part = vol[z_start : z_start + 28]
        axial_proj = head_part.max(axis=0)
        axial_proj /= axial_proj.max() + 1e-12
        mask_2d = ndimage.gaussian_filter(axial_proj, sigma=1) > threshold
        mask_2d = ndimage.binary_fill_holes(mask_2d)
        mask_2d = ndimage.binary_opening(mask_2d, iterations=1)
        mask_2d = ndimage.binary_closing(mask_2d, iterations=2)
        labels, n_labels = ndimage.label(mask_2d)
        if n_labels > 2:
            mask_2d = labels == np.bincount(labels.ravel())[1:].argmax() + 1
        coords = np.argwhere(mask_2d)
        cy, cx = coords.mean(axis=0).round().astype(int) if len(coords) else (ny // 2, nx // 2)
        side = max(8, min(int(round(target_fov_mm / gre_fov_mm * min(ny, nx))), ny, nx))
        roi = torch.zeros((nz, ny, nx), dtype=torch.bool, device=img.device)
        roi[z_start : z_start + 30, cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2] = True
        return roi

    def rovir(
        img: torch.Tensor, roi: torch.Tensor, *, n_compressed_coils: int, reg: float = 1e-6
    ) -> mrpro.operators.EinsumOp:
        """Create an orthonormal ROVir coil compression operator.

        The returned operator applies compression along the coil dimension of data
        shaped `(..., coil, z, y, x)`.

        Parameters
        ----------
        img
            Prewhitened coil images with shape `(coil, z, y, x)`.
        roi
            Boolean mask with shape `(z, y, x)`. Voxels inside the mask define the
            region to preserve.
        n_compressed_coils
            Number of virtual coils.
        reg
            Relative diagonal loading for the background covariance.

        Returns
        -------
            Operator mapping `(..., coil, z, y, x)` to `(..., compressed_coil, z, y, x)`.
        """
        if img.ndim != 4:
            raise ValueError(f'img must have shape (coil, z, y, x), got {tuple(img.shape)}')

        n_coils = img.shape[0]

        if roi.shape != img.shape[1:]:
            raise ValueError(f'roi must have shape {tuple(img.shape[1:])}, got {tuple(roi.shape)}')

        if not 1 <= n_compressed_coils <= n_coils:
            raise ValueError(f'n_compressed_coils must be in [1, {n_coils}], got {n_compressed_coils}')

        data = rearrange(img, 'coil ... -> (...) coil')
        roi_mask = roi.to(device=img.device, dtype=torch.bool).flatten()

        if not roi_mask.any():
            raise ValueError('roi does not contain any voxels')

        if roi_mask.all():
            raise ValueError('roi covers the full image; ROVir needs background voxels')

        roi_data = data[roi_mask]
        background_data = data[~roi_mask]

        roi_covariance = roi_data.conj().mT @ roi_data / roi_data.shape[0]
        background_covariance = background_data.conj().mT @ background_data / background_data.shape[0]

        identity = torch.eye(n_coils, dtype=img.dtype, device=img.device)

        # Stabilizes the generalized eigenvalue problem when the background covariance is close to singular.
        background_scale = background_covariance.diagonal().real.mean().clamp_min(1e-10)
        background_covariance = background_covariance + reg * background_scale * identity

        background_cholesky = torch.linalg.cholesky(background_covariance)
        background_whitener = torch.linalg.solve_triangular(background_cholesky, identity, upper=False)

        whitened_roi_covariance = background_whitener @ roi_covariance @ background_whitener.conj().mT
        whitened_roi_covariance = 0.5 * (whitened_roi_covariance + whitened_roi_covariance.conj().mT)

        eigenvalues, eigenvectors = torch.linalg.eigh(whitened_roi_covariance)
        component_order = torch.argsort(eigenvalues.real, descending=True)

        rovir_vectors = background_whitener.conj().mT @ eigenvectors[:, component_order[:n_compressed_coils]]
        rovir_vectors = rovir_vectors / (torch.linalg.norm(rovir_vectors, dim=0, keepdim=True) + 1e-12)

        orthonormal_vectors, _ = torch.linalg.qr(rovir_vectors, mode='reduced')
        compression_matrix = orthonormal_vectors.conj().mT.to(img.dtype)

        return mrpro.operators.EinsumOp(
            compression_matrix,
            'compressed_coil coil, ... coil z y x -> ... compressed_coil z y x',
        )

    gre_img = load_gre(input_path / f'training/{case}/raw_gre.npy', input_path / f'training/{case}/noise.npy')
    roi = get_roi(gre_img)
    return rovir(gre_img, roi, n_compressed_coils=n_compressed_coils)


rovir = get_rovir(input_path, case, n_compressed_coils)
kdata.data = rovir(kdata.data)[0]

# %%
print('Creating projection operator...')

shifts = torch.arange(0, kdata.header.recon_matrix.z, slice_width) + slice_width / 2 - kdata.header.recon_matrix.z / 2
proj_op = mrpro.operators.SliceProjectionOp(
    input_shape=kdata.header.recon_matrix,
    slice_shift=shifts,
    slice_profile=mrpro.utils.slice_profiles.SliceInterpolate(
        torch.arange(len(slice_profile)) - len(slice_profile) // 2, slice_profile
    ),
    optimize_for='both',
)
proj_rearrange = mrpro.operators.RearrangeOp('z time coil 1 y x -> time coil z y x')
proj = proj_rearrange @ proj_op
proj = proj.cuda()


# %%
print('Creating dictionary...')


def create_dictionary(n_components: int):
    signal = mrpro.operators.models.NeuroMRF().cuda()
    t1 = torch.linspace(0.2, 4, 100)[:, None].cuda()
    t2 = torch.linspace(0.01, 0.3, 100)[None, :].cuda()
    (d,) = signal(torch.tensor(1.0).cuda(), t1, t2)
    d = d.flatten(1).mT
    pca_op = mrpro.operators.PCACompressionOp(d, n_components, centering=False)
    return d.cpu(), pca_op.cpu()


dictionary, pca = create_dictionary(n_components)


# %%
print('Prepare batching...')
kdata = kdata.rearrange('(n batchsize) ... -> n batchsize ...', batchsize=batchsize)
fourier_ops = [mrpro.operators.FourierOp.from_kdata(k).cuda() for k in kdata]
dcf_ops = [dcf.as_operator().cuda() for dcf in dcf_data.split(0, batchsize)]


# %%
print('Adjoint reconstruction...')


def reco(kdata, dcf_ops, fourier_ops, recon_shape: tuple[int, int, int]):
    rhs = torch.empty(*kdata.shape[:-3], *recon_shape, dtype=torch.complex64)
    for i, (k, dcf, fi) in enumerate(zip(tqdm(kdata.data, leave=False), dcf_ops, fourier_ops)):
        (ri,) = proj(*fi.H(*dcf(k.cuda())))
        rhs[i].copy_(ri, non_blocking=True)
    torch.cuda.synchronize()
    return rhs


rhs = reco(kdata, dcf_ops, fourier_ops, (len(shifts), kdata.header.recon_matrix.y, kdata.header.recon_matrix.x))
del kdata, dcf_data
# %%
print('Estimating CSMs')


def estimate_csm(rhs: torch.Tensor, pca: mrpro.operators.PCACompressionOp) -> mrpro.operators.SensitivityOp:
    csm = torch.empty(1, *rhs.shape[-4:], dtype=torch.complex64)
    for z in tqdm(range(rhs.shape[-3]), leave=False):
        slice_rhs = rhs[..., z, None, :, :].flatten(0, 1).moveaxis(0, -1).to(pca._compression_matrix.device)
        coeff = pca(slice_rhs)[0][..., 0]
        coeff /= coeff.abs().max()
        csm[0, :, z, :, :] = mrpro.algorithms.csm.inati(coeff, smoothing_width=31).squeeze()
    return mrpro.operators.SensitivityOp(csm)


csm = estimate_csm(rhs, pca).cuda()


class CSMPenaltyOp(mrpro.operators.LinearOperator):
    def __init__(self, csm: mrpro.operators.SensitivityOp, lam):
        super().__init__()
        self.csm = csm
        self.lam = lam

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        out = torch.empty_like(x)
        for i, batch in enumerate(tqdm(x, leave=False)):
            batch = batch.to(device='cuda', copy=True)
            batch -= self.csm(*self.csm.adjoint(batch))[0]
            batch *= self.lam
            out[i].copy_(batch, non_blocking=True)
        torch.cuda.synchronize()
        return (out,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return self.forward(x)


# %%
class GramOp(mrpro.operators.LinearOperator):
    def __init__(self, proj, fourier_ops, dcf_ops):
        super().__init__()
        self.proj = proj
        self.fourier_ops = fourier_ops
        self.dcf_ops = dcf_ops

    def forward(self, rhs):
        out = torch.empty_like(rhs)
        for data, fourier, dcf, o in zip(tqdm(rhs, leave=False), self.fourier_ops, self.dcf_ops, out):
            (res,) = self.proj.adjoint(data.cuda())
            (res,) = fourier.adjoint(*dcf(*fourier(res)))
            (res,) = self.proj(res)
            o.copy_(res, non_blocking=True)
        torch.cuda.synchronize()
        return (out,)

    def adjoint(self, x):
        return self.forward(x)


gram = GramOp(proj, fourier_ops, dcf_ops)


# %%
class PerpOp(mrpro.operators.LinearOperator):
    def __init__(self, op, lam):
        super().__init__()
        self.gram_op = op.gram.cuda()
        self.lam = lam

    def forward(self, x):
        res = torch.empty_like(x)
        pattern = 'n batch coil z y x -> z coil y x (n batch)'
        res_view = rearrange(res, pattern)
        x_view = rearrange(x, pattern)
        for r, data in zip(tqdm(res_view, leave=False), x_view):
            data = data.to(device='cuda', copy=True)
            data -= self.gram_op(data)[0]
            data *= self.lam
            r.copy_(data, non_blocking=True)
        torch.cuda.synchronize()
        return (res,)

    def adjoint(self, x):
        return self.forward(x)


# %%
print('Creating mask...')


def create_mask(rhs: torch.Tensor, pca: mrpro.operators.PCACompressionOp) -> torch.Tensor:
    opening_iters = 1
    dilation_iters = 10
    closing_iters = 7
    sigma = 8
    rhs_view = rearrange(rhs, 'n batch coil z y x -> z coil y x (n batch)')
    mask = torch.zeros(rhs.shape[-3:], dtype=torch.float32)
    for z, data in enumerate(tqdm(rhs_view, leave=False)):
        img = pca(data.to(pca._compression_matrix.device))[0].norm(dim=-1).norm(dim=0).cpu().numpy()
        threshold = 1.42 * img[:20, :20].max()
        img = img > threshold
        img = ndimage.binary_fill_holes(img)
        img = ndimage.binary_closing(img, iterations=closing_iters)
        img = ndimage.binary_opening(img, iterations=opening_iters)
        img = ndimage.binary_fill_holes(img)
        labels, n_labels = ndimage.label(img)
        if n_labels > 2:
            img = labels == np.bincount(labels.ravel())[1:].argmax() + 1
        img = ndimage.binary_dilation(img, iterations=dilation_iters)
        img = ndimage.gaussian_filter((~img).astype(np.float32), sigma=sigma)
        mask[z] = torch.as_tensor(img, dtype=torch.float32)
    return mask


mask = create_mask(rhs, pca)
print('mask ratio:', mask.mean())
# %%
import gc

gc.collect()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()


# %%
def power_iteration(operator: mrpro.operators.LinearOperator, x: torch.Tensor, steps: int = 5) -> float:
    """Estimate the spectral radius of a linear operator."""
    for _ in (pbar := tqdm(range(steps), leave=False)):
        (x,) = operator(x)
        radius = torch.linalg.norm(x)
        x /= radius
        pbar.set_postfix({'radius': radius.item()})
    return radius.item()


sigma_max = 0.05  # power_iteration(gram, rhs.clone())
print(f'sigma_max: {sigma_max}')

# %%
from mrpro.algorithms.optimizers.cg import CGStatus


def cg(
    operator,
    right_hand_side: torch.Tensor,
    *,
    max_iterations: int = 128,
    callback=None,
) -> tuple[torch.Tensor]:
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    conjugate = residual.clone()
    residual_dot_residual = torch.vdot(residual.reshape(-1), residual.reshape(-1)).real

    for iteration in range(max_iterations):
        operator_conjugate = operator(conjugate)[0]
        alpha = (
            residual_dot_residual
            / torch.vdot(
                conjugate.reshape(-1),
                operator_conjugate.reshape(-1),
            ).real
        )

        solution.add_(conjugate, alpha=alpha.item())
        residual.sub_(operator_conjugate, alpha=alpha.item())

        residual_dot_residual_new = torch.vdot(
            residual.reshape(-1),
            residual.reshape(-1),
        ).real
        beta = residual_dot_residual_new / residual_dot_residual
        residual_dot_residual = residual_dot_residual_new

        conjugate.mul_(beta).add_(residual)

        if callback is not None:
            continue_iterations = callback(
                CGStatus(
                    solution=(solution,),
                    iteration_number=iteration,
                    residual=(residual,),
                    preconditioned_residual=(residual,),
                )
            )
            if continue_iterations is False:
                break

    return (solution,)


# %%
class NormalOp(mrpro.operators.LinearOperator):
    def __init__(self, gram, perp, csm_perp, tikonov):
        super().__init__()
        self._gram = gram
        self._perp = perp
        self._csm_perp = csm_perp
        self._tikonov = tikonov

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        out = self._gram(x)[0]
        out.add_(self._perp(x)[0])
        out.add_(self._csm_perp(x)[0])
        out.addcmul_(x, self._tikonov)
        return (out,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return self.forward(x)


csm_perp = CSMPenaltyOp(csm, lambda_csm * sigma_max)
perp = PerpOp(pca, lambda_perp * sigma_max)
tikonov = sigma_max * (lambda_mask * mask + lambda_global)
op = NormalOp(gram, perp, csm_perp, tikonov)

print('Running optimization...')
progress = tqdm(total=n_iterations)


def callback(status):
    progress.update(1)
    progress.set_postfix({'residual': status['residual'][0].abs().mean()})
    gc.collect()
    torch.cuda.empty_cache()


with torch.inference_mode():
    (res,) = cg(op, rhs, max_iterations=n_iterations, callback=callback)


# %%

print('Saving results...')
out_file = output_path / f'reconstructions_rovir_{case}.h5'
print(f'Saving to {out_file}')
res_slices = rearrange(res, 'n batch coil z y x -> z (n batch) coil 1 y x')

with h5py.File(out_file, 'w') as f:
    dset = f.create_dataset(
        'slices',
        shape=res_slices.shape,
        dtype=np.complex64,
        chunks=(1, *res_slices.shape[1:]),
        compression='lzf',
    )
    for z, slicedata in enumerate(tqdm(res_slices, leave=True)):
        dset[z] = slicedata.numpy()

    dset_coil_compression = f.create_dataset(
        'coil_compression',
        data=rovir.matrix.cpu().resolve_conj().numpy(),
        dtype=np.complex64,
    )
    f.attrs['case'] = case
    f.attrs['lambda_perp'] = lambda_perp
    f.attrs['lambda_mask'] = lambda_mask
    f.attrs['lambda_global'] = lambda_global
    f.attrs['lambda_csm'] = lambda_csm
    f.attrs['batchsize'] = batchsize
    f.attrs['n_compressed_coils'] = n_compressed_coils
    f.attrs['n_components'] = n_components
    f.attrs['n_iterations'] = n_iterations
    f.attrs['slice_width'] = slice_width
    f.attrs['sigma_max'] = sigma_max
#
