"""Tests for DELICS radial k-space preparation."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pinqi_mrf.prepare_delics_kspace import ConversionConfig, compress_coils, convert_all


def _write_source(path: Path, *, shape: tuple[int, ...] = (2, 3, 9, 1, 8, 8), singleton: int = 1) -> None:
    rng = np.random.default_rng(0)
    actual_shape = (*shape[:3], singleton, *shape[4:])
    frames = rng.normal(size=actual_shape) + 1j * rng.normal(size=actual_shape)
    frames = frames.astype(np.complex64)
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('slices', data=frames)
        handle.create_dataset('coil_compression', data=np.eye(shape[2], dtype=np.complex64))
        handle.attrs['case'] = 'case999'
        handle.attrs['n_components'] = 10


def test_compress_coils() -> None:
    """Compress one slice to eight orthonormal virtual coils."""
    frames = torch.randn(3, 9, 1, 4, 5, dtype=torch.complex64)

    compressed, compression = compress_coils(frames)

    assert compressed.shape == (3, 8, 1, 4, 5)
    assert compression.shape == (8, 9)
    torch.testing.assert_close(
        compression @ compression.conj().mT,
        torch.eye(8, dtype=torch.complex64),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason='DELICS k-space conversion assumes CUDA.')
def test_convert_delics_h5_to_radial_kspace(tmp_path: Path) -> None:
    """Convert a tiny synthetic DELICS H5 into the planned radial k-space schema."""
    input_root = tmp_path / 'input'
    output_root = tmp_path / 'output'
    input_root.mkdir()
    source_path = input_root / 'reconstructions_rovir_case999.h5'
    _write_source(source_path)

    outputs = convert_all(
        ConversionConfig(
            input_root=input_root,
            output_root=output_root,
            case_glob='*.h5',
            n_readout_samples=9,
        )
    )

    assert outputs == [output_root / 'delics_radial_case999.h5']
    with h5py.File(outputs[0], 'r') as handle:
        assert handle['kspace'].shape == (2, 3, 8, 1, 1, 9)
        assert handle['kspace'].dtype == np.complex64
        assert handle['mask'].shape == (2, 8, 8)
        assert handle['mask'].dtype == np.bool_
        assert handle['valid'].shape == (2,)
        assert handle['dcf'].shape == (3, 1, 1, 1, 9)
        assert handle['trajectory/kx'].shape == (3, 1, 1, 1, 9)
        assert handle['trajectory/ky'].shape == (3, 1, 1, 1, 9)
        assert handle['trajectory/kz'].shape == (1, 1, 1, 1, 1)
        assert 'coil_compression' not in handle
        assert 'source/coil_compression' not in handle
        assert handle['csm_initial'].shape == (2, 8, 1, 8, 8)
        assert handle['csm_initial'].dtype == np.complex64
        assert handle['csm_valid'].shape == (2,)
        assert handle['reference_m0'].shape == (2, 8, 8)
        assert handle['reference_t1'].shape == (2, 8, 8)
        assert handle['reference_t2'].shape == (2, 8, 8)
        assert handle['reference_m0'].dtype == np.complex64
        assert handle['reference_t1'].dtype == np.float32
        assert handle['reference_t2'].dtype == np.float32
        assert handle.attrs['case'] == 'case999'
        assert handle.attrs['n_coils'] == 8
        assert handle.attrs['n_source_coils'] == 9
        assert handle.attrs['schema_version'] == 1
        assert handle.attrs['source_attr_n_components'] == 10
        assert np.isfinite(handle['kspace'][()]).all()


def test_convert_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    """Existing output files are protected unless overwrite is set."""
    input_root = tmp_path / 'input'
    output_root = tmp_path / 'output'
    input_root.mkdir()
    output_root.mkdir()
    _write_source(input_root / 'reconstructions_rovir_case999.h5')
    (output_root / 'delics_radial_case999.h5').write_bytes(b'existing')

    with pytest.raises(FileExistsError):
        convert_all(
            ConversionConfig(
                input_root=input_root,
                output_root=output_root,
                case_glob='*.h5',
                n_readout_samples=9,
            )
        )


def test_invalid_singleton_dimension_raises(tmp_path: Path) -> None:
    """Input /slices must have singleton dimension one."""
    input_root = tmp_path / 'input'
    output_root = tmp_path / 'output'
    input_root.mkdir()
    _write_source(input_root / 'reconstructions_rovir_case999.h5', singleton=2)

    with pytest.raises(ValueError, match='singleton dimension'):
        convert_all(
            ConversionConfig(
                input_root=input_root,
                output_root=output_root,
                case_glob='*.h5',
                n_readout_samples=9,
            )
        )
