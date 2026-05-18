"""Tests for DELICS radial k-space datasets."""

from pathlib import Path

import h5py
import numpy as np
import torch

from pinqi_mrf.data import DelicsRadialDataModule, DelicsRadialDataset


def _write_delics_radial(path: Path, case_id: str = 'case001', valid: np.ndarray | None = None) -> None:
    if valid is None:
        valid = np.array([True, False])
    n_slices = len(valid)
    kspace = np.ones((n_slices, 3, 4, 1, 1, 5), dtype=np.complex64)
    mask = np.ones((n_slices, 6, 7), dtype=bool)
    reference_t1 = np.full((n_slices, 6, 7), 1.2, dtype=np.float32)
    reference_t2 = np.full((n_slices, 6, 7), 0.08, dtype=np.float32)
    with h5py.File(path, 'w') as handle:
        handle.attrs['case'] = case_id
        handle.create_dataset('kspace', data=kspace)
        handle.create_dataset('mask', data=mask)
        handle.create_dataset('valid', data=valid)
        handle.create_dataset('reference_t1', data=reference_t1)
        handle.create_dataset('reference_t2', data=reference_t2)


def test_delics_radial_dataset_returns_training_fields_only(tmp_path: Path) -> None:
    """Default samples contain only real-data-compatible training fields."""
    _write_delics_radial(tmp_path / 'delics_radial_case001.h5')

    dataset = DelicsRadialDataset(tmp_path, delics_noise_std=0.0)
    sample = dataset[0]

    assert set(sample) == {'case_id', 'slice_index', 'kspace', 'mask'}
    assert sample['case_id'] == 'case001'
    assert sample['slice_index'].item() == 0
    assert sample['kspace'].shape == (3, 4, 1, 1, 5)
    assert sample['kspace'].dtype == torch.complex64
    torch.testing.assert_close(sample['kspace'], torch.full((3, 4, 1, 1, 5), 250.0 + 0.0j, dtype=torch.complex64))
    assert sample['mask'].shape == (6, 7)
    assert len(dataset) == 1


def test_delics_radial_dataset_noise_is_deterministic(tmp_path: Path) -> None:
    """Configured DELICS noise stays fixed for the same case and slice."""
    _write_delics_radial(tmp_path / 'delics_radial_case001.h5')

    dataset = DelicsRadialDataset(tmp_path)
    sample_a = dataset[0]
    sample_b = dataset[0]
    reference = torch.full((3, 4, 1, 1, 5), 250.0 + 0.0j, dtype=torch.complex64)

    torch.testing.assert_close(sample_a['kspace'], sample_b['kspace'])
    assert (sample_a['kspace'] - reference).abs().amax() > 0


def test_delics_radial_reference_dataset_returns_reference_fields(tmp_path: Path) -> None:
    """Validation samples expose reference relaxation maps explicitly."""
    _write_delics_radial(tmp_path / 'delics_radial_case001.h5')

    dataset = DelicsRadialDataset(tmp_path, include_references=True, delics_noise_std=0.0)
    sample = dataset[0]

    assert set(sample) == {'case_id', 'slice_index', 'kspace', 'mask', 'reference_t1', 'reference_t2'}
    assert sample['reference_t1'].shape == (6, 7)
    assert sample['reference_t2'].shape == (6, 7)


def test_delics_radial_dataset_trims_subject_slices(tmp_path: Path) -> None:
    """Configured slice trimming removes low and high slice indices per subject."""
    _write_delics_radial(tmp_path / 'delics_radial_case001.h5', valid=np.ones(8, dtype=bool))

    dataset = DelicsRadialDataset(tmp_path, slice_trim=(2, 1), delics_noise_std=0.0)

    assert [dataset[index]['slice_index'].item() for index in range(len(dataset))] == [2, 3, 4, 5, 6]


def test_delics_radial_datamodule_filters_case_ids(tmp_path: Path) -> None:
    """Train and validation loaders use explicit subject splits."""
    _write_delics_radial(tmp_path / 'delics_radial_case001.h5', case_id='case001')
    _write_delics_radial(tmp_path / 'delics_radial_case002.h5', case_id='case002')

    datamodule = DelicsRadialDataModule(
        tmp_path,
        train_case_ids=('case001',),
        val_case_ids=('case002',),
        batch_size=1,
        num_workers=0,
    )
    datamodule.setup()

    train_batch = next(iter(datamodule.train_dataloader()))
    val_loader = datamodule.val_dataloader()
    assert val_loader is not None
    val_batch = next(iter(val_loader))

    assert train_batch['case_id'] == ['case001']
    assert val_batch['case_id'] == ['case002']
    assert train_batch['kspace'].shape == (1, 3, 4, 1, 1, 5)
    assert 'reference_t1' not in train_batch
    assert 'reference_t2' not in train_batch
    assert val_batch['reference_t1'].shape == (1, 6, 7)
    assert val_batch['reference_t2'].shape == (1, 6, 7)
