"""Tests for DELICS radial k-space datasets."""

from pathlib import Path

import h5py
import numpy as np
import torch

from pinqi_mrf.data import DelicsRadialDataModule, DelicsRadialDataset


def _write_delics_radial(path: Path, case_id: str = 'case001') -> None:
    kspace = np.ones((2, 3, 4, 1, 1, 5), dtype=np.complex64)
    mask = np.ones((2, 6, 7), dtype=bool)
    valid = np.array([True, False])
    with h5py.File(path, 'w') as handle:
        handle.attrs['case'] = case_id
        handle.create_dataset('kspace', data=kspace)
        handle.create_dataset('mask', data=mask)
        handle.create_dataset('valid', data=valid)


def test_delics_radial_dataset_returns_training_fields_only(tmp_path: Path) -> None:
    """Default samples contain only real-data-compatible training fields."""
    _write_delics_radial(tmp_path / 'delics_radial_case001.h5')

    dataset = DelicsRadialDataset(tmp_path)
    sample = dataset[0]

    assert set(sample) == {'case_id', 'slice_index', 'kspace', 'mask'}
    assert sample['case_id'] == 'case001'
    assert sample['slice_index'].item() == 0
    assert sample['kspace'].shape == (3, 4, 1, 1, 5)
    assert sample['kspace'].dtype == torch.complex64
    assert sample['mask'].shape == (6, 7)
    assert len(dataset) == 1


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
