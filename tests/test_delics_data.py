"""Tests for DELICS radial k-space datasets."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pinqi_mrf.data import DELICS_CASES, DelicsRadialDataset, SupervisedDataModule


def _write_delics_radial(path: Path, case_id: str = 'case001', valid: np.ndarray | None = None) -> None:
    if valid is None:
        valid = np.array([True, False])
    n_slices = len(valid)
    kspace = np.ones((n_slices, 3, 4, 1, 1, 5), dtype=np.complex64)
    mask = np.ones((n_slices, 6, 7), dtype=bool)
    csm_initial = np.ones((n_slices, 4, 1, 6, 7), dtype=np.complex64)
    reference_m0 = np.full((n_slices, 6, 7), 1.0 + 0.0j, dtype=np.complex64)
    reference_t1 = np.full((n_slices, 6, 7), 1.2, dtype=np.float32)
    reference_t2 = np.full((n_slices, 6, 7), 0.08, dtype=np.float32)
    with h5py.File(path, 'w') as handle:
        handle.attrs['case'] = case_id
        handle.create_dataset('kspace', data=kspace)
        handle.create_dataset('mask', data=mask)
        handle.create_dataset('csm_initial', data=csm_initial)
        handle.create_dataset('valid', data=valid)
        handle.create_dataset('reference_m0', data=reference_m0)
        handle.create_dataset('reference_t1', data=reference_t1)
        handle.create_dataset('reference_t2', data=reference_t2)


def test_delics_radial_dataset_noise_is_deterministic(tmp_path: Path) -> None:
    """Configured DELICS noise stays fixed for the same case and slice."""
    _write_delics_radial(tmp_path / 'delics_radial_case001.h5')

    dataset = DelicsRadialDataset(tmp_path)
    sample_a = dataset[0]
    sample_b = dataset[0]
    reference = torch.full((3, 4, 1, 1, 5), 1.0 + 0.0j, dtype=torch.complex64)

    torch.testing.assert_close(sample_a['kspace'], sample_b['kspace'])
    assert (sample_a['kspace'] - reference).abs().amax() > 0


def test_delics_radial_reference_dataset_returns_reference_fields(tmp_path: Path) -> None:
    """Validation samples expose reference relaxation maps explicitly."""
    _write_delics_radial(tmp_path / 'delics_radial_case001.h5')

    dataset = DelicsRadialDataset(tmp_path, delics_noise_std=0.0)
    sample = dataset[0]

    assert set(sample) == {
        'case_id',
        'slice_index',
        'kspace',
        'mask',
        'm0',
        't1',
        't2',
    }
    assert sample['m0'].shape == (6, 7)
    assert sample['t1'].shape == (6, 7)
    assert sample['t2'].shape == (6, 7)


def test_delics_radial_dataset_trims_subject_slices(tmp_path: Path) -> None:
    """Configured slice trimming removes low and high slice indices per subject."""
    _write_delics_radial(tmp_path / 'delics_radial_case001.h5', valid=np.ones(8, dtype=bool))

    dataset = DelicsRadialDataset(tmp_path, slice_trim=(2, 1), delics_noise_std=0.0)

    assert [dataset[index]['slice_index'].item() for index in range(len(dataset))] == [2, 3, 4, 5, 6]


@pytest.mark.parametrize('held_out', DELICS_CASES)
@pytest.mark.filterwarnings('ignore:ComplexHalf support is experimental:UserWarning')
def test_delics_leave_one_out_split(tmp_path: Path, held_out: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every fold trains on nine subjects, evaluates the tenth and rejects cross-fold resume."""
    frames = tmp_path / 'frames'
    frames.mkdir()
    for case in DELICS_CASES:
        _write_delics_radial(tmp_path / f'delics_radial_{case}.h5', case_id=case)
        with h5py.File(frames / f'{case}-t3-c4.h5', 'w') as cache:
            group = cache.create_group('z000')
            packed = np.zeros((3, 4, 6, 7), dtype=[('r', '<f2'), ('i', '<f2')])
            group['frames'] = packed
            for name in ('m0', 't1', 't2'):
                group[name] = np.ones((6, 7), dtype=np.complex64 if name == 'm0' else np.float32)
            group['mask'] = np.ones((6, 7), dtype=bool)

    data = SupervisedDataModule(
        tmp_path,
        frames,
        held_out_case=held_out,
        batch_size=1,
        num_workers=0,
        n_timepoints=3,
        n_coils=4,
    )
    original_file = h5py.File

    def training_file(path: Path, *args: object, **kwargs: object) -> h5py.File:
        assert Path(path).name != f'delics_radial_{held_out}.h5'
        return original_file(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(h5py, 'File', training_file)
        data.setup('fit')
    assert data.train_dataset is not None
    assert {case for _path, case in data.train_dataset.files} == set(DELICS_CASES) - {held_out}
    assert data.val_dataloader() == []
    train_batch = next(iter(data.train_dataloader()))
    assert train_batch['frames'].shape == (1, 3, 4, 6, 7)

    data.setup('validate')
    assert data.val_dataset is not None
    assert {case for _path, case in data.val_dataset.files} == {held_out}
    data.load_state_dict(data.state_dict())
    other = next(case for case in DELICS_CASES if case != held_out)
    with pytest.raises(ValueError, match='different DELICS fold'):
        data.load_state_dict({'case_ids': DELICS_CASES, 'held_out_case': other})
