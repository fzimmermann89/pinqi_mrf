"""DELICS reference inputs and leave-one-subject-out training splits."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import h5py
import lightning.pytorch as pl
import mr2
import torch

if TYPE_CHECKING:
    from pinqi_mrf.delics_frames import DelicsFrameDataset

DELICS_CASES = tuple(f'case{index:03d}' for index in range(10))


class ParameterBatch(TypedDict):
    """Reference parameter maps and their tissue mask."""

    m0: torch.Tensor
    t1: torch.Tensor
    t2: torch.Tensor
    mask: torch.Tensor


class DelicsRadialBatch(ParameterBatch):
    """One fixed radial acquisition and its full-acquisition reference maps."""

    case_id: torch.Tensor
    slice_index: torch.Tensor
    kspace: torch.Tensor


class DelicsRadialDataset(torch.utils.data.Dataset[DelicsRadialBatch]):
    """Fixed radial DELICS inputs with subject/slice-seeded complex noise."""

    def __init__(
        self,
        folder: Path,
        *,
        case_glob: str = 'delics_radial_case*.h5',
        case_ids: Sequence[str] | None = None,
        delics_scale: float = 1.0,
        delics_noise_std: float = 4e-3,
        slice_trim: tuple[int, int] = (0, 0),
    ) -> None:
        """Initialize the DELICS radial dataset."""
        self.folder = folder
        self.case_glob = case_glob
        self.delics_scale = delics_scale
        self.delics_noise_std = delics_noise_std
        self.slice_trim = slice_trim
        allowed_case_ids = set(case_ids) if case_ids is not None else None
        self.files: list[tuple[Path, str]] = []
        self.samples: list[tuple[int, int]] = []

        for path in sorted(folder.glob(case_glob)):
            case_id = path.stem.removeprefix('delics_radial_')
            if allowed_case_ids is not None and case_id not in allowed_case_ids:
                continue
            with h5py.File(path, 'r') as handle:
                if str(handle.attrs.get('case', case_id)) != case_id:
                    raise ValueError(f'DELICS subject attribute does not match its filename: {path}')
                file_index = len(self.files)
                self.files.append((path, case_id))
                valid = handle['valid'][()].astype(bool)
                first_slice = slice_trim[0]
                last_slice = len(valid) - slice_trim[1]
                for slice_index, is_valid in enumerate(valid):
                    if first_slice <= slice_index < last_slice and bool(is_valid):
                        self.samples.append((file_index, slice_index))

        if not self.samples:
            raise ValueError(f'No DELICS radial samples found in {folder} matching {case_glob!r}.')

    def __len__(self) -> int:
        """Return the number of available slices."""
        return len(self.samples)

    def __getitem__(self, index: int) -> DelicsRadialBatch:
        """Get one DELICS radial sample with optional reference maps."""
        file_index, slice_index = self.samples[index]
        path, case_id = self.files[file_index]
        with h5py.File(path, 'r') as handle:
            kspace = self.delics_scale * torch.as_tensor(handle['kspace'][slice_index], dtype=torch.complex64)
            if self.delics_noise_std:
                seed = int.from_bytes(
                    hashlib.blake2b(f'{case_id}:{slice_index}'.encode(), digest_size=8).digest(),
                    'little',
                )
                rng = mr2.utils.RandomGenerator(seed)
                noise_real = rng.randn_tensor(kspace.shape, torch.float32)
                noise_imag = rng.randn_tensor(kspace.shape, torch.float32)
                noise = (self.delics_noise_std / math.sqrt(2.0)) * torch.complex(noise_real, noise_imag)
                kspace = kspace + noise.to(torch.complex64)
            return {
                'case_id': torch.tensor(file_index, dtype=torch.int64),
                'slice_index': torch.tensor(slice_index, dtype=torch.int64),
                'kspace': kspace,
                'mask': torch.as_tensor(handle['mask'][slice_index], dtype=torch.bool),
                'm0': torch.as_tensor(handle['reference_m0'][slice_index], dtype=torch.complex64),
                't1': torch.as_tensor(handle['reference_t1'][slice_index], dtype=torch.float32),
                't2': torch.as_tensor(handle['reference_t2'][slice_index], dtype=torch.float32),
            }


class SupervisedDataModule(pl.LightningDataModule):
    """Train on the other subjects and evaluate exactly one held-out DELICS subject."""

    def __init__(
        self,
        folder: Path,
        frame_folder: Path,
        held_out_case: str,
        case_ids: Sequence[str] = DELICS_CASES,
        batch_size: int = 8,  # noqa: ARG002
        val_batch_size: int = 1,  # noqa: ARG002
        num_workers: int = 2,  # noqa: ARG002
        size: int = 256,  # noqa: ARG002
        n_timepoints: int = 352,  # noqa: ARG002
        n_coils: int = 8,  # noqa: ARG002
        noise: tuple[float, float] = (0.002, 0.006),  # noqa: ARG002
    ) -> None:
        """Record the cohort and derive a disjoint leave-one-subject-out fold."""
        super().__init__()
        if len(case_ids) < 2 or len(set(case_ids)) != len(case_ids):
            raise ValueError('The cohort must contain at least two distinct subjects.')
        if held_out_case not in case_ids:
            raise ValueError('The held-out subject must belong to the configured cohort.')
        self.save_hyperparameters()
        self.folder = folder
        self.frame_folder = frame_folder
        self.case_ids = tuple(case_ids)
        self.held_out_case = held_out_case
        self.train_case_ids = tuple(case for case in case_ids if case != held_out_case)
        self.train_dataset: DelicsFrameDataset | None = None
        self.val_dataset: DelicsRadialDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Read training caches for fit, and the held-out radial cache for validation."""
        from pinqi_mrf.delics_frames import DelicsFrameDataset

        if stage in (None, 'fit'):
            reference = self._reference_dataset(self.train_case_ids)
            self.train_dataset = DelicsFrameDataset(
                reference,
                self.frame_folder,
                n_timepoints=self.hparams.n_timepoints,
                n_coils=self.hparams.n_coils,
                noise=self.hparams.noise,
            )
        if stage in (None, 'validate'):
            self.val_dataset = self._reference_dataset((self.held_out_case,))

    def _reference_dataset(self, cases: Sequence[str]) -> DelicsRadialDataset:
        """Require every requested subject and reject aliased subject files."""
        dataset = DelicsRadialDataset(self.folder, case_ids=cases)
        if sorted(case for _path, case in dataset.files) != sorted(cases):
            raise ValueError('A requested DELICS subject is missing from the radial cache.')
        resolved = [path.resolve() for path, _case in dataset.files]
        if len(set(resolved)) != len(resolved):
            raise ValueError('DELICS subjects resolve to shared files.')
        return dataset

    def state_dict(self) -> dict[str, object]:
        """Keep the fold identity in resumable checkpoints."""
        return {'case_ids': self.case_ids, 'held_out_case': self.held_out_case}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Reject a checkpoint from another fold or cohort."""
        if state_dict != self.state_dict():
            raise ValueError('The checkpoint belongs to a different DELICS fold or cohort.')

    def _loader(self, dataset: torch.utils.data.Dataset, *, training: bool) -> torch.utils.data.DataLoader:
        """Spawn CPU workers without inheriting CUDA or HDF5 handles."""
        workers = self.hparams.num_workers
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size if training else self.hparams.val_batch_size,
            shuffle=training,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            multiprocessing_context='spawn' if workers else None,
            persistent_workers=workers > 0,
            prefetch_factor=1 if workers else None,
            timeout=120 if workers else 0,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Load augmented full frames from the nine training subjects."""
        if self.train_dataset is None:
            raise RuntimeError('Set up the training fold before requesting its loader.')
        return self._loader(self.train_dataset, training=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader | list[torch.utils.data.DataLoader]:
        """Expose the held-out subject only after standalone validation setup."""
        if self.val_dataset is None:
            return []
        return self._loader(self.val_dataset, training=False)
