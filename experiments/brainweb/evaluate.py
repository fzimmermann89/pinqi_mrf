"""Evaluate a trained method or the CG baseline on all 120 fixed BrainWeb test slices."""

import argparse
import csv
from pathlib import Path

import lightning.pytorch as pl
import torch
from data import SUBJECTS, BrainWebData
from models import cg_dictionary, metrics, normalize
from train import Experiment


def main() -> None:
    """Save per-slice brain scores and print equally weighted test-set means."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--brainweb', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output CSV path.')
    parser.add_argument('--checkpoint', type=Path, help='Omit for the CG + dictionary baseline.')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    model = Experiment.load_from_checkpoint(args.checkpoint).model.to(device).eval() if args.checkpoint else None
    data = BrainWebData(args.brainweb, batch_size=1, workers=args.workers)
    data.setup('test')
    rows = []
    for subject, loader in zip(SUBJECTS['test'], data.test_dataloader(), strict=True):
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.no_grad():
                sample = normalize(batch)
                m0, t1 = model.estimate(sample) if model is not None else cg_dictionary(sample)
                scores = metrics(m0, t1, sample)
            rows.append(
                {
                    'subject': subject,
                    'slice_index': int(batch['index'][0]),
                    **{name: float(value[0]) for name, value in scores.items()},
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ('t1_mae_s', 't1_rmse_s', 't1_nrmse', 't1_ssim', 'm0_nrmse'):
        mean = torch.tensor([row[name] for row in rows]).nanmean()
        print(f'{name}: {mean:.6f}')  # noqa: T201


if __name__ == '__main__':
    main()
