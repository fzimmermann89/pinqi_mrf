"""Train the five learned BrainWeb saturation-recovery methods.

Example: python train.py pinqi-dictionary --brainweb /path/to/brainweb --output runs/pinqi-dictionary
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import torch
from data import BrainWebData, Sample
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from models import PINQI, T1_BOUNDS, MoDL, masked_ssim, metrics, normalize

METHODS = ('modl', 'modl-dictionary', 'modl-lbfgs', 'pinqi-dictionary', 'pinqi-lbfgs')


class Experiment(pl.LightningModule):
    """Selected L1/SSIM recipe; fitting enters training only in the end-to-end methods."""

    def __init__(self, method: str, learning_rate: float | None = None) -> None:
        """Initialize the fixed BrainWeb experiment settings."""
        super().__init__()
        if method not in METHODS:
            raise ValueError(f'Unknown method: {method}')
        self.save_hyperparameters()
        self.method = method
        self.learning_rate = learning_rate if learning_rate is not None else (8e-4 if method == 'modl' else 4e-4)
        fitter: Literal['dictionary', 'lbfgs'] | None = (
            'dictionary' if method.endswith('-dictionary') else 'lbfgs' if method.endswith('-lbfgs') else None
        )
        if method.startswith('pinqi'):
            assert fitter is not None
            self.model: MoDL | PINQI = PINQI(fitter)
        else:
            self.model = MoDL(fitter)

    def parameter_loss(self, m0: torch.Tensor, t1: torch.Tensor, sample: Sample) -> torch.Tensor:
        """T1 L1 and complex M0 squared error, with downweighted non-brain head tissue."""
        head = sample['mask']
        weights = torch.where(sample['brain'][head], 1.0, 0.1)
        t1_error = (t1 - sample['t1'].clamp(*T1_BOUNDS)).abs()[head]
        m0_error = (m0 - sample['m0']).abs().square()
        brain_loss = (weights * (t1_error + 0.3 * m0_error[head])).sum() / weights.sum().clamp_min(1)
        background_loss = 0.1 * m0_error[~head].mean()
        return brain_loss + background_loss

    def loss(self, sample: Sample) -> torch.Tensor:
        """Image L1 + magnitude SSIM, or parameter L1 + SSIM + auxiliary image MSE."""
        images, estimates = self.model(sample)
        error = images - sample['target_images']
        if self.method == 'modl':
            mask = sample['mask'].unsqueeze(0).expand(images.shape[0], *sample['mask'].shape).flatten(0, 1)
            ssim = masked_ssim(images.abs().flatten(0, 1), sample['target_images'].abs().flatten(0, 1), mask).nanmean()
            return error.abs().mean() + 0.1 * (1 - ssim)
        losses = [self.parameter_loss(m0, t1, sample) for m0, t1 in estimates]
        parameter_loss = sum(0.05 * loss for loss in losses[:-1]) + losses[-1]
        scores = masked_ssim(estimates[-1][1], sample['t1'], sample['brain'])
        ssim_loss = torch.nan_to_num(1 - scores, nan=0.0).sum() / scores.isfinite().sum().clamp_min(1)
        return parameter_loss + 0.1 * error.abs().square().mean() + 0.1 * ssim_loss

    def training_step(self, batch: Sample, _batch_idx: int) -> torch.Tensor:
        """Train on a fresh simulated batch."""
        loss = self.loss(normalize(batch))
        self.log('train/loss', loss, prog_bar=True, batch_size=len(batch['index']))
        return loss

    def validation_step(self, batch: Sample, _batch_idx: int) -> None:
        """Select checkpoints by mean per-slice brain T1 MAE."""
        sample = normalize(batch)
        m0, t1 = self.model.estimate(sample)
        for name, value in metrics(m0, t1, sample).items():
            self.log(f'val/{name}', value.nanmean(), batch_size=len(batch['index']))

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """AdamW with a 20% warmup and cosine decay over the complete training run."""
        networks = [p for name, p in self.model.named_parameters() if name != 'raw_weights']
        optimizer = torch.optim.AdamW(
            [
                {'params': networks, 'weight_decay': 0.01},
                {'params': [self.model.raw_weights], 'weight_decay': 0.0},
            ]
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.learning_rate, 0.002],
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.2,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=3,
            cycle_momentum=False,
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


def main() -> None:
    """Train, retaining the full final state and the best validation-MAE checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=METHODS)
    parser.add_argument('--brainweb', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--wandb', action='store_true', help='Also log to the pinqi-mrf W&B project.')
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('high')
    data = BrainWebData(args.brainweb, args.batch_size, args.workers)
    model = Experiment(args.method, args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    loggers: list[Logger] = [CSVLogger(args.output, name='csv', version=0)]
    if args.wandb:
        loggers.append(WandbLogger(project='pinqi-mrf', name=args.output.name, save_dir=args.output))
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.epochs,
        default_root_dir=args.output,
        logger=loggers,
        inference_mode=False,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(
                dirpath=args.output / 'checkpoints',
                filename='best-mae',
                monitor='val/t1_mae_s',
                mode='min',
                save_last=True,
            ),
            LearningRateMonitor(logging_interval='step'),
        ],
    )
    trainer.fit(model, data, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
