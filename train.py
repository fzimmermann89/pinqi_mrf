"""Training entry point for PINQI-MRF experiments."""

import os

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NCCL_P2P_DISABLE'] = '1'

from argparse import ArgumentParser, Namespace
from pathlib import Path

import lightning.pytorch as pl
import matplotlib
import mr2
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from pinqi_mrf.data import DataModule
from pinqi_mrf.model import LogLambdasCallback, PinqiModule
from pinqi_mrf.signal_models import SubspaceLookup

matplotlib.use('Agg')
torch.set_num_threads(1)


def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument('--project', default='pinqi-mrf')
    parser.add_argument('--data-folder', type=Path, default=mr2.phantoms.brainweb.CACHE_DIR_BRAINWEB)
    parser.add_argument('--checkpoints-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--max-epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--size', type=int, default=192)
    parser.add_argument('--n-readout-samples', type=int, default=192)
    parser.add_argument('--n-coils', type=int, default=1)
    parser.add_argument('--n-subspace-coefficients', type=int, default=10)
    parser.add_argument('--trajectory', choices=['radial'], default='radial')
    parser.add_argument('--r1-steps', type=int, default=100)
    parser.add_argument('--r2-steps', type=int, default=100)
    parser.add_argument('--devices', default='auto')
    parser.add_argument('--accelerator', default='auto')
    parser.add_argument('--strategy', default='auto')
    return parser.parse_args()


def main() -> None:
    """Run training."""
    args = parse_args()
    torch.set_float32_matmul_precision('high')

    if not args.data_folder.exists():
        args.data_folder.mkdir(parents=True, exist_ok=True)
        mr2.phantoms.brainweb.download_brainweb(output_directory=args.data_folder, workers=2, progress=True)

    signalmodel = SubspaceLookup(
        signalmodel=mr2.operators.models.NeuroMRF(),
        n_subspace_coefficients=args.n_subspace_coefficients,
        parameter_ranges=((-3.5, 4.0, args.r1_steps), (-2.5, 4.0, args.r2_steps)),
    )
    datamodule = DataModule(
        folder=args.data_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        size=args.size,
        n_coils=args.n_coils,
    )
    model = PinqiModule(
        signalmodel=signalmodel,
        n_subspace_coefficients=args.n_subspace_coefficients,
        image_size=args.size,
        n_readout_samples=args.n_readout_samples,
        n_coils=args.n_coils,
        trajectory=args.trajectory,
    )

    wandb_logger = WandbLogger(log_model=False, project=args.project)
    run_name = str(wandb_logger.experiment.name)
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=2,
        dirpath=args.checkpoints_dir / run_name,
        filename='{epoch:02d}-{val/loss:.4f}',
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            checkpoint_callback,
            LogLambdasCallback(),
        ],
        log_every_n_steps=10,
        gradient_clip_algorithm='norm',
        gradient_clip_val=5.0,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
