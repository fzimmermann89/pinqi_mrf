"""Command-line entry point for PINQI-MRF experiments."""

import os
import sys
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NCCL_P2P_DISABLE'] = '1'

import matplotlib
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers import WandbLogger

from pinqi_mrf.data import DataModule
from pinqi_mrf.model import LogLambdasCallback, PinqiModule

matplotlib.use('Agg')
torch.set_num_threads(1)


class PinqiLightningCLI(LightningCLI):
    """LightningCLI with PINQI-MRF experiment defaults."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add run-level arguments and default model/data config."""
        parser.add_argument('--project', default='pinqi-mrf', help='W&B project name.')
        parser.add_argument(
            '--checkpoint_dir',
            type=Path,
            default=Path('checkpoints'),
            help='Base checkpoint directory.',
        )
        parser.add_argument('--run_name', type=str, required=True, help='W&B run name and checkpoint directory name.')

    def before_instantiate_classes(self) -> None:
        """Configure logging and checkpointing before Trainer instantiation."""
        if self.subcommand != 'fit':
            return

        config = getattr(self.config, self.subcommand)
        run_dir = Path(config.checkpoint_dir) / config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        config.trainer.default_root_dir = str(run_dir)
        config.trainer.logger = WandbLogger(
            name=config.run_name,
            project=config.project,
            log_model=False,
            save_dir='logs',
        )
        config.trainer.callbacks = [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                monitor='val/loss',
                mode='min',
                save_top_k=2,
                dirpath=str(run_dir),
                filename='{epoch:02d}-{val/loss:.4f}',
                save_last=True,
            ),
            LogLambdasCallback(),
        ]


def main() -> None:
    """Run the CLI."""
    if len(sys.argv) == 1:
        sys.argv.append('--help')
    elif len(sys.argv) > 1 and sys.argv[1].startswith('-'):
        sys.argv.insert(1, 'fit')

    torch.set_float32_matmul_precision('high')
    PinqiLightningCLI(
        PinqiModule,
        DataModule,
        save_config_kwargs={'overwrite': True},
        seed_everything_default=42,
        trainer_defaults={
            'max_epochs': 40,
            'accelerator': 'auto',
            'devices': 'auto',
            'strategy': 'auto',
            'log_every_n_steps': 10,
            'gradient_clip_algorithm': 'norm',
            'gradient_clip_val': 5.0,
        },
    )


if __name__ == '__main__':
    main()
