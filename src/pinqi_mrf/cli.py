"""Command-line entry point for PINQI-MRF experiments."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NCCL_P2P_DISABLE'] = '1'

import lightning.pytorch as pl
import matplotlib
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers import WandbLogger

from pinqi_mrf.data import DataModule, DelicsRadialDataModule
from pinqi_mrf.finetune import FineTuneModule
from pinqi_mrf.pretrain import PretrainModule
from pinqi_mrf.train import LogLambdasCallback

matplotlib.use('Agg')
torch.set_num_threads(4)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PinqiLightningCLI(LightningCLI):
    """LightningCLI with PINQI-MRF experiment defaults."""

    command_classes: ClassVar[dict[str, tuple[type[pl.LightningModule], type[pl.LightningDataModule]]]] = {
        'pretrain': (PretrainModule, DataModule),
        'finetune': (FineTuneModule, DelicsRadialDataModule),
    }
    fit_skip: ClassVar[set[str]] = {'model', 'train_dataloaders', 'val_dataloaders', 'datamodule'}

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        """Define PINQI-MRF training-regime subcommands."""
        return dict.fromkeys(PinqiLightningCLI.command_classes, PinqiLightningCLI.fit_skip)

    def _prepare_subcommand_parser(
        self,
        klass: type,
        subcommand: str,
        **kwargs: object,
    ) -> LightningArgumentParser:
        """Prepare regime-specific model and data arguments for custom subcommands."""
        model_class, datamodule_class = self.command_classes[subcommand]
        original_model_class, self._model_class = self._model_class, model_class
        original_datamodule_class, self._datamodule_class = self._datamodule_class, datamodule_class
        try:
            parser = super()._prepare_subcommand_parser(klass, subcommand, **kwargs)
        finally:
            self._model_class = original_model_class
            self._datamodule_class = original_datamodule_class
        if subcommand == 'pretrain':
            parser.link_arguments('data.n_coils', 'model.n_coils', apply_on='instantiate')
            parser.link_arguments('data.size', 'model.image_size', apply_on='instantiate')
        return parser

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add run-level arguments and default model/data config."""
        parser.add_argument('--project', default='pinqi-mrf', help='W&B project name.')
        parser.add_argument(
            '--checkpoint_dir',
            type=Path,
            default=Path('checkpoints'),
            help='Base checkpoint directory.',
        )
        parser.add_argument('--run_name', type=str, default=None, help='Optional W&B run name.')
        parser.add_argument(
            '--checkpoint_name',
            type=str,
            default=None,
            help='Checkpoint directory name. Defaults to a generated run id.',
        )

    def before_instantiate_classes(self) -> None:
        """Configure logging and checkpointing before Trainer instantiation."""
        if self.subcommand not in {'pretrain', 'finetune'}:
            return

        config = getattr(self.config, self.subcommand)
        run_id = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        checkpoint_name = config.checkpoint_name or config.run_name or run_id
        run_dir = Path(config.checkpoint_dir) / checkpoint_name
        run_dir.mkdir(parents=True, exist_ok=True)

        config.trainer.default_root_dir = str(run_dir)
        logger_kwargs: dict[str, Any] = {
            'id': run_id,
            'project': config.project,
            'log_model': False,
            'save_dir': 'logs',
        }
        if config.run_name is not None:
            logger_kwargs['name'] = config.run_name
        config.trainer.logger = WandbLogger(**logger_kwargs)
        checkpoint_callback: ModelCheckpoint
        if self.subcommand == 'pretrain':
            checkpoint_callback = ModelCheckpoint(
                monitor='val/loss',
                mode='min',
                save_top_k=2,
                dirpath=str(run_dir),
                filename='{epoch:02d}-{val/loss:.4f}',
                save_last=True,
                save_on_train_epoch_end=False,
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                save_top_k=0,
                dirpath=str(run_dir),
                filename='{epoch:02d}',
                save_last=True,
                save_on_train_epoch_end=True,
            )
        callbacks = getattr(config.trainer, 'callbacks', None)
        if callbacks is None:
            callbacks = []
        elif not isinstance(callbacks, list):
            callbacks = [callbacks]
        config.trainer.callbacks = [
            *callbacks,
            LearningRateMonitor(logging_interval='step'),
            checkpoint_callback,
            LogLambdasCallback(),
        ]


class PinqiTrainer(pl.Trainer):
    """Trainer with named PINQI-MRF training-regime commands."""

    def pretrain(
        self,
        model: pl.LightningModule,
        train_dataloaders: object | None = None,
        val_dataloaders: object | None = None,
        datamodule: pl.LightningDataModule | None = None,
        ckpt_path: str | Path | None = None,
    ) -> None:
        """Run BrainWeb pretraining."""
        self.fit(
            model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

    def finetune(
        self,
        model: pl.LightningModule,
        train_dataloaders: object | None = None,
        val_dataloaders: object | None = None,
        datamodule: pl.LightningDataModule | None = None,
        ckpt_path: str | Path | None = None,
    ) -> None:
        """Run DELICS radial k-space fine-tuning."""
        self.fit(
            model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )


def main() -> None:
    """Run the CLI."""
    PinqiLightningCLI(
        PretrainModule,
        DataModule,
        trainer_class=PinqiTrainer,
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
