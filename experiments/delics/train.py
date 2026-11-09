"""Train or validate supervised DELICS PINQI with an explicit experiment config."""

import torch
import torch._functorch.config
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from pinqi_mrf.supervised import SupervisedDataModule, SupervisedModule


class DelicsCLI(LightningCLI):
    """Keep shared acquisition dimensions and mixture weights consistent."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Link data settings to reconstruction settings."""
        for source, target in [
            ('brainweb_loss_weight', 'brainweb_loss_weight'),
            ('size', 'image_size'),
            ('n_coils', 'n_coils'),
            ('frame_coils', 'n_compressed_coils'),
            ('frame_timepoints', 'n_timepoints'),
        ]:
            parser.link_arguments(f'data.{source}', f'model.{target}', apply_on='instantiate')


def main() -> None:
    """Run Lightning's fit or validate command, including checkpoint resume."""
    torch.set_float32_matmul_precision('high')
    torch._functorch.config.activation_memory_budget = 0.7
    DelicsCLI(
        SupervisedModule,
        SupervisedDataModule,
        seed_everything_default=42,
        save_config_kwargs={'overwrite': True},
    )


if __name__ == '__main__':
    main()
