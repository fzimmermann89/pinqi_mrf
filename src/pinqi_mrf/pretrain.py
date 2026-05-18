"""BrainWeb supervised pretraining module for PINQI-MRF."""

import torch

from pinqi_mrf.data import BatchType
from pinqi_mrf.train import BaseModule, PretrainHParams


class PretrainModule(BaseModule):
    """Lightning module for BrainWeb supervised PINQI-MRF pretraining."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the pretraining module."""
        super().__init__(**kwargs)
        self.hparams: PretrainHParams  # type: ignore[assignment]

    def training_step(self, batch: BatchType, _batch_idx: int) -> torch.Tensor:
        """Run one training step."""
        coil_rhs, csm_ground_truth = self.simulate_coil_rhs(batch)
        csm_initial = csm_ground_truth if self.hparams.use_ground_truth_csm else self.estimate_initial_csm(coil_rhs)
        if self.global_step % 10 == 0:
            self.log_csm_diagnostics(csm_initial, prefix='debug/train/csm')
        _images, parameters = self(coil_rhs, csm_initial=csm_initial)
        loss, components = self.supervised_parameter_loss(parameters, batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.global_step % 10 == 0:
            self.log_dict(
                {f'debug/train/loss_{key}': value.detach() for key, value in components.items()},
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
        return loss
