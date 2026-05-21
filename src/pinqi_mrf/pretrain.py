"""BrainWeb supervised pretraining module for PINQI-MRF."""

from collections.abc import Sequence
from typing import Protocol

import torch

from pinqi_mrf.data import BatchType
from pinqi_mrf.train import BaseModule


class PretrainHParams(Protocol):
    """Typed view of pretraining hyperparameters."""

    n_coils: int
    n_compressed_coils: int | None
    use_ground_truth_csm: bool
    csm_estimation_iterations: int
    csm_estimation_smoothing_width: int
    csm_extrapolate: bool
    use_density_compensation: bool
    weight_scaling: bool
    normalize_inputs: bool
    n_subspace_coefficients: int
    use_image_net: bool
    use_parameter_net: bool
    use_dictionary_match: bool
    lr: float
    weight_decay: float
    loss_weights: Sequence[float]
    loss_l1: float
    loss_ssim: float


class PretrainModule(BaseModule):
    """Lightning module for BrainWeb supervised PINQI-MRF pretraining."""

    hparams: PretrainHParams

    def __init__(self, **kwargs: object) -> None:
        """Initialize the pretraining module."""
        super().__init__(**kwargs)  # type: ignore[arg-type]

    def training_step(self, batch: BatchType, _batch_idx: int) -> torch.Tensor:
        """Run one training step."""
        coil_rhs, csm_ground_truth = self.simulate_coil_rhs(batch)
        csm_initial = csm_ground_truth if self.hparams.use_ground_truth_csm else self.estimate_initial_csm(coil_rhs)
        if self.global_step % 10 == 0:
            self.log_csm_diagnostics(csm_initial, prefix='debug/train/csm')
        _images, parameters, _csm_final = self(coil_rhs, csm_initial=csm_initial)
        loss, components = self.supervised_parameter_loss(parameters, batch)
        self.log('train/loss', loss, on_epoch=True, prog_bar=True)  # on_step is true automatically in training_step
        if self.global_step % 10 == 0:
            self.log_dict(
                {f'debug/train/loss_{key}': value for key, value in components.items()},
                on_epoch=False,
            )
        return loss
