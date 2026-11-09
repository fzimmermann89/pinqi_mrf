# Supervised DELICS PINQI

Train PINQI on augmented reconstructed DELICS coil frames and validate on fixed radial inputs. The targets are maps fitted to the full acquisition. These experiments measure agreement with those reference maps; they do not provide an independent ground truth or held-out test cohort.

The scripts reuse the reconstruction, fitting, datasets and metrics in `src/pinqi_mrf`. `train.py` supplies Lightning's `fit` and `validate` commands, and `train.yaml` records the current experiment settings. Cache preparation is separate from training.

## Data

The radial cache must contain the reference maps, masks, valid slices and the `source_file` attribute identifying the reconstructed coil frames. Prepare the first 352 frames with eight compressed coils:

```bash
.venv/bin/python -m pinqi_mrf.prepare_delics_frames \
  --reference-folder data/delics/radial --output data/delics/frames \
  --case-ids case000 case001 case002 case003 case004 case005 case006 case007 case008 case009
```

The cache stores complex-half coil frames, the compression matrix, reference maps and provenance. Each slice is written once; existing subject caches are preserved. Set paths in `train.yaml` or override them on the command line. BrainWeb folders contain disjoint `s??.h5` subjects. Set `--data.brainweb_loss_weight 0` to train on DELICS alone.

The supplied split uses case001–006 and case008–009 for training, and case000/case007 for validation. All valid slices are used. The loader rejects overlapping or missing subjects. Validation inputs and BrainWeb draws are fixed; training augments spatial geometry, parameter maps, receive fields and noise.

## Train and validate

```bash
.venv/bin/python experiments/delics/train.py fit --config experiments/delics/train.yaml
.venv/bin/python experiments/delics/train.py fit --config experiments/delics/train.yaml \
  --ckpt_path runs/delics/version_0/checkpoints/last.ckpt
.venv/bin/python experiments/delics/train.py validate --config experiments/delics/train.yaml \
  --ckpt_path runs/delics/version_0/checkpoints/best-t1-rmse.ckpt
```

The default uses five PINQI stages, image U-Net widths 48/64/96/128, parameter U-Net widths 64/128/192/256, ten subspace coefficients, 352 spokes and a 256×256 matrix. Each update combines six DELICS and four BrainWeb slices, averaging each branch separately before adding 0.25 times the BrainWeb loss. The loss uses normalized-log T1/T2 L1, final-map SSIM with weight 0.3, P90-normalized M0 magnitude MSE with weight 0.5, and an outside-mask M0 penalty. Stage weights are 0.1/0.1/0.1/0.1/0.8. Training uses AdamW, learning rate 3e-4, scalar multiplier 2, and 80 epochs; validation uses EMA weights.

CSV metrics and the resolved config are saved under `runs/delics`. Checkpoints retain the complete training state in `last.ckpt` and the best validation T1 RMSE in `best-t1-rmse.ckpt`. Use `--ckpt_path` to resume optimizer, scheduler and EMA state; `--model.initial_checkpoint` loads weights for a new run. The default config is a reproducible recipe, not a claim that it is the best-performing recipe.

For optional W&B metrics and selected reference/baseline/prediction figures, replace `trainer.logger` with a list containing the CSV logger and `lightning.pytorch.loggers.WandbLogger`, setting its project and name through `init_args`. CSV-only runs do not require a W&B login. Validation reports MAE, RMSE, NRMSE, SSIM and HFEN for T1/T2 and the CG/dictionary baseline. Primary DELICS metrics use unclipped maps; clipped metrics have the `val_0_clipped` prefix. `val_1` is BrainWeb when enabled.

The compiled U-Nets use PyTorch's activation recomputation budget of 0.7 to match the current training recipe. The full default batch needs a large GPU; reduce batch sizes for smaller devices. Cluster launchers, recovery scripts and experiment sweeps are intentionally outside this directory.
