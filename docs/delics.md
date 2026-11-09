# Supervised DELICS PINQI

The default is the simple DELICS-only recipe: four PINQI stages, eight slices per batch, normalized log-L1 for T1/T2 with equal weights, P90-normalized M0 magnitude MSE (weight 0.5), and an outside-mask M0 penalty (weight 0.1). Deep supervision weights are 0.2/0.1/0.1/0.8. There is no auxiliary image loss or SSIM training loss.

The acquisition uses 352 spokes, 256×256 images, eight coils and ten subspace coefficients. Image U-Net widths are 48/64/96/128 and parameter U-Net widths are 64/128/192/256. Training retains the existing geometry, bias-field and noise augmentation, detached CG warm starts, EMA, clipping, and 80-epoch AdamW schedule. The default peak network LR is 3e-4 with scalar multiplier 2 and weight decay 1e-3.

## Data preparation

The radial cache contains the full-acquisition reference maps, masks, valid slices, and the `source_file` attribute pointing to reconstructed coil frames. Create the compressed frame cache once:

```sh
.venv/bin/python -m pinqi_mrf.prepare_delics_frames \
  --reference-folder data/delics/radial --output data/delics/frames \
  --case-ids case000 case001 case002 case003 case004 case005 case006 case007 case008 case009
```

This preserves provenance, the compression matrix and complex-half frames for every valid slice. Existing complete subject caches are reused. The three preparation modules cover raw reconstruction/maps, radial cache creation and frame caching; their inputs do not depend on trained PINQI models.

## Leave-one-subject-out training

`configs/delics.yaml` explicitly lists the ten-subject cohort, case000–case009. `data.held_out_case` selects one subject for evaluation; the other nine supply every training slice. Case100/case101 are outside this cohort. Training reads only the nine training subjects, including calibration, augmentation and frame data. Evaluation reads the held-out subject's fixed radial inputs. Slice noise is fixed by subject and anatomical slice.

Run each fold from scratch. For example, hold out case003:

```sh
.venv/bin/python -m pinqi_mrf.supervised fit --config configs/delics.yaml \
  --data.held_out_case case003
.venv/bin/python -m pinqi_mrf.supervised validate --config configs/delics.yaml \
  --data.held_out_case case003 \
  --ckpt_path runs/delics/case003/checkpoints/last.ckpt
```

Repeat with each of case000–case009 to obtain all ten folds. `--output` changes the common output root. CSV logs, the resolved config and checkpoints are separated by held-out subject. `--wandb` adds optional W&B metrics and reference/baseline/prediction figures.

Fitting runs for the fixed epoch count without evaluating the held-out subject. It saves the full final state as `checkpoints/last.ckpt`; no held-out metric selects a checkpoint. Standalone `validate` evaluates that final state with EMA weights. Fresh training refuses an output folder that already contains checkpoints, and validation requires an explicit checkpoint. Resume an interrupted fold with the same held-out subject and `--ckpt_path`; checkpoints record the cohort and fold and reject a different fold. Do not initialize one fold from another fold's model.

Freeze learning rate, epoch count and other recipe choices before running the final folds. The cohort already informed earlier recipe development, so describe this as leave-one-subject-out agreement with full-acquisition reference maps, not an independent external test or ground-truth accuracy study.

Validation reports per-slice-weighted MAE/RMSE/NRMSE, SSIM and HFEN for T1/T2 and the CG/dictionary baseline. `val_0` uses unclipped maps; additional clipped scores use `val_0_clipped`. Report each subject's scores and average subjects equally across the ten folds. The frame cache and full-acquisition reference fit for a subject use only that subject's data; no PINQI training or fitted cohort statistics enter them.

## Dependencies and memory

The compiled U-Nets use PyTorch's activation recomputation budget of 0.7. Batch eight needs a large GPU; reduce `--data.batch_size` for smaller devices. MR2 is declared as the `mrtwo` distribution and imported as `mr2`. The checked research checkout includes unpublished spherical-harmonic phantom helpers and optimizer changes. Publish and pin those MR2 changes before expecting a reproducible installation from released packages.
