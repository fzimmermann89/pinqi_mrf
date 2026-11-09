# BrainWeb saturation recovery

The comparison uses five saturation delays (0.5, 1, 1.5, 2, 8 s), eight simulated birdcage coils, Cartesian R8 undersampling with eight central lines, and 192×192 axial images. It compares ridge CG + dictionary matching, image-only MoDL followed by matching, and end-to-end MoDL/PINQI with dictionary or L-BFGS fitting. All learned methods use five stages and the same conditioned image U-Net (16/32/48/64 features). PINQI adds a 32/64/96/128 parameter U-Net.

Run from this folder in an environment with the repository dependencies. The implementation uses MR2's conditioned U-Net, differentiable dictionary matching and OptimizerOp; the original experiments used MR2 revision `7e6dd764f8d344c1192899e44563ffaab2b3a810`. BrainWeb inputs are the MR2-prepared `s??.h5` files.

```sh
python train.py pinqi-dictionary --brainweb /path/to/brainweb --output runs/pinqi-dictionary --wandb
python train.py pinqi-lbfgs --brainweb /path/to/brainweb --output runs/pinqi-lbfgs --wandb
python train.py modl-dictionary --brainweb /path/to/brainweb --output runs/modl-dictionary
python train.py modl-lbfgs --brainweb /path/to/brainweb --output runs/modl-lbfgs
python train.py modl --brainweb /path/to/brainweb --output runs/modl
```

The fixed recipe uses 80 epochs, batches of 8, AdamW, 20% warmup and cosine decay. End-to-end methods use T1 L1 + 0.1 SSIM, complex M0 loss weight 0.3 and auxiliary image MSE weight 0.1; earlier PINQI parameter estimates receive weight 0.05. The image-only model uses complex image L1 + 0.1 magnitude SSIM; fitting never enters its training loss. The default network learning rate is 4e-4 for the end-to-end methods , 8e-4 for image-only MoDL . `--learning-rate 8e-4` selects the higher-rate MoDL dictionary variant. Strengths use 2e-3. PINQI L-BFGS recomputes network activations during backward, preserving batch size 8 on the A6000. Change `--seed` for repeats; `--resume path/to/last.ckpt` restores full training state. CSV logging is always enabled; `--wandb` adds W&B (`WANDB_MODE=offline` is supported).

`checkpoints/last.ckpt` contains the final full training state; `best-mae.ckpt` contains the best validation-MAE state. Evaluation is separate, so checkpoints can be selected without using test results:

```sh
python evaluate.py --brainweb /path/to/brainweb --checkpoint runs/modl/checkpoints/last.ckpt --output runs/modl/test.csv
python evaluate.py --brainweb /path/to/brainweb --output runs/cg-dictionary/test.csv
```

The baseline uses ridge 0.001, CG cap 100 (absolute tolerance 1e-4), and 1000-entry dictionary matching without refinement. Learned dictionary fits use 1000 entries and one Gauss–Newton refinement; L-BFGS uses 20 iterations with implicit differentiation. MoDL fitting has fixed weak prior weight 1e-4; PINQI learns its prior weights. T1 is constrained to 0.05–6 s during fitting.

Training uses 16 subjects (`s04–s50`, explicitly listed in `data.py`), validation uses `s51` (20 slices), and testing uses `s52–s54` (120 slices). Training draws geometry, tissue values, sampling and noise afresh. Validation/test draws use SHA256 seeds incorporating subject, orientation, anatomical slice and random stream, independent of loader order. Their GM/WM base T1 ranges are 1.3–2.0 s and 0.7–1.3 s; spatial variation can extend beyond these bounds. Every split uses the original smooth/multiscale parameter variation and smooth M0 phase. Noise RMS is sampled uniformly from 0.001–0.04 in absolute signal units. Inputs and M0 targets are normalized per sample by the 95th percentile magnitude after two initial CG iterations.

Evaluation saves per-slice brain MAE/RMSE/NRMSE/SSIM and complex-M0 NRMSE. Means weight slices equally. SSIM uses 7×7 windows wholly inside the brain and each slice's ground-truth brain range. These scores are independent of plot color limits.

The CSV `slice_index` is the per-subject loader index, not the anatomical slice number. SSIM means exclude slices without a complete brain window (stored as NaN).
