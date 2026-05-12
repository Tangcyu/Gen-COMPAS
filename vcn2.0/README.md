# VCN 2.0

VCN 2.0 is a cleaned-up workflow for training a variational committor model, tracing committor-gradient pathways, clustering pathways, and diagnosing whether the learned committor gradient is dominated by a few high-variance CVs.

The key change from the older scripts is that input normalization is part of the saved model. Path finding and clustering use the normalized CV space by default, while outputs are also written back in raw CV units.

## Project Tree

```text
vcn2.0/
  run.py                       # Unified CLI: train/path/cluster/analyze/all
  config.full.yaml             # Fully commented template
  config.minimal.yaml          # Short template
  vcn2/
    config.py                  # YAML loading, device/output helpers
    data.py                    # Trajectory reading, time-lagging, normalizer stats
    model.py                   # CommittorNet with fixed InputNormalizer
    loss.py                    # VCN loss plus input/gradient regularization
    train.py                   # Training loop and model export
    path.py                    # Batched normalized-space gradient path finding
    cluster.py                 # GPU RMSD matrix + hierarchical clustering
    analysis.py                # q and gradient diagnostics by CV
```

## Quick Start

From the Gen-COMPAS root:

```bash
cd /home/ctang/1.Ongoing/1.Diffusion.Additional/Gen-COMPAS/vcn2.0
python run.py all config.minimal.yaml
```

Run individual stages:

```bash
python run.py train config.full.yaml
python run.py path config.full.yaml
python run.py cluster config.full.yaml
python run.py analyze config.full.yaml
python run.py plot config.full.yaml
```

## Input Regularization

The model receives raw CVs but immediately maps them to normalized coordinates:

```text
z_i = (x_i - loc_i) / scale_i
```

`loc` and `scale` are computed from the training time-lagged origin and target frames, then stored as TorchScript buffers inside `models/best_model.pt`. This prevents a high-variance distance CV from dominating the first network layer simply because it has a larger numerical range.

Intermediate lagged datasets are saved as `.csv.gz`; parquet support was intentionally left out to keep the dependency set small.

Available normalization methods:

```text
standard  mean/std
robust    median/IQR
range     center/half-range
none      no scaling
```

Additional controls:

```text
model.input_clip          clips z values after normalization
model.input_dropout       dropout after hidden activations during training
loss.input_noise_std      Gaussian noise in normalized space during training
loss.gradient_l2_scale    penalty on |dq/dz|^2 in normalized space
training.weight_decay     AdamW parameter regularization
```

## Path Finding

Path finding starts from trajectory frames with `q` near `initial_q_center` and integrates:

```text
z <- z +/- step_size * grad_z(q) / |grad_z(q)|
```

The path is reparametrized as a piecewise-linear polyline in normalized CV space. Each path is saved as:

```text
output/paths/path_00000.npz
  z       normalized CV path
  raw     raw CV path
  q       committor values along the path
  weight  initial-frame trajectory weight
```

`append_endpoints` is off by default. Turning it on can be useful for production strings, but diagnostics should first inspect unforced paths to avoid endpoint interpolation creating artificial curvature.

## Clustering

Clustering follows the old `cluster_paths.py` idea:

1. Load all `path_*.npz`.
2. Compute pairwise RMSD between normalized paths on CUDA with `torch.cdist`.
3. Run SciPy hierarchical average linkage.
4. Save cluster labels and weighted mean paths.

Outputs:

```text
output/clusters/path_rmsd_matrix.npy
output/clusters/cluster_labels.csv
output/clusters/cluster_summary.csv
output/clusters/cluster_001_mean_raw.txt
output/clusters/cluster_001_mean_z.txt
output/clusters/dendrogram.png
```

## Gradient Analysis

`python run.py analyze config.yaml` samples trajectory frames and writes:

```text
output/analysis/gradient_by_cv.csv
output/analysis/gradient_report.json
output/analysis/path_integrals.csv
output/analysis/gradient_share_by_cv.png
output/analysis/q_histogram.png
```

Important columns:

```text
grad_z_rms_share     fraction of normalized-space gradient RMS carried by each CV
near_zero_fraction   fraction of sampled frames with |dq/dz_i| below zero_grad_tol
delta_std            time-lagged std of CV changes
input_scale          scale used by the model normalizer
```

If one CV has a very high `grad_z_rms_share`, the report warns about gradient dominance. This is the main guard against the old failure mode where high-variance distances hid the useful gradients from smaller-amplitude CVs.

If `output/paths/path_*.npz` exists, the analysis stage also computes path line integrals on CUDA:

```text
line_integral_abs_grad_dz = integral |grad_z q| ds_z
min_grad_norm_z
median_grad_norm_z
monotone_positive_fraction
```

These values help distinguish a path driven by q from a curve whose shape mostly came from interpolation or reparametrization.

## Projection Plotting

`python run.py plot config.yaml` projects the trained q values onto selected raw CVs without slicing frames by q. Configure it with:

```yaml
plotting:
  cvs: [cv1, cv2, cv3]
```

For one CV it writes `q` versus that CV. For two or more CVs it writes all 2D pair scatter plots colored by q, plus `q_projection_data.csv.gz`.

## Notes

- CUDA is used for training, q prediction, path gradients, path RMSD matrices, and gradient diagnostics when available.
- SciPy is still used for hierarchical clustering after the RMSD matrix is computed.
- The saved TorchScript model accepts raw CVs, so downstream code can call `model(x_raw)` directly.
- For physical interpretation of gradients, prefer `grad_z_*` diagnostics first. Raw gradients scale as `grad_z / input_scale`.
