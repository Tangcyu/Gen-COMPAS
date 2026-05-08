# Gen-COMPAS Diffusion

Gen-COMPAS trains and samples a diffusion model for protein structure generation.

## Workflow

The command-line entry point supports two steps:

| Step | Description |
| --- | --- |
| `train_diffusion` | Train the protein-structure diffusion model from DCD trajectories and a PSF topology. |
| `sample_diffusion` | Generate protein conformations from a trained checkpoint. |

## Usage

```bash
python run.py --step train_diffusion --config config.yaml
python run.py --step sample_diffusion --config config.yaml
```

After installation, the same workflow is available through:

```bash
gen-compas --step train_diffusion --config config.yaml
gen-compas --step sample_diffusion --config config.yaml
```

## Configuration

All runtime settings live under the `Generative` section in `config.yaml`.

Key subsections:

- `data`: input DCD trajectory paths and PSF topology.
- `model`: embedding size, hidden dimensions, SchNet layers, attention layers, and k-nearest-neighbor graph size.
- `diffusion`: diffusion timesteps and beta schedule.
- `training`: optimizer, batch size, worker count, checkpoint cadence, and gradient clipping.
- `inference`: checkpoint path, output path, sample count, sample batch size, and noise scale.

## Dependencies

Core requirements:

- Python >= 3.9
- PyTorch >= 2.0
- MDTraj
- NumPy
- PyYAML
- tqdm

Install the package with:

```bash
pip install .
```

The package uses native PyTorch operations for scatter-mean steps, so `torch-scatter` is not required.

## Demo Data

The Trp-cage example dataset is located at:

```text
examples/0.DEMO_Trp-cage/Dataset/
```

On an NVIDIA L40s GPU, training the diffusion model for 50 epochs takes approximately 5 minutes, and generating 1,000 structures with batch size 200 takes approximately 1 minute.
