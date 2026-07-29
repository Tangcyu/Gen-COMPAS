# Gen-COMPAS v0.2.0

Gen-COMPAS v0.2.0 is the first release built around the complete iterative
workflow. It brings diffusion sampling, committor learning, molecular dynamics,
RiteWeight analysis, and free-energy estimation into one configurable and
restartable pipeline.

## Highlights

### Integrated iterative workflow

- Run bootstrap iteration 0 and subsequent committor-guided iterations through
  the unified `gen-compas` command or `workflow.py`.
- Execute the full sequence or select a range with `--start-at` and
  `--stop-after`.
- Inspect a resolved schedule before launching expensive calculations with
  `--dry-run`.
- Resume interrupted calculations from atomic per-iteration manifests with
  `--resume`.
- Inspect results between stages with `--stepwise`.
- Run or force one stage with `--run_step` and `--rerun_step`.
- Use iteration-specific diffusion epochs and sampling noise, optional diffusion
  warm starts, and cumulative trajectory inputs.
- Record a complete `effective_config.yaml`, stage status, timestamps, errors,
  and attempt counts for every iteration.

### Configuration GUI and helper

- Create a complete workflow configuration from project defaults.
- Load and expand an existing minimal or complete YAML file.
- Browse for input files and directories, validate required settings, preview
  the resolved configuration, and save it for later workflow execution.
- Add and remove independently editable FEL projection cards instead of editing
  the complete projections list manually.
- Use the installed `gen-compas-config` command interactively or in
  validation/output-only CLI workflows.

### Automated NAMD sampling

- Prepare independent target/protocol job directories from reusable templates.
- Run targeted MD followed by unbiased MD for every selected structure.
- Configure local CPU or GPU commands and concurrent NAMD jobs.
- Optionally include initial unbiased basin simulations in the iteration-0
  bootstrap data.

### RiteWeight and free-energy analysis

- Use a shared feature contract between RiteWeight and the Variational
  Committor Network.
- Construct lagged samples independently inside each source trajectory, avoiding
  artificial transitions between NAMD jobs.
- Preserve transition-origin weights for VCN training while providing symmetric
  start/end `fel_weight` values for equilibrium FEL estimation.
- Generate globally weighted one- and two-dimensional FEL projections in CV
  space.
- Separate the landscape calculation cap (`landscape_F_max`) from each plot's
  display cap (`F_max`); values above the display cap are rendered white while
  full landscape values remain in `.dat` and `.npz` outputs.
- Filter two routine native DCD-plugin messages without hiding other MDTraj
  diagnostics.

### Committor and target-selection improvements

- Keep RiteWeight and VCN internal-coordinate featurization consistent.
- Select committor-slice candidates around `q = 0.5` with configurable width and
  target count.
- Support optional committor-projection plots and stricter target-count checks.

### Examples, packaging, and documentation

- Add complete workflow configurations for Trp-cage, NANMA, trialanine, RBP,
  AAC apo/holo, Vo-domain, and nAChR examples.
- Install `gen-compas`, `gen-compas-workflow`, and `gen-compas-config` entry
  points.
- Document NAMD/Colvars setup, workflow controls, iteration layouts, parameter
  guidance, GUI usage, RiteWeight, and FEL estimation.
- Archive the legacy top-level runner and legacy reweighting/FEL scripts for
  reference.

## Upgrade notes

- The legacy top-level `run.py` interface is archived. Use `gen-compas` or
  `workflow.py`.
- Workflow stages now share the unified YAML schema in `config.yaml` and
  `common/config.py`.
- FEL estimation uses `fel_weight` by default. The legacy `weight` column remains
  the transition-origin weight used by VCN and is retained for compatibility.
- NAMD and Colvars remain external dependencies and are not installed by pip.
  NAMD 3.0.2 or newer is recommended.

## Validation

The release test suite completes with 74 passing tests and 3 environment-dependent
tests skipped.
