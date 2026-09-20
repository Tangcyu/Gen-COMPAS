# Configuration defaults

This directory is the source of defaults for the current Gen-COMPAS workflow.
`common/config.py` assembles these files for the CLI and configuration helper.
The root `config.yaml` remains an annotated run template; `minimal.yaml` contains
only system-specific inputs and selected overrides. Their explicit values take
precedence over these files, even when the template value differs from a default.

[Full.yaml](Full.yaml) is the complete run template: it expands the root
`config.yaml` with every default in this directory, including advanced AutoNoise
settings. Explicit template values are retained. Replace example paths and
system-specific basin/projection definitions before running. It is excluded from
automatic default-section loading; use `--config configs/Full.yaml` to select it.

## File layout

- `Section.yaml` contains that section's top-level fields, such as `VCN.val_ratio`.
- `Section.subsection.yaml` contains the fields under that exact subsection.
  For example, `Generative.model.yaml` starts with `node_feature_dim`, without
  repeating the `Generative:` or `model:` wrappers.
- Deeper mappings stay together in their subsection file, e.g. `cpu` and `gpu`
  inside `NAMD.execution.yaml`, and `geometry` inside `Generative.autonoise.yaml`.
- `Generative.autonoise.yaml` replaces the former `autonoise.yaml`.

| Configuration section | Default files |
| --- | --- |
| Clustering | [Clustering.yaml](Clustering.yaml) |
| FEL_estimate | [FEL_estimate.yaml](FEL_estimate.yaml) |
| Generative | [Generative.autonoise.yaml](Generative.autonoise.yaml), [Generative.coordinate_contract.yaml](Generative.coordinate_contract.yaml), [Generative.data.yaml](Generative.data.yaml), [Generative.diffusion.yaml](Generative.diffusion.yaml), [Generative.inference.yaml](Generative.inference.yaml), [Generative.model.yaml](Generative.model.yaml), [Generative.training.yaml](Generative.training.yaml), [Generative.yaml](Generative.yaml) |
| NAMD | [NAMD.execution.yaml](NAMD.execution.yaml), [NAMD.phases.yaml](NAMD.phases.yaml), [NAMD.targets.yaml](NAMD.targets.yaml), [NAMD.yaml](NAMD.yaml) |
| Occupancy | [Occupancy.yaml](Occupancy.yaml) |
| RiteWeight | [RiteWeight.colvars.yaml](RiteWeight.colvars.yaml), [RiteWeight.committor_labels.yaml](RiteWeight.committor_labels.yaml), [RiteWeight.features.yaml](RiteWeight.features.yaml), [RiteWeight.io.yaml](RiteWeight.io.yaml), [RiteWeight.outputs.yaml](RiteWeight.outputs.yaml), [RiteWeight.pairing.yaml](RiteWeight.pairing.yaml), [RiteWeight.pmf_output.yaml](RiteWeight.pmf_output.yaml), [RiteWeight.riteweight.yaml](RiteWeight.riteweight.yaml), [RiteWeight.yaml](RiteWeight.yaml) |
| VCN | [VCN.yaml](VCN.yaml) |
| Workflow | [Workflow.initial_diffusion_data.yaml](Workflow.initial_diffusion_data.yaml), [Workflow.yaml](Workflow.yaml) |

## Overrides and loading

Defaults are read from installed package resources, independently of the current
working directory. No additional CLI option or include directive is needed.
Mappings are merged recursively; explicit lists, scalars, and `null` replace the
default value. Unknown user keys are preserved. Each load produces independent
values, so editing a resolved run does not mutate the shared defaults.

For example, a run can override just one setting in each subsection:

```yaml
Generative:
  model:
    num_segment_layers: 3
  autonoise:
    search:
      max_samples: 1024
NAMD:
  execution:
    gpu:
      devices: ["0", "1"]
```

The remaining model, AutoNoise search, and NAMD execution settings keep their
defaults. Files must contain mappings and must not define the same configuration
path in both a parent file and a subsection file.

Previously implicit options such as `VCN.val_ratio`, `VCN.extra_label`,
`Clustering.random_seed`, RiteWeight histogram settings, and the optional legacy
`RiteWeight.pmf_output` are included here. Data-dependent values still resolve at
runtime: omitted RiteWeight `colvars.periodicities` means nonperiodic for each CV;
FEL projection ranges and names can be inferred from the data and CV names.
Per-projection options remain documented in the root `config.yaml`.

The workflow resolves iteration-specific input/output paths after applying run
overrides. AutoNoise runs before sampling only when enabled, and saves its
selection in the existing effective configuration and workflow manifest.
