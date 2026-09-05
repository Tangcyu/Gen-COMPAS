"""Shared configuration defaults and iteration-specific path resolution."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Union

import yaml


DEFAULT_CONFIG = {
    "Workflow": {
        "root_dir": "./Iterations",
        "initial_data_folders": [],
        "initial_diffusion_data": {"dcd_path": None, "topology_path": None},
        "run_initial_unbiased": False,
        "run_fel": True,
        "warm_start_diffusion": True,
        "iteration_noise_scales": {},
        "iteration_diffusion_epochs": {},
        "isolate_steps": True,
    },
    "Generative": {
        "save_dir": None,
        "device": "cuda:0",
        "random_seed": 42,
        "init_checkpoint_path": None,
        "data": {
            "dcd_path": None,
            "topology_path": None,
            "alignment_atomselect": "all",
        },
        "coordinate_contract": {
            "enabled": True,
            "source": None,
            "filename": "coordinate_contract.pt",
            "reference_filename": "canonical_reference.pdb",
            "alignment_atomselect": None,
        },
        "model": {
            "node_feature_dim": 64,
            "time_embedding_dim": 128,
            "hidden_dim": 128,
            "num_schnet_layers": 4,
            "num_gat_layers": 2,
            "residue_attn_heads": 4,
            "k_neighbors": 16,
            "num_segment_layers": 2,
            "segment_distance_rbf": 16,
        },
        "diffusion": {"timesteps": 200, "beta_schedule": "cosine"},
        "training": {
            "epochs": 50,
            "batch_size": 64,
            "lr": 1.0e-4,
            "weight_decay": 1.0e-6,
            "grad_clip": 1.0,
            "num_workers": 4,
            "save_interval": 50,
        },
        "inference": {
            "checkpoint": None,
            "output": None,
            "num_samples": 1000,
            "sample_batch": 100,
            "noise_scale": 1.5,
        },
    },
    "VCN": {
        "device": "cuda:0",
        "label": "committor",
        "sampling_path": ".",
        "z_matrix": True,
        "use_all": False,
        "pair_distance": False,
        "dcdfile": None,
        "traj_fns": None,
        "topfile": None,
        "atomindex": [],
        "atomselect": "protein and name CA",
        "stride": 1,
        "cvs": [],
        "periodic": False,
        "val_ratio": 0.1,
        "random_seed": 42,
        "time_shift": 1,
        "trajectory_column": "trajectory_id",
        "epochs": 5000,
        "learning_rate": 1.0e-4,
        "patience": 200,
        "batch_size_factor": 0.6,
        "num_layers": 4,
        "num_nodes": 64,
        "k": 1.0,
        "out_dir": None,
        "cvs_to_plot": None,
        "plot_committor_projections": False,
        "gendcdfile": None,
        "model_fn": None,
        "slice_dir": None,
        "q_variance": 0.1,
        "n_targets": 20,
        "require_n_targets": True,
    },
    "Clustering": {
        "topology": None,
        "trajectory": None,
        "atom_selection": "protein and name CA",
        "output_dir": "./output_clusters",
        "n_clusters": None,
        "n_per_cluster": 2,
        "max_k": 8,
        "select_farthest": True,
        "random_seed": 0,
    },
    "Occupancy": {
        "pdb_dir": None,
        "topology_file": None,
        "pdb_file": None,
        "output_dir": "./output_pdbs",
        "add_hydrogens": True,
        "selection": "protein and name CA",
    },
    "NAMD": {
        "namd_path": None,
        "template_path": None,
        "output_dir": "./output_namd",
        "tmd_force_constant": 10000.0,
        "existing_job_policy": "error",
        "targets": {
            "path": "./output_pdbs",
            "pattern": "*.pdb",
            "recursive": False,
            "target_filename": "output.pdb",
        },
        "protocols": [
            {
                "name": "A",
                "tmd_template": "TMD.A.conf",
                "unbiased_template": "Unbiased.A.conf",
                "initial_unbiased_template": "Initial.A.conf",
            },
            {
                "name": "B",
                "tmd_template": "TMD.B.conf",
                "unbiased_template": "Unbiased.B.conf",
                "initial_unbiased_template": "Initial.B.conf",
            },
        ],
        "phases": {"tmd": True, "unbiased": True},
        "execution": {
            "device": "cpu",
            "parallel_jobs": 1,
            "dry_run": False,
            "cpu": {
                "threads_per_job": 1,
                "command": ["{namd}", "+p{threads}", "{config}"],
            },
            "gpu": {
                "devices": ["0"],
                "threads_per_job": 1,
                "command": [
                    "{namd}", "+p{threads}", "+devices", "{device}", "{config}"
                ],
            },
        },
    },
    "RiteWeight": {
        "folders": [],
        "dcd_pattern": "*Unbiased.[AB].dcd",
        "colvars_pattern": "*Unbiased.[AB].colvars.traj",
        "tag_regex": r"\.([AB])(?:\.|$)",
        "io": {"topology": None, "out": "./output_riteweight", "stride": 1},
        "pairing": {"allow_skip_first_colvars": True, "strict": True},
        "features": {
            "mode": "internal_zmat",
            "internal_zmat": {
                "atomselect": "protein and name CA",
                "atom_order": None,
                "max_atoms": None,
                "order": "index",
            },
            "distances": {"atom_pairs": []},
            "cache": {
                "enabled": True,
                "format": "npz",
                "path": "./output_riteweight/features_internal_zmat.npz",
                "policy": "write_if_missing",
            },
        },
        "riteweight": {
            "n_clusters": 100,
            "n_iter": 200,
            "tol": 1.0e-6,
            "tol_window": 5,
            "avg_last": 20,
            "seed": 2026,
            "lag": 50,
        },
        "colvars": {"cv": ["CV1", "CV2"], "save_cols": "all", "periodic_cols": []},
        "committor_labels": {
            "enabled": True,
            "cvs_to_label": ["CV1", "CV2"],
            "basin_A": None,
            "basin_B": None,
            "basin_size": None,
            "k_prefactor": 1.0,
            "angle_unit": "degree",
        },
        "outputs": {
            "frame_csv": "frame_weights.csv",
            "segment_csv": "segment_weights.csv",
            "convergence_plot": "convergence_delta.png",
            "vcn": {"enabled": True, "filename": "vcn_training.pt"},
            "diffusion": {
                "enabled": True,
                "trajectory": "diffusion_training.dcd",
                "topology": "diffusion_training.pdb",
                "atomselect": "protein and not element H",
                "alignment_atomselect": None,
                "reference_path": None,
                "chunk_size": 1000,
            },
        },
    },
    "FEL_estimate": {
        "input": None,
        "output_dir": "./output_riteweight/fel",
        "weight_column": "fel_weight",
        "temperature_K": 300.0,
        "probability_floor": 1.0e-300,
        "landscape_F_max": 10.0,
        "projections": [],
    },
}


Q_CENTER = 0.5
DEFAULT_Q_VARIANCE = 0.1


def committor_slice_bounds(q_variance=DEFAULT_Q_VARIANCE):
    """Return the q=0.5 slice bounds for a validated half-width."""
    if isinstance(q_variance, bool):
        raise ValueError("VCN.q_variance must be a number between 0 and 0.5.")
    try:
        q_variance = float(q_variance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "VCN.q_variance must be a number between 0 and 0.5."
        ) from exc
    if not math.isfinite(q_variance) or not 0 <= q_variance <= 0.5:
        raise ValueError("VCN.q_variance must be a number between 0 and 0.5.")
    return Q_CENTER - q_variance, Q_CENTER + q_variance


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge mappings; lists and scalar values replace defaults."""
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def normalize_config_aliases(config: Mapping[str, Any]) -> dict:
    """Migrate supported legacy keys to their canonical configuration names."""
    normalized = deepcopy(dict(config))
    riteweight = normalized.get("RiteWeight")
    if isinstance(riteweight, Mapping):
        riteweight = deepcopy(dict(riteweight))
        normalized["RiteWeight"] = riteweight
        io_config = riteweight.get("io")
        if isinstance(io_config, Mapping):
            io_config = deepcopy(dict(io_config))
            riteweight["io"] = io_config
            if "top" in io_config:
                legacy = io_config.pop("top")
                if "topology" in io_config and io_config["topology"] != legacy:
                    raise ValueError(
                        "RiteWeight.io defines conflicting 'top' and 'topology' values."
                    )
                io_config.setdefault("topology", legacy)
    return normalized


def load_config(path: Union[str, Path]) -> dict:
    """Load a user YAML file and fill every omitted default value."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    if not isinstance(user_config, Mapping):
        raise TypeError("The top level of the configuration must be a mapping.")
    return deep_merge(DEFAULT_CONFIG, normalize_config_aliases(user_config))


def ordinal(value: int) -> str:
    """Return an English ordinal suitable for iteration directory names."""
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _rw_artifacts(directory: Path, config: Mapping[str, Any]) -> dict[str, str]:
    outputs = config["RiteWeight"]["outputs"]
    return {
        "dcd": str(directory / outputs["diffusion"]["trajectory"]),
        "topology": str(directory / outputs["diffusion"]["topology"]),
        "vcn": str(directory / outputs["vcn"]["filename"]),
    }


def _iteration_noise_scale(
    workflow: Mapping[str, Any], iteration: int, fallback: Any
) -> float:
    """Return a finite, non-negative sampling noise for one iteration."""
    overrides = workflow.get("iteration_noise_scales", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise TypeError("Workflow.iteration_noise_scales must be a mapping.")

    value = overrides.get(iteration, overrides.get(str(iteration), fallback))
    try:
        noise_scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sampling noise for iteration {iteration} must be a number; got {value!r}."
        ) from exc
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError(
            f"Sampling noise for iteration {iteration} must be finite and non-negative; "
            f"got {value!r}."
        )
    return noise_scale


def _iteration_diffusion_epochs(
    workflow: Mapping[str, Any], iteration: int, fallback: Any
) -> int:
    """Return a positive diffusion-training epoch count for one iteration."""
    overrides = workflow.get("iteration_diffusion_epochs", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise TypeError("Workflow.iteration_diffusion_epochs must be a mapping.")

    value = overrides.get(iteration, overrides.get(str(iteration), fallback))
    try:
        epochs = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Diffusion epochs for iteration {iteration} must be a positive integer; "
            f"got {value!r}."
        ) from exc
    if isinstance(value, bool) or epochs < 1:
        raise ValueError(
            f"Diffusion epochs for iteration {iteration} must be a positive integer; "
            f"got {value!r}."
        )
    try:
        is_integral = float(value) == epochs
    except (TypeError, ValueError, OverflowError):
        is_integral = False
    if not is_integral:
        raise ValueError(
            f"Diffusion epochs for iteration {iteration} must be a positive integer; "
            f"got {value!r}."
        )
    return epochs


def resolve_iteration_config(config: Mapping[str, Any], iteration: int) -> dict:
    """Derive all data-flow paths for one Gen-COMPAS iteration."""
    if iteration < 0:
        raise ValueError("Iteration must be non-negative.")
    resolved = deepcopy(dict(config))
    workflow = resolved["Workflow"]
    root = Path(workflow["root_dir"]).expanduser().resolve()
    iteration_dir = root / f"{ordinal(iteration)}.Iteration"
    previous_dir = root / f"{ordinal(iteration - 1)}.Iteration" if iteration else None

    paths = {
        "root": root,
        "iteration": iteration_dir,
        "initial_namd": root / "0th.Iteration" / "initial_namd",
        "diffusion_model": iteration_dir / "models" / "diffusion",
        "vcn_model": iteration_dir / "models" / "vcn",
        "generated": iteration_dir / "generated" / "candidates.dcd",
        "cluster": iteration_dir / "cluster_targets",
        "slice": iteration_dir / "committor_slice",
        "targets": iteration_dir / "targets",
        "namd": iteration_dir / "namd",
        "riteweight": iteration_dir / "riteweight",
    }

    if iteration == 0:
        initial_data = workflow.get("initial_diffusion_data", {})
        if not initial_data.get("dcd_path") or not initial_data.get("topology_path"):
            raise ValueError(
                "Iteration 0 requires Workflow.initial_diffusion_data.dcd_path "
                "and topology_path."
            )
        training_rw_dir = None
        training = {
            "dcd": str(Path(initial_data["dcd_path"]).expanduser().resolve()),
            "topology": str(
                Path(initial_data["topology_path"]).expanduser().resolve()
            ),
            "vcn": None,
        }
    else:
        training_rw_dir = previous_dir / "riteweight"
        training = _rw_artifacts(training_rw_dir, resolved)

    generative = resolved["Generative"]
    generative["data"].update(
        dcd_path=training["dcd"], topology_path=training["topology"]
    )
    generative["save_dir"] = str(paths["diffusion_model"])
    generative["inference"]["checkpoint"] = str(paths["diffusion_model"] / "best_model.pt")
    generative["inference"]["output"] = str(paths["generated"])
    generative["inference"]["noise_scale"] = _iteration_noise_scale(
        workflow, iteration, generative["inference"].get("noise_scale", 1.5)
    )
    generative["training"]["epochs"] = _iteration_diffusion_epochs(
        workflow, iteration, generative["training"].get("epochs", 50)
    )
    if iteration > 0 and workflow.get("warm_start_diffusion", True):
        generative["init_checkpoint_path"] = str(
            previous_dir / "models" / "diffusion" / "best_model.pt"
        )
        generative["coordinate_contract"]["source"] = str(
            previous_dir
            / "models"
            / "diffusion"
            / generative["coordinate_contract"].get(
                "filename", "coordinate_contract.pt"
            )
        )
    else:
        generative["init_checkpoint_path"] = None
        generative["coordinate_contract"]["source"] = None

    vcn = resolved["VCN"]
    vcn.update(
        sampling_path=".",
        dcdfile=[training["dcd"]],
        traj_fns=[training["vcn"]],
        topfile=training["topology"],
        out_dir=str(paths["vcn_model"]),
        label=f"iteration_{iteration}",
        gendcdfile=str(paths["generated"]),
        slice_dir=str(paths["slice"]),
    )
    label_suffix = f"{vcn['label']}_patience{vcn['patience']}"
    vcn["model_fn"] = str(paths["vcn_model"] / f"{label_suffix}_cpu_best_model.pt")

    resolved["Clustering"].update(
        topology=training["topology"],
        trajectory=str(paths["generated"]),
        output_dir=str(paths["cluster"]),
    )
    target_source = paths["cluster"] if iteration == 0 else paths["slice"] / "sliced_frames"
    occupancy = resolved["Occupancy"]
    occupancy.update(pdb_dir=str(target_source), output_dir=str(paths["targets"]))
    if occupancy.get("topology_file") is None:
        occupancy["topology_file"] = resolved["RiteWeight"]["io"]["topology"]

    resolved["NAMD"]["output_dir"] = str(paths["namd"])
    resolved["NAMD"]["targets"]["path"] = str(paths["targets"])

    if workflow.get("run_initial_unbiased", False):
        initial_folders = [str(paths["initial_namd"])]
    else:
        initial_folders = [
            str(Path(folder).expanduser().resolve())
            for folder in workflow.get("initial_data_folders", [])
        ]
    cumulative_folders = initial_folders + [
        str(root / f"{ordinal(index)}.Iteration" / "namd")
        for index in range(iteration + 1)
    ]
    riteweight = resolved["RiteWeight"]
    riteweight["folders"] = cumulative_folders
    riteweight["io"]["out"] = str(paths["riteweight"])
    riteweight["features"]["cache"]["path"] = str(
        paths["riteweight"] / "features_internal_zmat.npz"
    )
    if generative["coordinate_contract"].get("enabled", True):
        riteweight["outputs"]["diffusion"]["reference_path"] = str(
            paths["diffusion_model"]
            / generative["coordinate_contract"].get(
                "reference_filename", "canonical_reference.pdb"
            )
        )
    else:
        riteweight["outputs"]["diffusion"]["reference_path"] = None

    fel = resolved["FEL_estimate"]
    fel["input"] = str(paths["riteweight"] / riteweight["outputs"]["vcn"]["filename"])
    fel["output_dir"] = str(paths["riteweight"] / "fel")

    initial_namd = deepcopy(resolved["NAMD"])
    initial_namd["output_dir"] = str(paths["initial_namd"])
    initial_namd["phases"] = {"tmd": False, "unbiased": True}
    initial_namd.pop("targets", None)
    if workflow.get("run_initial_unbiased", False):
        for protocol in initial_namd["protocols"]:
            initial_template = protocol.get("initial_unbiased_template")
            if not initial_template:
                raise ValueError(
                    f"NAMD protocol {protocol.get('name')!r} needs "
                    "initial_unbiased_template when Workflow.run_initial_unbiased is true."
                )
            protocol["unbiased_template"] = initial_template

    workflow["runtime"] = {
        "iteration": iteration,
        "iteration_dir": str(iteration_dir),
        "training_riteweight_dir": (
            None if training_rw_dir is None else str(training_rw_dir)
        ),
        "initial_folders": initial_folders,
        "initial_namd": initial_namd,
    }
    return resolved


def write_effective_config(config: Mapping[str, Any], path: Union[str, Path]) -> None:
    """Write a resolved configuration without Python-only path objects."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
