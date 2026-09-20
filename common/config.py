"""Shared configuration defaults and iteration-specific path resolution."""

from __future__ import annotations

from copy import deepcopy
import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Union

import yaml

from configs import load_workflow_defaults


# Canonical values live in the packaged configs/*.yaml files.
DEFAULT_CONFIG = load_workflow_defaults()


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
    autonoise = generative.setdefault("autonoise", {"enabled": False})
    if not isinstance(autonoise, Mapping) or not isinstance(autonoise.get("enabled", False), bool):
        raise ValueError("Generative.autonoise.enabled must be a boolean.")
    if autonoise.get("enabled", False):
        if workflow.get("iteration_noise_scales"):
            warnings.warn("Generative.autonoise.enabled is true: Workflow.iteration_noise_scales "
                          "is ignored; AutoNoise selects the sampling noise for each iteration.",
                          UserWarning, stacklevel=2)
        generative["inference"]["noise_scale"] = None  # Resolved by autonoise_diffusion.
        autonoise["output_dir"] = str(iteration_dir / "autonoise")
    else:
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
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
    temporary.replace(output)
