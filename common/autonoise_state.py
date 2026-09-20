"""Lightweight AutoNoise handoff between workflow processes and resumed runs."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from common.config import deep_merge
from configs import load_defaults


def calibration_fingerprint(config):
    """Bind a selection to its iteration, model, coordinates, references and settings.

    Large reference DCDs use path/size/mtime identity; checkpoint and coordinate
    contract contents are hashed. Output locations do not affect the selection.
    """
    g = config["Generative"]
    settings = deep_merge(load_defaults("Generative.autonoise"), g.get("autonoise", {}))
    for key in ("enabled", "output_dir", "save_dcd"):
        settings.pop(key, None)
    checkpoint = Path(g["inference"]["checkpoint"])
    contract = checkpoint.parent / g.get("coordinate_contract", {}).get("filename", "coordinate_contract.pt")
    paths = {"checkpoint": checkpoint, "contract": contract,
             "topology": Path(g["data"].get("topology_path") or g["data"].get("psf_path")),
             "state_a": Path(settings["reference"]["state_a"] or ""),
             "state_b": Path(settings["reference"]["state_b"] or "")}
    inputs = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"AutoNoise {name} file not found: {path}")
        stat = path.stat()
        identity = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if name in ("checkpoint", "contract"):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            identity["sha256"] = digest.hexdigest()
        inputs[name] = identity
    payload = {"iteration": config["Workflow"]["runtime"]["iteration"], "inputs": inputs,
               "model": g["model"], "diffusion": g["diffusion"], "settings": settings}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def apply_autonoise_selection(config, selection=None):
    """Validate and apply a completed calibration; never fall back to manual noise."""
    runtime = config["Workflow"]["runtime"]
    if selection is None:
        selection = runtime.get("autonoise_selection")
    if not isinstance(selection, dict):
        raise RuntimeError("No completed AutoNoise selection. Run autonoise_diffusion before sample_diffusion.")
    noise = selection.get("noise_scale")
    if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not math.isfinite(noise) or noise < 0:
        raise RuntimeError("Invalid AutoNoise selection; rerun autonoise_diffusion.")
    if selection.get("fingerprint") != calibration_fingerprint(config):
        raise RuntimeError("AutoNoise inputs changed; rerun autonoise_diffusion for this iteration.")
    runtime["autonoise_selection"] = deepcopy(selection)
    config["Generative"]["inference"]["noise_scale"] = float(noise)
    config["Generative"]["autonoise"]["output_dir"] = selection["output_dir"]
    return runtime["autonoise_selection"]


def accept_autonoise_result(config, report, fingerprint):
    """Publish only a successful, still-current calibration in the runtime config."""
    recommendations = report.get("recommended_noise_scales", [])
    if report.get("status") != "ok" or not recommendations:
        raise RuntimeError(f"autonoise_diffusion returned {report.get('status')!r}; no noise selected. "
                           "Inspect the distribution plot and reference/search settings before retrying.")
    selection = {"noise_scale": recommendations[0], "recommended_noise_scales": recommendations,
                 "fingerprint": fingerprint, "output_dir": report["output_dir"]}
    return apply_autonoise_selection(config, selection)
