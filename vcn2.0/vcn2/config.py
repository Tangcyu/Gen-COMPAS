from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("VCN2", data)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_device(name: str):
    import torch

    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {name}, but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping.")
    return value
