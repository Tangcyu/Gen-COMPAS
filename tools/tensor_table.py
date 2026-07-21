"""Portable numeric table serialization built on safe ``torch.save`` payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import pandas as pd


FORMAT_NAME = "gen-compas.tensor-table"
FORMAT_VERSION = 1


def save_tensor_table(
    path: Union[str, Path],
    dataframe: pd.DataFrame,
    metadata: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Save all numeric DataFrame columns as a weights-only-compatible Torch payload."""
    import torch

    numeric = dataframe.select_dtypes(include=[np.number, "bool"])
    if numeric.shape[1] == 0:
        raise ValueError("Cannot save tensor table: the DataFrame has no numeric columns.")

    tensors = {
        column: torch.as_tensor(numeric[column].to_numpy(copy=True))
        for column in numeric.columns
    }
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "columns": list(numeric.columns),
        "tensors": tensors,
        "num_rows": len(numeric),
        "excluded_columns": [c for c in dataframe.columns if c not in numeric.columns],
        "metadata": dict(metadata or {}),
    }

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return list(numeric.columns)


def _torch_load_safe(path: Union[str, Path]):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # ``weights_only`` was added after older supported PyTorch releases.
        return torch.load(path, map_location="cpu")


def load_tensor_table(path: Union[str, Path]) -> pd.DataFrame:
    """Load a tensor table created by :func:`save_tensor_table`."""
    import torch

    payload = _torch_load_safe(path)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ValueError(f"{path} is not a {FORMAT_NAME} payload.")

    columns = payload.get("columns", [])
    tensors = payload.get("tensors", {})
    missing = [column for column in columns if column not in tensors]
    if missing:
        raise ValueError(f"Tensor table is missing columns: {missing}")

    data = {}
    expected_rows = int(payload.get("num_rows", -1))
    for column in columns:
        tensor = tensors[column]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Column {column!r} is not stored as a torch.Tensor.")
        values = tensor.detach().cpu().numpy().reshape(-1)
        if expected_rows >= 0 and len(values) != expected_rows:
            raise ValueError(
                f"Column {column!r} has {len(values)} rows; expected {expected_rows}."
            )
        data[column] = values
    return pd.DataFrame(data, columns=columns)
