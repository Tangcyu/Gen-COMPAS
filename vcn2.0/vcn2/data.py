from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_table(path: str | Path, columns: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, usecols=columns, nrows=nrows)
    if suffixes.endswith(".dat") or suffixes.endswith(".txt"):
        return pd.read_csv(path, sep=r"\s+", usecols=columns, nrows=nrows)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_trajectories(
    files: Iterable[str | Path],
    cvs: list[str],
    stride: int = 1,
    discard_before_step: float | None = None,
    step_column: str = "step",
    weight_column: str = "weight",
    extra_columns: Iterable[str] = ("Ka", "Kb", "center"),
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    optional = list(dict.fromkeys([weight_column, step_column, *extra_columns]))
    for filename in files:
        header = read_table(filename, columns=None, nrows=0)
        available = set(header.columns)
        missing_cvs = [name for name in cvs if name not in available]
        if missing_cvs:
            raise KeyError(f"{filename} is missing CV columns: {missing_cvs}")
        columns_to_read = list(dict.fromkeys([*cvs, *[name for name in optional if name in available]]))
        data = read_table(filename, columns=columns_to_read)
        if step_column in data.columns and discard_before_step is not None:
            data = data.loc[data[step_column] >= discard_before_step]
        if stride and stride > 1:
            data = data.iloc[::stride]
        if weight_column not in data.columns:
            data[weight_column] = 1.0
        for name in extra_columns:
            if name not in data.columns:
                if name == "center":
                    data[name] = 0.0
                else:
                    data[name] = 0.0
        required = list(dict.fromkeys([weight_column, *cvs, *extra_columns]))
        present = [name for name in required if name in data.columns]
        frames.append(data[present].reset_index(drop=True))
    if not frames:
        raise ValueError("No trajectory files were provided.")
    return pd.concat(frames, ignore_index=True)


def make_time_lagged(
    data: pd.DataFrame,
    time_shift: int,
    val_ratio: float,
    seed: int,
    weight_mode: str = "product",
    weight_column: str = "weight",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if time_shift < 1:
        raise ValueError("time_shift must be >= 1.")
    if len(data) <= time_shift:
        raise ValueError("Trajectory is shorter than time_shift.")

    origin = data.iloc[: len(data) - time_shift].copy()
    target = data.iloc[time_shift:].copy()
    origin.columns = [f"{name}_origin" for name in origin.columns]
    target.columns = [f"{name}_target" for name in target.columns]
    origin.reset_index(drop=True, inplace=True)
    target.reset_index(drop=True, inplace=True)
    lagged = pd.concat([origin, target], axis=1)

    w0 = lagged[f"{weight_column}_origin"].to_numpy(dtype=np.float64)
    wt = lagged[f"{weight_column}_target"].to_numpy(dtype=np.float64)
    if weight_mode == "product":
        weights = w0 * wt
    elif weight_mode == "sqrt_product":
        weights = np.sqrt(w0 * wt)
    elif weight_mode == "origin":
        weights = w0
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    weights = np.asarray(weights, dtype=np.float64)
    mean_weight = np.mean(weights)
    if mean_weight > 0:
        weights = weights / mean_weight
    lagged["weight"] = weights.astype(np.float32)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(lagged))
    n_val = int(round(len(lagged) * val_ratio))
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val:]
    return lagged, lagged.iloc[train_idx].reset_index(drop=True), lagged.iloc[val_idx].reset_index(drop=True)


def load_or_build_lagged(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cvs = list(config["cvs"])
    if config.get("train_set") and config.get("val_set"):
        train = pd.concat([read_table(path) for path in config["train_set"]], ignore_index=True)
        val = pd.concat([read_table(path) for path in config["val_set"]], ignore_index=True)
        all_data = pd.concat([train, val], ignore_index=True)
        return all_data, train, val

    traj = read_trajectories(
        config["traj_fns"],
        cvs=cvs,
        stride=int(config.get("stride", 1)),
        discard_before_step=config.get("discard_before_step"),
        step_column=config.get("step_column", "step"),
        weight_column=config.get("weight_column", "weight"),
    )
    return make_time_lagged(
        traj,
        time_shift=int(config.get("time_shift", 1)),
        val_ratio=float(config.get("val_ratio", 0.1)),
        seed=int(config.get("seed", 1234)),
        weight_mode=config.get("weight_mode", "product"),
        weight_column=config.get("weight_column", "weight"),
    )


@dataclass
class NormalizerStats:
    loc: np.ndarray
    scale: np.ndarray
    data_std: np.ndarray
    data_min: np.ndarray
    data_max: np.ndarray


def compute_normalizer_stats(
    lagged_train: pd.DataFrame,
    cvs: list[str],
    method: str = "standard",
    eps: float = 1e-8,
) -> NormalizerStats:
    cols = [f"{name}_origin" for name in cvs] + [f"{name}_target" for name in cvs]
    values = lagged_train[cols].to_numpy(dtype=np.float64).reshape(-1, len(cvs))
    data_std = np.std(values, axis=0)
    data_min = np.min(values, axis=0)
    data_max = np.max(values, axis=0)
    if method == "none":
        loc = np.zeros(len(cvs), dtype=np.float64)
        scale = np.ones(len(cvs), dtype=np.float64)
    elif method == "standard":
        loc = np.mean(values, axis=0)
        scale = data_std
    elif method == "robust":
        q25, q75 = np.percentile(values, [25, 75], axis=0)
        loc = np.median(values, axis=0)
        scale = (q75 - q25) / 1.349
    elif method == "range":
        loc = 0.5 * (data_max + data_min)
        scale = 0.5 * (data_max - data_min)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    scale = np.where(np.abs(scale) < eps, 1.0, scale)
    return NormalizerStats(loc=loc, scale=scale, data_std=data_std, data_min=data_min, data_max=data_max)


class TimeLaggedDataset(Dataset):
    def __init__(self, data: pd.DataFrame, cvs: list[str], device: torch.device | str = "cpu"):
        self.cvs = list(cvs)
        device = torch.device(device)
        x0_cols = [f"{name}_origin" for name in cvs]
        xt_cols = [f"{name}_target" for name in cvs]
        self.x0 = torch.as_tensor(data[x0_cols].to_numpy(np.float32), device=device)
        self.xt = torch.as_tensor(data[xt_cols].to_numpy(np.float32), device=device)
        self.weights = torch.as_tensor(data[["weight"]].to_numpy(np.float32), device=device)
        self.ka0 = self._column(data, "Ka_origin", device)
        self.kat = self._column(data, "Ka_target", device)
        self.kb0 = self._column(data, "Kb_origin", device)
        self.kbt = self._column(data, "Kb_target", device)
        self.center0 = self._column(data, "center_origin", device)
        self.centert = self._column(data, "center_target", device)

    @staticmethod
    def _column(data: pd.DataFrame, name: str, device: torch.device) -> torch.Tensor:
        if name in data.columns:
            arr = data[[name]].to_numpy(np.float32)
        else:
            arr = np.zeros((len(data), 1), dtype=np.float32)
        return torch.as_tensor(arr, device=device)

    def __len__(self) -> int:
        return int(self.x0.shape[0])

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.x0,
            self.xt,
            self.weights,
            self.ka0,
            self.kat,
            self.kb0,
            self.kbt,
            self.center0,
            self.centert,
        )

    def pin_memory(self) -> "TimeLaggedDataset":
        if self.x0.device.type != "cpu":
            return self
        self.x0 = self.x0.pin_memory()
        self.xt = self.xt.pin_memory()
        self.weights = self.weights.pin_memory()
        self.ka0 = self.ka0.pin_memory()
        self.kat = self.kat.pin_memory()
        self.kb0 = self.kb0.pin_memory()
        self.kbt = self.kbt.pin_memory()
        self.center0 = self.center0.pin_memory()
        self.centert = self.centert.pin_memory()
        return self

    def __getitem__(self, index: int):
        return (
            self.x0[index],
            self.xt[index],
            self.weights[index],
            self.ka0[index],
            self.kat[index],
            self.kb0[index],
            self.kbt[index],
            self.center0[index],
            self.centert[index],
        )


def sample_raw_points(
    files: Iterable[str | Path],
    cvs: list[str],
    max_points: int,
    seed: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    data = read_trajectories(files, cvs=cvs, stride=stride)
    x = data[cvs].to_numpy(np.float32)
    w = data["weight"].to_numpy(np.float32)
    if max_points and len(x) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_points, replace=False)
        x = x[idx]
        w = w[idx]
    return x, w
