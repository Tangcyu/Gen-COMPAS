from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from .config import ensure_dir, get_device, section
from .data import read_trajectories
from .model import load_model


def predict_q(model, x_raw: torch.Tensor, batch_size: int) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, x_raw.shape[0], batch_size):
            values.append(model(x_raw[start : start + batch_size]).detach().flatten())
    return torch.cat(values, dim=0)


def _sample_dataframe(data, max_points: int, seed: int):
    if max_points and len(data) > max_points:
        return data.sample(n=max_points, random_state=seed).reset_index(drop=True)
    return data.reset_index(drop=True)


def _scatter_2d(ax, x, y, q, xlabel: str, ylabel: str, cfg: dict):
    sc = ax.scatter(
        x,
        y,
        c=q,
        s=float(cfg.get("point_size", 4.0)),
        alpha=float(cfg.get("alpha", 0.75)),
        cmap=cfg.get("cmap", "viridis"),
        vmin=float(cfg.get("vmin", 0.0)),
        vmax=float(cfg.get("vmax", 1.0)),
        linewidths=0.0,
    )
    ax.set_xlabel(cfg.get("labels", {}).get(xlabel, xlabel))
    ax.set_ylabel(cfg.get("labels", {}).get(ylabel, ylabel))
    return sc


def _scatter_1d(ax, x, q, xlabel: str, cfg: dict):
    ax.scatter(
        x,
        q,
        s=float(cfg.get("point_size", 4.0)),
        alpha=float(cfg.get("alpha", 0.75)),
        color=cfg.get("one_d_color", "black"),
        linewidths=0.0,
    )
    ax.set_xlabel(cfg.get("labels", {}).get(xlabel, xlabel))
    ax.set_ylabel("q")


def plot_projection_from_config(config: dict) -> dict:
    data_cfg = section(config, "data")
    plot_cfg = section(config, "plotting")
    out_cfg = section(config, "output")
    model_cfg = section(config, "trained_model")

    device = get_device(plot_cfg.get("device", config.get("device", "cuda:0")))
    out_dir = ensure_dir(out_cfg.get("out_dir", "./vcn2_output"))
    plot_dir = ensure_dir(plot_cfg.get("out_dir", out_dir / "plots"))
    model_path = model_cfg.get("path", out_dir / "models" / "best_model.pt")
    model = load_model(model_path, device=device)

    model_cvs = list(data_cfg["cvs"])
    plot_cvs = list(plot_cfg.get("cvs", model_cvs[: min(3, len(model_cvs))]))
    all_cvs = list(dict.fromkeys([*model_cvs, *plot_cvs]))
    data = read_trajectories(
        data_cfg["traj_fns"],
        cvs=all_cvs,
        stride=int(plot_cfg.get("stride", data_cfg.get("stride", 1))),
        discard_before_step=data_cfg.get("discard_before_step"),
        step_column=data_cfg.get("step_column", "step"),
        weight_column=data_cfg.get("weight_column", "weight"),
    )
    data = _sample_dataframe(data, int(plot_cfg.get("max_points", 100000)), int(config.get("seed", 1234)))
    x_model = torch.as_tensor(data[model_cvs].to_numpy(np.float32), device=device)
    q = predict_q(model, x_model, int(plot_cfg.get("batch_size", 65536))).cpu().numpy()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("plotting requires matplotlib") from exc

    prefix = plot_cfg.get("prefix", "q_projection")
    dpi = int(plot_cfg.get("dpi", 250))
    written = []

    output_data = data[plot_cvs].copy()
    output_data["q"] = q
    output_data.to_csv(plot_dir / f"{prefix}_data.csv.gz", index=False)
    written.append(str(plot_dir / f"{prefix}_data.csv.gz"))

    if len(plot_cvs) == 1:
        fig, ax = plt.subplots(figsize=tuple(plot_cfg.get("figsize", [5.0, 3.5])))
        _scatter_1d(ax, data[plot_cvs[0]].to_numpy(), q, plot_cvs[0], plot_cfg)
        fig.tight_layout()
        out = plot_dir / f"{prefix}_{plot_cvs[0]}_q.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        written.append(str(out))
    else:
        pairs = list(combinations(plot_cvs, 2))
        for x_name, y_name in pairs:
            fig, ax = plt.subplots(figsize=tuple(plot_cfg.get("figsize", [5.0, 4.2])))
            sc = _scatter_2d(
                ax,
                data[x_name].to_numpy(),
                data[y_name].to_numpy(),
                q,
                x_name,
                y_name,
                plot_cfg,
            )
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("q")
            fig.tight_layout()
            out = plot_dir / f"{prefix}_{x_name}_vs_{y_name}.png"
            fig.savefig(out, dpi=dpi)
            plt.close(fig)
            written.append(str(out))

    return {"plot_dir": str(plot_dir), "files": written, "n_points": int(len(data)), "plot_cvs": plot_cvs}
