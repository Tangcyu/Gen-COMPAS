from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import ensure_dir, get_device, section
from .data import read_trajectories, sample_raw_points
from .model import load_model


def gradient_chunks(model, x_raw: torch.Tensor, chunk_size: int):
    scale = model.input_scale().detach()
    for start in range(0, x_raw.shape[0], chunk_size):
        raw = x_raw[start : start + chunk_size]
        z = model.normalize_input(raw).detach().clone().requires_grad_(True)
        q = model.forward_normalized(z)
        grad_z = torch.autograd.grad(q.sum(), z)[0].detach()
        grad_raw = grad_z / scale
        yield q.detach().flatten(), grad_z, grad_raw


def normalized_gradient_chunks(model, z_all: torch.Tensor, chunk_size: int):
    for start in range(0, z_all.shape[0], chunk_size):
        z = z_all[start : start + chunk_size].detach().clone().requires_grad_(True)
        q = model.forward_normalized(z)
        grad = torch.autograd.grad(q.sum(), z)[0].detach()
        yield q.detach().flatten(), grad


def analyze_path_integrals(model, path_dir: Path, analysis_dir: Path, device: torch.device, batch_size: int) -> int:
    files = sorted(path_dir.glob("path_*.npz"))
    if not files:
        return 0
    rows = []
    for file in files:
        data = np.load(file)
        z = torch.as_tensor(data["z"], dtype=torch.float32, device=device)
        q_parts = []
        grad_parts = []
        for q, grad in normalized_gradient_chunks(model, z, batch_size):
            q_parts.append(q)
            grad_parts.append(grad)
        q_path = torch.cat(q_parts)
        grad = torch.cat(grad_parts, dim=0)
        grad_norm = torch.linalg.norm(grad, dim=1)
        ds = torch.linalg.norm(torch.diff(z, dim=0), dim=1)
        integral = torch.sum(0.5 * (grad_norm[:-1] + grad_norm[1:]) * ds)
        dq = torch.diff(q_path)
        rows.append(
            {
                "path": file.name,
                "line_integral_abs_grad_dz": float(integral.detach().cpu()),
                "z_length": float(torch.sum(ds).detach().cpu()),
                "q_start": float(q_path[0].detach().cpu()),
                "q_end": float(q_path[-1].detach().cpu()),
                "q_delta": float((q_path[-1] - q_path[0]).detach().cpu()),
                "q_min": float(torch.min(q_path).detach().cpu()),
                "q_max": float(torch.max(q_path).detach().cpu()),
                "min_grad_norm_z": float(torch.min(grad_norm).detach().cpu()),
                "median_grad_norm_z": float(torch.median(grad_norm).detach().cpu()),
                "monotone_positive_fraction": float(torch.mean((dq >= -1e-6).float()).detach().cpu()),
            }
        )
    with open(analysis_dir / "path_integrals.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def analyze_gradients_from_config(config: dict) -> dict:
    data_cfg = section(config, "data")
    analysis_cfg = section(config, "analysis")
    out_cfg = section(config, "output")
    model_cfg = section(config, "trained_model")

    device = get_device(analysis_cfg.get("device", config.get("device", "cuda:0")))
    out_dir = ensure_dir(out_cfg.get("out_dir", "./vcn2_output"))
    analysis_dir = ensure_dir(out_dir / "analysis")
    model_path = model_cfg.get("path", out_dir / "models" / "best_model.pt")
    model = load_model(model_path, device=device)
    cvs = list(data_cfg["cvs"])

    x_np, _ = sample_raw_points(
        data_cfg["traj_fns"],
        cvs=cvs,
        max_points=int(analysis_cfg.get("max_points", 100000)),
        seed=int(config.get("seed", analysis_cfg.get("seed", 1234))),
        stride=int(analysis_cfg.get("stride", data_cfg.get("stride", 1))),
    )
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    q_values = []
    abs_sum_z = torch.zeros(len(cvs), device=device)
    sq_sum_z = torch.zeros(len(cvs), device=device)
    max_abs_z = torch.zeros(len(cvs), device=device)
    zero_count = torch.zeros(len(cvs), device=device)
    abs_sum_raw = torch.zeros(len(cvs), device=device)
    n_total = 0
    zero_tol = float(analysis_cfg.get("zero_grad_tol", 1e-7))
    for q, grad_z, grad_raw in gradient_chunks(model, x, int(analysis_cfg.get("batch_size", 32768))):
        q_values.append(q.cpu())
        abs_grad_z = torch.abs(grad_z)
        abs_grad_raw = torch.abs(grad_raw)
        abs_sum_z += torch.sum(abs_grad_z, dim=0)
        sq_sum_z += torch.sum(torch.square(grad_z), dim=0)
        max_abs_z = torch.maximum(max_abs_z, torch.max(abs_grad_z, dim=0).values)
        zero_count += torch.sum(abs_grad_z < zero_tol, dim=0)
        abs_sum_raw += torch.sum(abs_grad_raw, dim=0)
        n_total += grad_z.shape[0]

    mean_abs_z = abs_sum_z / max(n_total, 1)
    rms_z = torch.sqrt(sq_sum_z / max(n_total, 1))
    share = rms_z / torch.clamp(torch.sum(rms_z), min=1e-12)
    mean_abs_raw = abs_sum_raw / max(n_total, 1)
    near_zero = zero_count / max(n_total, 1)
    loc = model.input_loc().detach().cpu().numpy()
    scale = model.input_scale().detach().cpu().numpy()
    data_std = np.std(x_np, axis=0)

    delta_std = np.full(len(cvs), np.nan)
    try:
        traj = read_trajectories(data_cfg["traj_fns"], cvs=cvs, stride=int(data_cfg.get("stride", 1)))
        shift = int(data_cfg.get("time_shift", 1))
        if len(traj) > shift:
            delta_std = np.std(traj[cvs].to_numpy(np.float32)[shift:] - traj[cvs].to_numpy(np.float32)[:-shift], axis=0)
    except Exception as exc:
        print(f"Could not compute time-lag delta_std: {exc}")

    rows = []
    for i, cv in enumerate(cvs):
        rows.append(
            {
                "cv": cv,
                "input_loc": float(loc[i]),
                "input_scale": float(scale[i]),
                "data_std": float(data_std[i]),
                "delta_std": float(delta_std[i]),
                "grad_z_abs_mean": float(mean_abs_z[i].detach().cpu()),
                "grad_z_rms": float(rms_z[i].detach().cpu()),
                "grad_z_abs_max": float(max_abs_z[i].detach().cpu()),
                "grad_z_rms_share": float(share[i].detach().cpu()),
                "grad_raw_abs_mean": float(mean_abs_raw[i].detach().cpu()),
                "near_zero_fraction": float(near_zero[i].detach().cpu()),
            }
        )
    with open(analysis_dir / "gradient_by_cv.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    q_all = torch.cat(q_values).numpy()
    warnings = []
    dominance_threshold = float(analysis_cfg.get("dominance_share_warn", 0.5))
    max_share = float(torch.max(share).detach().cpu())
    if max_share > dominance_threshold:
        top = rows[int(torch.argmax(share).detach().cpu())]["cv"]
        warnings.append(f"Gradient RMS is dominated by {top}: share={max_share:.3f}.")
    flat_fraction = float(torch.mean((near_zero > 0.95).float()).detach().cpu())
    if np.nanmax(q_all) - np.nanmin(q_all) < float(analysis_cfg.get("q_range_warn", 0.1)):
        warnings.append("q has a narrow range on sampled trajectory points.")

    report = {
        "n_points": int(n_total),
        "q_min": float(np.min(q_all)),
        "q_max": float(np.max(q_all)),
        "q_mean": float(np.mean(q_all)),
        "max_gradient_share": max_share,
        "flat_fraction": flat_fraction,
        "warnings": warnings,
    }
    with open(analysis_dir / "gradient_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    path_dir = Path(analysis_cfg.get("path_dir", out_dir / "paths"))
    n_path_integrals = analyze_path_integrals(
        model,
        path_dir=path_dir,
        analysis_dir=analysis_dir,
        device=device,
        batch_size=int(analysis_cfg.get("path_integral_batch_size", analysis_cfg.get("batch_size", 32768))),
    )
    if n_path_integrals:
        report["n_path_integrals"] = n_path_integrals
        with open(analysis_dir / "gradient_report.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort([row["grad_z_rms"] for row in rows])[::-1]
        labels = [rows[i]["cv"] for i in order]
        values = [rows[i]["grad_z_rms_share"] for i in order]
        width = max(8.0, 0.35 * len(labels))
        plt.figure(figsize=(width, 4.0))
        plt.bar(np.arange(len(labels)), values)
        plt.xticks(np.arange(len(labels)), labels, rotation=90)
        plt.ylabel("RMS gradient share in normalized CV space")
        plt.tight_layout()
        plt.savefig(analysis_dir / "gradient_share_by_cv.png", dpi=220)
        plt.close()

        plt.figure(figsize=(5.0, 3.5))
        plt.hist(q_all, bins=50)
        plt.xlabel("q")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(analysis_dir / "q_histogram.png", dpi=220)
        plt.close()
    except Exception as exc:
        print(f"Could not write gradient plots: {exc}")

    return {"analysis_dir": str(analysis_dir), "warnings": warnings, "n_path_integrals": n_path_integrals}
