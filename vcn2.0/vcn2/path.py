from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import ensure_dir, get_device, section
from .data import sample_raw_points
from .model import load_model


def batched_q(model, x_raw: torch.Tensor, batch_size: int = 65536) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, x_raw.shape[0], batch_size):
            values.append(model(x_raw[start : start + batch_size]).detach())
    return torch.cat(values, dim=0).flatten()


def reparametrize_polyline(points: torch.Tensor, num_images: int) -> torch.Tensor:
    if points.shape[0] == num_images:
        return points
    if points.shape[0] < 2:
        return points.repeat(num_images, 1)
    segment = torch.linalg.norm(torch.diff(points, dim=0), dim=1)
    total = torch.sum(segment)
    if total <= 1e-12:
        return points[:1].repeat(num_images, 1)
    cumulative = torch.cat([torch.zeros(1, device=points.device), torch.cumsum(segment, dim=0)])
    target = torch.linspace(0.0, float(total.detach().cpu()), num_images, device=points.device)
    idx = torch.searchsorted(cumulative, target, right=True) - 1
    idx = torch.clamp(idx, 0, points.shape[0] - 2)
    left = cumulative[idx]
    right = cumulative[idx + 1]
    denom = torch.clamp(right - left, min=1e-12)
    alpha = ((target - left) / denom).unsqueeze(1)
    return points[idx] * (1.0 - alpha) + points[idx + 1] * alpha


def _active_step(model, z_active: torch.Tensor, direction: float, step_size: float, min_grad_norm: float, noise: float):
    z_req = z_active.detach().clone().requires_grad_(True)
    q = model.forward_normalized(z_req).sum()
    grad = torch.autograd.grad(q, z_req)[0]
    norms = torch.linalg.norm(grad, dim=1)
    good = norms > min_grad_norm
    versor = grad / torch.clamp(norms[:, None], min=1e-12)
    if noise > 0.0:
        trial = versor + torch.randn_like(versor) * noise
        trial = trial / torch.clamp(torch.linalg.norm(trial, dim=1, keepdim=True), min=1e-12)
        same_half_space = torch.sum(trial * versor, dim=1, keepdim=True) >= 0.0
        versor = torch.where(same_half_space, trial, -trial)
    z_next = z_active + float(direction) * step_size * versor
    return z_next.detach(), norms.detach(), good.detach()


def trace_direction(
    model,
    z_start: torch.Tensor,
    direction: float,
    max_steps: int,
    step_size: float,
    delta_q: float,
    min_grad_norm: float,
    stochasticity: float,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    z = z_start.detach().clone()
    paths = [[z[i].detach().clone()] for i in range(z.shape[0])]
    active = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
    min_norm_seen = torch.full((z.shape[0],), float("inf"), device=z.device)
    stopped_low_grad = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)

    for _ in range(max_steps):
        q = batched_q_normalized(model, z, batch_size=max(4096, z.shape[0]))
        reached = q >= 1.0 - delta_q if direction > 0.0 else q <= delta_q
        active = active & (~reached)
        if not bool(torch.any(active)):
            break
        active_idx = torch.where(active)[0]
        z_next, norms, good = _active_step(
            model,
            z[active_idx],
            direction=direction,
            step_size=step_size,
            min_grad_norm=min_grad_norm,
            noise=stochasticity,
        )
        min_norm_seen[active_idx] = torch.minimum(min_norm_seen[active_idx], norms)
        bad_idx = active_idx[~good]
        if bad_idx.numel() > 0:
            stopped_low_grad[bad_idx] = True
            active[bad_idx] = False
        good_idx = active_idx[good]
        if good_idx.numel() > 0:
            z[good_idx] = z_next[good]
            for local, global_idx in enumerate(good_idx.tolist()):
                paths[global_idx].append(z_next[good][local].detach().clone())

    stacked_paths = [torch.stack(items, dim=0) for items in paths]
    finite_min = min_norm_seen[torch.isfinite(min_norm_seen)]
    stats = {
        "min_grad_norm_seen": float(torch.min(finite_min).detach().cpu()) if finite_min.numel() else float("nan"),
        "low_grad_stops": int(torch.sum(stopped_low_grad).detach().cpu()),
    }
    return stacked_paths, stats


def batched_q_normalized(model, z: torch.Tensor, batch_size: int = 65536) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for start in range(0, z.shape[0], batch_size):
            values.append(model.forward_normalized(z[start : start + batch_size]).detach())
    return torch.cat(values, dim=0).flatten()


def save_path_npz(path: Path, z: torch.Tensor, raw: torch.Tensor, q: torch.Tensor, weight: float) -> None:
    np.savez_compressed(
        path,
        z=z.detach().cpu().numpy().astype(np.float32),
        raw=raw.detach().cpu().numpy().astype(np.float32),
        q=q.detach().cpu().numpy().astype(np.float32),
        weight=np.float32(weight),
    )


def find_paths_from_config(config: dict) -> dict:
    data_cfg = section(config, "data")
    path_cfg = section(config, "path")
    out_cfg = section(config, "output")
    model_cfg = section(config, "trained_model")

    device = get_device(path_cfg.get("device", config.get("device", "cuda:0")))
    out_dir = ensure_dir(out_cfg.get("out_dir", "./vcn2_output"))
    path_dir = ensure_dir(out_dir / "paths")
    model_path = model_cfg.get("path", out_dir / "models" / "best_model.pt")
    model = load_model(model_path, device=device)

    cvs = list(data_cfg["cvs"])
    x_np, weights_np = sample_raw_points(
        data_cfg["traj_fns"],
        cvs=cvs,
        max_points=int(path_cfg.get("candidate_max_points", 200000)),
        seed=int(config.get("seed", path_cfg.get("seed", 1234))),
        stride=int(path_cfg.get("candidate_stride", data_cfg.get("stride", 1))),
    )
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    q = batched_q(model, x, batch_size=int(path_cfg.get("q_batch_size", 65536)))
    window = float(path_cfg.get("initial_q_window", 0.1))
    center = float(path_cfg.get("initial_q_center", 0.5))
    mask = (q >= center - window) & (q <= center + window)
    candidates = torch.where(mask)[0]
    if candidates.numel() == 0:
        raise RuntimeError("No initial points found in the requested q window.")
    n_paths = int(path_cfg.get("n_paths", 100))
    rng = np.random.default_rng(int(config.get("seed", path_cfg.get("seed", 1234))))
    chosen_np = rng.choice(candidates.detach().cpu().numpy(), size=n_paths, replace=candidates.numel() < n_paths)
    chosen = torch.as_tensor(chosen_np, dtype=torch.long, device=device)
    x_start = x[chosen]
    path_weights = weights_np[chosen.detach().cpu().numpy()]
    z_start = model.normalize_input(x_start)

    max_steps = int(path_cfg.get("max_steps", 300))
    step_size = float(path_cfg.get("step_size", 0.05))
    delta_q = float(path_cfg.get("delta_q", 1e-3))
    min_grad_norm = float(path_cfg.get("min_grad_norm", 1e-8))
    stochasticity = float(path_cfg.get("stochasticity", 0.0))
    images = int(path_cfg.get("num_images", 64))

    paths_b, stats_b = trace_direction(
        model, z_start, 1.0, max_steps, step_size, delta_q, min_grad_norm, stochasticity
    )
    paths_a, stats_a = trace_direction(
        model, z_start, -1.0, max_steps, step_size, delta_q, min_grad_norm, stochasticity
    )

    endpoint_a = path_cfg.get("endpoint_A")
    endpoint_b = path_cfg.get("endpoint_B")
    z_endpoint_a = z_endpoint_b = None
    if bool(path_cfg.get("append_endpoints", False)):
        if endpoint_a is None or endpoint_b is None:
            raise ValueError("append_endpoints=true requires endpoint_A and endpoint_B.")
        z_endpoint_a = model.normalize_input(torch.as_tensor([endpoint_a], dtype=torch.float32, device=device))[0]
        z_endpoint_b = model.normalize_input(torch.as_tensor([endpoint_b], dtype=torch.float32, device=device))[0]

    summary_rows = []
    for i, (pa, pb) in enumerate(zip(paths_a, paths_b)):
        combined = torch.cat([torch.flip(pa, dims=[0]), pb[1:]], dim=0)
        if z_endpoint_a is not None and z_endpoint_b is not None:
            combined = torch.cat([z_endpoint_a[None, :], combined, z_endpoint_b[None, :]], dim=0)
        combined = reparametrize_polyline(combined, images)
        raw = model.denormalize_input(combined)
        q_path = batched_q_normalized(model, combined, batch_size=images)
        save_path_npz(path_dir / f"path_{i:05d}.npz", combined, raw, q_path, float(path_weights[i]))
        summary_rows.append(
            {
                "path": f"path_{i:05d}.npz",
                "weight": float(path_weights[i]),
                "q_start": float(q[chosen[i]].detach().cpu()),
                "q_min": float(torch.min(q_path).detach().cpu()),
                "q_max": float(torch.max(q_path).detach().cpu()),
                "z_length": float(torch.sum(torch.linalg.norm(torch.diff(combined, dim=0), dim=1)).detach().cpu()),
            }
        )

    with open(path_dir / "path_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(path_dir / "path_run_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"toward_A": stats_a, "toward_B": stats_b, "n_paths": n_paths}, handle, indent=2)
    return {"path_dir": str(path_dir), "n_paths": n_paths}
