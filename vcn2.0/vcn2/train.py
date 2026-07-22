from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .config import ensure_dir, get_device, section
from .data import TimeLaggedDataset, compute_normalizer_stats, load_or_build_lagged, write_table
from .loss import committor_loss
from .model import build_model, save_model_metadata


def _move_batch(batch, device: torch.device):
    return tuple(item if item.device == device else item.to(device, non_blocking=True) for item in batch)


def _iter_tensor_batches(
    dataset: TimeLaggedDataset,
    batch_size: int,
    shuffle: bool,
    target_device: torch.device,
):
    n_samples = len(dataset)
    source_tensors = dataset.tensors()
    source_device = source_tensors[0].device
    if shuffle:
        indices = torch.randperm(n_samples, device=source_device)
    else:
        indices = None
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        if indices is None:
            batch = tuple(t[start:stop] for t in source_tensors)
        else:
            batch_idx = indices[start:stop]
            batch = tuple(torch.index_select(t, 0, batch_idx) for t in source_tensors)
        yield _move_batch(batch, target_device)


def _epoch(
    model,
    dataset: TimeLaggedDataset,
    batch_size: int,
    optimizer,
    loss_cfg,
    device: torch.device,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    totals = {
        "loss": torch.zeros((), device=device),
        "jab": torch.zeros((), device=device),
        "endpoint": torch.zeros((), device=device),
        "grad_l2": torch.zeros((), device=device),
    }
    n = 0
    for batch in _iter_tensor_batches(dataset, batch_size, shuffle=train, target_device=device):
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = committor_loss(model, batch, loss_cfg)
            loss.backward()
            clip = float(loss_cfg.get("parameter_grad_clip", 0.0))
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        else:
            with torch.no_grad():
                loss, parts = committor_loss(
                    model,
                    batch,
                    {**loss_cfg, "gradient_l2_scale": 0.0, "input_noise_std": 0.0},
                )
        for key in totals:
            totals[key] = totals[key] + parts[key]
        n += 1
    inv_n = 1.0 / max(n, 1)
    # One GPU sync per epoch for logging, instead of one sync per batch.
    return {key: float((value * inv_n).detach().cpu()) for key, value in totals.items()}


def _estimate_dataset_gb(n_samples: int, n_cvs: int) -> float:
    # x0, xt have n_cvs each; weights plus six endpoint-restraint scalar columns.
    n_float32 = n_samples * (2 * n_cvs + 7)
    return n_float32 * 4.0 / (1024.0 ** 3)


def _resolve_dataset_device(train_cfg: dict, train_df_len: int, val_df_len: int, n_cvs: int, train_device: torch.device):
    requested = str(train_cfg.get("dataset_device", "cpu"))
    if requested != "auto":
        return torch.device(requested)
    estimated_gb = _estimate_dataset_gb(train_df_len + val_df_len, n_cvs)
    max_gb = float(train_cfg.get("gpu_dataset_max_gb", 4.0))
    if train_device.type == "cuda" and estimated_gb <= max_gb:
        print(f"Using GPU-resident dataset ({estimated_gb:.3f} GB estimated <= {max_gb:.3f} GB).")
        return train_device
    print(f"Using CPU-resident dataset ({estimated_gb:.3f} GB estimated; GPU limit {max_gb:.3f} GB).")
    return torch.device("cpu")


def train_from_config(config: dict) -> dict:
    data_cfg = section(config, "data")
    model_cfg = section(config, "model")
    train_cfg = section(config, "training")
    loss_cfg = section(config, "loss")
    out_cfg = section(config, "output")

    seed = int(config.get("seed", train_cfg.get("seed", 1234)))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = get_device(train_cfg.get("device", config.get("device", "cuda:0")))
    out_dir = ensure_dir(out_cfg.get("out_dir", "./vcn2_output"))
    data_dir = ensure_dir(out_dir / "data")
    model_dir = ensure_dir(out_dir / "models")

    lagged, train_df, val_df = load_or_build_lagged(data_cfg)
    if out_cfg.get("save_lagged_data", True):
        write_table(lagged, data_dir / "lagged_all.csv.gz")
        write_table(train_df, data_dir / "lagged_train.csv.gz")
        write_table(val_df, data_dir / "lagged_val.csv.gz")

    cvs = list(data_cfg["cvs"])
    norm_cfg = section(config, "normalization")
    stats = compute_normalizer_stats(
        train_df,
        cvs=cvs,
        method=norm_cfg.get("method", "standard"),
        eps=float(norm_cfg.get("eps", 1e-8)),
    )
    model = build_model(model_cfg, stats.loc, stats.scale).to(device)
    print(model)

    dataset_device = _resolve_dataset_device(train_cfg, len(train_df), len(val_df), len(cvs), device)
    train_set = TimeLaggedDataset(train_df, cvs=cvs, device=dataset_device)
    val_set = TimeLaggedDataset(val_df, cvs=cvs, device=dataset_device)
    if train_cfg.get("batch_size"):
        batch_size = int(train_cfg["batch_size"])
    else:
        batch_size = max(1, int(len(train_set) ** float(train_cfg.get("batch_size_factor", 0.6))))
        batch_size = max(batch_size, int(train_cfg.get("min_batch_size", 1)))
        if train_cfg.get("max_batch_size"):
            batch_size = min(batch_size, int(train_cfg["max_batch_size"]))
    pin_memory = bool(train_cfg.get("pin_memory", dataset_device.type == "cpu" and device.type == "cuda"))
    if pin_memory:
        train_set.pin_memory()
        val_set.pin_memory()
    val_batch_size = int(train_cfg.get("val_batch_size", max(batch_size, 8192)))
    print(
        f"Training samples: {len(train_set)} | Validation samples: {len(val_set)} | "
        f"Batch size: {batch_size} | Val batch size: {val_batch_size} | "
        f"Dataset device: {dataset_device} | Train device: {device}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(train_cfg.get("epochs", 500))
    patience = int(train_cfg.get("patience", 30))
    validation_interval = max(1, int(train_cfg.get("validation_interval", 1)))
    save_best_each_improvement = bool(train_cfg.get("save_best_each_improvement", False))
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0
    history_path = out_dir / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_jab",
                "train_endpoint",
                "train_grad_l2",
                "val_loss",
                "val_jab",
                "val_endpoint",
            ],
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_metrics = _epoch(model, train_set, batch_size, optimizer, loss_cfg, device, train=True)
            run_validation = epoch == 1 or epoch % validation_interval == 0 or epoch == epochs
            if run_validation:
                val_metrics = _epoch(model, val_set, val_batch_size, optimizer, loss_cfg, device, train=False)
            else:
                val_metrics = {"loss": best_val, "jab": np.nan, "endpoint": np.nan, "grad_l2": 0.0}
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_jab": train_metrics["jab"],
                "train_endpoint": train_metrics["endpoint"],
                "train_grad_l2": train_metrics["grad_l2"],
                "val_loss": val_metrics["loss"],
                "val_jab": val_metrics["jab"],
                "val_endpoint": val_metrics["endpoint"],
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch {epoch:5d} train {train_metrics['loss']:.6g} "
                f"val {val_metrics['loss']:.6g} jab {val_metrics['jab']:.6g}"
            )
            if run_validation and val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_state = deepcopy(model.state_dict())
                stale = 0
                if save_best_each_improvement:
                    model.load_state_dict(best_state)
                    scripted = torch.jit.script(model.eval().cpu())
                    scripted.save(str(model_dir / "best_model.pt"))
                    model.to(device)
            elif run_validation:
                stale += 1
                if stale >= patience:
                    print(f"Early stopping after {epoch} epochs.")
                    break

    model.load_state_dict(best_state)
    model.eval().cpu()
    scripted = torch.jit.script(model)
    scripted.save(str(model_dir / "best_model.pt"))
    torch.save({"model_state_dict": model.state_dict(), "best_val_loss": best_val}, model_dir / "best_model_state.pt")
    metadata = {
        "cvs": cvs,
        "normalization": {
            "method": norm_cfg.get("method", "standard"),
            "loc": stats.loc.tolist(),
            "scale": stats.scale.tolist(),
            "data_std": stats.data_std.tolist(),
            "data_min": stats.data_min.tolist(),
            "data_max": stats.data_max.tolist(),
            "input_clip": float(model_cfg.get("input_clip", 0.0)),
        },
        "model": model_cfg,
        "best_val_loss": best_val,
    }
    save_model_metadata(model_dir / "metadata.json", metadata)
    np.savetxt(model_dir / "input_loc.txt", stats.loc)
    np.savetxt(model_dir / "input_scale.txt", stats.scale)
    return {"model_path": str(model_dir / "best_model.pt"), "metadata": metadata, "out_dir": str(out_dir)}
