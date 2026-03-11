#!/usr/bin/env python3
"""
Train an ensemble of committor models from one cached dataset, then estimate uncertainty.

Workflow
--------
1. Build dataset once from traj_fns (and optional z-matrix path if enabled).
2. Save cached dataset + metadata to disk.
3. Train N ensemble members from the cached dataset.
4. Predict committor for all frames with all trained models.
5. Estimate uncertainty as std across ensemble predictions.
6. Project uncertainty onto 2 CVs and save CSV/PDF.

Recommended for large datasets because host-memory-heavy preprocessing happens only once.
"""

import os
import sys
import gc
import glob
import copy
import yaml
import torch
import random
import logging
import argparse
import traceback
import warnings
import numpy as np
import pandas as pd
import mdtraj as md
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from multiprocessing import get_context
from matplotlib import font_manager

from vcn.loss import loss_vcns_soft_endpoints
from vcn.main import CommittorDataset
from vcn.custom_dataloader import MyDataLoader
from vcn.train import train_model
from vcn.model import Encoder
from vcn.process_traj import preprocess_traj
from vcn.zmatrix import (
    get_internal_coordinates,
    get_pair_distances,
    get_minimal_internal_coordinates,
)

warnings.filterwarnings("ignore", category=UserWarning)

colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
          "#0072B2", "#D55E00", "#CC79A7", "#000000"]


# =========================================================
# Utility
# =========================================================

def cm2inch(value):
    return value / 2.54


def load_yaml_config(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def prepare_output_dir(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def ensure_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def resolve_path(path0: str, p: str) -> str:
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(path0, p)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(device_str: str) -> torch.device:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# =========================================================
# Logging
# =========================================================

def setup_logger(out_dir: str, log_name: str = "run.log", logger_name: str = "VCN"):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, log_name)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(stream=sys.__stdout__)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_path


@contextmanager
def redirect_prints_to_file(file_obj):
    with redirect_stdout(file_obj), redirect_stderr(file_obj):
        yield


# =========================================================
# CSV / DCD loading
# =========================================================

def load_one_csv(csv_path: str, stride=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if stride is not None:
        df = df[::int(stride)]
    return df.reset_index(drop=True)


def expand_dcd_paths(path0: str, dcd_entry):
    if dcd_entry is None:
        return []

    entry = resolve_path(path0, dcd_entry)

    if isinstance(entry, list):
        return sorted([resolve_path(path0, x) for x in entry])

    if "*" in entry or os.path.isdir(entry):
        if os.path.isdir(entry):
            paths = glob.glob(os.path.join(entry, "*.dcd"))
        else:
            paths = glob.glob(entry)
        return sorted(paths)

    return [entry]

def add_periodic_features(traj: pd.DataFrame, cvs):
    """
    For each periodic CV in degrees, add sin/cos columns:
        s<cv> = sin(cv * pi/180)
        c<cv> = cos(cv * pi/180)
    """
    for cv in cvs:
        if cv not in traj.columns:
            raise KeyError(f"Periodic CV '{cv}' not found in dataframe columns.")
        values = traj[cv].to_numpy(dtype=np.float32, copy=False)
        radians = values * np.pi / 180.0
        traj["s" + cv] = np.sin(radians).astype(np.float32, copy=False)
        traj["c" + cv] = np.cos(radians).astype(np.float32, copy=False)
    return traj

def _compute_internal_from_chunk(traj_chunk, atomindex_1based, use_all: bool, pair_distance: bool):
    if use_all:
        labels, values = get_internal_coordinates(traj_chunk, atomindex_1based)
    elif pair_distance:
        labels, values = get_pair_distances(traj_chunk, atomindex_1based)
    else:
        labels, values = get_minimal_internal_coordinates(traj_chunk, atomindex_1based)
    return labels, values



def convert_to_zmatrix_iterload(
    dcd_paths,
    top_path: str,
    atomselect,
    atomindex,
    use_all: bool,
    pair_distance: bool,
    stride,
    chunk_size: int,
    logger,
    log_every: int,
):
    top_path = os.path.abspath(top_path)
    dcd_paths = [os.path.abspath(p) for p in dcd_paths]

    if atomselect is not None:
        top = md.load_topology(top_path)
        atomindex_1based = top.select(atomselect) + 1
        logger.info(f"Atom selection '{atomselect}' -> {len(atomindex_1based)} atoms (1-based).")
    else:
        if not atomindex:
            raise ValueError("Either atomselect or atomindex must be provided.")
        atomindex_1based = np.array(atomindex, dtype=int)
        logger.info(f"Using provided atomindex: {len(atomindex_1based)} atoms (1-based).")

    all_values = []
    labels_ref = None
    total_frames = 0

    stride_i = int(stride) if stride else 1
    chunk_size = int(chunk_size)
    log_every = max(int(log_every), 1)

    logger.info(f"Start streaming DCD -> Z-matrix. chunk_size={chunk_size}, stride={stride_i}")

    for dcd in dcd_paths:
        logger.info(f"Reading DCD: {dcd}")
        iterator = md.iterload(dcd, top=top_path, chunk=chunk_size)
        chunk_idx = 0

        for traj_chunk in iterator:
            if stride_i != 1:
                traj_chunk = traj_chunk[::stride_i]
            if traj_chunk.n_frames == 0:
                continue

            labels, values = _compute_internal_from_chunk(
                traj_chunk, atomindex_1based, use_all, pair_distance
            )

            if labels_ref is None:
                labels_ref = labels
                logger.info(f"Z-matrix feature dim = {len(labels_ref)}")

            if len(labels) != len(labels_ref):
                raise RuntimeError(
                    f"Inconsistent feature dim across chunks: {len(labels)} vs {len(labels_ref)}"
                )

            all_values.append(values.astype(np.float32, copy=False))
            total_frames += traj_chunk.n_frames
            chunk_idx += 1

            if chunk_idx == 1 or (chunk_idx % log_every) == 0:
                logger.info(f"  processed chunks={chunk_idx}, total_frames={total_frames}")

        logger.info(f"Finished file: {dcd}")

    if labels_ref is None:
        raise RuntimeError("No frames were read from DCD(s).")

    values_full = np.vstack(all_values).astype(np.float32, copy=False)
    z_data = pd.DataFrame(values_full, columns=labels_ref, dtype=np.float32)

    logger.info(f"Finished Z-matrix. total_frames={total_frames}, dim={len(labels_ref)}")
    return z_data, labels_ref, total_frames


# =========================================================
# Build dataset once
# =========================================================

def build_full_dataframe_and_training_cvs(config: dict, logger=None):
    path0 = config.get("Sampling_path", "./")
    use_z_matrix = config.get("z_matrix", False)
    use_all = config.get("use_all", False)
    pair_distance = config.get("pair_distance", False)

    dcdfile = config.get("dcdfile", None)
    topfile = config.get("topfile", None)
    atomindex = config.get("atomindex", [])
    atomselect = config.get("atomselect", None)

    traj_fns = config.get("traj_fns", None)
    stride = config.get("stride", None)

    cvs = config.get("cvs", [])
    periodic = config.get("periodic", False)

    dcd_chunk_size = int(config.get("dcd_chunk_size", 2000))
    zmatrix_log_every = int(config.get("zmatrix_log_every", 10))

    traj_fns_list = ensure_list(traj_fns)
    dcd_list = ensure_list(dcdfile)

    if len(traj_fns_list) == 0:
        raise ValueError("traj_fns is empty.")
    if use_z_matrix and len(dcd_list) == 0:
        raise ValueError("z_matrix=True but dcdfile is empty.")
    if use_z_matrix and len(traj_fns_list) != len(dcd_list):
        raise ValueError("traj_fns and dcdfile must have the same length when z_matrix=True.")

    top_path = resolve_path(path0, topfile) if topfile is not None else None
    if use_z_matrix and (top_path is None or not os.path.isfile(top_path)):
        raise FileNotFoundError(f"Top file not found: {top_path}")

    original_cvs = list(cvs)
    training_cvs = list(cvs)

    pair_dfs = []
    labels_ref = None

    for i, csv_fn in enumerate(traj_fns_list):
        csv_path = resolve_path(path0, csv_fn)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[PAIR {i}] CSV not found: {csv_path}")

        this_traj = load_one_csv(csv_path, stride=stride)

        if use_z_matrix:
            dcd_entry = dcd_list[i]
            dcd_paths = expand_dcd_paths(path0, dcd_entry)
            if len(dcd_paths) == 0:
                raise FileNotFoundError(f"[PAIR {i}] No DCD files resolved from entry: {dcd_entry}")

            z_data, labels, _ = convert_to_zmatrix_iterload(
                dcd_paths=dcd_paths,
                top_path=top_path,
                atomselect=atomselect,
                atomindex=atomindex,
                use_all=use_all,
                pair_distance=pair_distance,
                stride=stride,
                chunk_size=dcd_chunk_size,
                logger=logger,
                log_every=zmatrix_log_every,
            )

            if len(this_traj) != len(z_data):
                raise RuntimeError(
                    f"[PAIR {i}] Frame mismatch: CSV rows={len(this_traj)} vs Z-matrix frames={len(z_data)}"
                )

            if labels_ref is None:
                labels_ref = labels
            else:
                if list(labels) != list(labels_ref):
                    raise RuntimeError(f"[PAIR {i}] Z-matrix labels inconsistent across trajectories.")

            this_traj = this_traj.join(z_data)
            original_cvs = list(labels_ref)
            training_cvs = list(labels_ref)

        this_traj["traj_id"] = i
        this_traj["source_csv"] = os.path.abspath(csv_path)
        pair_dfs.append(this_traj)

    traj = pd.concat(pair_dfs, ignore_index=True)

    # Cast original CVs / z-matrix features to float32
    for cv in original_cvs:
        if cv not in traj.columns:
            raise KeyError(f"CV '{cv}' not found in dataframe columns.")
        traj[cv] = traj[cv].astype(np.float32, copy=False)

    # Add sin/cos features for periodic CVs
    if periodic:
        traj = add_periodic_features(traj, original_cvs)
        training_cvs = ["s" + cv for cv in original_cvs] + ["c" + cv for cv in original_cvs]
    else:
        training_cvs = list(original_cvs)

    if logger is not None:
        logger.info(f"Final dataset rows: {len(traj)}")
        logger.info(f"Original CVs: {original_cvs}")
        logger.info(f"Training CVs: {training_cvs}")

    return traj, training_cvs

def prepare_cached_dataset(config: dict):
    out_dir = prepare_output_dir(config.get("out_dir", "./output_ensemble"))
    cache_dir = prepare_output_dir(os.path.join(out_dir, "dataset_cache"))
    logger, _ = setup_logger(cache_dir, "dataset_cache.log", "VCN_DATASET_CACHE")

    logger.info("Building dataset once and caching it...")
    traj, training_cvs = build_full_dataframe_and_training_cvs(config, logger=logger)

    dataset_path = os.path.join(cache_dir, "full_dataset.pkl")
    traj.to_pickle(dataset_path)

    meta = {
        "dataset_path": os.path.abspath(dataset_path),
        "training_cvs": training_cvs,
        "n_frames": int(len(traj)),
        "columns": list(traj.columns),
    }
    meta_path = os.path.join(cache_dir, "dataset_meta.yaml")
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    logger.info(f"Cached dataset: {dataset_path}")
    logger.info(f"Metadata file: {meta_path}")
    return dataset_path, training_cvs


def load_cached_dataset(config: dict):
    dataset_path = config.get("cached_dataset_path", None)
    training_cvs = config.get("cached_training_cvs", None)

    if dataset_path is None or training_cvs is None:
        raise ValueError("cached_dataset_path and cached_training_cvs are required.")

    traj = pd.read_pickle(dataset_path)

    for cv in training_cvs:
        traj[cv] = traj[cv].astype(np.float32, copy=False)

    return traj, training_cvs


# =========================================================
# Model helpers
# =========================================================

def build_encoder(num_input_features: int, num_layers: int, num_nodes: int, device):
    model = Encoder(num_input_features=num_input_features)
    model.build(
        [num_nodes for _ in range(num_layers)] + [1],
        [nn.ELU() for _ in range(num_layers)] + [nn.Identity()],
    )
    model.to(device)
    return model


def calc_committors_sig_batched(model, positions, periodic=False, device="cpu", batch_size=16384):
    preds = []
    model.eval()

    with torch.no_grad():
        n = len(positions)
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            chunk = positions[start:stop]

            input_tensor = torch.tensor(chunk, dtype=torch.float32, device=device)
            output_tensor = model(input_tensor)
            preds.append(output_tensor.detach().cpu().numpy().reshape(-1))

    return np.concatenate(preds, axis=0)
    

# =========================================================
# Training
# =========================================================

def train_one_ensemble_member(member_idx: int, config: dict):
    seed_base = int(config.get("seed_base", 12345))
    seed = seed_base + member_idx

    out_dir_root = prepare_output_dir(config.get("out_dir", "./output_ensemble"))
    member_out_dir = os.path.join(out_dir_root, f"ensemble_{member_idx:02d}")
    prepare_output_dir(member_out_dir)

    available_devices = ensure_list(config.get("ensemble_devices", [f"cuda:{i}" for i in range(8)]))
    device_str = available_devices[member_idx % len(available_devices)]

    label = config.get("label", "committor")
    extra_label = config.get("extra_label", "")
    patience = config.get("patience", 20)
    val_ratio = config.get("val_ratio", 0.1)
    epochs = config.get("epochs", 500)
    num_layers = config.get("num_layers", 1)
    num_nodes = config.get("num_nodes", 32)
    batch_size_factor = config.get("batch_size_factor", 1.0)
    k_scale = config.get("k", 1000.0)
    checkpoint_path = config.get("checkpoint_path", None)

    logger, log_path = setup_logger(
        member_out_dir,
        log_name=f"{label}_member_{member_idx:02d}.log",
        logger_name=f"VCN_MEMBER_{member_idx:02d}"
    )

    logger.info(f"=== Start member {member_idx:02d} ===")
    logger.info(f"seed={seed}")
    logger.info(f"device={device_str}")

    set_global_seed(seed)
    device = setup_device(device_str)

    traj, training_cvs = load_cached_dataset(config)
    logger.info(f"Loaded cached dataset with {len(traj)} rows")

    checkpoint = torch.load(checkpoint_path, map_location=device) if checkpoint_path is not None else None

    train_val_data, train_data, val_data = preprocess_traj(
        data=traj, val_ratio=val_ratio, time_shift=1
    )

    # for cv in training_cvs:
    #     train_data[cv] = train_data[cv].astype(np.float32, copy=False)
    #     val_data[cv] = val_data[cv].astype(np.float32, copy=False)

    train_set = CommittorDataset(data=train_data, variables=training_cvs, device=device)
    val_set = CommittorDataset(data=val_data, variables=training_cvs, device=device)

    del traj
    del train_val_data
    gc.collect()

    label_suffix = f"{label}{extra_label}_member{member_idx:02d}_seed{seed}_patience{patience}"
    model_name = os.path.join(member_out_dir, label_suffix)

    model = build_encoder(
        num_input_features=len(training_cvs),
        num_layers=num_layers,
        num_nodes=num_nodes,
        device=device,
    )

    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Val size: {len(val_data)}")
    logger.info("Start training...")

    with open(log_path, "a", buffering=1) as f:
        with redirect_prints_to_file(f):
            train_model(
                model_to_train=model,
                output_prefix=model_name,
                train_set=train_set,
                val_set=val_set,
                loss_function=loss_vcns_soft_endpoints,
                epochs=epochs,
                patience=patience,
                batch_size_factor=batch_size_factor,
                dataloader=MyDataLoader,
                old_checkpoint=checkpoint,
                k_scale=k_scale,
            )

    logger.info("Training finished.")

    best_model_path = f"{model_name}_best_model.pt"
    if not os.path.isfile(best_model_path):
        candidates = sorted(glob.glob(os.path.join(member_out_dir, "*_best_model.pt")))
        if len(candidates) == 0:
            raise FileNotFoundError(f"Could not find best model for member {member_idx:02d}")
        best_model_path = candidates[-1]

    best_model = torch.jit.load(best_model_path, map_location="cpu")
    cpu_model_path = os.path.join(member_out_dir, f"ensemble_{member_idx:02d}_cpu_best_model.pt")
    best_model.save(cpu_model_path)

    meta = {
        "member_idx": member_idx,
        "seed": seed,
        "device": device_str,
        "best_model_path": os.path.abspath(best_model_path),
        "cpu_model_path": os.path.abspath(cpu_model_path),
        "training_cvs": training_cvs,
    }
    with open(os.path.join(member_out_dir, "member_metadata.yaml"), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    logger.info(f"=== Finished member {member_idx:02d} ===")
    return meta


def _worker_train(member_idx: int, config: dict):
    try:
        return train_one_ensemble_member(member_idx, config)
    except Exception as e:
        traceback.print_exc()
        return {
            "member_idx": member_idx,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def train_ensemble(config: dict):
    n_models = int(config.get("n_ensemble", 8))
    out_dir = prepare_output_dir(config.get("out_dir", "./output_ensemble"))
    logger, _ = setup_logger(out_dir, "ensemble_master.log", "VCN_ENSEMBLE_MASTER")

    dataset_path, training_cvs = prepare_cached_dataset(config)

    config = copy.deepcopy(config)
    config["cached_dataset_path"] = dataset_path
    config["cached_training_cvs"] = training_cvs

    max_parallel = int(config.get("max_parallel_jobs", 1))
    logger.info(f"n_ensemble={n_models}")
    logger.info(f"max_parallel_jobs={max_parallel}")

    ctx = get_context("spawn")
    with ctx.Pool(processes=min(n_models, max_parallel)) as pool:
        results = pool.starmap(_worker_train, [(i, config) for i in range(n_models)])

    summary_path = os.path.join(out_dir, "ensemble_training_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    n_failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
    logger.info(f"Training done. failed={n_failed}/{n_models}")

    if n_failed > 0:
        raise RuntimeError(f"{n_failed} ensemble members failed. Check {summary_path}")

    return config, results


# =========================================================
# Plotting
# =========================================================

def load_matplotlib_local_fonts(config):
    plt.rcParams.update({
        "axes.labelsize": config.get("label_font_size", 7),
        "axes.linewidth": config.get("line_width", 0.4),
        "font.size": config.get("font_size", 7),
        "axes.unicode_minus": False,
    })
    font_path = config.get("font_path", "")
    try:
        if font_path and os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            matplotlib.rc("font", family="sans-serif")
            matplotlib.rcParams.update({"font.sans-serif": prop.get_name()})
    except Exception as e:
        print(f"Error loading fonts: {e}")


def project_scalar_to_2d(x, y, values, xedges, yedges):
    denominator, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    nominator, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=values)
    avg = nominator / denominator.clip(1)
    avg[denominator == 0] = np.nan
    return avg, denominator


def plot_scalar_projection_with_string(
    x,
    y,
    scalar_values,
    save_figure=None,
    plot_config=None,
    colorbar_title=r"$\sigma_q$",
    cmap_name="viridis",
    periodic_lines=False,
):
    if plot_config is None:
        plot_config = {}

    side = cm2inch(plot_config.get("figure_width_cm", 3.5))
    fig = plt.figure(figsize=(side, side / plot_config.get("aspect_ratio", 1.0)))

    x_lim = plot_config.get("x_lim", [np.min(x), np.max(x)])
    y_lim = plot_config.get("y_lim", [np.min(y), np.max(y)])
    n_bins = plot_config.get("bins", 50)

    xedges = np.linspace(x_lim[0], x_lim[1], n_bins + 1)
    yedges = np.linspace(y_lim[0], y_lim[1], n_bins + 1)

    avg_scalar, counts = project_scalar_to_2d(x, y, scalar_values, xedges, yedges)
    cf = plt.pcolormesh(xedges, yedges, avg_scalar.T, cmap=matplotlib.colormaps.get_cmap(cmap_name).copy())

    strings = plot_config.get("strings_fn", [])
    p1 = []

    if len(strings) > 0:
        plot_indices = [plot_config["cvs"].index(cv) for cv in plot_config["plot_cvs"]]
        for string, color in zip(strings, colors):
            try:
                string_data = np.loadtxt(string)
                x_str = string_data.T[plot_indices[0]]
                y_str = string_data.T[plot_indices[1]]

                if periodic_lines:
                    dx = np.abs(np.diff(x_str))
                    dy = np.abs(np.diff(y_str))
                    split_indices = np.where((dx > 180) | (dy > 180))[0] + 1
                    segments = np.split(np.arange(len(x_str)), split_indices)
                    first_handle = None
                    for seg in segments:
                        if len(seg) < 2:
                            continue
                        h, = plt.plot(x_str[seg], y_str[seg], color=color)
                        if first_handle is None:
                            first_handle = h
                    if first_handle is not None:
                        p1.append(first_handle)
                else:
                    h, = plt.plot(x_str, y_str, color=color)
                    p1.append(h)

            except Exception as e:
                print(f"Could not load or plot string from {string}: {e}")

    ax = plt.gca()
    plt.xlabel(plot_config.get("x_label", "CV$_\\mathrm{1}$"))
    plt.ylabel(plot_config.get("y_label", "CV$_\\mathrm{2}$"))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if p1 and plot_config.get("legend", True):
        legend_labels = [f"Path {i+1}" for i in range(len(p1))]
        ax.legend(p1, legend_labels, loc="best", framealpha=1.0, fancybox=False, edgecolor="w")

    lw = plot_config.get("line_width", 0.4)
    ax.xaxis.set_tick_params(width=lw)
    ax.yaxis.set_tick_params(width=lw)

    clb = plt.colorbar(
        cf,
        orientation=plot_config.get("colorbar_orientation", "vertical"),
        location=plot_config.get("colorbar_location", "right"),
    )
    clb.ax.set_title(colorbar_title, pad=5)
    clb.ax.tick_params(width=lw)

    if save_figure is not None:
        plt.savefig(save_figure, dpi=300, bbox_inches="tight", transparent=False)

    plt.show()
    plt.close(fig)

    return avg_scalar, counts, xedges, yedges


# =========================================================
# Uncertainty analysis
# =========================================================

def find_trained_model_paths(config: dict):
    out_dir = config.get("out_dir", "./output_ensemble")
    n_models = int(config.get("n_ensemble", 8))
    model_paths = []

    for i in range(n_models):
        member_dir = os.path.join(out_dir, f"ensemble_{i:02d}")
        cpu_model_path = os.path.join(member_dir, f"ensemble_{i:02d}_cpu_best_model.pt")
        if not os.path.isfile(cpu_model_path):
            raise FileNotFoundError(f"Missing CPU model: {cpu_model_path}")
        model_paths.append(cpu_model_path)

    return model_paths


# def analyze_uncertainty(config: dict):
#     out_dir = prepare_output_dir(config.get("out_dir", "./output_ensemble"))
#     analysis_dir = prepare_output_dir(os.path.join(out_dir, "uncertainty_analysis"))
#     logger, _ = setup_logger(analysis_dir, "uncertainty_analysis.log", "VCN_UNCERTAINTY")

#     traj, training_cvs = load_cached_dataset(config)
#     logger.info(f"Loaded cached dataset for analysis: {len(traj)} rows")

#     model_paths = find_trained_model_paths(config)
#     periodic = bool(config.get("periodic", False))
#     inference_device = config.get("inference_device", "cpu")
#     if inference_device.startswith("cuda") and not torch.cuda.is_available():
#         inference_device = "cpu"
#     batch_size = int(config.get("inference_batch_size", 16384))

#     positions = np.asarray(traj[training_cvs], dtype=np.float32)
#     ensemble_predictions = []

#     for i, model_path in enumerate(model_paths):
#         logger.info(f"Loading model {i}: {model_path}")
#         model = torch.jit.load(model_path, map_location=inference_device)
#         preds = calc_committors_sig_batched(
#             model=model,
#             positions=positions,
#             periodic=False,
#             device=inference_device,
#             batch_size=batch_size,
#         )
#         ensemble_predictions.append(preds)

#     ensemble_predictions = np.vstack(ensemble_predictions)
#     q_mean = np.mean(ensemble_predictions, axis=0)
#     q_std = np.std(ensemble_predictions, axis=0, ddof=0)

#     plot_cvs = config.get("uncertainty_plot_cvs", config.get("plot_cvs", None))
#     if plot_cvs is None or len(plot_cvs) != 2:
#         raise ValueError("Please provide uncertainty_plot_cvs or plot_cvs with exactly 2 CV names.")

#     keep_cols = [c for c in traj.columns if c in set(plot_cvs + training_cvs + ["traj_id", "source_csv"])]
#     frame_df = traj[keep_cols].copy()

#     for i in range(ensemble_predictions.shape[0]):
#         frame_df[f"q_model_{i:02d}"] = ensemble_predictions[i]
#     frame_df["q_mean"] = q_mean
#     frame_df["q_std"] = q_std

#     frame_csv = os.path.join(analysis_dir, "per_frame_committor_uncertainty.csv")
#     frame_df.to_csv(frame_csv, index=False)
#     logger.info(f"Saved per-frame CSV: {frame_csv}")

#     load_matplotlib_local_fonts(config)

#     x = traj[plot_cvs[0]].to_numpy()
#     y = traj[plot_cvs[1]].to_numpy()

#     plot_config = copy.deepcopy(config)
#     plot_config["plot_cvs"] = plot_cvs

#     pdf_name = config.get("uncertainty_pdf_name", f"uncertainty_{plot_cvs[0]}_{plot_cvs[1]}.pdf")
#     pdf_path = os.path.join(analysis_dir, pdf_name)

#     avg_sigma, counts, xedges, yedges = plot_scalar_projection_with_string(
#         x=x,
#         y=y,
#         scalar_values=q_std,
#         save_figure=pdf_path,
#         plot_config=plot_config,
#         colorbar_title=config.get("uncertainty_colorbar_title", r"$\sigma_q$"),
#         cmap_name=config.get("uncertainty_cmap", "viridis"),
#         periodic_lines=bool(config.get("periodic", False)),
#     )
#     logger.info(f"Saved PDF: {pdf_path}")

#     xcenters = 0.5 * (xedges[:-1] + xedges[1:])
#     ycenters = 0.5 * (yedges[:-1] + yedges[1:])
#     xx, yy = np.meshgrid(xcenters, ycenters, indexing="ij")

#     grid_df = pd.DataFrame({
#         plot_cvs[0]: xx.ravel(),
#         plot_cvs[1]: yy.ravel(),
#         "uncertainty_mean": avg_sigma.ravel(),
#         "counts": counts.ravel(),
#     })
#     grid_csv = os.path.join(analysis_dir, f"uncertainty_projection_{plot_cvs[0]}_{plot_cvs[1]}.csv")
#     grid_df.to_csv(grid_csv, index=False)
#     logger.info(f"Saved projected grid CSV: {grid_csv}")
def analyze_uncertainty(config: dict):
    out_dir = prepare_output_dir(config.get("out_dir", "./output_ensemble"))
    analysis_dir = prepare_output_dir(os.path.join(out_dir, "uncertainty_analysis"))
    logger, _ = setup_logger(analysis_dir, "uncertainty_analysis.log", "VCN_UNCERTAINTY")

    traj, training_cvs = load_cached_dataset(config)
    logger.info(f"Loaded cached dataset for analysis: {len(traj)} rows")

    model_paths = find_trained_model_paths(config)
    periodic = bool(config.get("periodic", False))
    inference_device = config.get("inference_device", "cpu")
    if inference_device.startswith("cuda") and not torch.cuda.is_available():
        inference_device = "cpu"
    batch_size = int(config.get("inference_batch_size", 16384))

    positions = np.asarray(traj[training_cvs], dtype=np.float32)
    ensemble_predictions = []

    for i, model_path in enumerate(model_paths):
        logger.info(f"Loading model {i}: {model_path}")
        model = torch.jit.load(model_path, map_location=inference_device)
        preds = calc_committors_sig_batched(
            model=model,
            positions=positions,
            periodic=False,
            device=inference_device,
            batch_size=batch_size,
        )
        ensemble_predictions.append(preds)

    ensemble_predictions = np.vstack(ensemble_predictions)
    q_mean = 1 - np.mean(ensemble_predictions, axis=0)
    q_std = np.std(ensemble_predictions, axis=0, ddof=0)

    plot_cvs = config.get("uncertainty_plot_cvs", config.get("plot_cvs", None))
    if plot_cvs is None or len(plot_cvs) != 2:
        raise ValueError("Please provide uncertainty_plot_cvs or plot_cvs with exactly 2 CV names.")

    keep_cols = [c for c in traj.columns if c in set(plot_cvs + training_cvs + ["traj_id", "source_csv"])]
    frame_df = traj[keep_cols].copy()

    for i in range(ensemble_predictions.shape[0]):
        frame_df[f"q_model_{i:02d}"] = ensemble_predictions[i]
    frame_df["q_mean"] = q_mean
    frame_df["q_std"] = q_std

    frame_csv = os.path.join(analysis_dir, "per_frame_committor_uncertainty.csv")
    frame_df.to_csv(frame_csv, index=False)
    logger.info(f"Saved per-frame CSV: {frame_csv}")

    load_matplotlib_local_fonts(config)

    x = traj[plot_cvs[0]].to_numpy()
    y = traj[plot_cvs[1]].to_numpy()

    plot_config = copy.deepcopy(config)
    plot_config["plot_cvs"] = plot_cvs

    # --- plot uncertainty ---
    sigma_pdf_name = config.get(
        "uncertainty_pdf_name",
        f"uncertainty_{plot_cvs[0]}_{plot_cvs[1]}.pdf"
    )
    sigma_pdf_path = os.path.join(analysis_dir, sigma_pdf_name)

    avg_sigma, counts_sigma, xedges_sigma, yedges_sigma = plot_scalar_projection_with_string(
        x=x,
        y=y,
        scalar_values=q_std,
        save_figure=sigma_pdf_path,
        plot_config=plot_config,
        colorbar_title=config.get("uncertainty_colorbar_title", r"$\sigma_q$"),
        cmap_name=config.get("uncertainty_cmap", "viridis"),
        periodic_lines=periodic,
    )
    logger.info(f"Saved uncertainty PDF: {sigma_pdf_path}")

    # --- plot ensemble mean committor ---
    mean_pdf_name = config.get(
        "committor_mean_pdf_name",
        f"committor_mean_{plot_cvs[0]}_{plot_cvs[1]}.pdf"
    )
    mean_pdf_path = os.path.join(analysis_dir, mean_pdf_name)

    avg_qmean, counts_qmean, xedges_qmean, yedges_qmean = plot_scalar_projection_with_string(
        x=x,
        y=y,
        scalar_values=q_mean,
        save_figure=mean_pdf_path,
        plot_config=plot_config,
        colorbar_title=config.get("committor_mean_colorbar_title", r"${q}$"),
        cmap_name=config.get("committor_mean_cmap", "RdBu_r"),
        periodic_lines=periodic,
    )
    logger.info(f"Saved ensemble-mean committor PDF: {mean_pdf_path}")

    # --- save projected uncertainty grid ---
    xcenters_sigma = 0.5 * (xedges_sigma[:-1] + xedges_sigma[1:])
    ycenters_sigma = 0.5 * (yedges_sigma[:-1] + yedges_sigma[1:])
    xx_sigma, yy_sigma = np.meshgrid(xcenters_sigma, ycenters_sigma, indexing="ij")

    sigma_grid_df = pd.DataFrame({
        plot_cvs[0]: xx_sigma.ravel(),
        plot_cvs[1]: yy_sigma.ravel(),
        "uncertainty_mean": avg_sigma.ravel(),
        "counts": counts_sigma.ravel(),
    })
    sigma_grid_csv = os.path.join(
        analysis_dir,
        f"uncertainty_projection_{plot_cvs[0]}_{plot_cvs[1]}.csv"
    )
    sigma_grid_df.to_csv(sigma_grid_csv, index=False)
    logger.info(f"Saved projected uncertainty grid CSV: {sigma_grid_csv}")

    # --- save projected mean-committor grid ---
    xcenters_qmean = 0.5 * (xedges_qmean[:-1] + xedges_qmean[1:])
    ycenters_qmean = 0.5 * (yedges_qmean[:-1] + yedges_qmean[1:])
    xx_qmean, yy_qmean = np.meshgrid(xcenters_qmean, ycenters_qmean, indexing="ij")

    qmean_grid_df = pd.DataFrame({
        plot_cvs[0]: xx_qmean.ravel(),
        plot_cvs[1]: yy_qmean.ravel(),
        "committor_mean": avg_qmean.ravel(),
        "counts": counts_qmean.ravel(),
    })
    qmean_grid_csv = os.path.join(
        analysis_dir,
        f"committor_mean_projection_{plot_cvs[0]}_{plot_cvs[1]}.csv"
    )
    qmean_grid_df.to_csv(qmean_grid_csv, index=False)
    logger.info(f"Saved projected ensemble-mean committor grid CSV: {qmean_grid_csv}")

# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Cached-dataset ensemble committor training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default="all", choices=["cache", "train", "analyze", "all"])
    args = parser.parse_args()

    config_all = load_yaml_config(args.config)
    config = config_all["VCN"] if "VCN" in config_all else config_all

    if args.mode == "cache":
        prepare_cached_dataset(config)
        return

    if args.mode in ["train", "all"]:
        config, _ = train_ensemble(config)

    elif args.mode == "analyze":
        cache_meta_path = os.path.join(config.get("out_dir", "./output_ensemble"), "dataset_cache", "dataset_meta.yaml")
        with open(cache_meta_path, "r") as f:
            meta = yaml.safe_load(f)
        config = copy.deepcopy(config)
        config["cached_dataset_path"] = meta["dataset_path"]
        config["cached_training_cvs"] = meta["training_cvs"]

    if args.mode in ["analyze", "all"]:
        if "cached_dataset_path" not in config or "cached_training_cvs" not in config:
            cache_meta_path = os.path.join(config.get("out_dir", "./output_ensemble"), "dataset_cache", "dataset_meta.yaml")
            with open(cache_meta_path, "r") as f:
                meta = yaml.safe_load(f)
            config["cached_dataset_path"] = meta["dataset_path"]
            config["cached_training_cvs"] = meta["training_cvs"]

        analyze_uncertainty(config)


if __name__ == "__main__":
    main()