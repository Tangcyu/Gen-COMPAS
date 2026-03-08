import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import glob
import mdtraj as md
import torch.nn as nn
import argparse
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# === Import modules from your project ===
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

# =========================================================
# === Utility functions
# =========================================================

def load_yaml_config(file_path: str) -> dict:
    """Load YAML configuration file safely."""
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Config file not found at {file_path}")
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Error reading YAML file: {exc}")


def setup_device(device_str: str) -> torch.device:
    """Initialize torch device."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def prepare_output_dir(out_dir: str) -> str:
    """Create output directory if missing."""
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def ensure_list(x):
    """Ensure config entries are lists."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def resolve_path(path0: str, p: str) -> str:
    """Resolve a path relative to path0 unless it is absolute."""
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(path0, p)


# =========================================================
# === Logging utilities (console + file)
# =========================================================

def setup_logger(out_dir: str, log_name: str = "run.log"):
    """Create a logger that writes to both console and a log file."""
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, log_name)

    logger = logging.getLogger("VCN")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler(stream=sys.__stdout__)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_path


@contextmanager
def redirect_prints_to_file(file_obj):
    """
    Redirect stdout/stderr into a real file object.
    This avoids custom stdout objects (Torch Dynamo may call sys.stdout.isatty()).
    """
    with redirect_stdout(file_obj), redirect_stderr(file_obj):
        yield


# =========================================================
# === CSV trajectory loading
# =========================================================

def load_one_csv(csv_path: str, stride) -> pd.DataFrame:
    """Load a single CSV and apply stride if requested."""
    df = pd.read_csv(csv_path)
    if stride is not None:
        df = df[::int(stride)]
    return df.reset_index(drop=True)


# =========================================================
# === DCD utilities
# =========================================================

def expand_dcd_paths(path0: str, dcd_entry):
    """
    Expand one dcd entry into a list of paths.
    - If entry is a direct file -> [file]
    - If entry contains '*' -> glob
    - If entry is a directory -> glob *.dcd
    """
    if dcd_entry is None:
        return []

    entry = resolve_path(path0, dcd_entry)

    if isinstance(entry, list):
        # Not expected here, but keep safe
        paths = [resolve_path(path0, x) for x in entry]
        return sorted(paths)

    if "*" in entry or os.path.isdir(entry):
        base = entry
        if os.path.isdir(base):
            paths = glob.glob(os.path.join(base, "*.dcd"))
        else:
            paths = glob.glob(base)
        return sorted(paths)

    return [entry]


# =========================================================
# === Z-matrix conversion (Scheme B: md.iterload streaming)
# =========================================================

def _compute_internal_from_chunk(traj_chunk, atomindex_1based, use_all: bool, pair_distance: bool):
    """Compute internal coordinates for one trajectory chunk."""
    if use_all:
        labels, values = get_internal_coordinates(traj_chunk, atomindex_1based)
    elif pair_distance:
        labels, values = get_pair_distances(traj_chunk, atomindex_1based)
    else:
        labels, values = get_minimal_internal_coordinates(traj_chunk, atomindex_1based)
    return labels, values  # values: (n_frames_chunk, n_features)


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
    """
    Stream DCD(s) chunk-by-chunk, compute Z-matrix features, and return a DataFrame.
    This avoids loading the entire trajectory into memory.
    """
    top_path = os.path.abspath(top_path)
    dcd_paths = [os.path.abspath(p) for p in dcd_paths]

    # Determine atom indices once (1-based indices expected by your zmatrix routines)
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
            # Manual stride (portable across mdtraj versions)
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

    values_full = np.vstack(all_values)  # (total_frames, n_features)
    z_data = pd.DataFrame(values_full, columns=labels_ref)

    logger.info(f"Finished Z-matrix. total_frames={total_frames}, dim={len(labels_ref)}")
    return z_data, labels_ref, total_frames


# =========================================================
# === Training pipeline
# =========================================================

def train_committor_model(config: dict):
    """Main training routine."""

    # --- Extract configuration values ---
    label = config.get("label", "default_label")
    extra_label = config.get("extra_label", None)
    path0 = config.get("Sampling_path", "./")
    out_dir = prepare_output_dir(config.get("out_dir", "./output/"))
    device = setup_device(config.get("device", "cuda:0"))

    # --- New config entries (recommended) ---
    # dcd_chunk_size: chunk size for md.iterload
    # zmatrix_log_every: log progress every N chunks
    dcd_chunk_size = int(config.get("dcd_chunk_size", 2000))
    zmatrix_log_every = int(config.get("zmatrix_log_every", 10))

    # --- Logger ---
    logger, log_path = setup_logger(out_dir, log_name=f"{label}_train.log")
    logger.info(f"Log file: {log_path}")

    # --- Feature / IO flags ---
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
    val_ratio = config.get("val_ratio", 0.1)

    # --- Training hyperparameters ---
    epochs = config.get("epochs", 500)
    patience = config.get("patience", 20)
    num_layers = config.get("num_layers", 1)
    num_nodes = config.get("num_nodes", 32)
    batch_size_factor = config.get("batch_size_factor", 1.0)
    k_scale = config.get("k", 1000.0)
    checkpoint_path = config.get("checkpoint_path", None)
    checkpoint = torch.load(checkpoint_path) if checkpoint_path is not None else None

    # --- Normalize list configs for 1-1 mapping ---
    traj_fns_list = ensure_list(traj_fns)
    dcd_list = ensure_list(dcdfile)

    if len(traj_fns_list) == 0:
        raise ValueError("traj_fns is empty. Please provide at least one CSV file.")
    if use_z_matrix and len(dcd_list) == 0:
        raise ValueError("z_matrix=True but dcdfile is empty. Please provide DCD file(s).")

    # Enforce 1-1 mapping when z_matrix is enabled
    if use_z_matrix and (len(traj_fns_list) != len(dcd_list)):
        raise ValueError(
            f"traj_fns and dcdfile must have the same length for 1-1 mapping when z_matrix=True; "
            f"got {len(traj_fns_list)} vs {len(dcd_list)}"
        )

    # Resolve topfile once
    top_path = resolve_path(path0, topfile) if topfile is not None else None
    if use_z_matrix and (top_path is None or not os.path.isfile(top_path)):
        raise FileNotFoundError(f"Top file not found: {top_path}")

    # Periodic CV expansion
    if periodic:
        cvs = ["s" + cv for cv in cvs] + ["c" + cv for cv in cvs]

    # --- Build dataset pair-by-pair, then concatenate ---
    pair_dfs = []
    labels_ref = None

    logger.info(f"Found CSV files: {[resolve_path(path0, x) for x in traj_fns_list]}")
    if use_z_matrix:
        logger.info(f"Found DCD entries: {[resolve_path(path0, x) for x in dcd_list]}")

    for i, csv_fn in enumerate(traj_fns_list):
        csv_path = resolve_path(path0, csv_fn)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[PAIR {i}] CSV not found: {csv_path}")

        logger.info(f"[PAIR {i}] CSV: {csv_path}")
        this_traj = load_one_csv(csv_path, stride=stride)

        if use_z_matrix:
            dcd_entry = dcd_list[i]
            dcd_paths = expand_dcd_paths(path0, dcd_entry)
            if len(dcd_paths) == 0:
                raise FileNotFoundError(f"[PAIR {i}] No DCD files resolved from entry: {dcd_entry}")

            logger.info(f"[PAIR {i}] DCD paths: {dcd_paths}")

            z_data, labels, n_frames = convert_to_zmatrix_iterload(
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

            # Per-pair alignment check (this is the key for 1-1 mapping)
            if len(this_traj) != len(z_data):
                raise RuntimeError(
                    f"[PAIR {i}] Frame mismatch: CSV rows={len(this_traj)} vs Z-matrix frames={len(z_data)}. "
                    f"CSV={csv_path}, DCD_entry={dcd_entry}"
                )

            # Enforce consistent Z-matrix feature labels across pairs
            if labels_ref is None:
                labels_ref = labels
            else:
                if list(labels) != list(labels_ref):
                    raise RuntimeError(
                        f"[PAIR {i}] Z-matrix labels inconsistent across trajectories."
                    )

            this_traj = this_traj.join(z_data)
            cvs = labels_ref  # Override variables with Z-matrix labels

        # Add pair id for debugging and downstream grouping
        this_traj["traj_id"] = i
        this_traj["source_csv"] = os.path.abspath(csv_path)
        if use_z_matrix:
            this_traj["source_dcd_entry"] = str(resolve_path(path0, dcd_list[i]))

        pair_dfs.append(this_traj)

    traj = pd.concat(pair_dfs, ignore_index=True)
    logger.info(f"Final concatenated dataset rows: {len(traj)}")

    # --- Prepare training and validation sets ---
    train_val_data, train_data, val_data = preprocess_traj(
        data=traj, val_ratio=val_ratio, time_shift=1
    )
    train_set = CommittorDataset(data=train_data, variables=cvs, device=device)
    val_set = CommittorDataset(data=val_data, variables=cvs, device=device)

    # --- Build model ---
    label_suffix = f"{label}{extra_label}_patience{patience}" if extra_label else f"{label}_patience{patience}"
    model_name = os.path.join(out_dir, label_suffix)

    model = Encoder(num_input_features=len(cvs))
    model.build(
        [num_nodes for _ in range(num_layers)] + [1],
        [nn.ELU() for _ in range(num_layers)] + [nn.Identity()],
    )
    model.to(device)

    # --- Train model ---
    logger.info("Start training...")

    # Redirect training prints into the log file (safe for Torch; file object supports isatty())
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

    # --- Save CPU copy ---
    best_model = torch.jit.load(os.path.join(out_dir, f"{label_suffix}_best_model.pt"))
    cpu_model_path = os.path.join(out_dir, f"{label_suffix}_cpu_best_model.pt")
    best_model.to("cpu").save(cpu_model_path)
    logger.info(f"Saved CPU model at {cpu_model_path}")


# =========================================================
# === Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Train VCN model with config file")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        print(f"Error: Config file {config_path} does not exist.")
        sys.exit(1)

    config = load_yaml_config(config_path)
    if "VCN" not in config:
        raise KeyError("Top-level key 'VCN' not found in YAML config.")

    train_committor_model(config["VCN"])


if __name__ == "__main__":
    main()
