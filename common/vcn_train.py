import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import glob
import mdtraj as md
import torch.nn as nn
from tqdm import tqdm
import argparse


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
# === Utility functions ===
# =========================================================

def load_yaml_config(file_path):
    """Load YAML configuration file safely."""
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Config file not found at {file_path}")
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Error reading YAML file: {exc}")


def setup_device(device_str):
    """Initialize the torch device."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def prepare_output_dir(out_dir):
    """Create output directory if missing."""
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# =========================================================
# === Trajectory Loading ===
# =========================================================

def load_csv_trajectories(path0, label, traj_fns, stride):
    """Load CSV trajectory files."""
    traj_fns = [os.path.join(path0, fn) for fn in traj_fns]
    print(f"Found CSV files: {traj_fns}")

    if len(traj_fns) == 1:
        traj = pd.read_csv(traj_fns[0])
    else:
        traj = pd.concat([pd.read_csv(fn) for fn in traj_fns], ignore_index=True)

    if stride is not None:
        traj = traj[::int(stride)]

    return traj


def load_dcd_trajectories(path0, dcdfile, topfile, stride):
    """Load and concatenate DCD trajectories."""
    dcd_paths = []

    if isinstance(dcdfile, list):
        dcd_paths = [os.path.join(path0, fn) for fn in dcdfile]
    elif isinstance(dcdfile, str):
        if "*" in dcdfile or os.path.isdir(os.path.join(path0, dcdfile)):
            base_path = os.path.join(path0, dcdfile)
            dcd_paths = (
                glob.glob(os.path.join(base_path, "*.dcd"))
                if os.path.isdir(base_path)
                else glob.glob(base_path)
            )
        else:
            dcd_paths = [os.path.join(path0, dcdfile)]
    else:
        raise ValueError("dcdfile must be a string, list, or directory path.")

    if len(dcd_paths) == 0:
        raise FileNotFoundError("No DCD files found!")

    print(f"Found DCD files: {dcd_paths}")
    loaded_trajs = [
        md.load(dcd, stride=int(stride) if stride else 1, top=os.path.join(path0, topfile))
        for dcd in sorted(dcd_paths)
    ]

    dcdtraj = loaded_trajs[0].join(loaded_trajs[1:]) if len(loaded_trajs) > 1 else loaded_trajs[0]
    print(f"Total frames loaded from DCD files: {dcdtraj.n_frames}")
    return dcdtraj


def convert_to_zmatrix(dcdtraj, atomselect, atomindex, topfile, path0, use_all, pair_distance):
    """Convert DCD trajectory to internal (Z-matrix) coordinates."""
    if atomselect is not None:
        atomindex = dcdtraj.topology.select(atomselect) + 1
        print(f"Selected atom indices: {atomindex}")
    elif not atomindex:
        raise ValueError("Either atomselect or atomindex must be provided.")

    print("Converting to Z-matrix...")
    if use_all:
        labels, values = get_internal_coordinates(dcdtraj, atomindex)
    elif pair_distance:
        labels, values = get_pair_distances(dcdtraj, atomindex)
    else:
        labels, values = get_minimal_internal_coordinates(dcdtraj, atomindex)

    z_data = pd.DataFrame({label: value for label, value in zip(labels, values.T)})
    print("DCD trajectory converted to Z-matrix.")
    return z_data, labels


# =========================================================
# === Training pipeline ===
# =========================================================

def train_committor_model(config):
    """Main training routine."""

    # --- Extract configuration values ---
    label = config.get("label", "default_label")
    extra_label = config.get("extra_label", None)
    path0 = config.get("Sampling_path", "./")
    out_dir = prepare_output_dir(config.get("out_dir", "./output/"))
    device = setup_device(config.get("device", "cuda:0"))

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

    epochs = config.get("epochs", 500)
    patience = config.get("patience", 20)
    num_layers = config.get("num_layers", 1)
    num_nodes = config.get("num_nodes", 32)
    batch_size_factor = config.get("batch_size_factor", 1.0)
    k_scale = config.get("k", 1000.0)

    # --- Load trajectories ---
    traj = load_csv_trajectories(path0, label, traj_fns, stride)

    if periodic:
        cvs = ["s" + cv for cv in cvs] + ["c" + cv for cv in cvs]

    if use_z_matrix:
        dcdtraj = load_dcd_trajectories(path0, dcdfile, topfile, stride)
        z_data, labels = convert_to_zmatrix(dcdtraj, atomselect, atomindex, topfile, path0, use_all, pair_distance)
        traj = traj.reset_index(drop=True).join(z_data)
        cvs = labels
        print(f"Z-matrix joined. Final dimension: {len(labels)}")

    # --- Prepare training and validation sets ---
    train_val_data, train_data, val_data = preprocess_traj(data=traj, val_ratio=val_ratio, time_shift=1)
    train_set = CommittorDataset(data=train_data, variables=cvs, device=device)
    val_set = CommittorDataset(data=val_data, variables=cvs, device=device)

    # --- Build model ---
    label_suffix = f"{label}{extra_label}_patience{patience}" if extra_label else f"{label}_patience{patience}"
    model_name = os.path.join(out_dir, label_suffix)

    model = Encoder(num_input_features=len(cvs))
    model.build([num_nodes for _ in range(num_layers)] + [1],
                [nn.ELU() for _ in range(num_layers)] + [nn.Identity()])
    model.to(device)

    # --- Train model ---
    best_model = train_model(
        model_to_train=model,
        output_prefix=model_name,
        train_set=train_set,
        val_set=val_set,
        loss_function=loss_vcns_soft_endpoints,
        epochs=epochs,
        patience=patience,
        batch_size_factor=batch_size_factor,
        dataloader=MyDataLoader,
        k_scale=k_scale,
    )

    # --- Save CPU copy ---
    best_model = torch.jit.load(os.path.join(out_dir, f"{label_suffix}_best_model.pt"))
    cpu_model_path = os.path.join(out_dir, f"{label_suffix}_cpu_best_model.pt")
    best_model.to("cpu").save(cpu_model_path)
    print(f"Saved CPU model at {cpu_model_path}")


# =========================================================
# === Main ===
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Train VCN model with config file")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        print(f"Error: Config file {config_path} does not exist.")
        sys.exit(1)
    config = load_yaml_config(config_path)
    train_committor_model(config["VCN"])


if __name__ == "__main__":
    main()
