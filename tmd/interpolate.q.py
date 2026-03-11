#!/usr/bin/env python3
import os
import glob
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import mdtraj as md

from vcn.zmatrix import (
    get_internal_coordinates,
    get_minimal_internal_coordinates,
    get_pair_distances,
)


# =========================================================
# === Utility functions ===
# =========================================================

def load_yaml_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def calc_committors_sig(model, positions, periodic=False, device="cpu"):
    """Calculate committor values for input coordinates."""
    positions = np.asarray(positions)

    if periodic:
        input_tensor = torch.tensor(
            np.concatenate(
                [np.sin(positions * np.pi / 180.0), np.cos(positions * np.pi / 180.0)],
                axis=1,
            ),
            dtype=torch.float32,
            device=device,
        )
    else:
        input_tensor = torch.tensor(positions, dtype=torch.float32, device=device)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    return output_tensor.detach().cpu().numpy().flatten()


def resolve_topology_path(dcd_dir, topfile):
    """Resolve topology path from config."""
    if os.path.isabs(topfile):
        return topfile
    return os.path.abspath(os.path.join(dcd_dir, topfile))


def normalize_atomindex(atomindex):
    """
    Normalize atomindex into a list/array of integers.
    Assumes the downstream vcn.zmatrix functions expect 1-based indexing,
    consistent with the user's original script.
    """
    if atomindex is None:
        return []

    atomindex = np.asarray(atomindex, dtype=int).flatten()

    if atomindex.size == 0:
        return []

    return atomindex.tolist()


def load_dcd_data(dcd_path, topfile, atomselect=None, atomindex=None):
    """
    Load DCD trajectory and determine atom indices.

    Priority:
      1. atomindex from config
      2. atomselect from config
      3. empty list
    """
    traj = md.load(dcd_path, top=topfile)

    if atomindex is not None:
        atomindex = normalize_atomindex(atomindex)
        print(f"Using atomindex from config with {len(atomindex)} atoms.")
    elif atomselect is not None:
        # mdtraj select returns 0-based indices; original script converted to +1
        atomindex = (traj.topology.select(atomselect) + 1).tolist()
        print(f"Using atomselect='{atomselect}' -> {len(atomindex)} atoms.")
    else:
        atomindex = []
        print("No atomindex/atomselect provided; using default empty atom index list.")

    return traj, atomindex


def convert_to_zmatrix(traj, atomindex, use_all=False, pair_distance=False):
    """Convert Cartesian trajectory to internal coordinates."""
    if use_all:
        labels, values = get_internal_coordinates(traj, atomindex)
    elif pair_distance:
        labels, values = get_pair_distances(traj, atomindex)
    else:
        labels, values = get_minimal_internal_coordinates(traj, atomindex)

    print(f"Converted trajectory to internal coordinates with {len(labels)} variables.")
    return pd.DataFrame(np.vstack(values), columns=labels), labels


def find_dcd_files(root_dir, patterns=("TMD_noh.A.dcd", "TMD_noh.B.dcd")):
    """Recursively find target DCD files."""
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(root_dir, "**", pattern), recursive=True))
    return sorted(matches)


def process_single_dcd(dcd_path, model, config, device):
    """Compute q for one DCD and save CSV in same folder."""
    dcd_dir = os.path.dirname(os.path.abspath(dcd_path))
    dcd_name = os.path.basename(dcd_path)
    csv_name = os.path.splitext(dcd_name)[0] + ".csv"
    csv_path = os.path.join(dcd_dir, csv_name)

    topfile = resolve_topology_path(dcd_dir, config["topfile"])
    atomselect = config.get("atomselect", None)
    atomindex = config.get("atomindex", None)

    use_all = config.get("use_all", False)
    pair_distance = config.get("pair_distance", False)
    use_z_matrix = config.get("z_matrix", False)
    periodic = config.get("periodic", False)
    cvs_to_plot = config.get("cvs_to_plot", [])

    if not os.path.exists(topfile):
        raise FileNotFoundError(f"Topology file not found for {dcd_path}: {topfile}")

    print(f"\nProcessing: {dcd_path}")
    print(f"Using topology: {topfile}")

    dcdtraj, atomindex_used = load_dcd_data(
        dcd_path,
        topfile,
        atomselect=atomselect,
        atomindex=atomindex,
    )

    traj_df, labels = convert_to_zmatrix(
        dcdtraj,
        atomindex_used,
        use_all=use_all,
        pair_distance=pair_distance,
    )

    cvs0 = labels if use_z_matrix else cvs_to_plot
    if not cvs0:
        raise ValueError(
            "No CVs selected for model input. "
            "Set z_matrix: true or provide cvs_to_plot in config."
        )

    missing = [cv for cv in cvs0 if cv not in traj_df.columns]
    if missing:
        raise ValueError(
            f"Missing CV columns in {dcd_path}: {missing}\n"
            f"Available columns: {list(traj_df.columns)}"
        )

    traj_values = traj_df[cvs0].to_numpy()
    q_values = calc_committors_sig(model, traj_values, periodic=periodic, device=device)

    out_df = traj_df.copy()
    out_df.insert(0, "frame", np.arange(len(out_df), dtype=int))
    out_df["q"] = q_values

    out_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return csv_path, len(out_df)


# =========================================================
# === Main ===
# =========================================================

def run_batch_interpolation(config, root_dir):
    device = torch.device(
        config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    )

    model_fn = config["model_fn"]
    model = torch.jit.load(model_fn, map_location=device)
    model.eval()

    dcd_files = find_dcd_files(root_dir, patterns=("TMD_noh.A.dcd", "TMD_noh.B.dcd"))

    if not dcd_files:
        print(f"No TMD_noh.A.dcd or TMD_noh.B.dcd found under: {root_dir}")
        return

    print(f"Found {len(dcd_files)} DCD file(s).")

    n_ok = 0
    n_fail = 0

    for dcd_path in dcd_files:
        try:
            process_single_dcd(dcd_path, model, config, device)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[ERROR] Failed on {dcd_path}")
            print(f"        {type(e).__name__}: {e}")

    print("\n=== Summary ===")
    print(f"Success: {n_ok}")
    print(f"Failed : {n_fail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolate q values for every TMD.A.dcd / TMD.B.dcd under a folder."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root folder to search recursively for TMD.A.dcd and TMD.B.dcd",
    )
    args = parser.parse_args()

    config_all = load_yaml_config(args.config)
    config = config_all["VCN"]

    run_batch_interpolation(config, os.path.abspath(args.root))