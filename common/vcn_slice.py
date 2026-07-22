import os
import numpy as np
import pandas as pd
import mdtraj as md
import yaml
import matplotlib.pyplot as plt
import torch
from vcn.zmatrix import (
    get_internal_coordinates,
    get_minimal_internal_coordinates,
    get_pair_distances,
)
from common.feature_contract import validate_riteweight_vcn_featurization
from common.target_selection import committor_slice_bounds, select_slice_targets


# =========================================================
# === Utility functions ===
# =========================================================

def load_yaml_config(file_path):
    """Load YAML configuration safely."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def committor_projection_plotting_enabled(config):
    """Return the explicit workflow switch for committor projection plots."""
    enabled = config.get("plot_committor_projections", False)
    if not isinstance(enabled, bool):
        raise TypeError("VCN.plot_committor_projections must be true or false.")
    return enabled


@torch.no_grad()
def calc_committors_sig(model, positions, periodic=False, device='cpu'):
    """Calculate committor values for input coordinates."""
    if periodic:
        dim = positions.shape[1]
        input_tensor = torch.tensor(
            np.concatenate(
                [np.sin(positions * np.pi / 180), np.cos(positions * np.pi / 180)], axis=1
            ),
            dtype=torch.float,
            device=device,
        )
    else:
        input_tensor = torch.tensor(positions, dtype=torch.float, device=device)

    output_tensor = model(input_tensor)
    return output_tensor.cpu().detach().numpy().flatten()


def load_dcd_data(path0, dcdfile, topfile, atomselect):
    """Load DCD trajectory and apply atom selection if given."""
    dcd_path = dcdfile if os.path.isabs(dcdfile) else os.path.join(path0, dcdfile)
    top_path = topfile if os.path.isabs(topfile) else os.path.join(path0, topfile)
    traj = md.load(dcd_path, top=top_path)
    if atomselect is not None:
        atomindex = traj.topology.select(atomselect) + 1
    else:
        atomindex = []
    return traj, atomindex


def convert_to_zmatrix(traj, atomindex, use_all=False, pair_distance=False):
    """Convert Cartesian trajectory to internal coordinates."""
    if use_all:
        labels, values = get_internal_coordinates(traj, atomindex)
    elif pair_distance:
        labels, values = get_pair_distances(traj, atomindex)
    else:
        labels, values = get_minimal_internal_coordinates(traj, atomindex)

    print(f"Converted trajectory to Z-matrix with {len(labels)} variables.")
    return pd.DataFrame(values, columns=labels), labels


# =========================================================
# === Plotting ===
# =========================================================

def plot_committor_2d(x, y, q_values, out_path, title_suffix=""):
    """2D scatter plot of committor values."""
    plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap('RdBu_r', 20)
    sc = plt.scatter(x, y, c=q_values, cmap=cmap, vmin=0, vmax=1)
    clb = plt.colorbar(sc)
    clb.ax.set_title(r'$q$', pad=12.0)
    clb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def plot_committor_pairs(traj, q_values, cvs, out_dir, prefix):
    """Plot all 2D combinations of three CVs as 2D plots."""
    if len(cvs) == 3:
        combinations = [(cvs[0], cvs[1]), (cvs[1], cvs[2]), (cvs[0], cvs[2])]
        for x, y in combinations:
            out_path = os.path.join(out_dir, f"{prefix}_{x}_vs_{y}.png")
            plot_committor_2d(traj[x], traj[y], q_values, out_path)
    elif len(cvs) == 2:
        out_path = os.path.join(out_dir, f"{prefix}_{cvs[0]}_vs_{cvs[1]}.png")
        plot_committor_2d(traj[cvs[0]], traj[cvs[1]], q_values, out_path)
    else:
        print("Warning: plot_committor_pairs supports only 2 or 3 CVs.")


# =========================================================
# === Main slicing ===
# =========================================================

def run_committor_slice(config, riteweight_config=None):
    """Slice generated frames using a consistently featurized VCN model."""

    if riteweight_config is not None:
        validate_riteweight_vcn_featurization(config, riteweight_config)

    label = config.get("label", "default_label")
    model_fn = config["model_fn"]
    path0 = config.get("sampling_path", config.get("Sampling_path", "./"))
    out_dir = config.get("slice_dir", "./output/")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(
        config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    )

    # Load settings
    use_z_matrix = config.get("z_matrix", False)
    pair_distance = config.get("pair_distance", False)
    use_all = config.get("use_all", False)
    dcdfile = config["gendcdfile"]
    topfile = config["topfile"]
    atomindex  = config.get("atomindex", [])
    atomselect = config.get("atomselect", None)
    cvs_to_plot = config.get("cvs_to_plot", None)
    plot_projections = committor_projection_plotting_enabled(config)
    periodic = config.get("periodic", False)
    q_bounds = committor_slice_bounds(config.get("q_variance", 0.1))
    n_targets = int(config.get("n_targets", 20))
    require_n_targets = bool(config.get("require_n_targets", True))

    # Load trajectory
    print(dcdfile, topfile, atomselect)
    if atomselect is not None:
        dcdtraj, atomindex = load_dcd_data(path0, dcdfile, topfile, atomselect)
    elif len(atomindex) > 0:
        dcdtraj, _ = load_dcd_data(path0, dcdfile, topfile, None)
    else:
        raise ValueError("Either atomselect or atomindex must be provided.")  
    traj, labels = convert_to_zmatrix(dcdtraj, atomindex, use_all, pair_distance)

    cvs0 = labels if use_z_matrix else cvs_to_plot
    model = torch.jit.load(model_fn, map_location=device).to(device)
    model.eval()
    traj_values = traj[cvs0].to_numpy()

    q_values = calc_committors_sig(model, traj_values, periodic=periodic, device=device)
    if len(q_values) != dcdtraj.n_frames or not np.all(np.isfinite(q_values)):
        raise ValueError("The committor model returned invalid values for generated frames.")
    traj["committor"] = q_values
    committor_path = os.path.join(out_dir, "committor.csv")
    traj.to_csv(committor_path, index=False)
    # Stage 1: one configurable interval centered on q=0.5.
    q_min, q_max = q_bounds
    mask = (q_values >= q_min) & (q_values <= q_max)
    if not np.any(mask):
        raise ValueError(
            "No generated frame lies in the configured committor range "
            f"[{q_min:g}, {q_max:g}]."
        )

    candidate_indices = np.flatnonzero(mask)
    candidate_points = traj.iloc[candidate_indices].copy()
    candidate_points.to_csv(
        os.path.join(out_dir, "committor_candidates.csv"), index=False
    )

    # Stage 2: retain the first 20 (or configured count) candidates in trajectory
    # order. n_targets is only a count limit and does not perform clustering.
    selected_indices = select_slice_targets(
        candidate_indices,
        n_targets,
        require_n_targets=require_n_targets,
        q_bounds=q_bounds,
    )

    sliced_points = traj.iloc[selected_indices].copy()
    sliced_points.to_csv(os.path.join(out_dir, "sliced.csv"), index=False)

    selected_frames = dcdtraj[selected_indices]
    slice_dir = os.path.join(out_dir, "sliced_frames")
    os.makedirs(slice_dir, exist_ok=True)

    # Save PDBs
    for i, frame in enumerate(selected_frames):
        frame.save_pdb(
            os.path.join(slice_dir, f"sliced_{i+1:0{len(str(len(selected_frames)))+1}d}.pdb")
        )
    selected_frames.save(os.path.join(out_dir, "sliced_frames.dcd"))
    print(f"Saved {len(selected_frames)} sliced frames around q=0.5 to {out_dir}/sliced_frames.dcd and {slice_dir}/sliced_*.pdb.")

    # Plot committor maps (2D or 3×2D)
    if (
        plot_projections
        and cvs_to_plot is not None
        and all(cv in traj.columns for cv in cvs_to_plot)
    ):
        plot_committor_pairs(traj, q_values, cvs_to_plot, out_dir, prefix="all")
        plot_committor_pairs(
            sliced_points, q_values[selected_indices], cvs_to_plot, out_dir,
            prefix="sliced"
        )
    elif plot_projections and cvs_to_plot:
        print("Warning: skipping committor plots because requested CVs are not model features.")

    print("Committor slicing completed successfully.")
    return {
        "trajectory": os.path.join(out_dir, "sliced_frames.dcd"),
        "pdb_dir": slice_dir,
        "committor_table": committor_path,
        "count": len(selected_indices),
        "candidate_count": int(mask.sum()),
    }


# =========================================================
# === Entry point ===
# =========================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Slice generated frames with a VCN model")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    run_committor_slice(config["VCN"], config.get("RiteWeight"))
