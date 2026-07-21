import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from MDAnalysis import Universe
from MDAnalysis.coordinates.DCD import DCDWriter
from tqdm import tqdm
import pandas as pd
import yaml
import argparse

# Boltzmann constant in kcal/mol/K
kB = 0.0019872041


# =========================================================
# === Helper Functions ===
# =========================================================

def compute_descriptor_from_distances(distances, method="pca", ndim=2):
    """Compute low-dimensional descriptors from pairwise distances."""
    if method == "mean":
        return np.mean(distances, axis=1, keepdims=True)
    elif method == "pca":
        pca = PCA(n_components=ndim)
        return pca.fit_transform(distances)
    else:
        raise ValueError(f"Unknown descriptor method '{method}'.")


def read_colvars(colvars_path, index_mismatch=True, skip_rows=1):
    """Read a .colvars.traj file and remove duplicated column names."""
    with open(colvars_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                raw_headers = line[1:].strip().split()
                break

    data = np.loadtxt(colvars_path, comments=["#", "@"], skiprows=skip_rows)
    if index_mismatch:
        data = data[1:]

    seen, keep_indices, headers = {}, [], []
    for i, name in enumerate(raw_headers):
        if name not in seen:
            seen[name] = True
            keep_indices.append(i)
            headers.append(name)

    data = data[:, keep_indices]
    print(f"Loaded {colvars_path} with {len(headers)} unique columns.")
    return headers, data


def determine_AB_functor(basin_A, basin_B, basin_size):
    """Create a function to classify a CV point as 'A', 'B', or 'M'."""
    basin_A, basin_B = np.array(basin_A), np.array(basin_B)
    basin_size = np.full_like(basin_A, basin_size, dtype=float) if np.isscalar(basin_size) else np.array(basin_size)

    def in_basin(cvs, center):
        return np.all(np.abs(np.array(cvs) - center) <= basin_size)

    def determine_AB(cvs):
        if in_basin(cvs, basin_A):
            return "A"
        elif in_basin(cvs, basin_B):
            return "B"
        return "M"

    return determine_AB


# =========================================================
# === Main Reweighting Function ===
# =========================================================

def run_reweighting(config):
    psf_path = config["topology_file"]
    dcd_folder = config["dcd_folder"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    output_dcd = os.path.join(output_dir, "concat.dcd")
    output_psf = os.path.join(output_dir, "concat.psf")
    weight_csv = os.path.join(output_dir, "weights.csv")

    # --- Selections ---
    selection_weights = config.get("sel_weights", "protein and not name H*")
    selection_output = config.get("sel_output", "protein and not name H*")

    # --- Parameters ---
    temperature = config.get("temperature", 300)
    method = config.get("method", "pca")
    ndim = config.get("ndim", 2)
    split = config.get("split", 0.1)
    every = config.get("every", 1)
    index_mismatch = config.get("colvars_mismatch", True)
    relabel = config.get("Relabel", False)
    periodic = config.get("periodic", False)
    k = config.get("k_prefactor", 1.0)
    beta = 1 / (kB * temperature)

    cvs0 = config["cvs_to_label"]
    basin_A, basin_B = np.array(config["basin_A"]), np.array(config["basin_B"])
    basin_size = config["basin_size"]

    determine_AB = determine_AB_functor(basin_A, basin_B, basin_size)

    # --- Find DCD files ---
    dcd_files = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(dcd_folder)
        for f in files if f.startswith(config["match"]) and f.endswith(".dcd")
    ])

    if relabel:
        # --- Re-label existing weights file ---
        df = pd.read_csv(weight_csv)
        positions = df[cvs0].to_numpy()
        df["state"] = np.apply_along_axis(determine_AB, 1, positions)

        df["label"] = df["state"].map({"A": 0, "B": 1, "M": -1})
        df["center"] = df["state"].map({"A": 0.0, "B": 1.0, "M": -1})
        df["Ka"] = np.where(df["state"] == "A", k, 0.0)
        df["Kb"] = np.where(df["state"] == "B", k, 0.0)
        df.to_csv(weight_csv, index=False)
        print(f"Re-labeled and updated {weight_csv}")
        return

    # --- Full reweighting mode ---
    all_descriptors, all_colvars, all_universes = [], [], []

    for dcd_path in tqdm(dcd_files, desc="Processing trajectories"):
        base = os.path.splitext(dcd_path)[0]
        colvars_path = base + ".colvars.traj"

        if not os.path.exists(colvars_path):
            print(f"⚠️ Missing colvars for {dcd_path}, skipping.")
            continue

        u = Universe(psf_path, dcd_path)
        sel_weights = u.select_atoms(selection_weights)
        sel_output = u.select_atoms(selection_output)

        descriptors = []
        for ts in u.trajectory[::every]:
            d = pdist(sel_weights.positions)
            descriptors.append(d)
        descriptors = np.array(descriptors)

        all_universes.append(sel_output.universe)

        desc_proj = compute_descriptor_from_distances(descriptors, method, ndim)
        all_descriptors.append(desc_proj)

        headers, colvars_data = read_colvars(colvars_path, index_mismatch)
        colvars_data = colvars_data[::every]
        if len(colvars_data) != len(desc_proj):
            raise ValueError(f"Frame mismatch: {dcd_path}")
        all_colvars.append(colvars_data)

    # Stack data
    descriptor_all = np.vstack(all_descriptors)
    colvars_all = np.vstack(all_colvars)

    # --- Compute ΔF ---
    desc_init = np.vstack([d[: int(split * len(d))] for d in all_descriptors])
    desc_final = np.vstack([d[-int(split * len(d)) :] for d in all_descriptors])

    xbins = np.linspace(np.min(descriptor_all[:, 0]), np.max(descriptor_all[:, 0]), 10)
    ybins = np.linspace(np.min(descriptor_all[:, 1]), np.max(descriptor_all[:, 1]), 10)
    H_init, _, _ = np.histogram2d(desc_init[:, 0], desc_init[:, 1], bins=(xbins, ybins), density=True)
    H_final, _, _ = np.histogram2d(desc_final[:, 0], desc_final[:, 1], bins=(xbins, ybins), density=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        deltaF = -kB * temperature * np.log(H_final / (H_init + 1e-10))
        deltaF -= np.nanmin(deltaF[np.isfinite(deltaF)])

    X, Y = np.meshgrid(0.5 * (xbins[:-1] + xbins[1:]), 0.5 * (ybins[:-1] + ybins[1:]))
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, deltaF.T, levels=20, cmap="viridis")
    plt.colorbar(label="ΔF (kcal/mol)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("ΔF in 2D PCA projection")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "deltaF_pca.png"))

    # --- Compute weights ---
    weights = []
    for frame_descriptor in descriptor_all:
        x, y = frame_descriptor
        ix = np.digitize(x, xbins) - 1
        iy = np.digitize(y, ybins) - 1
        if 0 <= ix < deltaF.shape[0] and 0 <= iy < deltaF.shape[1]:
            df = deltaF[ix, iy]
            weight = np.exp(-beta * df) if np.isfinite(df) else 0.0
        else:
            weight = 0.0
        weights.append(weight)

    weights = np.array(weights)
    weights /= np.sum(weights)

    # --- Save weights + CVs ---
    df = pd.DataFrame(colvars_all, columns=headers)
    df.insert(0, "frame", np.arange(len(weights)))
    df["weight"] = weights

    if periodic:
        for cv in cvs0:
            df[f"s{cv}"] = np.sin(df[cv] * np.pi / 180.0)
            df[f"c{cv}"] = np.cos(df[cv] * np.pi / 180.0)
    positions = df[cvs0].to_numpy()
    df["state"] = np.apply_along_axis(determine_AB, 1, positions)
    df["label"] = df["state"].map({"A": 0, "B": 1, "M": -1})
    df["center"] = df["state"].map({"A": 0.0, "B": 1.0, "M": -1})
    df["Ka"] = np.where(df["state"] == "A", k, 0.0)
    df["Kb"] = np.where(df["state"] == "B", k, 0.0)
    df.to_csv(weight_csv, index=False)
    print(f"Saved frame weights + CVs to {weight_csv}")

    # --- Write DCD and PSF for output selection ---
    sel_output = all_universes[0].select_atoms(selection_output)
    with DCDWriter(output_dcd, sel_output.n_atoms) as writer:
        for u in all_universes:
            for ts in u.trajectory[::every]:
                writer.write(u.atoms)
    print(f"Saved concatenated DCD: {output_dcd}")


# =========================================================
# === Entry Point ===
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reweighting and free energy reconstruction")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_reweighting(config)
