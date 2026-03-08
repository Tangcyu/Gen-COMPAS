import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from MDAnalysis import Universe
from MDAnalysis.coordinates.DCD import DCDWriter
from MDAnalysis.lib.distances import calc_bonds, calc_angles, calc_dihedrals
from tqdm import tqdm
import pandas as pd
import yaml
import argparse

# Boltzmann constant in kcal/mol/K
kB = 0.0019872041


# =========================================================
# === Helper Functions ===
# =========================================================

def compute_descriptor_from_features(features, method="pca", ndim=2):
    """Compute low-dimensional descriptors from generic high-dim features."""
    if method == "mean":
        return np.mean(features, axis=1, keepdims=True)
    elif method == "pca":
        pca = PCA(n_components=ndim)
        return pca.fit_transform(features)
    else:
        raise ValueError(f"Unknown descriptor method '{method}'.")


def build_min_zmatrix_indices(n_atoms: int):
    """
    Build a minimal Z-matrix-like internal coordinate definition (3N-6).
    Uses a simple sequential topology:
      bonds:    (i, i-1)
      angles:   (i, i-1, i-2)
      dihedral: (i, i-1, i-2, i-3)
    """
    if n_atoms < 4:
        raise ValueError("Need at least 4 atoms to form a minimal (3N-6) internal coordinate set.")

    bonds = np.array([[i, i - 1] for i in range(1, n_atoms)], dtype=np.int32)          # (N-1, 2)
    angles = np.array([[i, i - 1, i - 2] for i in range(2, n_atoms)], dtype=np.int32)  # (N-2, 3)
    diheds = np.array([[i, i - 1, i - 2, i - 3] for i in range(3, n_atoms)], dtype=np.int32)  # (N-3, 4)
    return bonds, angles, diheds


def internal_coords_min_zmatrix(positions: np.ndarray, bonds, angles, diheds):
    """
    Compute the (3N-6) internal coordinates for one frame:
      - bond lengths (Å)
      - angles (radians)
      - dihedrals (radians, in [-pi, pi])
    """
    b = calc_bonds(
        positions[bonds[:, 0]],
        positions[bonds[:, 1]],
    )
    a = calc_angles(
        positions[angles[:, 0]],
        positions[angles[:, 1]],
        positions[angles[:, 2]],
    )
    d = calc_dihedrals(
        positions[diheds[:, 0]],
        positions[diheds[:, 1]],
        positions[diheds[:, 2]],
        positions[diheds[:, 3]],
    )
    return np.concatenate([b, a, d], axis=0)


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


def kmeans_metastable_labeling(X, n_clusters, quantile=0.9, random_state=0):
    """
    KMeans clustering + distance-based 'intermediate' filtering.

    Returns:
      state_id: shape (n_frames,), in {0..n_clusters-1, -1}
      kmeans: fitted KMeans object
      dist_to_centroid: shape (n_frames,)
      cluster_thresholds: shape (n_clusters,), per-cluster distance cutoff
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D array for KMeans.")
    if len(X) < n_clusters:
        raise ValueError(f"Not enough samples ({len(X)}) for n_clusters={n_clusters}.")

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(X)

    centers = kmeans.cluster_centers_
    dists = np.linalg.norm(X - centers[labels], axis=1)

    thresholds = np.zeros(n_clusters, dtype=float)
    for c in range(n_clusters):
        mask = labels == c
        if np.any(mask):
            thresholds[c] = np.quantile(dists[mask], quantile)
        else:
            thresholds[c] = np.inf

    # intermediate if too far from its assigned centroid
    state_id = labels.copy()
    too_far = dists > thresholds[labels]
    state_id[too_far] = -1

    return state_id, kmeans, dists, thresholds


def add_pairwise_committor_columns(df, state_col, n_states, prefix="q"):
    """
    Add C(N,2) pairwise labels for committor-vector training.
    For each pair (i,j), create a column:
      state==i -> 0
      state==j -> 1
      else     -> -1
    """
    states = df[state_col].to_numpy()
    for i in range(n_states):
        for j in range(i + 1, n_states):
            col = f"{prefix}_{i}_{j}"
            arr = np.full_like(states, -1, dtype=np.int32)
            arr[states == i] = 0
            arr[states == j] = 1
            df[col] = arr
    return df


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
    beta = 1 / (kB * temperature)

    # --- Meta-stable state (KMeans) config ---
    n_states = int(config.get("n_states", 4))
    kmeans_space = config.get("kmeans_space", "descriptor")  # "descriptor" or "cvs"
    cvs_to_cluster = config.get("cvs_to_cluster", [])         # used if kmeans_space == "cvs"
    inter_quantile = float(config.get("intermediate_quantile", 0.9))
    kmeans_random_state = int(config.get("kmeans_random_state", 0))

    # --- Pairwise committor-vector columns ---
    make_pairwise = bool(config.get("make_pairwise_committor", True))
    pairwise_prefix = config.get("pairwise_prefix", "q")

    # --- CVs for saving (and optional periodic encoding) ---
    cvs0 = config.get("cvs_to_save", config.get("cvs_to_label", []))  # backward compatible

    # --- Find DCD files ---
    dcd_files = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(dcd_folder)
        for f in files if f.startswith(config["match"]) and f.endswith(".dcd")
    ])

    if relabel:
        # --- Re-label existing weights file using existing columns ---
        df = pd.read_csv(weight_csv)

        if kmeans_space == "descriptor":
            # relabel mode can't rebuild descriptors unless you store them.
            # So we require columns exist in CSV, e.g. PC1/PC2 saved earlier.
            pc_cols = config.get("descriptor_cols", ["PC1", "PC2"])
            if not all(c in df.columns for c in pc_cols):
                raise ValueError(
                    f"Relabel with kmeans_space='descriptor' requires descriptor_cols {pc_cols} in CSV."
                )
            X_cluster = df[pc_cols].to_numpy()
        elif kmeans_space == "cvs":
            if not cvs_to_cluster:
                raise ValueError("Relabel with kmeans_space='cvs' requires cvs_to_cluster in config.")
            if not all(c in df.columns for c in cvs_to_cluster):
                raise ValueError("Some cvs_to_cluster not found in CSV.")
            X_cluster = df[cvs_to_cluster].to_numpy()
        else:
            raise ValueError("kmeans_space must be 'descriptor' or 'cvs'.")

        state_id, kmeans, dists, thresholds = kmeans_metastable_labeling(
            X_cluster, n_clusters=n_states, quantile=inter_quantile, random_state=kmeans_random_state
        )
        df["meta_state"] = state_id
        df["is_intermediate"] = (df["meta_state"] == -1).astype(int)
        df["dist_to_centroid"] = dists

        # Optional pairwise labels
        if make_pairwise:
            df = add_pairwise_committor_columns(df, "meta_state", n_states, prefix=pairwise_prefix)

        df.to_csv(weight_csv, index=False)
        print(f"Re-labeled (KMeans) and updated {weight_csv}")
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

        # build minimal Z-matrix indices once per trajectory
        n_atoms = sel_weights.n_atoms
        bonds, angles, diheds = build_min_zmatrix_indices(n_atoms)

        # compute (3N-6) internal-coordinate features per frame
        features = []
        for ts in u.trajectory[::every]:
            z = internal_coords_min_zmatrix(sel_weights.positions, bonds, angles, diheds)
            features.append(z.astype(np.float32))
        features = np.asarray(features, dtype=np.float32)

        all_universes.append(sel_output.universe)

        desc_proj = compute_descriptor_from_features(features, method, ndim)
        all_descriptors.append(desc_proj)

        headers, colvars_data = read_colvars(colvars_path, index_mismatch)
        colvars_data = colvars_data[::every]
        if len(colvars_data) != len(desc_proj):
            raise ValueError(f"Frame mismatch: {dcd_path}")
        all_colvars.append(colvars_data)

    if len(all_descriptors) == 0:
        raise RuntimeError("No valid trajectories found. Check match / dcd_folder / colvars.")

    # Stack data
    descriptor_all = np.vstack(all_descriptors)
    colvars_all = np.vstack(all_colvars)

    # --- Compute ΔF in descriptor space (same as before) ---
    desc_init = np.vstack([d[: int(split * len(d))] for d in all_descriptors])
    desc_final = np.vstack([d[-int(split * len(d)) :] for d in all_descriptors])

    xbins = np.linspace(np.min(descriptor_all[:, 0]), np.max(descriptor_all[:, 0]), 10)
    ybins = np.linspace(np.min(descriptor_all[:, 1]), np.max(descriptor_all[:, 1]), 10)
    H_init, _, _ = np.histogram2d(desc_init[:, 0], desc_init[:, 1], bins=(xbins, ybins), density=True)
    H_final, _, _ = np.histogram2d(desc_final[:, 0], desc_final[:, 1], bins=(xbins, ybins), density=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        deltaF = -kB * temperature * np.log(H_final / (H_init + 1e-10))
        deltaF -= np.nanmin(deltaF[np.isfinite(deltaF)])

    Xg, Yg = np.meshgrid(0.5 * (xbins[:-1] + xbins[1:]), 0.5 * (ybins[:-1] + ybins[1:]))
    plt.figure(figsize=(6, 5))
    plt.contourf(Xg, Yg, deltaF.T, levels=20, cmap="viridis")
    plt.colorbar(label="ΔF (kcal/mol)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("ΔF in 2D PCA projection")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "deltaF_pca.png"))

    # --- Compute weights (same as before) ---
    weights = []
    for frame_descriptor in descriptor_all:
        x, y = frame_descriptor
        ix = np.digitize(x, xbins) - 1
        iy = np.digitize(y, ybins) - 1
        if 0 <= ix < deltaF.shape[0] and 0 <= iy < deltaF.shape[1]:
            df_ = deltaF[ix, iy]
            weight = np.exp(-beta * df_) if np.isfinite(df_) else 0.0
        else:
            weight = 0.0
        weights.append(weight)

    weights = np.array(weights, dtype=float)
    s = np.sum(weights)
    if s <= 0:
        raise RuntimeError("All weights are zero. Check deltaF binning / projection.")
    weights /= s

    # --- Save weights + CVs ---
    df = pd.DataFrame(colvars_all, columns=headers)
    df.insert(0, "frame", np.arange(len(weights)))
    df["weight"] = weights

    # Save descriptor columns for later relabeling / debugging
    if descriptor_all.shape[1] >= 2:
        df["PC1"] = descriptor_all[:, 0]
        df["PC2"] = descriptor_all[:, 1]
    else:
        df["PC1"] = descriptor_all[:, 0]

    # periodic encoding if you need it downstream
    if periodic and cvs0:
        for cv in cvs0:
            if cv in df.columns:
                df[f"s{cv}"] = np.sin(df[cv] * np.pi / 180.0)
                df[f"c{cv}"] = np.cos(df[cv] * np.pi / 180.0)

    # --- KMeans meta-stable labeling ---
    if kmeans_space == "descriptor":
        X_cluster = descriptor_all
    elif kmeans_space == "cvs":
        if not cvs_to_cluster:
            raise ValueError("kmeans_space='cvs' requires cvs_to_cluster in config.")
        missing = [c for c in cvs_to_cluster if c not in df.columns]
        if missing:
            raise ValueError(f"cvs_to_cluster columns missing in colvars: {missing}")
        X_cluster = df[cvs_to_cluster].to_numpy()
    else:
        raise ValueError("kmeans_space must be 'descriptor' or 'cvs'.")

    meta_state, kmeans, dists, thresholds = kmeans_metastable_labeling(
        X_cluster, n_clusters=n_states, quantile=inter_quantile, random_state=kmeans_random_state
    )
    df["meta_state"] = meta_state
    df["is_intermediate"] = (df["meta_state"] == -1).astype(int)
    df["dist_to_centroid"] = dists

    # Optional: C(N,2) pairwise labels for committor-vector training
    if make_pairwise:
        df = add_pairwise_committor_columns(df, "meta_state", n_states, prefix=pairwise_prefix)

    df.to_csv(weight_csv, index=False)
    print(f"Saved frame weights + meta-state labels to {weight_csv}")

    # --- Write DCD and PSF for output selection ---
    sel_output = all_universes[0].select_atoms(selection_output)
    with DCDWriter(output_dcd, sel_output.n_atoms) as writer:
        for u in all_universes:
            for ts in u.trajectory[::every]:
                writer.write(u.atoms)
    print(f"Saved concatenated DCD: {output_dcd}")

    # Optional: also save cluster thresholds
    thr_path = os.path.join(output_dir, "kmeans_thresholds.txt")
    with open(thr_path, "w") as f:
        f.write(f"n_states={n_states}\n")
        f.write(f"intermediate_quantile={inter_quantile}\n")
        f.write("thresholds (per cluster):\n")
        for i, t in enumerate(thresholds):
            f.write(f"{i} {t}\n")
    print(f"Saved KMeans thresholds: {thr_path}")


# =========================================================
# === Entry Point ===
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reweighting + KMeans metastable labeling + pairwise committor labels")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_reweighting(config)
