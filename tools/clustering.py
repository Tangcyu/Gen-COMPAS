import os
import yaml
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from itertools import combinations
from sklearn.cluster import KMeans
from MDAnalysis import Universe
from MDAnalysis.coordinates.PDB import PDBWriter


# =========================================================
# === Utility functions ===
# =========================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error reading YAML configuration: {e}")


def load_universe(topology: str, trajectory: str | None = None) -> Universe:
    """Load MDAnalysis Universe from topology and trajectory."""
    return Universe(topology, trajectory) if trajectory else Universe(topology)


def extract_internal_coordinates(universe: Universe, atom_selection: str = "all") -> np.ndarray:
    """Extract pairwise distances between atoms for each frame."""
    sel = universe.select_atoms(atom_selection)
    n_atoms = len(sel)
    print(f"Number of selected atoms: {n_atoms}")

    pairs = np.array(list(combinations(range(n_atoms), 2)))
    n_pairs = len(pairs)
    n_frames = len(universe.trajectory)

    coords = np.zeros((n_frames, n_pairs), dtype=np.float32)

    for i, ts in enumerate(tqdm(universe.trajectory, desc="Extracting coordinates")):
        coords[i] = np.linalg.norm(
            sel.positions[pairs[:, 0]] - sel.positions[pairs[:, 1]], axis=1
        )

    return coords


def optimal_k_elbow(data: np.ndarray, max_k: int = 5) -> int:
    """Determine optimal cluster count via elbow method."""
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        distortions.append(kmeans.inertia_)

    deltas = np.diff(distortions)
    double_deltas = np.diff(deltas)
    elbow = np.argmax(double_deltas) + 2 if len(double_deltas) > 0 else 1
    print(f"Elbow method suggests k = {elbow}")
    return elbow


def cluster_and_select_representatives(
    data: np.ndarray,
    n_clusters: int,
    n_per_cluster: int = 1,
    select_farthest: bool = False
) -> Tuple[np.ndarray, List[int]]:
    """Cluster data and select representatives closest (and optionally farthest) to cluster centers."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(data)
    selected_points = []

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_points = data[cluster_indices]
        center = kmeans.cluster_centers_[cluster_id]
        dists = np.linalg.norm(cluster_points - center, axis=1)
        sorted_idx = np.argsort(dists)

        # Select closest
        closest_idx = sorted_idx[:n_per_cluster]
        selected_points.extend(cluster_indices[closest_idx])

        # Optionally select farthest
        if select_farthest:
            farthest_idx = sorted_idx[-n_per_cluster:]
            selected_points.extend(cluster_indices[farthest_idx])

    return labels, selected_points


def save_selected_structures(universe: Universe, frame_indices: List[int], output_dir: str):
    """Save selected frames as PDB files (no REMARK lines)."""
    os.makedirs(output_dir, exist_ok=True)
    digits = len(str(len(frame_indices))) + 1

    for i, frame_idx in enumerate(frame_indices):
        universe.trajectory[frame_idx]
        filename = os.path.join(output_dir, f"frame_{i+1:0{digits}d}.pdb")
        with PDBWriter(filename, multiframe=False) as pdb:
            pdb.write(universe.atoms)

    print(f"Saved {len(frame_indices)} representative structures to {output_dir}")


# =========================================================
# === Main clustering routine ===
# =========================================================

def run_clustering(config):
    """Run clustering pipeline using parameters from config['Clustering']."""

    # === Load parameters from config ===
    topology = config["topology"]
    trajectory = config.get("trajectory", None)
    atom_selection = config.get("atom_selection", "all")
    output_dir = config.get("output_dir", "./clusters")
    n_clusters = config.get("n_clusters", None)
    n_per_cluster = config.get("n_per_cluster", 1)
    max_k = config.get("max_k", 5)
    select_farthest = config.get("select_farthest", False)

    # === Load universe ===
    universe = load_universe(topology, trajectory)

    print("Extracting internal coordinates...")
    coords = extract_internal_coordinates(universe, atom_selection)

    # === Determine cluster count if not provided ===
    if n_clusters is None:
        print("Estimating optimal number of clusters...")
        n_clusters = optimal_k_elbow(coords, max_k=max_k)
        print(f"Optimal cluster count: {n_clusters}")

    # === Perform clustering and select frames ===
    print(f"Running KMeans with {n_clusters} clusters...")
    _, representatives = cluster_and_select_representatives(
        coords,
        n_clusters=n_clusters,
        n_per_cluster=n_per_cluster,
        select_farthest=select_farthest,
    )

    # === Save results ===
    print("Saving selected structures...")
    save_selected_structures(universe, representatives, output_dir)
    print("Clustering completed successfully.")


# =========================================================
# === Entry Point ===
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster molecular structures and extract representative frames.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    run_clustering(args.config)
