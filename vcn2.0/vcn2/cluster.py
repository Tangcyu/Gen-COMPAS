from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .config import ensure_dir, get_device, section
from .path import reparametrize_polyline


def load_paths(path_dir: str | Path) -> tuple[list[Path], torch.Tensor, torch.Tensor, np.ndarray]:
    files = sorted(Path(path_dir).glob("path_*.npz"))
    if not files:
        raise FileNotFoundError(f"No path_*.npz files found in {path_dir}")
    z_paths = []
    raw_paths = []
    weights = []
    for file in files:
        data = np.load(file)
        z_paths.append(data["z"])
        raw_paths.append(data["raw"])
        weights.append(float(data["weight"]) if "weight" in data.files else 1.0)
    return files, torch.as_tensor(np.stack(z_paths), dtype=torch.float32), torch.as_tensor(np.stack(raw_paths), dtype=torch.float32), np.asarray(weights)


def pairwise_rmsd(paths: torch.Tensor, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    flat = paths.reshape(paths.shape[0], -1).to(device)
    denom = float(flat.shape[1]) ** 0.5
    rows = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            rows.append(torch.cdist(flat[start : start + batch_size], flat) / denom)
    return torch.cat(rows, dim=0).cpu().numpy()


def cluster_from_config(config: dict) -> dict:
    cluster_cfg = section(config, "clustering")
    out_cfg = section(config, "output")
    device = get_device(cluster_cfg.get("device", config.get("device", "cuda:0")))
    out_dir = ensure_dir(out_cfg.get("out_dir", "./vcn2_output"))
    path_dir = Path(cluster_cfg.get("path_dir", out_dir / "paths"))
    cluster_dir = ensure_dir(out_dir / "clusters")

    files, z_paths, raw_paths, weights = load_paths(path_dir)
    n_images_out = int(cluster_cfg.get("mean_num_images", z_paths.shape[1]))
    dist = pairwise_rmsd(z_paths, device=device, batch_size=int(cluster_cfg.get("distance_batch_size", 2048)))
    np.save(cluster_dir / "path_rmsd_matrix.npy", dist)

    import scipy.cluster.hierarchy as hierarchy
    from scipy.cluster.hierarchy import fcluster
    from scipy.spatial.distance import squareform

    linkage = hierarchy.linkage(squareform(dist, checks=False), method=cluster_cfg.get("linkage", "average"))
    threshold = float(cluster_cfg.get("threshold", 0.25))
    labels = fcluster(linkage, threshold, criterion=cluster_cfg.get("criterion", "distance"))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hierarchy.dendrogram(linkage, count_sort="descendent", no_labels=True)
        plt.ylabel("Path RMSD in normalized CV space")
        plt.xlabel("Path")
        plt.tight_layout()
        plt.savefig(cluster_dir / "dendrogram.png", dpi=200)
        plt.close()
    except Exception as exc:
        print(f"Could not write dendrogram: {exc}")

    with open(cluster_dir / "cluster_labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "cluster", "weight"])
        writer.writeheader()
        for file, label, weight in zip(files, labels, weights):
            writer.writerow({"path": file.name, "cluster": int(label), "weight": float(weight)})

    z_device = z_paths.to(device)
    raw_device = raw_paths.to(device)
    mean_rows = []
    for cluster_id in sorted(set(labels.tolist())):
        mask = labels == cluster_id
        w = torch.as_tensor(weights[mask], dtype=torch.float32, device=device)
        w = w / torch.clamp(torch.sum(w), min=1e-12)
        mean_z = torch.sum(z_device[mask] * w[:, None, None], dim=0)
        mean_raw = torch.sum(raw_device[mask] * w[:, None, None], dim=0)
        mean_z = reparametrize_polyline(mean_z, n_images_out)
        mean_raw = reparametrize_polyline(mean_raw, n_images_out)
        np.savetxt(cluster_dir / f"cluster_{cluster_id:03d}_mean_z.txt", mean_z.detach().cpu().numpy())
        np.savetxt(cluster_dir / f"cluster_{cluster_id:03d}_mean_raw.txt", mean_raw.detach().cpu().numpy())
        mean_rows.append(
            {
                "cluster": int(cluster_id),
                "n_paths": int(np.sum(mask)),
                "weight_sum": float(np.sum(weights[mask])),
                "mean_raw": f"cluster_{cluster_id:03d}_mean_raw.txt",
                "mean_z": f"cluster_{cluster_id:03d}_mean_z.txt",
            }
        )

    with open(cluster_dir / "cluster_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mean_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mean_rows)
    return {"cluster_dir": str(cluster_dir), "n_clusters": len(mean_rows)}
