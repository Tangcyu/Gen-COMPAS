#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import DBSCAN, KMeans

try:
    import hdbscan
except ImportError:
    hdbscan = None

try:
    import yaml
except ImportError as e:
    raise SystemExit("Need pyyaml. Install: pip install pyyaml") from e

try:
    import mdtraj as md
except ImportError as e:
    raise SystemExit("Need mdtraj. Install: pip install mdtraj") from e


def find_matching(root: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


def pair_by_subdir_tag(files: List[str], root: str, tag_re: str) -> Dict[Tuple[str, str], str]:
    matched: Dict[Tuple[str, str], str] = {}
    cre = re.compile(tag_re)
    for file_path in files:
        rel = os.path.relpath(file_path, root)
        subdir = os.path.dirname(rel)
        bname = os.path.basename(file_path)
        mm = cre.search(bname)
        if mm:
            matched[(subdir, mm.group(1))] = file_path
    return matched


def find_pairs_dcd_colvars(
    roots: List[str],
    match_dcd: str,
    match_colvars: str,
    tag_re: str = r"([ABM])",
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for root in roots:
        dcds = find_matching(root, f"*{match_dcd}*.dcd")
        cols = find_matching(root, f"*{match_colvars}*.colvars.traj")

        if not dcds:
            print(f"[WARN] No DCD files under {root} matching '*{match_dcd}*.dcd'")
            continue
        if not cols:
            print(f"[WARN] No colvars files under {root} matching '*{match_colvars}*.colvars.traj'")
            continue

        d_map = pair_by_subdir_tag(dcds, root, tag_re)
        c_map = pair_by_subdir_tag(cols, root, tag_re)
        common = sorted(set(d_map) & set(c_map))
        local_pairs = [(d_map[key], c_map[key]) for key in common]
        if not local_pairs:
            print(f"[WARN] No matching (dcd,colvars) pairs found in root={root} with tag_re={tag_re}")
        pairs.extend(local_pairs)

    if not pairs:
        raise FileNotFoundError("No matching (dcd, colvars) pairs found across all folders.")
    return sorted(pairs)


def read_colvars_traj(path: str) -> pd.DataFrame:
    colnames = None
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                tokens = stripped.lstrip("#").strip().split()
                if len(tokens) >= 2 and all("=" not in token for token in tokens):
                    colnames = tokens
            else:
                break

    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    if colnames is not None and len(colnames) == df.shape[1]:
        df.columns = colnames
    else:
        df.columns = [f"col{i}" for i in range(df.shape[1])]
    return df


def maybe_align_colvars(df: pd.DataFrame, n_frames: int, allow_skip_first: bool) -> Tuple[pd.DataFrame, str]:
    if len(df) == n_frames:
        return df, "ok"
    if allow_skip_first and len(df) == n_frames + 1:
        return df.iloc[1:].reset_index(drop=True), "skip_first_colvars"
    return df, "mismatch"


def count_dcd_frames(dcd_path: str, top_path: str, stride: int) -> int:
    total = 0
    for chunk in md.iterload(dcd_path, top=top_path, stride=stride, chunk=1000):
        total += chunk.n_frames
    return total


@dataclass
class PairBuildResult:
    frame_table: pd.DataFrame
    report: pd.DataFrame


def build_frame_table(
    pairs: List[Tuple[str, str]],
    top_path: str,
    stride: int,
    allow_skip_first: bool,
    strict: bool,
) -> PairBuildResult:
    frame_tables: List[pd.DataFrame] = []
    report_rows: List[dict] = []
    global_offset = 0

    for pair_id, (dcd_path, colvars_path) in enumerate(pairs):
        n_frames = count_dcd_frames(dcd_path, top_path, stride)
        raw_colvars = read_colvars_traj(colvars_path)
        strided_colvars = raw_colvars.iloc[::stride].reset_index(drop=True) if stride != 1 else raw_colvars
        aligned_colvars, action = maybe_align_colvars(strided_colvars, n_frames, allow_skip_first)
        ok = action != "mismatch"

        report_rows.append(
            {
                "pair_id": pair_id,
                "dcd": dcd_path,
                "colvars": colvars_path,
                "dcd_frames": n_frames,
                "colvars_rows": len(raw_colvars),
                "colvars_rows_after_stride": len(strided_colvars),
                "action": action,
                "aligned_rows": len(aligned_colvars) if ok else np.nan,
                "ok": ok,
            }
        )

        if not ok:
            msg = (
                f"Frame/colvars mismatch for\n"
                f"  dcd={dcd_path}\n"
                f"  colvars={colvars_path}\n"
                f"  dcd_frames={n_frames}, colvars_rows_after_stride={len(strided_colvars)}"
            )
            if strict:
                raise ValueError(msg)
            print(f"[WARN] {msg}\n[WARN] Skipping this pair.")
            continue

        local_df = aligned_colvars.reset_index(drop=True).copy()
        local_df.insert(0, "pair_id", pair_id)
        local_df.insert(1, "frame", np.arange(global_offset, global_offset + len(local_df), dtype=np.int64))
        local_df.insert(2, "local_frame", np.arange(len(local_df), dtype=np.int64))
        local_df.insert(3, "traj_frame", np.arange(0, len(local_df) * stride, stride, dtype=np.int64))
        local_df.insert(4, "dcd_path", dcd_path)
        local_df.insert(5, "colvars_path", colvars_path)
        frame_tables.append(local_df)
        global_offset += len(local_df)

    if not frame_tables:
        raise SystemExit("No aligned (dcd, colvars) pairs were available for clustering.")

    return PairBuildResult(
        frame_table=pd.concat(frame_tables, ignore_index=True),
        report=pd.DataFrame(report_rows),
    )


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_output_path(base_dir: str, maybe_relative: str) -> str:
    if os.path.isabs(maybe_relative):
        return maybe_relative
    return os.path.join(base_dir, maybe_relative)


def apply_cv_filters(df: pd.DataFrame, filters: Dict[str, List[float]]) -> pd.DataFrame:
    filtered = df
    for column, bounds in filters.items():
        if column not in filtered.columns:
            raise SystemExit(f"CV filter column '{column}' was not found.")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise SystemExit(f"Filter for '{column}' must be [min, max].")
        low, high = float(bounds[0]), float(bounds[1])
        filtered = filtered[(filtered[column] >= low) & (filtered[column] <= high)]
    return filtered.reset_index(drop=True)


def load_gromacs_fel_2d(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.rstrip() for line in handle if line.strip()]

    header = [line for line in lines if line.startswith("#")]
    data_lines = [line for line in lines if not line.startswith("#")]
    if len(header) < 3:
        raise SystemExit(f"FEL file does not have the expected GROMACS-style header: {path}")

    try:
        ndim = int(header[0].lstrip("#").strip())
    except ValueError as exc:
        raise SystemExit(f"Could not read dimensionality from FEL header: {path}") from exc
    if ndim != 2:
        raise SystemExit(f"Only 2D FEL .dat files are supported for basin filtering, got ndim={ndim} in {path}")

    def parse_axis(header_line: str) -> Tuple[float, float, int]:
        parts = header_line.lstrip("#").split()
        if len(parts) < 3:
            raise SystemExit(f"Malformed FEL axis header in {path}: {header_line}")
        return float(parts[0]), float(parts[1]), int(parts[2])

    x_min, x_binw, x_nbin = parse_axis(header[1])
    y_min, y_binw, y_nbin = parse_axis(header[2])

    raw = np.loadtxt(data_lines)
    raw = np.atleast_2d(raw)
    if raw.shape[1] < 3:
        raise SystemExit(f"FEL data in {path} must have at least 3 columns: x y F")

    x_centers = x_min + (np.arange(x_nbin, dtype=float) + 0.5) * x_binw
    y_centers = y_min + (np.arange(y_nbin, dtype=float) + 0.5) * y_binw
    F = np.full((x_nbin, y_nbin), np.nan, dtype=float)

    for row in raw:
        x_val, y_val, f_val = float(row[0]), float(row[1]), float(row[2])
        ix = int(np.argmin(np.abs(x_centers - x_val)))
        iy = int(np.argmin(np.abs(y_centers - y_val)))
        F[ix, iy] = f_val

    return x_centers, y_centers, F


def annotate_fel(
    df: pd.DataFrame,
    cv_columns: List[str],
    fel_grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> pd.DataFrame:
    if len(cv_columns) != 2:
        raise SystemExit("FEL basin filtering requires exactly two clustering CVs.")

    x_centers, y_centers, F = fel_grid
    x_vals = df[cv_columns[0]].to_numpy(dtype=float)
    y_vals = df[cv_columns[1]].to_numpy(dtype=float)

    x_idx = np.abs(x_vals[:, None] - x_centers[None, :]).argmin(axis=1)
    y_idx = np.abs(y_vals[:, None] - y_centers[None, :]).argmin(axis=1)
    fel_values = F[x_idx, y_idx]

    out = df.copy()
    out["fel_x_idx"] = x_idx
    out["fel_y_idx"] = y_idx
    out["fel_kcal_per_mol"] = fel_values
    return out


def apply_fel_cutoff(
    df: pd.DataFrame,
    cv_columns: List[str],
    fel_grid: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    fel_cutoff: Optional[float],
) -> pd.DataFrame:
    if fel_grid is None:
        return df

    annotated = annotate_fel(df, cv_columns=cv_columns, fel_grid=fel_grid)
    valid = np.isfinite(annotated["fel_kcal_per_mol"].to_numpy(dtype=float))
    annotated = annotated.loc[valid].reset_index(drop=True)
    if annotated.empty:
        raise SystemExit("No frames could be mapped onto finite FEL values from the supplied .dat file.")

    if fel_cutoff is None:
        return annotated

    filtered = annotated[annotated["fel_kcal_per_mol"] <= float(fel_cutoff)].reset_index(drop=True)
    if filtered.empty:
        raise SystemExit(
            f"No frames remain after applying FEL cutoff <= {fel_cutoff} kcal/mol from the supplied FEL grid."
        )
    return filtered


def standardize_features(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0.0] = 1.0
    return (data - mean) / std, mean, std


def build_feature_matrix(
    df: pd.DataFrame,
    cv_columns: List[str],
    use_fel_as_feature: bool,
) -> Tuple[np.ndarray, List[str]]:
    missing = [column for column in cv_columns if column not in df.columns]
    if missing:
        raise SystemExit(f"Clustering CV columns missing from aligned data: {missing}")

    features = [df[cv_columns].to_numpy(dtype=float)]
    names = list(cv_columns)

    if use_fel_as_feature:
        if "fel_kcal_per_mol" not in df.columns:
            raise SystemExit("use_fel_as_feature=true requires FEL values. Set cv_clustering.fel_dat.")
        features.append(df[["fel_kcal_per_mol"]].to_numpy(dtype=float))
        names.append("fel_kcal_per_mol")

    matrix = np.concatenate(features, axis=1)
    matrix, _, _ = standardize_features(matrix)
    return matrix, names


def compute_elbow_curve(
    data: np.ndarray,
    max_k: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    k_values = np.arange(1, max_k + 1, dtype=int)
    inertias = []
    for k in k_values:
        model = KMeans(n_clusters=int(k), random_state=random_state, n_init=20)
        model.fit(data)
        inertias.append(model.inertia_)
    return k_values, np.asarray(inertias, dtype=float)


def pick_k_by_elbow(k_values: np.ndarray, inertias: np.ndarray) -> int:
    if len(k_values) == 1:
        return int(k_values[0])
    if len(k_values) == 2:
        return int(k_values[1])

    first = np.diff(inertias)
    second = np.diff(first)
    elbow_index = int(np.argmax(np.abs(second))) + 2
    return int(k_values[elbow_index - 1])


def fit_kmeans(
    data: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> KMeans:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50)
    model.fit(data)
    return model


def fit_dbscan(
    data: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
    return model.fit_predict(data)


def fit_hdbscan(
    data: np.ndarray,
    min_cluster_size: int,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0,
) -> np.ndarray:
    if hdbscan is None:
        raise SystemExit(
            "hdbscan is not installed in this environment. Install it or use method: dbscan/fel_components/kmeans."
        )
    model = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
    )
    return model.fit_predict(data)


def fit_fel_components(
    df: pd.DataFrame,
    fel_grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
    fel_cutoff: Optional[float],
    connectivity: int = 8,
    min_component_frames: int = 1,
) -> np.ndarray:
    if fel_cutoff is None:
        raise SystemExit("method: fel_components requires cv_clustering.fel_cutoff_kcal_per_mol.")
    if "fel_x_idx" not in df.columns or "fel_y_idx" not in df.columns:
        raise SystemExit("FEL grid indices are missing; supply cv_clustering.fel_dat for fel_components.")

    _, _, F = fel_grid
    low_mask = np.isfinite(F) & (F <= float(fel_cutoff))
    if connectivity == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    elif connectivity == 8:
        structure = ndimage.generate_binary_structure(2, 2)
    else:
        raise SystemExit("fel_components.connectivity must be 4 or 8.")

    comp_grid, _ = ndimage.label(low_mask, structure=structure)
    raw_labels = comp_grid[
        df["fel_x_idx"].to_numpy(dtype=int),
        df["fel_y_idx"].to_numpy(dtype=int),
    ]

    counts = pd.Series(raw_labels).value_counts()
    keep = {int(label) for label, count in counts.items() if int(label) > 0 and int(count) >= int(min_component_frames)}

    mapped = np.full(len(df), -1, dtype=int)
    relabel = {old: new for new, old in enumerate(sorted(keep))}
    for i, label in enumerate(raw_labels):
        if int(label) in relabel:
            mapped[i] = relabel[int(label)]
    return mapped


def choose_representatives(
    df: pd.DataFrame,
    data: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    n_select: int,
) -> pd.DataFrame:
    selected_chunks: List[pd.DataFrame] = []

    for cluster_id in sorted(np.unique(labels)):
        if int(cluster_id) < 0:
            continue
        cluster_index = np.where(labels == cluster_id)[0]
        cluster_data = data[cluster_index]
        center = centers[cluster_id]
        distances = np.linalg.norm(cluster_data - center, axis=1)
        cluster_df = df.iloc[cluster_index].copy()
        cluster_df["distance_to_center"] = distances
        cluster_df = cluster_df.sort_values(
            by=["distance_to_center", "fel_kcal_per_mol", "frame"],
            ascending=[True, True, True],
        )
        selected_chunks.append(cluster_df.head(n_select))

    if not selected_chunks:
        raise SystemExit("No representatives could be selected from the clustering result.")

    return pd.concat(selected_chunks, ignore_index=True)


def save_selected_structures(
    selected_df: pd.DataFrame,
    top_path: str,
    out_dir: str,
    structure_format: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    digits = max(4, len(str(len(selected_df))))

    for i, row in selected_df.reset_index(drop=True).iterrows():
        traj = md.load_frame(row["dcd_path"], int(row["traj_frame"]), top=top_path)
        suffix = structure_format.lower()
        out_path = os.path.join(
            out_dir,
            f"cluster_{int(row['cluster']):02d}_frame_{int(row['frame']):0{digits}d}.{suffix}",
        )
        if suffix == "pdb":
            traj.save_pdb(out_path)
        elif suffix in {"xtc", "dcd"}:
            getattr(traj, f"save_{suffix}")(out_path)
        else:
            raise SystemExit(f"Unsupported structure_format '{structure_format}'. Use pdb, xtc, or dcd.")


def plot_elbow(k_values: np.ndarray, inertias: np.ndarray, chosen_k: int, out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(k_values, inertias, marker="o")
    plt.axvline(chosen_k, color="tab:red", linestyle="--", linewidth=1.2, label=f"k = {chosen_k}")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_clusters(
    df: pd.DataFrame,
    plot_columns: List[str],
    selected_frames: pd.DataFrame,
    out_path: str,
    fel_cutoff: Optional[float] = None,
    fel_grid: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    color_values = df["cluster"].to_numpy(dtype=int)
    if len(plot_columns) < 2:
        column = plot_columns[0]
        plt.figure(figsize=(7, 4.5))
        plt.scatter(
            df[column],
            df["fel_kcal_per_mol"],
            c=color_values,
            s=12,
            cmap="tab20",
            alpha=0.75,
        )
        plt.xlabel(column)
        plt.ylabel("FEL (kcal/mol)")
    else:
        xcol, ycol = plot_columns[:2]
        plt.figure(figsize=(6.5, 5.5))
        plt.scatter(
            df[xcol],
            df[ycol],
            c=color_values,
            s=12,
            cmap="tab20",
            alpha=0.75,
        )
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        if fel_cutoff is not None and fel_grid is not None:
            x_centers, y_centers, F = fel_grid
            if len(x_centers) > 1 and len(y_centers) > 1:
                XX, YY = np.meshgrid(x_centers, y_centers, indexing="ij")
                plt.contour(
                    XX,
                    YY,
                    F,
                    levels=[float(fel_cutoff)],
                    colors="black",
                    linewidths=1.0,
                    linestyles="--",
                )

    if not selected_frames.empty:
        sel = selected_frames
        if len(plot_columns) < 2:
            plt.scatter(
                sel[plot_columns[0]],
                sel["fel_kcal_per_mol"],
                s=60,
                facecolors="none",
                edgecolors="black",
                linewidths=0.9,
            )
        else:
            plt.scatter(
                sel[plot_columns[0]],
                sel[plot_columns[1]],
                s=60,
                facecolors="none",
                edgecolors="black",
                linewidths=0.9,
            )

    # Label each cluster near the centroid in the plotted projection.
    if len(plot_columns) < 2:
        label_positions = df[df["cluster"] >= 0].groupby("cluster")[plot_columns[0]].mean().to_frame()
        label_positions["y"] = df.groupby("cluster")["fel_kcal_per_mol"].mean()
        for cluster_id, row in label_positions.iterrows():
            plt.text(
                row[plot_columns[0]],
                row["y"],
                str(int(cluster_id)),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
    else:
        label_positions = df[df["cluster"] >= 0].groupby("cluster")[plot_columns[:2]].mean()
        for cluster_id, row in label_positions.iterrows():
            plt.text(
                row[plot_columns[0]],
                row[plot_columns[1]],
                str(int(cluster_id)),
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def build_cluster_centers(
    data: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, np.ndarray]:
    centers: Dict[int, np.ndarray] = {}
    for cluster_id in sorted(np.unique(labels)):
        if int(cluster_id) < 0:
            continue
        cluster_index = np.where(labels == cluster_id)[0]
        if cluster_index.size == 0:
            continue
        centers[int(cluster_id)] = data[cluster_index].mean(axis=0)
    return centers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster aligned structures using selected CVs and FEL basin energies."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config used by riteweight.")
    parsed = parser.parse_args()

    cfg = load_yaml(parsed.config)
    cluster_cfg = cfg.get("cv_clustering", {})
    if not cluster_cfg:
        raise SystemExit("config.yaml must define a 'cv_clustering' section for cvclusters.py")

    roots = cfg.get("folders", [])
    if not roots:
        raise SystemExit("config.yaml must define 'folders: [..]'")

    match_dcd = cfg.get("match_dcd", "")
    match_colvars = cfg.get("match_colvars", "")
    tag_re = cfg.get("tag_regex", r"([AB])")
    top_path = cfg["io"]["top"]
    stride = int(cfg["io"].get("stride", 1))
    allow_skip_first = bool(cfg.get("pairing", {}).get("allow_skip_first_colvars", True))
    strict = bool(cfg.get("pairing", {}).get("strict", False))

    cv_columns = cluster_cfg.get("cvs")
    if not cv_columns:
        cv_columns = cfg.get("colvars", {}).get("cv")
    if isinstance(cv_columns, str):
        cv_columns = [cv_columns]
    if not cv_columns:
        raise SystemExit("cv_clustering.cvs or colvars.cv must define at least one CV column.")

    base_dir = os.path.dirname(os.path.abspath(parsed.config))
    output_dir = resolve_output_path(base_dir, cluster_cfg.get("out", os.path.join(cfg["io"].get("out", "rw_out"), "cvclusters")))
    os.makedirs(output_dir, exist_ok=True)

    method = str(cluster_cfg.get("method", "kmeans")).lower()
    use_fel_as_feature = bool(cluster_cfg.get("use_fel_as_feature", True))
    filters = cluster_cfg.get("filters", {})
    fel_path = cluster_cfg.get("fel_dat")
    if fel_path:
        fel_path = resolve_output_path(base_dir, fel_path)
    fel_cutoff = cluster_cfg.get("fel_cutoff_kcal_per_mol")
    n_select = int(cluster_cfg.get("n_select", 20))
    max_k = int(cluster_cfg.get("max_k", 10))
    random_state = int(cluster_cfg.get("seed", cfg.get("riteweight", {}).get("seed", 2026)))
    structure_format = str(cluster_cfg.get("structure_format", "pdb")).lower()
    save_structures = bool(cluster_cfg.get("save_structures", True))
    plot_columns = cluster_cfg.get("plot_cvs", cv_columns[:2])
    if isinstance(plot_columns, str):
        plot_columns = [plot_columns]

    pairs = find_pairs_dcd_colvars(roots, match_dcd, match_colvars, tag_re=tag_re)
    print(f"[INFO] Found {len(pairs)} candidate (dcd, colvars) pairs.")

    build = build_frame_table(
        pairs=pairs,
        top_path=top_path,
        stride=stride,
        allow_skip_first=allow_skip_first,
        strict=strict,
    )
    build.report.to_csv(os.path.join(output_dir, "pair_alignment_report.csv"), index=False)
    print(build.report[["pair_id", "dcd_frames", "colvars_rows", "action", "aligned_rows", "ok"]].to_string(index=False))

    fel_grid = load_gromacs_fel_2d(fel_path) if fel_path else None
    merged = build.frame_table.copy()
    merged = apply_cv_filters(merged, filters)
    merged = apply_fel_cutoff(
        merged,
        cv_columns=cv_columns,
        fel_grid=fel_grid,
        fel_cutoff=fel_cutoff,
    )
    if merged.empty:
        raise SystemExit("No frames remain after applying the configured CV/FEL basin filters.")
    if "fel_kcal_per_mol" not in merged.columns:
        raise SystemExit("FEL values are required for clustering. Set cv_clustering.fel_dat.")
    missing_plot = [column for column in plot_columns if column not in merged.columns]
    if missing_plot:
        raise SystemExit(f"Plot CV columns missing from merged frame data: {missing_plot}")

    feature_matrix, feature_names = build_feature_matrix(
        merged,
        cv_columns=cv_columns,
        use_fel_as_feature=use_fel_as_feature,
    )

    centers: Dict[int, np.ndarray]
    if method == "kmeans":
        requested_k = cluster_cfg.get("n_clusters")
        if requested_k is None:
            upper_k = min(max_k, len(merged))
            if upper_k < 1:
                raise SystemExit("Not enough frames available to determine k.")
            k_values, inertias = compute_elbow_curve(
                feature_matrix,
                max_k=upper_k,
                random_state=random_state,
            )
            n_clusters = pick_k_by_elbow(k_values, inertias)
            pd.DataFrame({"k": k_values, "inertia": inertias}).to_csv(
                os.path.join(output_dir, "elbow_curve.csv"),
                index=False,
            )
            plot_elbow(k_values, inertias, n_clusters, os.path.join(output_dir, "elbow_curve.png"))
        else:
            n_clusters = int(requested_k)

        if n_clusters < 1:
            raise SystemExit("n_clusters must be >= 1")
        if n_clusters > len(merged):
            raise SystemExit("n_clusters cannot exceed the number of available frames.")

        model = fit_kmeans(
            feature_matrix,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        labels = model.labels_.astype(int)
        centers = {int(i): center for i, center in enumerate(model.cluster_centers_)}
    elif method == "dbscan":
        dbscan_cfg = cluster_cfg.get("dbscan", {})
        eps = dbscan_cfg.get("eps", cluster_cfg.get("eps", 0.25))
        min_samples = dbscan_cfg.get("min_samples", cluster_cfg.get("min_samples", 10))
        labels = fit_dbscan(feature_matrix, eps=eps, min_samples=min_samples).astype(int)
        centers = build_cluster_centers(feature_matrix, labels)
    elif method == "hdbscan":
        hdbscan_cfg = cluster_cfg.get("hdbscan", {})
        min_cluster_size = hdbscan_cfg.get("min_cluster_size", cluster_cfg.get("min_cluster_size", 20))
        min_samples = hdbscan_cfg.get("min_samples", cluster_cfg.get("min_samples", None))
        cluster_selection_epsilon = hdbscan_cfg.get(
            "cluster_selection_epsilon",
            cluster_cfg.get("cluster_selection_epsilon", 0.0),
        )
        labels = fit_hdbscan(
            feature_matrix,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        ).astype(int)
        centers = build_cluster_centers(feature_matrix, labels)
    elif method == "fel_components":
        fc_cfg = cluster_cfg.get("fel_components", {})
        connectivity = int(fc_cfg.get("connectivity", cluster_cfg.get("connectivity", 8)))
        min_component_frames = int(fc_cfg.get("min_component_frames", cluster_cfg.get("min_component_frames", 1)))
        if fel_grid is None:
            raise SystemExit("method: fel_components requires cv_clustering.fel_dat.")
        labels = fit_fel_components(
            merged,
            fel_grid=fel_grid,
            fel_cutoff=fel_cutoff,
            connectivity=connectivity,
            min_component_frames=min_component_frames,
        ).astype(int)
        centers = build_cluster_centers(feature_matrix, labels)
    else:
        raise SystemExit("Unsupported cv_clustering.method. Use kmeans, dbscan, hdbscan, or fel_components.")

    merged["cluster"] = labels
    non_noise = merged["cluster"] >= 0
    if not non_noise.any():
        raise SystemExit(f"Clustering method '{method}' did not produce any non-noise clusters.")

    representatives = choose_representatives(
        merged.loc[non_noise].reset_index(drop=True),
        data=feature_matrix[non_noise.to_numpy()],
        labels=labels[non_noise.to_numpy()],
        centers=centers,
        n_select=n_select,
    )
    representatives["selected_rank"] = representatives.groupby("cluster")["distance_to_center"].rank(method="first")

    merged["selected"] = False
    merged.loc[merged["frame"].isin(representatives["frame"]), "selected"] = True

    merged.to_csv(os.path.join(output_dir, "clustered_frames.csv"), index=False)
    representatives.to_csv(os.path.join(output_dir, "selected_structures.csv"), index=False)

    cluster_sizes = merged.loc[non_noise].groupby("cluster").size().rename("n_frames")
    cluster_fel = merged.loc[non_noise].groupby("cluster")["fel_kcal_per_mol"].mean().rename("mean_fel_kcal_per_mol")
    cluster_summary = pd.concat([cluster_sizes, cluster_fel], axis=1).reset_index()
    cluster_summary.to_csv(os.path.join(output_dir, "cluster_summary.csv"), index=False)

    plot_clusters(
        merged,
        plot_columns=plot_columns,
        selected_frames=representatives,
        out_path=os.path.join(output_dir, "clusters_on_cvs.png"),
        fel_cutoff=fel_cutoff,
        fel_grid=fel_grid,
    )

    if save_structures:
        save_selected_structures(
            representatives,
            top_path=top_path,
            out_dir=os.path.join(output_dir, "selected_structures"),
            structure_format=structure_format,
        )

    if fel_path:
        print(f"[OK] FEL basin filter: {fel_path} with cutoff <= {fel_cutoff} kcal/mol")
    print(f"[OK] clustering method = {method}")
    print(f"[OK] Feature columns used for clustering: {feature_names}")
    print(f"[OK] discovered clusters = {int(merged.loc[merged['cluster'] >= 0, 'cluster'].nunique())}")
    if (merged["cluster"] < 0).any():
        print(f"[OK] noise/unassigned frames = {int((merged['cluster'] < 0).sum())}")
    print(f"[OK] Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
