#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import re
import glob
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


NODE_RE = re.compile(r"node_(\d+)")


def read_colvars_traj(path: str) -> Tuple[List[str], np.ndarray]:
    """
    Read NAMD colvars .traj-like file.
    Returns (headers, data) where data is float ndarray.
    - Header expected in a line starting with '#', containing column names (including 'step' usually).
    - Handles multiple comment lines.
    """
    headers = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                toks = s.lstrip("#").strip().split()
                # pick the first plausible header line
                if len(toks) >= 2 and all("=" not in t for t in toks):
                    headers = toks
            else:
                break

    data = np.loadtxt(path, comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if headers is None or len(headers) != data.shape[1]:
        # fallback generic names
        headers = [f"col{i}" for i in range(data.shape[1])]
    else:
        # de-duplicate headers: keep first occurrence
        seen = set()
        keep = []
        new_headers = []
        for i, h in enumerate(headers):
            if h not in seen:
                seen.add(h)
                keep.append(i)
                new_headers.append(h)
        data = data[:, keep]
        headers = new_headers

    return headers, data


def maybe_skip_first_row_if_needed(data: np.ndarray, expected_len: Optional[int], allow_skip_first: bool) -> np.ndarray:
    """
    If expected_len is given, and data has expected_len+1 rows, optionally skip first row.
    In this script we usually don't know expected_len (no DCD), so we only apply skip when asked
    AND file looks like it has a duplicated first step (common colvars mismatch symptom).
    """
    if not allow_skip_first:
        return data
    if data.shape[0] >= 2:
        # heuristic: if first two rows have identical 'step' (col0 often step) or step decreases
        # this is a weak heuristic but matches common colvars mismatch cases.
        step0 = data[0, 0]
        step1 = data[1, 0]
        if step1 <= step0:
            return data[1:, :]
    return data


def find_node_traj_files(roots: List[str], file_glob: str) -> List[Tuple[int, str]]:
    """
    Return list of (node_id, filepath) for files under roots matching node_XXXX/**/file_glob
    """
    out = []
    for root in roots:
        pattern = os.path.join(root, "node_*", "**", file_glob)
        for fp in sorted(glob.glob(pattern, recursive=True)):
            mm = NODE_RE.search(fp)
            if mm:
                node_id = int(mm.group(1))
                out.append((node_id, fp))
    return out


def downsample_per_node(df: pd.DataFrame, node_col: str, max_points: Optional[int], seed: int = 2026) -> pd.DataFrame:
    if max_points is None:
        return df
    rng = np.random.default_rng(seed)
    parts = []
    for node, grp in df.groupby(node_col, sort=True):
        if len(grp) > max_points:
            idx = rng.choice(grp.index.to_numpy(), size=max_points, replace=False)
            parts.append(grp.loc[np.sort(idx)])
        else:
            parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def violin_plot(df: pd.DataFrame, node_col: str, value_col: str, out_path: str, dpi: int = 200):
    # nodes sorted
    nodes = sorted(df[node_col].unique().tolist())
    data = [df.loc[df[node_col] == n, value_col].to_numpy(dtype=float) for n in nodes]

    plt.figure(figsize=(max(6, 0.35 * len(nodes)), 4.5))
    parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    # x tick labels
    plt.xticks(np.arange(1, len(nodes) + 1), [str(n) for n in nodes], rotation=90)
    plt.xlabel("path node")
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def write_pathway(df: pd.DataFrame, node_col: str, cvs: List[str], method: str, out_txt: str):
    """
    Build a pathway as node-dependent aggregated CV values.
    Output format: plain text, space separated, no header
    """

    if method not in ("median", "mean"):
        raise ValueError(f"Unsupported pathway method: {method}")

    rows = []
    for node, grp in df.groupby(node_col, sort=True):

        vals = []
        for cv in cvs:
            data = grp[cv].to_numpy(dtype=float)

            if method == "median":
                vals.append(np.nanmedian(data))
            else:
                vals.append(np.nanmean(data))

        rows.append(vals)

    path = np.array(rows, dtype=float)

    np.savetxt(
        out_txt,
        path,
        fmt="%.18e",
        delimiter=" "
    )

    print(f"[OK] Saved pathway TXT: {out_txt}")

    return path

def main():
    ap = argparse.ArgumentParser(description="Make violin plots of CV distributions per node from node_XXXX/*traj files.")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vc = cfg.get("violin_plot", {})
    out_dir = vc.get("out_dir", "violin_out")
    os.makedirs(out_dir, exist_ok=True)

    roots = vc.get("roots", [])
    if isinstance(roots, str):
        roots = [roots]
    if not roots:
        raise SystemExit("config.yaml: violin_plot.roots is required")

    file_glob = vc.get("file_glob", "*traj")
    cvs = vc.get("cvs", [])
    if isinstance(cvs, str):
        cvs = [cvs]
    if not cvs:
        raise SystemExit("config.yaml: violin_plot.cvs must be a list of CV column names")

    allow_skip_first = bool(vc.get("allow_skip_first_colvars", True))
    stride = int(vc.get("stride", 1))

    combined_csv_name = vc.get("combined_csv", "combined_cvs.csv")
    combined_csv_path = os.path.join(out_dir, combined_csv_name)

    plot_cfg = vc.get("plot", {})
    fmt = str(plot_cfg.get("format", "png")).lower()
    dpi = int(plot_cfg.get("dpi", 200))
    node_min = plot_cfg.get("node_min", None)
    node_max = plot_cfg.get("node_max", None)
    max_points_per_node = plot_cfg.get("max_points_per_node", None)

    # 1) find files
    files = find_node_traj_files(roots, file_glob)
    if not files:
        raise SystemExit(f"No files found for roots={roots} with file_glob='{file_glob}' under node_*")

    # optional node range filter
    if node_min is not None:
        files = [(n, fp) for (n, fp) in files if n >= int(node_min)]
    if node_max is not None:
        files = [(n, fp) for (n, fp) in files if n <= int(node_max)]

    if not files:
        raise SystemExit("No files left after node_min/node_max filtering.")

    print(f"[INFO] Found {len(files)} traj files.")

    # 2) read and collect
    rows = []
    for node_id, fp in files:
        headers, data = read_colvars_traj(fp)
        data = maybe_skip_first_row_if_needed(data, expected_len=None, allow_skip_first=allow_skip_first)

        if stride != 1:
            data = data[::stride, :]

        h2i = {h: i for i, h in enumerate(headers)}
        missing = [c for c in cvs if c not in h2i]
        if missing:
            raise SystemExit(f"Missing columns {missing} in file: {fp}\nAvailable columns: {headers}")

        # build records
        sub = data[:, [h2i[c] for c in cvs]].astype(np.float64)
        for i in range(sub.shape[0]):
            rec = {"node": node_id, "file": fp, "row": i}
            for j, c in enumerate(cvs):
                rec[c] = float(sub[i, j])
            rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No data collected.")

    # 3) optional downsample per node
    if max_points_per_node is not None:
        df = downsample_per_node(df, "node", int(max_points_per_node))

    # 4) save combined csv
    df.to_csv(combined_csv_path, index=False)
    print(f"[OK] Saved combined CSV: {combined_csv_path}  rows={len(df)}")

    # 5) violin plots (one per CV)
    for c in cvs:
        out_path = os.path.join(out_dir, f"violin_{c}.{fmt}")
        violin_plot(df, node_col="node", value_col=c, out_path=out_path, dpi=dpi)
        print(f"[OK] Saved violin plot: {out_path}")

    # 6) pathway output
    path_cfg = vc.get("pathway", {})
    if bool(path_cfg.get("enabled", False)):
        path_cvs = path_cfg.get("cvs", cvs)
        if isinstance(path_cvs, str):
            path_cvs = [path_cvs]

        missing = [c for c in path_cvs if c not in df.columns]
        if missing:
            raise SystemExit(f"pathway.cvs missing from combined dataframe: {missing}")

        method = str(path_cfg.get("method", "median")).lower()
        path_csv = os.path.join(out_dir, path_cfg.get("output_txt", "pathway.txt"))

        path_df = write_pathway(df, node_col="node", cvs=path_cvs, method=method, out_txt=path_csv)

if __name__ == "__main__":
    main()