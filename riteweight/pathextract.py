#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, re, glob, argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import yaml

import MDAnalysis as mda
from MDAnalysis.coordinates.DCD import DCDWriter


# =========================================================
# === Pairing: (dcd, colvars) recursive + (subdir, A/B) ===
# =========================================================

def find_matching(root: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))

def pair_by_subdir_tag(files: List[str], root: str, tag_re: str) -> Dict[Tuple[str, str], str]:
    m: Dict[Tuple[str, str], str] = {}
    cre = re.compile(tag_re)
    for fp in files:
        rel = os.path.relpath(fp, root)
        sub = os.path.dirname(rel)
        mm = cre.search(os.path.basename(fp))
        if mm:
            tag = mm.group(1)
            m[(sub, tag)] = fp
    return m

def find_pairs_dcd_colvars(roots: List[str], match_dcd: str, match_colvars: str, tag_re: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for root in roots:
        dcds = find_matching(root, f"*{match_dcd}*.dcd")
        cols = find_matching(root, f"*{match_colvars}*.colvars.traj")
        if not dcds or not cols:
            continue
        m_d = pair_by_subdir_tag(dcds, root, tag_re)
        m_c = pair_by_subdir_tag(cols, root, tag_re)
        for k in sorted(set(m_d) & set(m_c)):
            pairs.append((m_d[k], m_c[k]))
    if not pairs:
        raise FileNotFoundError("No matching (dcd, colvars) pairs found.")
    return sorted(pairs)


# =========================================================
# === Colvars reading + mismatch alignment ===
# =========================================================

def read_colvars_traj(path: str) -> pd.DataFrame:
    colnames = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                toks = s.lstrip("#").strip().split()
                if len(toks) >= 2 and all("=" not in t for t in toks):
                    colnames = toks
            else:
                break
    df = pd.read_csv(path, delim_whitespace=True, comment="#", header=None)
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


# =========================================================
# === Reaction path distance (with periodic support) ===
# =========================================================

def periodic_diff(x: np.ndarray, y: np.ndarray, period: float) -> np.ndarray:
    """Return minimal signed difference x-y on a periodic domain."""
    d = x - y
    d = (d + 0.5 * period) % period - 0.5 * period
    return d

def frame_to_path_min_dist(
    X: np.ndarray,                     # (Nframes, D)
    path: np.ndarray,                  # (Npath, D)
    periodic: np.ndarray,              # (D,) 0/1
    period: np.ndarray,                # (D,)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each frame: compute min distance to any path node and argmin node index.
    Uses Euclidean norm with periodic dims handled by minimal image difference.
    """
    N, D = X.shape
    M = path.shape[0]
    min_d2 = np.full(N, np.inf, dtype=np.float64)
    argmin = np.full(N, -1, dtype=np.int32)

    # process path nodes one by one to avoid gigantic (N,M,D) allocation
    for j in range(M):
        pj = path[j][None, :]  # (1,D)
        diff = X - pj
        # periodic dims
        for k in range(D):
            if periodic[k]:
                diff[:, k] = periodic_diff(X[:, k], pj[0, k], period[k])
        d2 = np.sum(diff * diff, axis=1)
        better = d2 < min_d2
        min_d2[better] = d2[better]
        argmin[better] = j

    return np.sqrt(min_d2), argmin


# =========================================================
# === Main extraction ===
# =========================================================

def load_reaction_path(path_file: str) -> np.ndarray:
    arr = np.loadtxt(path_file, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def main():
    ap = argparse.ArgumentParser(description="Extract structures closest to a CV-defined reaction path.")
    ap.add_argument("--config", required=True, help="config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    roots = cfg.get("folders", [])
    if not roots:
        raise SystemExit("config.yaml must define folders: [..]")
    match_dcd = cfg.get("match_dcd", "")
    match_colvars = cfg.get("match_colvars", "")
    tag_re = cfg.get("tag_regex", r"([AB])")

    top = cfg["io"]["top"]

    pe = cfg.get("path_extract", {})
    if not pe.get("enabled", True):
        print("[INFO] path_extract.enabled is false; nothing to do.")
        return

    path_file = pe["reaction_path_file"]
    cv_names = pe["cv_names"]
    periodic = np.asarray(pe.get("periodic", [0]*len(cv_names)), dtype=int)
    period = np.asarray(pe.get("period", [360.0]*len(cv_names)), dtype=float)

    if len(cv_names) != len(periodic) or len(cv_names) != len(period):
        raise SystemExit("path_extract: cv_names / periodic / period lengths must match.")

    n_per_node = int(pe.get("n_per_node", 1))
    max_total = pe.get("max_total", None)
    max_total = int(max_total) if max_total is not None else None

    selection_output = pe.get("selection_output", "all")
    out_dir = pe.get("out_dir", "path_extract_out")
    out_dcd = pe.get("out_dcd", "near_path.dcd")
    out_pdb = pe.get("out_pdb", "near_path.pdb")
    out_csv = pe.get("out_csv", "selected_frames.csv")

    stride = int(pe.get("stride", cfg.get("io", {}).get("stride", 1)))
    allow_skip_first = bool(pe.get("allow_skip_first_colvars", True))

    os.makedirs(out_dir, exist_ok=True)
    out_dcd_path = os.path.join(out_dir, out_dcd)
    out_pdb_path = os.path.join(out_dir, out_pdb)
    out_csv_path = os.path.join(out_dir, out_csv)

    # load path
    path_pts = load_reaction_path(path_file)
    D = path_pts.shape[1]
    if D != len(cv_names):
        raise SystemExit(f"Reaction path has {D} columns, but cv_names has {len(cv_names)}.")

    # find pairs
    pairs = find_pairs_dcd_colvars(roots, match_dcd, match_colvars, tag_re)
    print(f"[INFO] Found {len(pairs)} (dcd,colvars) pairs.")

    # pass 1: compute per-frame distance and keep metadata (no coordinates loaded into memory)
    rows = []
    global_idx = 0
    for (dcd_path, col_path) in pairs:
        # read colvars
        df = read_colvars_traj(col_path)
        # read trajectory length (fast in MDAnalysis)
        u = mda.Universe(top, dcd_path)
        n_frames = len(u.trajectory)

        if stride != 1:
            # note: MDAnalysis stride happens during iteration; for colvars we stride rows
            df = df.iloc[::stride].reset_index(drop=True)
            n_frames_eff = (n_frames + stride - 1) // stride
        else:
            n_frames_eff = n_frames

        df2, action = maybe_align_colvars(df, n_frames_eff, allow_skip_first)
        if action == "mismatch":
            print(f"[WARN] mismatch, skipped: dcd={dcd_path} frames_eff={n_frames_eff} colvars={len(df)}")
            continue
        if action == "skip_first_colvars":
            print(f"[INFO] skip_first_colvars: {col_path}")

        missing = [c for c in cv_names if c not in df2.columns]
        if missing:
            raise SystemExit(f"Missing CV columns in {col_path}: {missing}")

        X = df2[cv_names].to_numpy(dtype=np.float64)
        if X.shape[0] != n_frames_eff:
            print(f"[WARN] post-align mismatch, skipped: {dcd_path} X={X.shape[0]} frames_eff={n_frames_eff}")
            continue

        dist, node = frame_to_path_min_dist(X, path_pts, periodic, period)

        # store per-frame metadata
        # local_frame is the *effective* frame index after stride (0..n_frames_eff-1)
        for i in range(n_frames_eff):
            rows.append((dcd_path, top, i, global_idx, int(node[i]), float(dist[i])))
            global_idx += 1

    if not rows:
        raise SystemExit("No usable frames found after mismatch checks.")

    meta = pd.DataFrame(rows, columns=["dcd", "top", "local_frame_eff", "global_frame", "path_node", "dist"])
    print(f"[INFO] Total usable frames: {len(meta)}")

    # selection: pick nearest n_per_node per node
    picked = []
    for node_id, grp in meta.groupby("path_node", sort=True):
        grp2 = grp.sort_values("dist", ascending=True)
        take = grp2.head(n_per_node)
        picked.append(take)
    picked_df = pd.concat(picked, ignore_index=True)

    # optional cap total
    if max_total is not None and len(picked_df) > max_total:
        picked_df = picked_df.sort_values("dist", ascending=True).head(max_total).reset_index(drop=True)

    # For reproducible writing order: sort by path_node then dist
    picked_df = picked_df.sort_values(["path_node", "dist"], ascending=[True, True]).reset_index(drop=True)

    print(f"[INFO] Selected frames: {len(picked_df)}  (n_per_node={n_per_node}, max_total={max_total})")



    # ---------------------------------------------------------
    # Write global manifest (optional, convenient)
    # ---------------------------------------------------------
    all_csv_name = pe.get("all_csv", "selected_frames_all.csv")
    all_csv_path = os.path.join(out_dir, all_csv_name)
    picked_df.to_csv(all_csv_path, index=False)
    print(f"[OK] Wrote global selection manifest: {all_csv_path}")

    # ---------------------------------------------------------
    # Per-node outputs: folder + DCD/PDB/CSV
    # ---------------------------------------------------------
    node_dir_fmt = pe.get("node_dir_fmt", "node_{node:04d}")
    node_dcd_fmt = pe.get("node_dcd_fmt", "near_path_node{node:04d}.dcd")
    node_pdb_fmt = pe.get("node_pdb_fmt", "near_path_node{node:04d}.pdb")
    node_csv_fmt = pe.get("node_csv_fmt", "selected_frames_node{node:04d}.csv")

    # We will reuse the same atom selection string, but write per-node files.
    # For each node, we:
    #  1) write a node-specific CSV manifest
    #  2) write a node-specific PDB (topology for selected atoms)
    #  3) write a node-specific DCD containing only that node's frames
    #
    # Efficiency: for each node we group by dcd and then read only required frames.

    # Helper to build selection atoms for a given dcd (same natoms required across nodes)
    def make_selection_universe(dcd_path: str):
        u = mda.Universe(top, dcd_path)
        sel_atoms = u.select_atoms(selection_output)
        if sel_atoms.n_atoms == 0:
            raise SystemExit(f"selection_output matched 0 atoms: {selection_output}")
        return u, sel_atoms

    # Determine natoms once, and also write a reference PDB if you want (optional)
    # Here we will write per-node PDB anyway, but we still need natoms for DCDWriter.
    ref_dcd = picked_df.iloc[0]["dcd"]
    u_ref, sel_ref = make_selection_universe(ref_dcd)
    natoms = sel_ref.n_atoms

    # Iterate nodes in order
    for node_id, node_grp in picked_df.groupby("path_node", sort=True):
        node_id = int(node_id)
        node_subdir = os.path.join(out_dir, node_dir_fmt.format(node=node_id))
        os.makedirs(node_subdir, exist_ok=True)

        node_csv_path = os.path.join(node_subdir, node_csv_fmt.format(node=node_id))
        node_dcd_path = os.path.join(node_subdir, node_dcd_fmt.format(node=node_id))
        node_pdb_path = os.path.join(node_subdir, node_pdb_fmt.format(node=node_id))

        # save node manifest
        node_grp = node_grp.sort_values(["dist"], ascending=True).reset_index(drop=True)
        node_grp.to_csv(node_csv_path, index=False)

        # write node PDB topology from the first frame's dcd in this node
        first_dcd = node_grp.iloc[0]["dcd"]
        u0, sel0 = make_selection_universe(first_dcd)
        if sel0.n_atoms != natoms:
            raise SystemExit(
                f"Atom selection size changed across DCDs: ref={natoms}, node{node_id} first_dcd={sel0.n_atoms}. "
                "Use a selection that yields consistent atom counts."
            )
        sel0.write(node_pdb_path)

        # write node DCD
        by_dcd = node_grp.groupby("dcd", sort=False)

        with DCDWriter(node_dcd_path, natoms) as W:
            for dcd_path, grp in by_dcd:
                u = mda.Universe(top, dcd_path)
                sel_u = u.select_atoms(selection_output)
                if sel_u.n_atoms != natoms:
                    raise SystemExit(
                        f"Atom selection size changed across DCDs: ref={natoms}, got={sel_u.n_atoms} for {dcd_path}."
                    )

                frames_eff = grp["local_frame_eff"].to_numpy(dtype=int)
                for i_eff in frames_eff:
                    i_actual = int(i_eff) * stride
                    if i_actual >= len(u.trajectory):
                        print(f"[WARN] frame out of range, skip: node={node_id} {dcd_path} eff={i_eff} actual={i_actual}")
                        continue
                    u.trajectory[i_actual]
                    W.write(sel_u)

        print(f"[OK] node {node_id:04d}: wrote {len(node_grp)} frames")
        print(f"     CSV: {node_csv_path}")
        print(f"     PDB: {node_pdb_path}")
        print(f"     DCD: {node_dcd_path}")


if __name__ == "__main__":
    main()