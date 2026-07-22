#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, glob, os, re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy import ndimage
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.feature_contract import FEATURE_SCHEMA
from common.config import normalize_config_aliases
from tools.tensor_table import save_tensor_table

try:
    import yaml
except ImportError as e:
    raise SystemExit("Need pyyaml. Install: pip install pyyaml") from e

try:
    import mdtraj as md
except ImportError as e:
    raise SystemExit("Need mdtraj. Install: pip install mdtraj") from e

from vcn.zmatrix import get_minimal_internal_coordinates

KB_KCAL_PER_MOL_K = 0.00198720425864083


# =========================================================
# === File finding & pairing (recursive + (subdir, tag)) ===
# =========================================================

def find_matching(root: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))

def pair_by_subdir_tag(files: List[str], root: str, tag_re: str) -> Dict[Tuple[str, str], str]:
    m: Dict[Tuple[str, str], str] = {}
    cre = re.compile(tag_re)
    for fp in files:
        rel = os.path.relpath(fp, root)
        sub = os.path.dirname(rel)
        bname = os.path.basename(fp)
        mm = cre.search(bname)
        if mm:
            tag = mm.group(1)
            key = (sub, tag)
            if key in m:
                raise ValueError(
                    f"Multiple files map to the same RiteWeight pair key {key}: "
                    f"{m[key]} and {fp}. Tighten the file patterns."
                )
            m[key] = fp
    return m

def find_pairs_dcd_colvars(
    roots: List[str],
    dcd_pattern: str,
    colvars_pattern: str,
    tag_re: str = r"([ABM])",
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for root in roots:
        dcds = find_matching(root, dcd_pattern)
        cols = find_matching(root, colvars_pattern)

        if not dcds:
            print(f"[WARN] No DCD files under {root} matching {dcd_pattern!r}")
            continue
        if not cols:
            print(f"[WARN] No colvars files under {root} matching {colvars_pattern!r}")
            continue

        m_d = pair_by_subdir_tag(dcds, root, tag_re)
        m_c = pair_by_subdir_tag(cols, root, tag_re)
        common = sorted(set(m_d) & set(m_c))
        local_pairs = [(m_d[k], m_c[k]) for k in common]
        if not local_pairs:
            print(f"[WARN] No matching (dcd,colvars) pairs found in root={root} with tag_re={tag_re}")
        pairs.extend(local_pairs)

    if not pairs:
        raise FileNotFoundError("No matching (dcd, colvars) pairs found across all folders.")
    return sorted(pairs)


# =========================================================
# === Colvars reader & alignment ===
# =========================================================

def read_colvars_traj(path: str) -> pd.DataFrame:
    colnames = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                tokens = s.lstrip("#").strip().split()
                if len(tokens) >= 2 and all("=" not in t for t in tokens):
                    colnames = tokens
            else:
                break

    df = pd.read_csv(path, sep=r'\s+', comment="#", header=None)
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

def dedup_columns_keep_first(cols: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def determine_AB_functor(basin_A, basin_B, basin_size):
    basin_A = np.asarray(basin_A, float)
    basin_B = np.asarray(basin_B, float)
    if np.isscalar(basin_size):
        basin_size = np.full_like(basin_A, float(basin_size))
    else:
        basin_size = np.asarray(basin_size, float)

    def in_basin(x, center):
        return np.all(np.abs(np.asarray(x, float) - center) <= basin_size)

    def determine_AB(x):
        if in_basin(x, basin_A):
            return "A"
        if in_basin(x, basin_B):
            return "B"
        return "M"

    return determine_AB


# =========================================================
# === Features: distances / internal_zmat (atom_order or atomselect) ===
# =========================================================

def features_distances(traj: "md.Trajectory", atom_pairs: List[List[int]]) -> np.ndarray:
    pairs = np.array(atom_pairs, dtype=int)
    return md.compute_distances(traj, pairs)  # nm

def resolve_zmat_atoms(
    top: "md.Topology",
    internal_cfg: Dict[str, Any],
) -> List[int]:
    """
    Resolve Z-matrix atom indices from either:
      - internal_cfg["atom_order"] : explicit list
      - internal_cfg["atomselect"] : mdtraj selection DSL
    For reproducibility, default ordering is sorted indices ("index").
    """
    if "atom_order" in internal_cfg and internal_cfg["atom_order"] is not None:
        ao = [int(x) for x in internal_cfg["atom_order"]]
        if len(ao) < 4:
            raise ValueError("internal_zmat.atom_order must have at least 4 atoms.")
        return ao

    sel = internal_cfg.get("atomselect", None)
    if not sel:
        raise ValueError("internal_zmat must define either atom_order or atomselect.")

    idx = top.select(sel)
    if idx.size < 4:
        raise ValueError(f"atomselect picked {idx.size} atoms (<4). Selection: {sel}")

    # limit number of atoms if requested
    max_atoms = internal_cfg.get("max_atoms", None)
    if max_atoms is not None:
        max_atoms = int(max_atoms)
        idx = idx[:max_atoms]
        if idx.size < 4:
            raise ValueError("max_atoms made selected atoms <4; need at least 4.")

    order = internal_cfg.get("order", "index")
    if order == "index":
        ao = sorted(map(int, idx.tolist()))
    else:
        raise ValueError(f"Unsupported internal_zmat.order: {order}")

    return ao

def features_internal_zmat(traj: "md.Trajectory", atom_order: List[int]) -> np.ndarray:
    """
    Return the exact internal-coordinate representation used by VCN.

    ``get_minimal_internal_coordinates`` accepts one-based atom indices;
    RiteWeight resolves zero-based MDTraj indices, so convert only the index
    convention here and keep all feature computation in the shared function.
    """
    ao = list(map(int, atom_order))
    if len(ao) < 4:
        raise ValueError("internal_zmat needs N>=4 atoms.")
    _, values = get_minimal_internal_coordinates(traj, [index + 1 for index in ao])
    return values


# =========================================================
# === RiteWeight core ===
# =========================================================

def assign_clusters_random_centers(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    M = X.shape[0]
    if n_clusters >= M:
        raise ValueError(f"n_clusters ({n_clusters}) must be < #configs ({M}).")
    center_idx = rng.choice(M, size=n_clusters, replace=False)
    C = X[center_idx]
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(C * C, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (X @ C.T)
    return np.argmin(d2, axis=1).astype(np.int32)

def build_transition_matrix(start_labels, end_labels, w, n_clusters, eps=1e-15):
    T_num = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    wI = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(wI, start_labels, w)
    np.add.at(T_num, (start_labels, end_labels), w)
    T = T_num / (wI[:, None] + eps)
    zero = wI <= eps
    if np.any(zero):
        T[zero, :] = 0.0
        T[zero, np.where(zero)[0]] = 1.0
    T = np.clip(T, 0.0, 1.0)
    rs = T.sum(axis=1, keepdims=True)
    T = T / (rs + eps)
    return T, wI

def stationary_distribution(T):
    vals, vecs = eig(T.T)
    idx = np.argmin(np.abs(vals - 1.0))
    v = np.real(vecs[:, idx])
    v = np.abs(v)
    s = v.sum()
    if s <= 0:
        raise RuntimeError("Invalid stationary distribution.")
    return v / s

@dataclass
class RiteWeightResult:
    w_segment: np.ndarray
    w_frame_nonzero: np.ndarray
    frame_map: np.ndarray
    delta_history: List[float]

def riteweight(X, seg_start_idx, seg_end_idx, n_clusters, n_iter, tol, tol_window, avg_last, seed):
    rng = np.random.default_rng(seed)
    Nseg = seg_start_idx.size
    w = np.full(Nseg, 1.0 / Nseg, dtype=np.float64)
    w_prev = w.copy()
    delta_hist: List[float] = []
    ok = 0
    accum: List[np.ndarray] = []

    for it in tqdm(range(1, n_iter + 1)):
        labels = assign_clusters_random_centers(X, n_clusters, rng)
        start_lab = labels[seg_start_idx]
        end_lab = labels[seg_end_idx]

        T, wI = build_transition_matrix(start_lab, end_lab, w, n_clusters)
        pi = stationary_distribution(T)

        scale = pi / (wI + 1e-15)
        w_new = w * scale[start_lab]
        w_new = np.clip(w_new, 0.0, np.inf)
        w_new /= w_new.sum()

        delta = float(np.sum(np.abs(w_new - w_prev)))
        delta_hist.append(delta)
        ok = ok + 1 if delta < tol else 0

        if it > max(1, n_iter - avg_last):
            accum.append(w_new.copy())

        w_prev = w
        w = w_new
        if ok >= tol_window:
            break

    if accum:
        w_final = np.mean(np.stack(accum, axis=0), axis=0)
        w_final = np.clip(w_final, 0.0, np.inf)
        w_final /= w_final.sum()
    else:
        w_final = w

    w_frame = np.zeros(X.shape[0], dtype=np.float64)
    np.add.at(w_frame, seg_start_idx, w_final)
    mask = w_frame > 0
    frame_map = np.where(mask)[0]
    w_frame_nonzero = w_frame[mask]
    w_frame_nonzero /= w_frame_nonzero.sum()

    return RiteWeightResult(
        w_segment=w_final,
        w_frame_nonzero=w_frame_nonzero,
        frame_map=frame_map,
        delta_history=delta_hist,
    )


# =========================================================
# === FEL / PMF utilities (Gaussian smoothing + GROMACS-like .dat) ===
# =========================================================

def fes_from_prob(P, T_K, floor=1e-300):
    kT = KB_KCAL_PER_MOL_K * T_K
    P2 = np.clip(P, floor, 1.0)
    F = -kT * np.log(P2)
    F -= np.nanmin(F[np.isfinite(F)])
    return F

def gaussian_smooth_F(F: np.ndarray, sigma_bins: Optional[List[float]], F_max: Optional[float] = None) -> np.ndarray:
    if sigma_bins is None:
        return F
    sig = np.asarray(sigma_bins, float)
    if np.all(sig <= 0):
        return F
    G = ndimage.gaussian_filter(F, sigma=sig, mode="nearest")
    G -= np.nanmin(G[np.isfinite(G)])
    if F_max is not None:
        G = np.minimum(G, float(F_max))
    return G

def save_gromacs_like(filename: str, F: np.ndarray, edges: List[np.ndarray], cvs: List[str],
                     periodicities: List[int], xlim=None, ylim=None):
    ndim = F.ndim
    ctrs = [0.5 * (e[:-1] + e[1:]) for e in edges]
    nbin = [len(e) - 1 for e in edges]

    # apply xlim/ylim by re-gridding (same bins, new extents)
    xlim = xlim or [edges[0][0], edges[0][-1]]
    new_edges = [np.linspace(float(xlim[0]), float(xlim[1]), nbin[0] + 1)]
    if ndim == 2:
        ylim = ylim or [edges[1][0], edges[1][-1]]
        new_edges.append(np.linspace(float(ylim[0]), float(ylim[1]), nbin[1] + 1))
    new_ctrs = [0.5 * (e[:-1] + e[1:]) for e in new_edges]

    if len(periodicities) != ndim:
        raise ValueError(f"periodicities length {len(periodicities)} != ndim {ndim}")

    with open(filename, "w") as f:
        f.write(f"# {ndim}\n")
        for d in range(ndim):
            rmin = new_edges[d][0]
            binw = new_edges[d][1] - new_edges[d][0]
            f.write(f"# {rmin: .14e} {binw: .14e} {nbin[d]:8d} {int(periodicities[d])}\n")

        if ndim == 1:
            for i, x in enumerate(new_ctrs[0]):
                f.write(f"{x: .14e} {F[i]: .14e}\n")
        else:
            for i, x in enumerate(new_ctrs[0]):
                for j, y in enumerate(new_ctrs[1]):
                    f.write(f"{x: .14e} {y: .14e} {F[i, j]: .14e}\n")
                f.write("\n")

    # a quick pdf (like your reference code)
    pdf = filename + ".png"
    if ndim == 1:
        plt.figure()
        plt.plot(new_ctrs[0], F)
        plt.xlabel(cvs[0]); plt.ylabel("Free Energy (kcal/mol)")
        plt.tight_layout(); plt.savefig(pdf); plt.close()
    else:
        plt.figure()
        Xp, Yp = np.meshgrid(new_ctrs[0], new_ctrs[1], indexing="ij")
        plt.contourf(Xp, Yp, F, levels=20, cmap="RdBu_r")
        plt.colorbar(label="Free Energy (kcal/mol)")
        plt.xlabel(cvs[0]); plt.ylabel(cvs[1])
        plt.tight_layout(); plt.savefig(pdf); plt.close()

def weighted_hist_1d(x, w, bins, xlim=None):
    if xlim is None:
        xlim = (float(np.min(x)), float(np.max(x)))
    hist, edges = np.histogram(x, bins=bins, range=xlim, weights=w)
    P = hist.astype(np.float64)
    P /= P.sum() if P.sum() > 0 else 1.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    return P, centers, edges

def weighted_hist_2d(x, y, w, bins, xlim=None, ylim=None):
    if xlim is None:
        xlim = (float(np.min(x)), float(np.max(x)))
    if ylim is None:
        ylim = (float(np.min(y)), float(np.max(y)))
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim], weights=w)
    P = H.astype(np.float64)
    P /= P.sum() if P.sum() > 0 else 1.0
    xcent = 0.5 * (xedges[:-1] + xedges[1:])
    ycent = 0.5 * (yedges[:-1] + yedges[1:])
    return P, xcent, ycent, xedges, yedges

def save_scaled_pmf_if_requested(
    *,
    F: np.ndarray,
    edges: List[np.ndarray],
    cvs: List[str],
    periodicities: List[int],
    xlim,
    ylim,
    out_dir: str,
    scaled_bins: Optional[int],
    scaled_filename: str,
):
    """
    Resample F onto a coarser grid (scaled_bins per dimension) via linear interpolation,
    then save in gromacs-like .dat (+ .pdf).
    """
    if scaled_bins is None:
        return
    scaled_bins = int(scaled_bins)
    if scaled_bins <= 1:
        print("[WARN] scaled_bins <= 1, skip scaled output.")
        return

    ndim = F.ndim
    print(f"Generating scaled-down free energy ({scaled_bins} bins/dim)...")

    # original centers
    ctrs = [0.5 * (e[:-1] + e[1:]) for e in edges]

    # scaled edges/centers
    scaled_edges = [np.linspace(float(e[0]), float(e[-1]), scaled_bins + 1) for e in edges]
    scaled_ctrs = [0.5 * (e[:-1] + e[1:]) for e in scaled_edges]

    # interpolate
    if ndim == 1:
        xi = scaled_ctrs[0][:, None]
        F_scaled = interpn(
            ctrs, F, xi,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        ).reshape(scaled_bins)
    elif ndim == 2:
        Xn, Yn = np.meshgrid(scaled_ctrs[0], scaled_ctrs[1], indexing="ij")
        xi = np.stack([Xn, Yn], axis=-1).reshape(-1, 2)
        F_scaled = interpn(
            ctrs, F, xi,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        ).reshape(scaled_bins, scaled_bins)
    else:
        raise ValueError("Only 1D/2D PMF supported for scaled output.")

    # shift min to 0 (ignore nan)
    F_scaled -= np.nanmin(F_scaled)

    scaled_out_path = os.path.join(out_dir, scaled_filename)
    save_gromacs_like(scaled_out_path, F_scaled, scaled_edges, cvs, periodicities, xlim, ylim)
    print(f"Saved scaled free energy to {scaled_out_path} (+ .pdf)")

# =========================================================
# === Feature caching (with metadata validation) ===
# =========================================================


def cache_meta_dict(mode: str, cfg_features: dict, top_path: str, stride: int,
                    lag: int, pairs: List[Tuple[str, str]]) -> dict:
    """Minimal metadata to validate cache reproducibility."""
    meta = {
        "mode": mode,
        "feature_schema": FEATURE_SCHEMA,
        "top": os.path.abspath(top_path),
        "top_size": os.path.getsize(top_path),
        "top_mtime_ns": os.stat(top_path).st_mtime_ns,
        "stride": int(stride),
        "lag": int(lag),
        "sources": [
            {
                "dcd": os.path.abspath(dcd),
                "dcd_size": os.path.getsize(dcd),
                "dcd_mtime_ns": os.stat(dcd).st_mtime_ns,
                "colvars": os.path.abspath(colvars),
                "colvars_size": os.path.getsize(colvars),
                "colvars_mtime_ns": os.stat(colvars).st_mtime_ns,
            }
            for dcd, colvars in pairs
        ],
    }
    if mode == "internal_zmat":
        meta["internal_zmat"] = cfg_features.get("internal_zmat", {})
    elif mode == "distances":
        meta["distances"] = cfg_features.get("distances", {})
    return meta

def meta_to_string(meta: dict) -> str:
    # stable-ish string; YAML dump is convenient but keep dependency-free
    import json
    return json.dumps(meta, sort_keys=True)

def save_features_csv(path: str, X: np.ndarray, meta: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = "META=" + meta_to_string(meta)
    np.savetxt(path, X, delimiter=",", header=header, comments="")
    print(f"[INFO] Saved features CSV: {path}  shape={X.shape}")

def load_features_csv(path: str) -> Tuple[np.ndarray, Optional[dict]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    meta = None
    if first.startswith("META="):
        import json
        meta = json.loads(first[len("META="):])
        # load remainder as numeric
        X = np.loadtxt(path, delimiter=",", comments="#", skiprows=1)
    else:
        X = np.loadtxt(path, delimiter=",", comments="#")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    print(f"[INFO] Loaded features CSV: {path}  shape={X.shape}")
    return X, meta

def save_features_npz(path: str, X: np.ndarray, meta: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, X=X, meta=meta_to_string(meta))
    print(f"[INFO] Saved features NPZ: {path}  shape={X.shape}")

def load_features_npz(path: str) -> Tuple[np.ndarray, Optional[dict]]:
    z = np.load(path, allow_pickle=False)
    X = z["X"]
    meta = None
    if "meta" in z:
        import json
        meta = json.loads(str(z["meta"]))
    print(f"[INFO] Loaded features NPZ: {path}  shape={X.shape}")
    return X, meta

def load_or_compute_features_with_cache(
    *,
    cfg: dict,
    pairs: List[Tuple[str, str]],
    top_path: str,
    stride: int,
    allow_skip_first: bool,
    strict: bool,
    feat_mode: str,
    # internal_zmat support
    zmat_atom_order: Optional[List[int]],
    # distances support
    atom_pairs: Optional[List[List[int]]],
    # lag support (if you added lag)
    lag: int = 1,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    List[Tuple[str, str]],
    List[int],
]:
    """
    Returns:
      X_all, CV_all, seg_start, seg_end, df_save_all, used_pairs, nframes_each
    It will:
      - if cache policy says 'read' and cache exists: load X_all and still parse CV/segments by reading colvars and counting frames
      - else compute X_all normally and write cache
    """
    feats_cfg = cfg.get("features", {})
    cache_cfg = feats_cfg.get("cache", {})
    cache_enabled = bool(cache_cfg.get("enabled", False))
    cache_format = str(cache_cfg.get("format", "csv")).lower()
    cache_path = cache_cfg.get("path", None)
    policy = str(cache_cfg.get("policy", "write_if_missing"))

    # --- Decide cache action ---
    cache_exists = cache_enabled and cache_path and os.path.exists(cache_path)
    force_recompute = (policy == "force_recompute")
    require_cache = (feat_mode == "internal_zmat_cached")

    # meta for validation
    meta_expected = cache_meta_dict(
        mode=("internal_zmat" if feat_mode in ("internal_zmat", "internal_zmat_cached") else feat_mode),
        cfg_features=feats_cfg,
        top_path=top_path,
        stride=stride,
        lag=lag,
        pairs=pairs,
    )

    def load_cache():
        if cache_format == "npz":
            Xc, meta = load_features_npz(cache_path)
        else:
            Xc, meta = load_features_csv(cache_path)
        # validate meta if present
        matches = meta is not None and meta_to_string(meta) == meta_to_string(meta_expected)
        return Xc, matches

    # We always need CV_all and segments, so we still loop through pairs to:
    # - align colvars
    # - build CV_all
    # - build seg indices & total frame count
    CV_list = []
    seg_start_list, seg_end_list = [], []
    nframes_each = []
    used_pairs = []
    df_save_list = []

    offset = 0
    for pair_index, (dcd_path, col_path) in enumerate(tqdm(pairs)):
        # load dcd just to know n_frames (mdtraj cheap-ish) unless you want to infer from colvars only
        traj = md.load(dcd_path, top=top_path)
        df = read_colvars_traj(col_path)

        if stride != 1:
            traj = traj[::stride]
            df = df.iloc[::stride].reset_index(drop=True)

        df2, action = maybe_align_colvars(df, traj.n_frames, allow_skip_first)
        if action == "mismatch":
            msg = f"[WARN] mismatch -> skipped: dcd={traj.n_frames} colvars={len(df)} | {dcd_path}"
            if strict:
                raise SystemExit(msg)
            print(msg)
            continue

        cv_cols = cfg["colvars"]["cv"]
        if isinstance(cv_cols, str):
            cv_cols = [cv_cols]
        for c in cv_cols:
            if c not in df2.columns:
                raise SystemExit(f"CV column '{c}' not in {col_path}.")

        # --- CVs for PMF (as before) ---
        CV = df2[cv_cols].to_numpy(dtype=np.float64)

        # --- Full colvars for output CSV ---
        save_cols = cfg.get("colvars", {}).get("save_cols", "all")
        if save_cols == "all":
            cols_out = dedup_columns_keep_first(list(df2.columns))
        else:
            cols_out = list(save_cols)

        # ensure requested columns exist
        missing = [c for c in cols_out if c not in df2.columns]
        if missing:
            raise SystemExit(f"Requested colvars.save_cols missing in {col_path}: {missing}")

        df_save = df2[cols_out].copy()

        nF = CV.shape[0]
        if nF < 2:
            continue

        # segments with lag
        L = int(lag)
        if nF <= L:
            print(f"[WARN] too short for lag={L} -> skipped: {dcd_path}")
            continue

        df_save.insert(0, "trajectory_frame", np.arange(nF, dtype=np.int64))
        df_save.insert(0, "trajectory_id", pair_index)

        starts = np.arange(0, nF - L, dtype=np.int64) + offset
        ends = starts + L
        seg_start_list.append(starts)
        seg_end_list.append(ends)

        CV_list.append(CV)
        nframes_each.append(nF)
        used_pairs.append((dcd_path, col_path))
        df_save_list.append(df_save)
        offset += nF

    if not used_pairs:
        raise SystemExit("No usable segments after alignment checks.")

    CV_all = np.concatenate(CV_list, axis=0)
    df_save_all = pd.concat(df_save_list, ignore_index=True)
    seg_start = np.concatenate(seg_start_list, axis=0)
    seg_end = np.concatenate(seg_end_list, axis=0)

    # --- Feature cache logic ---
    if require_cache:
        if not cache_exists:
            raise SystemExit("features.mode=internal_zmat_cached but cache file not found.")
        X_all, cache_matches = load_cache()
        if not cache_matches:
            raise SystemExit(
                "Cached features were created with a different feature schema or config. "
                "Use features.mode=internal_zmat and cache.policy=force_recompute first."
            )
        if X_all.shape[0] != CV_all.shape[0]:
            raise SystemExit(f"Cached features rows ({X_all.shape[0]}) != total frames from colvars ({CV_all.shape[0]}). "
                             "Cache does not match current dataset.")
        return X_all, CV_all, seg_start, seg_end, df_save_all, used_pairs, nframes_each

    if cache_enabled and cache_exists and not force_recompute:
        print(f"[INFO] Loading features from cache: {cache_path}")
        X_all, cache_matches = load_cache()
        if cache_matches:
            if X_all.shape[0] != CV_all.shape[0]:
                raise SystemExit(f"Cached features rows ({X_all.shape[0]}) != total frames from colvars ({CV_all.shape[0]}). "
                                 "Cache does not match current dataset.")
            return X_all, CV_all, seg_start, seg_end, df_save_all, used_pairs, nframes_each
        print("[INFO] Feature cache schema/config changed; recomputing the cache.")

    # --- Compute features because cache missing or force_recompute ---
    X_list = []
    offset_check = 0
    for (dcd_path, col_path), nF in tqdm(zip(used_pairs, nframes_each)):
        traj = md.load(dcd_path, top=top_path)
        if stride != 1:
            traj = traj[::stride]
        # compute features for exactly nF frames
        if traj.n_frames != nF:
            # mismatch after alignment handling should not happen, but just in case
            traj = traj[:nF]

        if feat_mode in ("internal_zmat", "internal_zmat_cached"):
            X = features_internal_zmat(traj, zmat_atom_order)
        elif feat_mode == "distances":
            X = features_distances(traj, atom_pairs)
        else:
            raise SystemExit(f"Unknown features.mode: {feat_mode}")

        if X.shape[0] != nF:
            raise SystemExit(f"Feature frames ({X.shape[0]}) != colvars frames ({nF}) for {dcd_path}")

        X_list.append(X)
        offset_check += nF

    X_all = np.concatenate(X_list, axis=0)

    if cache_enabled and cache_path:
        if cache_format == "npz":
            save_features_npz(cache_path, X_all, meta_expected)
        else:
            save_features_csv(cache_path, X_all, meta_expected)

    return X_all, CV_all, seg_start, seg_end, df_save_all, used_pairs, nframes_each

# =========================================================
# === Mismatch report ===
# =========================================================

def check_mismatch_report(pairs, top, stride, allow_skip_first):
    rows = []
    for dcd_path, col_path in pairs:
        traj = md.load(dcd_path, top=top)
        df = read_colvars_traj(col_path)
        if stride != 1:
            traj = traj[::stride]
            df = df.iloc[::stride].reset_index(drop=True)

        nF = traj.n_frames
        nC = len(df)
        aligned, action = maybe_align_colvars(df, nF, allow_skip_first)
        ok = (len(aligned) == nF) and (action in ("ok", "skip_first_colvars"))
        rows.append([dcd_path, col_path, nF, nC, action, len(aligned), ok])

    rep = pd.DataFrame(rows, columns=["dcd", "colvars", "dcd_frames", "colvars_rows", "action", "aligned_rows", "ok"])
    print(rep[["dcd_frames","colvars_rows","action","aligned_rows","ok","dcd","colvars"]].to_string(index=False))
    bad = rep[~rep["ok"]]
    print(f"\n[INFO] total pairs: {len(rep)}, ok: {int(rep['ok'].sum())}, bad: {len(bad)}")


def write_diffusion_training_data(
    *,
    used_pairs: List[Tuple[str, str]],
    expected_frames: List[int],
    top_path: str,
    stride: int,
    atomselect: Optional[str],
    alignment_atomselect: str,
    dcd_path: str,
    topology_path: str,
    chunk_size: int,
):
    """RMSD-align RiteWeight frames, then write a selected DCD and topology."""
    topology = md.load_topology(top_path)
    if atomselect:
        atom_indices = topology.select(atomselect)
    else:
        atom_indices = np.arange(topology.n_atoms, dtype=int)
    if len(atom_indices) == 0:
        raise ValueError(f"Diffusion atom selection matched no atoms: {atomselect!r}")

    alignment_indices = topology.select(alignment_atomselect)
    if len(alignment_indices) < 3:
        raise ValueError(
            "Diffusion RMSD alignment needs at least three atoms; "
            f"{alignment_atomselect!r} matched {len(alignment_indices)}."
        )

    reference = md.load_frame(used_pairs[0][0], 0, top=top_path)
    if reference.n_frames != 1:
        raise RuntimeError("Could not load the diffusion RMSD reference frame.")

    os.makedirs(os.path.dirname(dcd_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(topology_path) or ".", exist_ok=True)
    wrote_topology = False
    total_frames = 0

    with md.formats.DCDTrajectoryFile(
        dcd_path, mode="w", force_overwrite=True
    ) as writer:
        for (dcd_source, _), expected in zip(used_pairs, expected_frames):
            source_frames = 0
            for chunk in md.iterload(
                dcd_source,
                top=top_path,
                chunk=max(int(chunk_size), 1),
                stride=max(int(stride), 1),
            ):
                # Calculate the rigid fit from the alignment selection, then
                # apply that transform to every atom before selecting output.
                chunk.superpose(reference, atom_indices=alignment_indices)
                selected = chunk.atom_slice(atom_indices)
                if not wrote_topology and selected.n_frames:
                    selected[0].save_pdb(topology_path, force_overwrite=True)
                    wrote_topology = True

                if selected.unitcell_lengths is None:
                    writer.write(selected.xyz * 10.0)  # MDTraj stores xyz in nm; DCD writer expects Å.
                else:
                    writer.write(
                        selected.xyz * 10.0,
                        cell_lengths=selected.unitcell_lengths * 10.0,
                        cell_angles=selected.unitcell_angles,
                    )
                source_frames += selected.n_frames
                total_frames += selected.n_frames

            if source_frames != expected:
                raise RuntimeError(
                    f"Diffusion DCD frame mismatch for {dcd_source}: "
                    f"wrote {source_frames}, expected {expected}."
                )

    if not wrote_topology:
        raise RuntimeError("No trajectory frames were available for diffusion output.")
    print(
        f"[OK] Saved diffusion training trajectory: {dcd_path} "
        f"({total_frames} frames, {len(atom_indices)} atoms)"
    )
    print(
        "[OK] RMSD-aligned diffusion frames using "
        f"{len(alignment_indices)} atoms from {alignment_atomselect!r}"
    )
    print(f"[OK] Saved matching diffusion topology: {topology_path}")
    return total_frames, len(atom_indices)


# =========================================================
# === Main ===
# =========================================================

def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return normalize_config_aliases({"RiteWeight": loaded})["RiteWeight"]

def run_riteweight(cfg: Dict, check_mismatch: bool = False):

    roots = cfg.get("folders", [])
    if not roots:
        raise SystemExit("config.yaml must define 'folders: [..]'")

    dcd_pattern = cfg.get("dcd_pattern")
    colvars_pattern = cfg.get("colvars_pattern")
    if not dcd_pattern:
        dcd_pattern = f"*{cfg.get('match_dcd', '')}*.dcd"
    if not colvars_pattern:
        colvars_pattern = f"*{cfg.get('match_colvars', '')}*.colvars.traj"
    tag_re = cfg.get("tag_regex", r"([AB])")

    top_path = cfg["io"]["topology"]
    out = cfg["io"].get("out", "rw_out")
    stride = int(cfg["io"].get("stride", 1))
    output_cfg = cfg.get("outputs", {})

    allow_skip_first = bool(cfg.get("pairing", {}).get("allow_skip_first_colvars", True))
    strict = bool(cfg.get("pairing", {}).get("strict", False))

    pairs = find_pairs_dcd_colvars(
        roots, dcd_pattern, colvars_pattern, tag_re=tag_re
    )
    print(f"[INFO] Found {len(pairs)} (dcd,colvars) pairs across folders.")

    if check_mismatch:
        check_mismatch_report(pairs, top=top_path, stride=stride, allow_skip_first=allow_skip_first)
        return

    os.makedirs(out, exist_ok=True)

    # CV settings
    cv_cols = cfg["colvars"]["cv"]
    # --- colvars output controls ---
    save_cols = cfg.get("colvars", {}).get("save_cols", "all")
    periodic_cols = cfg.get("colvars", {}).get("periodic_cols", [])
    periodic_cols = [periodic_cols] if isinstance(periodic_cols, str) else list(periodic_cols)

    # --- committor label controls ---
    lab_cfg = cfg.get("committor_labels", {})
    label_enabled = bool(lab_cfg.get("enabled", False))
    if label_enabled:
        cvs_to_label = lab_cfg["cvs_to_label"]
        basin_A = lab_cfg["basin_A"]
        basin_B = lab_cfg["basin_B"]
        basin_size = lab_cfg["basin_size"]
        if not cvs_to_label or basin_A is None or basin_B is None or basin_size is None:
            raise ValueError(
                "Enabled committor labels require cvs_to_label, basin_A, "
                "basin_B, and basin_size."
            )
        if len(basin_A) != len(cvs_to_label) or len(basin_B) != len(cvs_to_label):
            raise ValueError("Committor basin centers must match cvs_to_label dimensions.")
        if not np.isscalar(basin_size) and len(basin_size) != len(cvs_to_label):
            raise ValueError("committor_labels.basin_size has the wrong dimension.")
        k_pref = float(lab_cfg.get("k_prefactor", 1.0))
        angle_unit = str(lab_cfg.get("angle_unit", "degree")).lower()
        determine_AB = determine_AB_functor(basin_A, basin_B, basin_size)


    if isinstance(cv_cols, str):
        cv_cols = [cv_cols]
    if len(cv_cols) not in (1, 2):
        raise SystemExit("colvars.cv must be 1 or 2 columns.")

    bins = cfg["colvars"].get("bins", 200)
    T_K = float(cfg["colvars"].get("temperature_K", 300.0))
    xlim = cfg["colvars"].get("xlim", None)
    ylim = cfg["colvars"].get("ylim", None)
    periodicities = cfg["colvars"].get("periodicities", [0] * len(cv_cols))

    # PMF output
    pmf_cfg = cfg.get("pmf_output", {})
    pmf_enabled = bool(pmf_cfg.get("enabled", False))
    pmf_filename = pmf_cfg.get("filename", "pmf.dat")
    F_max = pmf_cfg.get("F_max", None)
    sigma_bins = pmf_cfg.get("sigma_bins", None)
    scaled_bins = pmf_cfg.get("scaled_bins", None)
    scaled_filename = pmf_cfg.get("scaled_filename", "pmf_scaled.dat")

    # Features settings
    feat_mode = cfg["features"]["mode"]
    atom_pairs = cfg["features"].get("distances", {}).get("atom_pairs", None)
    internal_cfg = cfg["features"].get("internal_zmat", {})

    # RiteWeight params
    rw = cfg["riteweight"]
    n_clusters = int(rw["n_clusters"])
    n_iter = int(rw.get("n_iter", 200))
    tol = float(rw.get("tol", 1e-6))
    tol_window = int(rw.get("tol_window", 5))
    avg_last = int(rw.get("avg_last", 20))
    seed = int(rw.get("seed", 2026))
    lag = int(rw.get("lag", 50))

    # We resolve atomselect based on topology once (deterministic).
    top_for_sel = md.load_topology(top_path)

    zmat_atom_order: Optional[List[int]] = None
    if feat_mode in ("internal_zmat", "internal_zmat_cached"):
        zmat_atom_order = resolve_zmat_atoms(top_for_sel, internal_cfg)
        print(f"[INFO] internal_zmat atoms (N={len(zmat_atom_order)}): {zmat_atom_order}")


    (
        X_all,
        CV_all,
        seg_start,
        seg_end,
        df_save_all,
        used_pairs,
        nframes_each,
    ) = load_or_compute_features_with_cache(
        cfg=cfg,
        pairs=pairs,
        top_path=top_path,
        stride=stride,
        allow_skip_first=allow_skip_first,
        strict=strict,
        feat_mode=feat_mode,
        zmat_atom_order=zmat_atom_order,
        atom_pairs=atom_pairs,
        lag=lag,
    )

    res = riteweight(
        X_all, seg_start, seg_end,
        n_clusters=n_clusters, n_iter=n_iter,
        tol=tol, tol_window=tol_window,
        avg_last=avg_last, seed=seed
    )

    # frame weights
    CV_use = CV_all[res.frame_map]
    w_use = res.w_frame_nonzero

    # save weights

    # w_frame_full = np.zeros(X_all.shape[0], dtype=np.float64)
    # w_frame_full[res.frame_map] = w_use
    # np.savetxt(os.path.join(out, "frame_weights.csv"),
    #            np.column_stack([np.arange(X_all.shape[0]), w_frame_full]),
    #            delimiter=",", header="frame_index,weight", comments="")
    # --- frame weights (full length, zeros for frames not used as segment starts) ---
    w_frame_full = np.zeros(X_all.shape[0], dtype=np.float64)
    w_frame_full[res.frame_map] = w_use

    # --- build output table ---
    out_df = df_save_all.copy()
    if len(out_df) != len(w_frame_full):
        raise RuntimeError(
            f"RiteWeight output alignment failed: table rows={len(out_df)}, "
            f"weights={len(w_frame_full)}."
        )
    out_df.insert(0, "frame", np.arange(len(out_df), dtype=np.int64))
    out_df["weight"] = w_frame_full

    # --- periodic sin/cos columns ---
    if periodic_cols:
        # sin/cos assume degrees by default (like your reference script)
        if "angle_unit" in locals() and angle_unit == "radian":
            scale = 1.0
        else:
            scale = np.pi / 180.0
        for cv in periodic_cols:
            if cv not in out_df.columns:
                print(f"[WARN] periodic_cols '{cv}' not found in saved colvars columns; skip sin/cos.")
                continue
            out_df[f"s{cv}"] = np.sin(out_df[cv].to_numpy(dtype=float) * scale)
            out_df[f"c{cv}"] = np.cos(out_df[cv].to_numpy(dtype=float) * scale)

    # --- committor training A/B labels ---
    if label_enabled:
        miss = [c for c in cvs_to_label if c not in out_df.columns]
        if miss:
            raise SystemExit(f"committor_labels.cvs_to_label missing from saved columns: {miss}. "
                            f"Add them to colvars.save_cols or set save_cols: all")

        pos = out_df[cvs_to_label].to_numpy(dtype=float)
        states = np.apply_along_axis(determine_AB, 1, pos)
        out_df["state"] = states
        out_df["label"] = pd.Series(states).map({"A": 0, "B": 1, "M": -1}).to_numpy(dtype=int)
        out_df["center"] = pd.Series(states).map({"A": 0.0, "B": 1.0, "M": -1.0}).to_numpy(dtype=float)
        out_df["Ka"] = np.where(states == "A", k_pref, 0.0)
        out_df["Kb"] = np.where(states == "B", k_pref, 0.0)

    # --- write CSV ---
    frame_csv_path = os.path.join(out, output_cfg.get("frame_csv", "frame_weights.csv"))
    out_df.to_csv(frame_csv_path, index=False)
    print(f"[OK] Saved extended frame weights + colvars (+ labels) to: {frame_csv_path}")

    segment_csv_path = os.path.join(out, output_cfg.get("segment_csv", "segment_weights.csv"))
    np.savetxt(segment_csv_path,
               np.column_stack([np.arange(res.w_segment.size), res.w_segment]),
               delimiter=",", header="segment_index,weight", comments="")

    # Torch-native numeric table consumed directly by common.vcn_train.
    vcn_cfg = output_cfg.get("vcn", {})
    vcn_path = None
    if bool(vcn_cfg.get("enabled", True)):
        vcn_path = os.path.join(out, vcn_cfg.get("filename", "vcn_training.pt"))
        saved_columns = save_tensor_table(
            vcn_path,
            out_df,
            metadata={
                "source": "RiteWeight",
                "frame_csv": os.path.abspath(frame_csv_path),
                "topology": os.path.abspath(top_path),
                "stride": stride,
                "lag": lag,
                "feature_schema": FEATURE_SCHEMA,
                "feature_mode": feat_mode,
                "feature_dimension": int(X_all.shape[1]),
                "feature_atomselect": internal_cfg.get("atomselect"),
            },
        )
        print(
            f"[OK] Saved VCN Torch training table: {vcn_path} "
            f"({len(out_df)} rows, {len(saved_columns)} numeric columns)"
        )

    # Aligned selected trajectory consumed directly by the diffusion trainer.
    diffusion_cfg = output_cfg.get("diffusion", {})
    diffusion_dcd_path = None
    diffusion_topology_path = None
    if bool(diffusion_cfg.get("enabled", True)):
        diffusion_dcd_path = os.path.join(
            out, diffusion_cfg.get("trajectory", "diffusion_training.dcd")
        )
        diffusion_topology_path = os.path.join(
            out, diffusion_cfg.get("topology", "diffusion_training.pdb")
        )
        write_diffusion_training_data(
            used_pairs=used_pairs,
            expected_frames=nframes_each,
            top_path=top_path,
            stride=stride,
            atomselect=diffusion_cfg.get("atomselect"),
            alignment_atomselect=(
                diffusion_cfg.get("alignment_atomselect")
                or internal_cfg.get("atomselect")
                or diffusion_cfg.get("atomselect")
                or "all"
            ),
            dcd_path=diffusion_dcd_path,
            topology_path=diffusion_topology_path,
            chunk_size=int(diffusion_cfg.get("chunk_size", 1000)),
        )

    # convergence
    plt.figure()
    plt.plot(np.arange(1, len(res.delta_history) + 1), res.delta_history)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("L1 change in segment weights")
    plt.tight_layout()
    convergence_path = os.path.join(
        out, output_cfg.get("convergence_plot", "convergence_delta.png")
    )
    plt.savefig(convergence_path, dpi=200)
    plt.close()

    output_paths = {
        "frame_csv": frame_csv_path,
        "segment_csv": segment_csv_path,
        "vcn_pt": vcn_path,
        "diffusion_dcd": diffusion_dcd_path,
        "diffusion_topology": diffusion_topology_path,
        "convergence_plot": convergence_path,
    }

    # Weighted FEL projection now lives in tools/felestimate.py. This optional
    # branch is retained only for compatibility with older RiteWeight configs.
    if not pmf_enabled:
        print(f"[OK] RiteWeight outputs written to: {out}")
        return output_paths

    # build PMF/FEL
    kT = KB_KCAL_PER_MOL_K * T_K

    if len(cv_cols) == 1:
        x = CV_use[:, 0]
        nb = int(bins) if not isinstance(bins, (list, tuple)) else int(bins[0])
        P, centers, edges = weighted_hist_1d(x, w_use, bins=nb, xlim=xlim)
        F = fes_from_prob(P, T_K)
        if F_max is not None:
            F = np.minimum(F, float(F_max))
        F = gaussian_smooth_F(F, sigma_bins=sigma_bins, F_max=F_max)

        # save dat
        dat_path = os.path.join(out, pmf_filename)
        save_gromacs_like(dat_path, F, [edges], cv_cols, periodicities, xlim=xlim)
        save_scaled_pmf_if_requested(
            F=F,
            edges=[edges],
            cvs=cv_cols,
            periodicities=periodicities,
            xlim=xlim,
            ylim=None,
            out_dir=out,
            scaled_bins=scaled_bins,
            scaled_filename=scaled_filename,
        )

        # optional handy csv
        np.savetxt(os.path.join(out, "fes_1d.csv"),
                   np.column_stack([centers, P, F]),
                   delimiter=",", header=f"{cv_cols[0]}_center,prob,fes_kcal_per_mol", comments="")

    else:
        x, y = CV_use[:, 0], CV_use[:, 1]
        if isinstance(bins, (list, tuple)) and len(bins) == 2:
            nb = (int(bins[0]), int(bins[1]))
        else:
            nb = (200, 200)
        P, xcent, ycent, xedges, yedges = weighted_hist_2d(x, y, w_use, bins=nb, xlim=xlim, ylim=ylim)
        F = fes_from_prob(P, T_K)
        if F_max is not None:
            F = np.minimum(F, float(F_max))
        F = gaussian_smooth_F(F, sigma_bins=sigma_bins, F_max=F_max)

        dat_path = os.path.join(out, pmf_filename)
        save_gromacs_like(dat_path, F, [xedges, yedges], cv_cols, periodicities, xlim=xlim, ylim=ylim)

        save_scaled_pmf_if_requested(
            F=F,
            edges=[xedges, yedges],
            cvs=cv_cols,
            periodicities=periodicities,
            xlim=xlim,
            ylim=ylim,
            out_dir=out,
            scaled_bins=scaled_bins,
            scaled_filename=scaled_filename,
        )

        # optional grid csv
        XX, YY = np.meshgrid(xcent, ycent, indexing="ij")
        grid = np.column_stack([XX.ravel(), YY.ravel(), P.ravel(), F.ravel()])
        np.savetxt(os.path.join(out, "fes_2d_grid.csv"),
                   grid, delimiter=",",
                   header=f"{cv_cols[0]}_center,{cv_cols[1]}_center,prob,fes_kcal_per_mol",
                   comments="")

    print(f"[OK] Outputs written to: {out}")
    print(f"[OK] PMF dat written: {os.path.join(out, pmf_filename)} (and .pdf)")
    return output_paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config.yaml")
    ap.add_argument("--check-mismatch", action="store_true", help="Only report mismatches and exit.")
    args = ap.parse_args(argv)
    run_riteweight(load_yaml(args.config), check_mismatch=args.check_mismatch)


if __name__ == "__main__":
    main()
