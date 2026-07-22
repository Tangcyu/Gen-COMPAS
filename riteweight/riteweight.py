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
from matplotlib.patches import Rectangle
from tqdm import tqdm

try:
    import yaml
except ImportError as e:
    raise SystemExit("Need pyyaml. Install: pip install pyyaml") from e

try:
    import mdtraj as md
except ImportError as e:
    raise SystemExit("Need mdtraj. Install: pip install mdtraj") from e

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
            m[(sub, tag)] = fp
    return m

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

def normalize_cv_pair(value: Any, default: Optional[List[str]], config_key: str) -> Optional[List[str]]:
    if value is None:
        if default is None:
            return None
        value = default

    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()] if "," in value else [value.strip()]
    else:
        parts = [str(p) for p in list(value)]

    if len(parts) != 2:
        raise SystemExit(f"{config_key} must contain exactly two CV column names.")
    if parts[0] == parts[1]:
        raise SystemExit(f"{config_key} must contain two different CV column names.")
    return parts

def committor_label_check_cv_pair(lab_cfg: dict, cvs_to_label: List[str]) -> Optional[List[str]]:
    check_cfg = lab_cfg.get("check", {})
    if isinstance(check_cfg, dict):
        if not bool(check_cfg.get("enabled", True)):
            return None
        cv_pair_value = check_cfg.get("cv_pair", lab_cfg.get("check_cv_pair", None))
    else:
        if not bool(check_cfg):
            return None
        cv_pair_value = lab_cfg.get("check_cv_pair", None)

    default_pair = list(cvs_to_label) if len(cvs_to_label) == 2 else None
    cv_pair = normalize_cv_pair(cv_pair_value, default_pair, "committor_labels.check_cv_pair")
    if cv_pair is None:
        print("[WARN] committor_labels.check_cv_pair is not set and cvs_to_label is not 2D; skip label projection check.")
    return cv_pair

def committor_label_check_option(lab_cfg: dict, key: str, default: Any) -> Any:
    check_cfg = lab_cfg.get("check", {})
    if isinstance(check_cfg, dict) and key in check_cfg:
        return check_cfg[key]
    flat_key = f"check_{key}"
    return lab_cfg.get(flat_key, default)

def add_basin_overlay_if_possible(
    ax,
    cv_pair: List[str],
    cvs_to_label: List[str],
    basin_A: Any,
    basin_B: Any,
    basin_size: Any,
) -> None:
    if not all(cv in cvs_to_label for cv in cv_pair):
        return

    idx = [cvs_to_label.index(cv) for cv in cv_pair]
    centers = {"A": np.asarray(basin_A, dtype=float), "B": np.asarray(basin_B, dtype=float)}
    size = np.asarray(basin_size, dtype=float)
    if size.ndim == 0:
        size = np.full(len(cvs_to_label), float(size))

    if size.size < len(cvs_to_label) or any(center.size < len(cvs_to_label) for center in centers.values()):
        return

    colors = {"A": "tab:blue", "B": "tab:red"}
    for state, center_full in centers.items():
        center = center_full[idx]
        half = size[idx]
        rect = Rectangle(
            (center[0] - half[0], center[1] - half[1]),
            2.0 * half[0],
            2.0 * half[1],
            fill=False,
            edgecolor=colors[state],
            linewidth=1.3,
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.scatter([center[0]], [center[1]], marker="x", s=70, color=colors[state], linewidths=1.4)

def save_committor_label_check(
    df: pd.DataFrame,
    *,
    cv_pair: List[str],
    out_dir: str,
    cvs_to_label: List[str],
    basin_A: Any,
    basin_B: Any,
    basin_size: Any,
    lab_cfg: dict,
) -> None:
    required = list(cv_pair) + ["state", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"committor_labels check columns missing from frame table: {missing}. "
            f"Add the selected CVs to colvars.save_cols or set save_cols: all."
        )

    os.makedirs(out_dir, exist_ok=True)
    counts_filename = str(committor_label_check_option(lab_cfg, "counts_filename", "committor_labels_counts.csv"))
    counts_path = os.path.join(out_dir, counts_filename)
    counts = df.groupby(["state", "label"], dropna=False).size().reset_index(name="count")
    counts["fraction"] = counts["count"] / max(len(df), 1)
    counts.to_csv(counts_path, index=False)

    plot_df = df.dropna(subset=cv_pair + ["label"]).copy()
    max_points = committor_label_check_option(lab_cfg, "max_points", 200000)
    if max_points is not None:
        max_points = int(max_points)
        if max_points > 0 and len(plot_df) > max_points:
            seed = int(committor_label_check_option(lab_cfg, "seed", 2026))
            plot_df = plot_df.sample(n=max_points, random_state=seed)

    filename = str(committor_label_check_option(lab_cfg, "filename", "committor_labels_projection.png"))
    out_path = os.path.join(out_dir, filename)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    state_specs = [
        ("M", -1, "0.65", ".", 10, 0.35),
        ("A", 0, "tab:blue", "o", 18, 0.75),
        ("B", 1, "tab:red", "^", 18, 0.75),
    ]
    for state, label, color, marker, size, alpha in state_specs:
        mask = plot_df["label"].eq(label)
        if "state" in plot_df.columns:
            mask = mask | plot_df["state"].eq(state)
        n_full = int((df["label"].eq(label) | df["state"].eq(state)).sum())
        if not mask.any():
            continue
        ax.scatter(
            plot_df.loc[mask, cv_pair[0]],
            plot_df.loc[mask, cv_pair[1]],
            s=size,
            c=color,
            marker=marker,
            alpha=alpha,
            linewidths=0,
            label=f"{state} label={label} (n={n_full})",
            rasterized=True,
        )

    add_basin_overlay_if_possible(ax, cv_pair, cvs_to_label, basin_A, basin_B, basin_size)
    ax.set_xlabel(cv_pair[0])
    ax.set_ylabel(cv_pair[1])
    ax.legend(frameon=False, loc="best")
    ax.set_title("Committor labels projected on selected CVs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved committor label check: {out_path}")
    print(f"[OK] Saved committor label counts: {counts_path}")

def relabel_only(cfg: dict):
    lab_cfg = cfg.get("committor_labels", {})
    if not lab_cfg or not bool(lab_cfg.get("enabled", False)):
        raise SystemExit("committor_labels.enabled must be true for --relabel-only")

    cvs_to_label = lab_cfg["cvs_to_label"]
    if isinstance(cvs_to_label, str):
        cvs_to_label = [cvs_to_label]
    basin_A = lab_cfg["basin_A"]
    basin_B = lab_cfg["basin_B"]
    basin_size = lab_cfg["basin_size"]
    k_pref = float(lab_cfg.get("k_prefactor", 1.0))
    check_cv_pair = committor_label_check_cv_pair(lab_cfg, cvs_to_label)
    weight_mask_cfg = cfg.get("weight_mask", cfg.get("cv_weight_mask", {}))

    determine_AB = determine_AB_functor(basin_A, basin_B, basin_size)

    out_dir = cfg.get("io", {}).get("out", "rw_out")
    fw_name = cfg.get("io", {}).get("frame_weights_csv", "frame_weights.csv")
    in_csv = fw_name if os.path.isabs(fw_name) else os.path.join(out_dir, fw_name)

    relabeled = cfg.get("io", {}).get("relabeled_csv", None)
    out_csv = (relabeled if os.path.isabs(relabeled) else os.path.join(out_dir, relabeled)) if relabeled else in_csv

    if not os.path.exists(in_csv):
        raise SystemExit(f"--relabel-only: file not found: {in_csv}")

    df = pd.read_csv(in_csv)

    missing = [c for c in cvs_to_label if c not in df.columns]
    if missing:
        raise SystemExit(
            f"--relabel-only: required columns missing in {in_csv}: {missing}. "
            f"Make sure you saved them in frame_weights.csv (e.g. colvars.save_cols: all)."
        )

    pos = df[cvs_to_label].to_numpy(dtype=float)
    states = np.apply_along_axis(determine_AB, 1, pos)

    df["state"] = states
    df["label"] = pd.Series(states).map({"A": 0, "B": 1, "M": -1}).to_numpy(dtype=int)
    df["center"] = pd.Series(states).map({"A": 0.0, "B": 1.0, "M": -1.0}).to_numpy(dtype=float)
    df["Ka"] = np.where(states == "A", k_pref, 0.0)
    df["Kb"] = np.where(states == "B", k_pref, 0.0)

    if _weight_mask_conditions(weight_mask_cfg):
        if "weight" not in df.columns:
            raise SystemExit("--relabel-only with weight_mask requires a 'weight' column in frame_weights.csv.")
        masked_weights, _ = apply_cv_weight_mask(df, df["weight"].to_numpy(dtype=np.float64), weight_mask_cfg)
        df["weight"] = masked_weights

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    if out_csv == in_csv:
        print(f"[OK] Relabeled in-place: {out_csv}")
    else:
        print(f"[OK] Relabeled and saved to: {out_csv}")

    if check_cv_pair is not None:
        save_committor_label_check(
            df,
            cv_pair=check_cv_pair,
            out_dir=os.path.dirname(out_csv) or ".",
            cvs_to_label=cvs_to_label,
            basin_A=basin_A,
            basin_B=basin_B,
            basin_size=basin_size,
            lab_cfg=lab_cfg,
        )

# =========================================================
# === CV-based weight masking ===
# =========================================================

def _weight_mask_conditions(mask_cfg: Any) -> List[dict]:
    if not mask_cfg:
        return []
    if isinstance(mask_cfg, list):
        return list(mask_cfg)
    if not isinstance(mask_cfg, dict):
        raise SystemExit("weight_mask must be a mapping or a list of conditions.")
    if not bool(mask_cfg.get("enabled", True)):
        return []
    conditions = mask_cfg.get("conditions", mask_cfg.get("criteria", None))
    if conditions is None and ("cv" in mask_cfg or "column" in mask_cfg):
        conditions = [mask_cfg]
    if conditions is None:
        return []
    if isinstance(conditions, dict):
        conditions = [conditions]
    return list(conditions)

def _condition_threshold(cond: dict) -> float:
    value = cond.get("threshold", cond.get("value", None))
    if value is None:
        raise SystemExit(f"weight_mask condition for {cond.get('cv', cond.get('column', '<missing cv>'))} needs threshold/value.")
    return float(value)

def _condition_range(cond: dict) -> Tuple[Optional[float], Optional[float]]:
    lo = cond.get("min", cond.get("lower", None))
    hi = cond.get("max", cond.get("upper", None))
    if lo is None and hi is None:
        raise SystemExit(f"weight_mask range condition for {cond.get('cv', cond.get('column', '<missing cv>'))} needs min/max.")
    return (None if lo is None else float(lo), None if hi is None else float(hi))

def cv_condition_mask(df: pd.DataFrame, cond: dict) -> np.ndarray:
    """
    Return frames matched by one CV condition. Matched frames are the ones
    whose weights will be zeroed by apply_cv_weight_mask().
    """
    cv = cond.get("cv", cond.get("column", None))
    if not cv:
        raise SystemExit("Each weight_mask condition needs a 'cv' or 'column' key.")
    if cv not in df.columns:
        raise SystemExit(
            f"weight_mask CV column '{cv}' is not in the output table. "
            f"Add it to colvars.save_cols or set colvars.save_cols: all."
        )

    x = pd.to_numeric(df[cv], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(x)
    op = str(cond.get("op", cond.get("operator", ""))).strip().lower()
    if not op and any(k in cond for k in ("min", "max", "lower", "upper")):
        op = "outside"

    if op in (">", "gt", "greater", "greater_than"):
        mask = x > _condition_threshold(cond)
    elif op in (">=", "ge", "greater_equal", "greater_than_or_equal"):
        mask = x >= _condition_threshold(cond)
    elif op in ("<", "lt", "less", "less_than"):
        mask = x < _condition_threshold(cond)
    elif op in ("<=", "le", "less_equal", "less_than_or_equal"):
        mask = x <= _condition_threshold(cond)
    elif op in ("==", "eq", "equal"):
        mask = x == _condition_threshold(cond)
    elif op in ("!=", "ne", "not_equal"):
        mask = x != _condition_threshold(cond)
    elif op in ("inside", "between"):
        lo, hi = _condition_range(cond)
        mask = np.ones(len(df), dtype=bool)
        if lo is not None:
            mask &= x >= lo
        if hi is not None:
            mask &= x <= hi
    elif op in ("outside", "out_of_range"):
        lo, hi = _condition_range(cond)
        mask = np.zeros(len(df), dtype=bool)
        if lo is not None:
            mask |= x < lo
        if hi is not None:
            mask |= x > hi
    else:
        raise SystemExit(
            f"Unsupported weight_mask op '{op}' for CV '{cv}'. "
            "Use >, >=, <, <=, ==, !=, inside, or outside."
        )

    return mask & finite

def apply_cv_weight_mask(
    df: pd.DataFrame,
    weights: np.ndarray,
    mask_cfg: Any,
    *,
    weight_col: str = "weight",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set frame weights to zero for frames matching CV-based conditions.

    Example config:
      weight_mask:
        enabled: true
        combine: any
        renormalize: true
        conditions:
          - {cv: CV1, op: ">", threshold: 2.0}
    """
    conditions = _weight_mask_conditions(mask_cfg)
    w = np.asarray(weights, dtype=np.float64).copy()
    if w.shape[0] != len(df):
        raise SystemExit(f"weight_mask weights length ({w.shape[0]}) != table rows ({len(df)}).")
    if not conditions:
        return w, np.zeros(len(df), dtype=bool)

    combine = "any"
    renormalize = True
    if isinstance(mask_cfg, dict):
        combine = str(mask_cfg.get("combine", "any")).lower()
        renormalize = bool(mask_cfg.get("renormalize", True))

    masks = [cv_condition_mask(df, cond) for cond in conditions]
    if combine == "all":
        zero_mask = np.logical_and.reduce(masks)
    elif combine == "any":
        zero_mask = np.logical_or.reduce(masks)
    else:
        raise SystemExit("weight_mask.combine must be 'any' or 'all'.")

    before = w.copy()
    w[zero_mask] = 0.0
    if weight_col in df.columns:
        df[weight_col] = w

    if renormalize:
        total = float(w.sum())
        if total <= 0.0:
            raise SystemExit("weight_mask zeroed all frame weights; relax the CV mask.")
        w /= total
        if weight_col in df.columns:
            df[weight_col] = w

    matched = int(zero_mask.sum())
    weighted_matched = int(np.count_nonzero(zero_mask & (before > 0.0)))
    print(
        f"[INFO] CV weight mask zeroed {weighted_matched} weighted frames "
        f"({matched} total matched); weight sum {before.sum():.8g} -> {w.sum():.8g}."
    )
    return w, zero_mask

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
    Deterministic 3N-6 internal coordinates via a simple Z-matrix-like definition.

      - r01 = dist(a0,a1)
      - r02 = dist(a0,a2)
      - ang102 = angle(a1,a0,a2)
      - for k>=3:
          r0k  = dist(a0, ak)
          ang10k = angle(a1,a0, ak)
          dih210k = dihedral(a2,a1,a0, ak)

    Total features: 3 + 3*(N-3) = 3N-6 (nonlinear).
    """
    ao = list(map(int, atom_order))
    if len(ao) < 4:
        raise ValueError("internal_zmat needs N>=4 atoms.")

    a0, a1, a2 = ao[0], ao[1], ao[2]

    dist_pairs = [(a0, a1), (a0, a2)] + [(a0, ak) for ak in ao[3:]]
    D = md.compute_distances(traj, np.array(dist_pairs, dtype=int))  # (F, 2 + N-3)

    ang_triples = [(a1, a0, a2)] + [(a1, a0, ak) for ak in ao[3:]]
    A = md.compute_angles(traj, np.array(ang_triples, dtype=int))    # (F, 1 + N-3)

    dih_quads = [(a2, a1, a0, ak) for ak in ao[3:]]
    H = md.compute_dihedrals(traj, np.array(dih_quads, dtype=int))   # (F, N-3)

    X = np.concatenate(
        [
            D[:, [0]],          # r01
            D[:, [1]],          # r02
            A[:, [0]],          # ang102
            D[:, 2:],           # r0k
            A[:, 1:],           # ang10k
            H,                  # dih210k
        ],
        axis=1,
    )
    return X


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
        # kld = kl_divergence(w_new, w_prev)
        # delta_hist.append(kld)

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
# == Convergence Check (KL/JS divergence) ===
# =========================================================

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-300) -> float:
    """
    KL(p || q) with clipping for numerical stability.
    p, q must be nonnegative and sum to 1 (we will renormalize defensively).
    """
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = np.clip(p, 0.0, np.inf)
    q = np.clip(q, 0.0, np.inf)
    sp = p.sum()
    sq = q.sum()
    if sp <= 0 or sq <= 0:
        return float("nan")
    p = p / sp
    q = q / sq
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-300) -> float:
    """
    Jensen–Shannon divergence (symmetric, finite), using natural log.
    JS(p,q) = 0.5 KL(p||m) + 0.5 KL(q||m), m = 0.5(p+q)
    """
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = np.clip(p, 0.0, np.inf)
    q = np.clip(q, 0.0, np.inf)
    sp = p.sum()
    sq = q.sum()
    if sp <= 0 or sq <= 0:
        return float("nan")
    p = p / sp
    q = q / sq
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)

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


def cache_meta_dict(mode: str, cfg_features: dict, top_path: str, stride: int, lag: int) -> dict:
    """Minimal metadata to validate cache reproducibility."""
    meta = {
        "mode": mode,
        "top": os.path.abspath(top_path),
        "stride": int(stride),
        "lag": int(lag),
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
    save_cols: Any = "all",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_all, CV_all, seg_start, seg_end
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
    )

    def load_cache():
        if cache_format == "npz":
            Xc, meta = load_features_npz(cache_path)
        else:
            Xc, meta = load_features_csv(cache_path)
        # validate meta if present
        if meta is not None:
            if meta_to_string(meta) != meta_to_string(meta_expected):
                msg = "[WARN] Feature cache metadata differs from current config/top/stride/lag. " \
                      "Results may be inconsistent. Consider force_recompute."
                print(msg)
        return Xc

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
    for dcd_path, col_path in tqdm(pairs):
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
        if save_cols == "all":
            cols_out = dedup_columns_keep_first(list(df2.columns))
        else:
            cols_out = list(save_cols)

        # ensure requested columns exist
        missing = [c for c in cols_out if c not in df2.columns]
        if missing:
            raise SystemExit(f"Requested colvars.save_cols missing in {col_path}: {missing}")

        nF = CV.shape[0]
        if nF < 2:
            continue

        L = int(lag)
        if nF <= L:
            print(f"[WARN] too short for lag={L} -> skipped: {dcd_path}")
            continue

        df_save = df2[cols_out].copy()
        df_save_list.append(df_save)

        starts = np.arange(0, nF - L, dtype=np.int64) + offset
        ends = starts + L
        seg_start_list.append(starts)
        seg_end_list.append(ends)

        CV_list.append(CV)
        nframes_each.append(nF)
        used_pairs.append((dcd_path, col_path))
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
        X_all = load_cache()
        if X_all.shape[0] != CV_all.shape[0]:
            raise SystemExit(f"Cached features rows ({X_all.shape[0]}) != total frames from colvars ({CV_all.shape[0]}). "
                             "Cache does not match current dataset.")
        return X_all, CV_all, seg_start, seg_end, df_save_all

    if cache_enabled and cache_exists and not force_recompute:
        print(f"[INFO] Loading features from cache: {cache_path}")
        X_all = load_cache()
        if X_all.shape[0] != CV_all.shape[0]:
            raise SystemExit(f"Cached features rows ({X_all.shape[0]}) != total frames from colvars ({CV_all.shape[0]}). "
                             "Cache does not match current dataset.")
        return X_all, CV_all, seg_start, seg_end, df_save_all

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

    return X_all, CV_all, seg_start, seg_end, df_save_all

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


# =========================================================
# === Main ===
# =========================================================

def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config.yaml")
    ap.add_argument("--check-mismatch", action="store_true", help="Only report mismatches and exit.")
    ap.add_argument("--relabel-only", action="store_true",
                help="Only relabel existing frame_weights.csv using committor_labels; do not rerun RiteWeight.")

    args = ap.parse_args()

    cfg = load_yaml(args.config)

    roots = cfg.get("folders", [])
    if not roots:
        raise SystemExit("config.yaml must define 'folders: [..]'")

    match_dcd = cfg.get("match_dcd", "")
    match_colvars = cfg.get("match_colvars", "")
    tag_re = cfg.get("tag_regex", r"([AB])")

    top_path = cfg["io"]["top"]
    out = cfg["io"].get("out", "rw_out")
    stride = int(cfg["io"].get("stride", 1))

    allow_skip_first = bool(cfg.get("pairing", {}).get("allow_skip_first_colvars", True))
    strict = bool(cfg.get("pairing", {}).get("strict", False))

    pairs = find_pairs_dcd_colvars(roots, match_dcd, match_colvars, tag_re=tag_re)
    print(f"[INFO] Found {len(pairs)} (dcd,colvars) pairs across folders.")

    if args.check_mismatch:
        check_mismatch_report(pairs, top=top_path, stride=stride, allow_skip_first=allow_skip_first)
        return
    if args.relabel_only:
        relabel_only(cfg)
        return

    os.makedirs(out, exist_ok=True)

    # CV settings
    cv_cols = cfg["colvars"]["cv"]
    # --- colvars output controls ---
    save_cols = cfg.get("colvars", {}).get("save_cols", "all")
    periodic_cols = cfg.get("colvars", {}).get("periodic_cols", [])
    periodic_cols = [periodic_cols] if isinstance(periodic_cols, str) else list(periodic_cols)
    weight_mask_cfg = cfg.get("weight_mask", cfg.get("cv_weight_mask", {}))

    # --- committor label controls ---
    lab_cfg = cfg.get("committor_labels", {})
    label_enabled = bool(lab_cfg.get("enabled", False))
    check_cv_pair = None
    if label_enabled:
        cvs_to_label = lab_cfg["cvs_to_label"]
        if isinstance(cvs_to_label, str):
            cvs_to_label = [cvs_to_label]
        basin_A = lab_cfg["basin_A"]
        basin_B = lab_cfg["basin_B"]
        basin_size = lab_cfg["basin_size"]
        k_pref = float(lab_cfg.get("k_prefactor", 1.0))
        angle_unit = str(lab_cfg.get("angle_unit", "degree")).lower()
        determine_AB = determine_AB_functor(basin_A, basin_B, basin_size)
        check_cv_pair = committor_label_check_cv_pair(lab_cfg, cvs_to_label)


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

    X_list, CV_list = [], []
    seg_start_list, seg_end_list = [], []
    offset = 0
    used = 0

    # We resolve atomselect based on topology once (deterministic).
    top_for_sel = md.load_topology(top_path)

    zmat_atom_order: Optional[List[int]] = None
    if feat_mode == "internal_zmat":
        zmat_atom_order = resolve_zmat_atoms(top_for_sel, internal_cfg)
        print(f"[INFO] internal_zmat atoms (N={len(zmat_atom_order)}): {zmat_atom_order}")


    X_all, CV_all, seg_start, seg_end, df_save_all = load_or_compute_features_with_cache(
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
        save_cols=save_cols,
    )

    res = riteweight(
        X_all, seg_start, seg_end,
        n_clusters=n_clusters, n_iter=n_iter,
        tol=tol, tol_window=tol_window,
        avg_last=avg_last, seed=seed
    )

    # --- frame weights (full length, zeros for frames not used as segment starts) ---
    w_frame_full = np.zeros(X_all.shape[0], dtype=np.float64)
    w_frame_full[res.frame_map] = res.w_frame_nonzero

    # --- build output table ---
    out_df = df_save_all.copy()
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

    # --- optional CV-region masking: matched frames get zero frame weight ---
    w_frame_full, _ = apply_cv_weight_mask(out_df, w_frame_full, weight_mask_cfg)
    out_df["weight"] = w_frame_full

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

        if check_cv_pair is not None:
            save_committor_label_check(
                out_df,
                cv_pair=check_cv_pair,
                out_dir=out,
                cvs_to_label=cvs_to_label,
                basin_A=basin_A,
                basin_B=basin_B,
                basin_size=basin_size,
                lab_cfg=lab_cfg,
            )

    # PMF uses the masked frame weights.
    pmf_frame_mask = w_frame_full > 0.0
    if not np.any(pmf_frame_mask):
        raise SystemExit("No nonzero frame weights remain for PMF after CV weight masking.")
    CV_use = CV_all[pmf_frame_mask]
    w_use = w_frame_full[pmf_frame_mask]
    w_sum = float(w_use.sum())
    if w_sum <= 0.0:
        raise SystemExit("No positive PMF weight remains after CV weight masking.")
    w_use = w_use / w_sum

    # --- write CSV ---
    frame_csv_path = os.path.join(out, "frame_weights.csv")
    out_df.to_csv(frame_csv_path, index=False)
    print(f"[OK] Saved extended frame weights + colvars (+ labels) to: {frame_csv_path}")

    np.savetxt(os.path.join(out, "segment_weights.csv"),
               np.column_stack([np.arange(seg_start.size), w_frame_full[seg_start]]),
               delimiter=",", header="segment_index,weight", comments="")

    # convergence
    plt.figure()
    plt.plot(np.arange(1, len(res.delta_history) + 1), res.delta_history)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("L1 change in segment weights")
    # plt.ylabel("KL divergence in segment weights (log scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "convergence_delta.png"), dpi=200)
    plt.close()

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


if __name__ == "__main__":
    main()
