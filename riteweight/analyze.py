#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, re, glob, argparse
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import yaml
from tqdm import tqdm

import mdtraj as md

KB_KCAL_PER_MOL_K = 0.00198720425864083


# =========================================================
# === Pairing logic (same as your main script) ===
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
    tag_re: str = r"([AB])",
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
        pairs.extend([(m_d[k], m_c[k]) for k in common])
    if not pairs:
        raise FileNotFoundError("No matching (dcd, colvars) pairs found across all folders.")
    return sorted(pairs)


# =========================================================
# === Colvars IO + alignment (same policy) ===
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
    df = pd.read_csv(path, sep="\s+", comment="#", header=None)
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
# === PMF / FEL helpers ===
# =========================================================

def fes_from_prob_mass(P: np.ndarray, T_K: float, floor: float = 1e-300) -> np.ndarray:
    """F = -kT ln P + const; P is probability mass on bins (sum(P)=1)."""
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
    nbin = [len(e) - 1 for e in edges]

    xlim = xlim or [edges[0][0], edges[0][-1]]
    new_edges = [np.linspace(float(xlim[0]), float(xlim[1]), nbin[0] + 1)]
    if ndim == 2:
        ylim = ylim or [edges[1][0], edges[1][-1]]
        new_edges.append(np.linspace(float(ylim[0]), float(ylim[1]), nbin[1] + 1))
    new_ctrs = [0.5 * (e[:-1] + e[1:]) for e in new_edges]

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

    pdf = filename + ".pdf"
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

def build_edges_from_union(data_list: List[np.ndarray], bins, mins=None, maxs=None) -> List[np.ndarray]:
    all_data = np.vstack(data_list)
    lo = np.min(all_data, axis=0) if mins is None else np.asarray(mins, float)
    hi = np.max(all_data, axis=0) if maxs is None else np.asarray(maxs, float)
    bins_list = list(bins) if isinstance(bins, (list, tuple)) else [int(bins)] * all_data.shape[1]
    return [np.linspace(a, b, n + 1) for (a, b), n in zip(zip(lo, hi), bins_list)]

def weighted_hist_prob_mass(samples: np.ndarray, weights: np.ndarray, edges: List[np.ndarray]) -> np.ndarray:
    H, _ = np.histogramdd(samples, bins=edges, weights=weights, density=False)
    P = H.astype(np.float64)
    s = P.sum()
    if s > 0:
        P /= s
    return P


# =========================================================
# === Core: load CV per trajectory + slice frame weights ===
# =========================================================

def load_frame_weights_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if "weight" not in df.columns:
        raise ValueError(f"{path} must have column 'weight'")
    return df["weight"].to_numpy(dtype=np.float64)

def load_traj_cvs_and_weights(
    cfg: dict,
    pairs: List[Tuple[str, str]],
    frame_weights: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Returns:
      cvs_list: list of (nFi, dim)
      w_list:   list of (nFi,) (includes zeros for last frame etc.)
      edges_basis_data: list of CV arrays for building union edges (same as cvs_list)
      frame_idx_list: list of global frame indices for each traj (nFi,)
    """
    top = cfg["io"]["top"]
    stride = int(cfg["io"].get("stride", 1))
    allow_skip_first = bool(cfg.get("pairing", {}).get("allow_skip_first_colvars", True))
    strict = bool(cfg.get("pairing", {}).get("strict", False))

    cv_cols = cfg["colvars"]["cv"]
    if isinstance(cv_cols, str):
        cv_cols = [cv_cols]
    dim = len(cv_cols)

    cvs_list: List[np.ndarray] = []
    w_list: List[np.ndarray] = []
    frame_idx_list: List[np.ndarray] = []

    offset = 0
    for dcd_path, col_path in tqdm(pairs, desc="Loading trajectories"):
        traj = md.load(dcd_path, top=top)
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

        for c in cv_cols:
            if c not in df2.columns:
                raise SystemExit(f"CV column '{c}' not found in {col_path}.")

        CV = df2[cv_cols].to_numpy(dtype=np.float64)
        nF = CV.shape[0]
        if nF == 0:
            continue

        # global frame indices for this trajectory chunk
        gidx = np.arange(offset, offset + nF, dtype=np.int64)

        # slice weights
        nW = frame_weights.shape[0]
        need = offset + nF
        if need > nW:
            overflow = need - nW
            # Allow Overflow and output error message, but truncate to available weights
            print(f"[WARN] frame_weights.csv shorter than reconstructed frames by {overflow} rows.")
            print(f"       Offending pair: {dcd_path}")
            print(f"       colvars rows (after alignment/stride) = {nF}, remaining weights = {nW - offset}")

            # cut CV and weights to available frames
            new_nF = max(0, nW - offset)
            if new_nF == 0:
                print("[WARN] No remaining weights; stop reading further pairs.")
                break

            CV = CV[:new_nF, :]
            nF = new_nF
            gidx = np.arange(offset, offset + nF, dtype=np.int64)

        W = frame_weights[gidx].astype(np.float64)

        # Basic shape check
        if CV.shape[1] != dim:
            raise SystemExit("CV dimension mismatch.")

        cvs_list.append(CV)
        w_list.append(W)
        frame_idx_list.append(gidx)

        offset += nF

    if offset != frame_weights.shape[0]:
        print(f"[WARN] Total frames reconstructed from paired colvars = {offset}, "
            f"but frame_weights has {frame_weights.shape[0]}. "
            f"Difference={offset - frame_weights.shape[0]}. "
            f"This often indicates a small mismatch/skip_first difference in one segment.")
            
    return cvs_list, w_list, cvs_list, frame_idx_list


# =========================================================
# === Error: trajectory bootstrap on F ===
# =========================================================

def bootstrap_F(
    cvs_list: List[np.ndarray],
    w_list: List[np.ndarray],
    edges: List[np.ndarray],
    T_K: float,
    sigma_bins: Optional[List[float]],
    F_max: Optional[float],
    n_bootstrap: int,
    ci: float,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = len(cvs_list)
    Fs = []

    for b in tqdm(range(n_bootstrap), desc="Bootstrap"):
        idx = rng.integers(0, N, size=N)  # resample trajectories with replacement
        CV = np.vstack([cvs_list[i] for i in idx])
        W = np.concatenate([w_list[i] for i in idx])
        # normalize weights within bootstrap sample
        s = W.sum()
        if s <= 0:
            continue
        W = W / s

        P = weighted_hist_prob_mass(CV, W, edges)
        F = fes_from_prob_mass(P, T_K)
        if F_max is not None:
            F = np.minimum(F, float(F_max))
        F = gaussian_smooth_F(F, sigma_bins=sigma_bins, F_max=F_max)
        Fs.append(F)

    Fs = np.stack(Fs, axis=0)  # (B, ...)
    F_mean = np.nanmean(Fs, axis=0)
    F_std = np.nanstd(Fs, axis=0)

    alpha = 1.0 - float(ci)
    lo = np.nanpercentile(Fs, 100 * (alpha / 2), axis=0)
    hi = np.nanpercentile(Fs, 100 * (1 - alpha / 2), axis=0)

    return F_mean, lo, hi, F_std


# =========================================================
# === Convergence: permutations of trajectory accumulation ===
# =========================================================

def rmse_masked(A: np.ndarray, B: np.ndarray, mask: np.ndarray) -> float:
    D = (A - B)[mask]
    if D.size == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean(D * D)))

def convergence_curve(
    cvs_list: List[np.ndarray],
    w_list: List[np.ndarray],
    edges: List[np.ndarray],
    T_K: float,
    sigma_bins: Optional[List[float]],
    F_max: Optional[float],
    p_rel_threshold: float,
    n_permute: int,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = len(cvs_list)

    # Final (all trajectories)
    CV_all = np.vstack(cvs_list)
    W_all = np.concatenate(w_list)
    W_all = W_all / W_all.sum()
    P_final = weighted_hist_prob_mass(CV_all, W_all, edges)
    F_final = fes_from_prob_mass(P_final, T_K)
    if F_max is not None:
        F_final = np.minimum(F_final, float(F_max))
    F_final = gaussian_smooth_F(F_final, sigma_bins=sigma_bins, F_max=F_max)

    # mask based on probability threshold relative to max
    thr = float(p_rel_threshold) * float(np.max(P_final)) if np.max(P_final) > 0 else 0.0
    mask = P_final > thr

    curves = np.full((n_permute, N), np.nan, dtype=np.float64)

    for r in range(n_permute):
        perm = rng.permutation(N)
        CV_acc = []
        W_acc = []
        for k in range(1, N + 1):
            i = perm[k - 1]
            CV_acc.append(cvs_list[i])
            W_acc.append(w_list[i])
            CV = np.vstack(CV_acc)
            W = np.concatenate(W_acc)
            s = W.sum()
            if s <= 0:
                continue
            W = W / s
            Pk = weighted_hist_prob_mass(CV, W, edges)
            Fk = fes_from_prob_mass(Pk, T_K)
            if F_max is not None:
                Fk = np.minimum(Fk, float(F_max))
            Fk = gaussian_smooth_F(Fk, sigma_bins=sigma_bins, F_max=F_max)
            curves[r, k - 1] = rmse_masked(Fk, F_final, mask)

    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0)
    return mean, std


# =========================================================
# === Entry ===
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="Estimate FEL/PMF error bars and convergence using existing frame weights.")
    ap.add_argument("--config", required=True, help="YAML config used for the main RiteWeight run (plus analysis section).")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # pairing config
    roots = cfg.get("folders", [])
    if not roots:
        raise SystemExit("config.yaml must define folders: [..]")
    match_dcd = cfg.get("match_dcd", "")
    match_colvars = cfg.get("match_colvars", "")
    tag_re = cfg.get("tag_regex", r"([AB])")

    pairs = find_pairs_dcd_colvars(roots, match_dcd, match_colvars, tag_re=tag_re)
    print(f"[INFO] Found {len(pairs)} (dcd,colvars) pairs.")

    # analysis config
    ana = cfg.get("analysis", {})
    fw_csv = ana.get("frame_weights_csv", None)
    if not fw_csv:
        raise SystemExit("config.yaml: analysis.frame_weights_csv is required.")
    out_dir = ana.get("out", "analysis_out")
    os.makedirs(out_dir, exist_ok=True)

    boot = ana.get("bootstrap", {})
    n_bootstrap = int(boot.get("n_bootstrap", 200))
    ci = float(boot.get("ci", 0.95))

    conv = ana.get("convergence", {})
    n_permute = int(conv.get("n_permute", 20))

    mask_cfg = ana.get("mask", {})
    p_rel_threshold = float(mask_cfg.get("p_rel_threshold", 1e-4))

    outs = ana.get("outputs", {})
    pmf_mean_name = outs.get("pmf_mean", "pmf_mean.dat")
    pmf_lo_name = outs.get("pmf_ci_low", "pmf_ci_low.dat")
    pmf_hi_name = outs.get("pmf_ci_high", "pmf_ci_high.dat")
    pmf_std_name = outs.get("pmf_ci_std", "pmf_ci_std.dat")
    conv_csv_name = outs.get("convergence_csv", "convergence_rmse.csv")
    conv_pdf_name = outs.get("convergence_pdf", "convergence_rmse.pdf")

    # FEL config
    cv_cols = cfg["colvars"]["cv"]
    if isinstance(cv_cols, str):
        cv_cols = [cv_cols]
    ndim = len(cv_cols)
    if ndim not in (1, 2):
        raise SystemExit("Only 1D/2D CV supported.")

    bins = cfg["colvars"].get("bins", 200)
    if isinstance(bins, int):
        bins = [bins] * ndim
    T_K = float(cfg["colvars"].get("temperature_K", 300.0))
    xlim = cfg["colvars"].get("xlim", None)
    ylim = cfg["colvars"].get("ylim", None)
    periodicities = cfg["colvars"].get("periodicities", [0] * ndim)

    # smoothing / cap (reuse pmf_output if present)
    pmf_cfg = cfg.get("pmf_output", {})
    sigma_bins = pmf_cfg.get("sigma_bins", None)
    F_max = pmf_cfg.get("F_max", None)

    # load weights + data
    frame_weights = load_frame_weights_csv(fw_csv)
    cvs_list, w_list, edges_basis_data, _ = load_traj_cvs_and_weights(cfg, pairs, frame_weights)
    print(f"[INFO] Loaded {len(cvs_list)} trajectories worth of CV+weights for analysis.")

    # Build edges from union (consistent with your other scripts)
    # edges = build_edges_from_union(edges_basis_data, bins=bins, mins=None, maxs=None)

    if ndim == 1:
        if xlim is not None:
            edges = [np.linspace(float(xlim[0]), float(xlim[1]), int(bins[0]) + 1)]
        else:
            edges = build_edges_from_union(edges_basis_data, bins=bins, mins=None, maxs=None)
    else:
        if xlim is not None and ylim is not None:
            edges = [
                np.linspace(float(xlim[0]), float(xlim[1]), int(bins[0]) + 1),
                np.linspace(float(ylim[0]), float(ylim[1]), int(bins[1]) + 1),
            ]
        else:
            edges = build_edges_from_union(edges_basis_data, bins=bins, mins=None, maxs=None)

    # Bootstrap error bars
    print(f"[INFO] Bootstrapping (trajectory-level) n_bootstrap={n_bootstrap}, CI={ci} ...")
    F_mean, F_lo, F_hi, F_std = bootstrap_F(
        cvs_list=cvs_list,
        w_list=w_list,
        edges=edges,
        T_K=T_K,
        sigma_bins=sigma_bins,
        F_max=F_max,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=2026,
    )

    save_gromacs_like(os.path.join(out_dir, pmf_mean_name), F_mean, edges, cv_cols, periodicities, xlim, ylim)
    save_gromacs_like(os.path.join(out_dir, pmf_lo_name), F_lo, edges, cv_cols, periodicities, xlim, ylim)
    save_gromacs_like(os.path.join(out_dir, pmf_hi_name), F_hi, edges, cv_cols, periodicities, xlim, ylim)
    save_gromacs_like(os.path.join(out_dir, pmf_std_name), F_std, edges, cv_cols, periodicities, xlim, ylim)
    print("[OK] Saved PMF mean/CI .dat (+ .pdf).")

    # Convergence curves
    print(f"[INFO] Convergence permutations n_permute={n_permute} ...")
    mean_rmse, std_rmse = convergence_curve(
        cvs_list=cvs_list,
        w_list=w_list,
        edges=edges,
        T_K=T_K,
        sigma_bins=sigma_bins,
        F_max=F_max,
        p_rel_threshold=p_rel_threshold,
        n_permute=n_permute,
        seed=2027,
    )

    # Save CSV + plot
    k = np.arange(1, len(mean_rmse) + 1)
    conv_df = pd.DataFrame({"n_traj": k, "rmse_mean": mean_rmse, "rmse_std": std_rmse})
    conv_csv_path = os.path.join(out_dir, conv_csv_name)
    conv_df.to_csv(conv_csv_path, index=False)
    print(f"[OK] Saved convergence CSV: {conv_csv_path}")

    plt.figure()
    plt.plot(k, mean_rmse)
    plt.fill_between(k, mean_rmse - std_rmse, mean_rmse + std_rmse, alpha=0.3)
    plt.xlabel("Number of trajectories included")
    plt.ylabel("RMSE vs final FEL (kcal/mol)")
    plt.tight_layout()
    conv_pdf_path = os.path.join(out_dir, conv_pdf_name)
    plt.savefig(conv_pdf_path)
    plt.close()
    print(f"[OK] Saved convergence plot: {conv_pdf_path}")


if __name__ == "__main__":
    main()