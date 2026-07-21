import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import linalg, ndimage
from scipy.interpolate import interpn
import yaml

KB_KCAL_PER_MOLK = 0.00198720425864083  # kcal/(mol*K)


# =========================================================
# === File and Data Utilities ===
# =========================================================

def find_matching(root, pattern):
    """Recursively find files matching pattern under root."""
    return sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))


def pair_by_subdir_ab(files, root):
    """Map (subdir, A|B) → full path."""
    m = {}
    for fp in files:
        rel = os.path.relpath(fp, root)
        sub = os.path.dirname(rel)
        abm = re.search(r"([AB])", os.path.basename(fp))
        if abm:
            m[(sub, abm.group(1))] = fp
    return m


def find_pairs_colvars_bias(root, match_colvars, match_bias):
    """Pair colvars and bias logs by subdir and A/B identifier."""
    colvars = find_matching(root, f"*{match_colvars}*.colvars.traj")
    blogs = find_matching(root, f"*{match_bias}*")
    if not colvars:
        raise FileNotFoundError(f"No files matching '*{match_colvars}*.colvars.traj' under {root}")
    if not blogs:
        raise FileNotFoundError(f"No files matching '*{match_bias}*' under {root}")

    m_c = pair_by_subdir_ab(colvars, root)
    m_b = pair_by_subdir_ab(blogs, root)
    pairs = [(cf, m_b[k]) for k, cf in m_c.items() if k in m_b]
    if not pairs:
        raise FileNotFoundError("No matching (colvars, bias log) pairs found.")
    return sorted(pairs)


def parse_header_indices(filepath, cv_names):
    """Return column indices for selected CVs from colvars header."""
    with open(filepath) as f:
        for line in f:
            if line.startswith("#") and "step" in line:
                header = line.strip("# \n").split()
                break
        else:
            raise ValueError(f"No header with 'step' found in {filepath}")
    return [header.index(name) for name in cv_names]


def load_selected_cvs(filename, cv_indices, stride=1):
    """Load selected CVs from colvars file."""
    data = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if line.startswith("#"):
                continue
            if i % stride == 0:
                parts = line.split()
                data.append([float(parts[j]) for j in cv_indices])
    return np.asarray(data, float)


def extract_energy(filepath, column_index=12):
    """Extract bias energies from NAMD log file (ENERGY: lines)."""
    pattern = re.compile(r"^ENERGY:\s+\d+")
    series = []
    with open(filepath) as f:
        for line in f:
            if pattern.search(line):
                cols = line.split()
                val = float(cols[column_index - 1])
                series.append(val)
    if not series:
        raise ValueError(f"No ENERGY lines found in {filepath}")
    return np.asarray(series, float)

def detect_stable_region(B_series, window=100, tol=5e-3, min_fraction=0.1):
    B = np.asarray(B_series, float)
    n = len(B)

    if n <= window:
        return float(np.mean(B)), 0

    running = np.convolve(B, np.ones(window) / window, mode='valid')
    diffs = np.abs(np.diff(running))   # 比 gradient 稳定

    # 平滑 diff
    if len(diffs) > 5:
        diffs = np.convolve(diffs, np.ones(5)/5, mode="same")

    # 默认起点（保证至少 min_fraction 的尾巴）
    max_idx_start = int((1 - min_fraction) * n)
    idx_start = max_idx_start

    # 从尾部寻找最后一次超过 tol 的点
    for i in range(len(diffs) - 1, -1, -1):
        if diffs[i] > tol:
            idx_start = min(i + 1, max_idx_start)
            break

    stable_mean = float(np.mean(B[idx_start:]))
    # print(idx_start)
    return stable_mean, idx_start


# =========================================================
# === Grid and Histogram Utilities ===
# =========================================================

def make_edges_from_union(data_list, bins, mins=None, maxs=None):
    """Generate grid edges covering all datasets."""
    all_data = np.vstack(data_list)
    lo = np.min(all_data, axis=0) if mins is None else np.asarray(mins, float)
    hi = np.max(all_data, axis=0) if maxs is None else np.asarray(maxs, float)
    return [np.linspace(a, b, n + 1) for (a, b), n in zip(zip(lo, hi), bins)]


def cell_volume(edges):
    """Volume of a single grid cell."""
    widths = [e[1] - e[0] for e in edges]
    return float(np.prod(widths))


def hist_density(samples, edges):
    """Uniform histogram density estimate."""
    H, _ = np.histogramdd(samples, bins=edges, density=True)
    H[H < 0] = 0.0
    return H


# =========================================================
# === Overlap Alignment ===
# =========================================================

def pairwise_deltas_from_overlap(p_list, kT, overlap_threshold=1e-4, min_overlap_bins=20):
    """Compute Δ_ij = C_i - C_j using bins where both p_i and p_j have mass."""
    N = len(p_list)
    deltas = {}
    for i in range(N):
        pi = p_list[i]
        thr_i = overlap_threshold * pi.max()
        for j in range(i + 1, N):
            pj = p_list[j]
            thr_j = overlap_threshold * pj.max()
            mask = (pi > thr_i) & (pj > thr_j)
            if mask.sum() >= min_overlap_bins:
                delta_ij = float(np.mean(kT * (np.log(pi[mask]) - np.log(pj[mask]))))
                deltas[(i, j)] = delta_ij
                deltas[(j, i)] = -delta_ij
    return deltas


def solve_offsets(N, deltas):
    """Solve constants C_i from Δ_ij constraints with gauge ∑C_i=0."""
    rows, rhs = [], []
    for (i, j), d in deltas.items():
        row = np.zeros(N)
        row[i], row[j] = 1, -1
        rows.append(row)
        rhs.append(d)
    if not rows:
        return np.zeros(N), False
    A, b = np.vstack(rows), np.array(rhs)
    A = np.vstack([A, np.ones(N)])
    b = np.concatenate([b, [0.0]])
    C, *_ = linalg.lstsq(A, b)
    return C, True


# =========================================================
# === Combine Probabilities and Compute Free Energy ===
# =========================================================

def combine_probabilities(p_list, C, kT, dV, traj_weights):
    """Combine per-trajectory densities into total probability."""
    shape = p_list[0].shape
    P = np.zeros(shape, float)
    for i, p in enumerate(p_list):
        P += traj_weights[i] * p * np.exp(-C[i] / kT)
    norm = (P * dV).sum()
    if norm <= 0:
        raise ValueError("Combined probability integral ≤ 0.")
    return P / norm


def free_energy_from_prob(P, kT, F_max=None):
    """Convert probability density to free energy (kcal/mol)."""
    eps = 1e-300
    F = -kT * np.log(np.where(P > 0, P, eps))
    F -= np.nanmin(F)
    if F_max is not None:
        F = np.minimum(F, F_max)
    return F


def gaussian_smooth_F(F, sigma_bins, F_max=None):
    """Gaussian-smooth the free energy surface."""
    G = ndimage.gaussian_filter(F, sigma=sigma_bins, mode="nearest")
    G -= np.nanmin(G)
    if F_max is not None:
        G = np.minimum(G, F_max)
    return G


# =========================================================
# === Output Writer ===
# =========================================================

def save_gromacs_like(filename, F, edges, cvs, periodicities, xlim=None, ylim=None):
    """Save 1D/2D free energy in GROMACS-style format (+ PDF)."""
    ndim = F.ndim
    ctrs = [0.5 * (e[:-1] + e[1:]) for e in edges]
    nbin = [len(e) - 1 for e in edges]

    xlim = xlim or [edges[0][0], edges[0][-1]]
    ylim = ylim or ([edges[1][0], edges[1][-1]] if ndim == 2 else None)
    new_edges = [np.linspace(xlim[0], xlim[1], nbin[0] + 1)]
    if ndim == 2:
        new_edges.append(np.linspace(ylim[0], ylim[1], nbin[1] + 1))
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
            plt.plot(new_ctrs[0], F)
            plt.xlabel(cvs[0]); plt.ylabel("Free Energy (kcal/mol)")
            plt.tight_layout(); plt.savefig(filename + ".pdf"); plt.close()
        else:
            for i, x in enumerate(new_ctrs[0]):
                for j, y in enumerate(new_ctrs[1]):
                    f.write(f"{x: .14e} {y: .14e} {F[i, j]: .14e}\n")
                f.write("\n")
            Xp, Yp = np.meshgrid(new_ctrs[0], new_ctrs[1], indexing="ij")
            plt.contourf(Xp, Yp, F, levels=20, cmap="RdBu_r")
            plt.colorbar(label="Free Energy (kcal/mol)")
            plt.xlabel(cvs[0]); plt.ylabel(cvs[1])
            plt.tight_layout(); plt.savefig(filename + ".pdf"); plt.close()


# =========================================================
# === Core Function ===
# =========================================================

def run_fel_estimate(config):
    """Perform overlap-constant matching and free energy reconstruction."""
    roots = config["folders"]
    match = config["match"]
    match_bias = config["match_bias"]
    log_column = config.get("log_column", 12)
    cvs = config["cv_names"]
    temperature = config["temperature"]
    bins = config.get("bins", [200])
    cvmins = config.get("cvmins", None)
    cvmaxs = config.get("cvmaxs", None)
    overlap_threshold = config.get("overlap_threshold", 1e-4)
    min_overlap_bins = config.get("min_overlap_bins", 20)
    periodicities = config.get("periodicities", [0])
    sigma = config.get("sigma", [1.0])
    F_max = config.get("F_max", None)
    xlim = config.get("xlim", None)
    ylim = config.get("ylim", None)
    stride = config.get("stride", 1)
    out_file = config.get("output", "free_energy.dat")
    scaled_bins = config.get("scaled_bins", None)
    scaled_out = config.get("scaled_output", "free_energy_scaled.dat")
    verbose = config.get("verbose", False)

    kT = KB_KCAL_PER_MOLK * temperature

    # Find and pair files
    pairs = []
    for folder in roots:
        pairs += find_pairs_colvars_bias(folder, match, match_bias)
    if not pairs:
        raise RuntimeError("No (colvars, bias) pairs found.")

    # Dimensionality
    col_idx = parse_header_indices(pairs[0][0], cvs)
    ndim = len(col_idx)
    if isinstance(bins, int):
        bins = [bins] * ndim

    # Load trajectories and compute stable biases
    data_list, bias_values = [], []
    for cf, bf in tqdm(pairs, desc="Loading trajectories & biases"):
        X = load_selected_cvs(cf, col_idx, stride)
        B_series = extract_energy(bf, column_index=log_column)
        _,idx_stable = detect_stable_region(B_series, window=100, tol=1e-3)
        B_delta = -kT * np.log(np.mean(np.exp(-B_series) / np.exp(np.mean(-B_series[idx_stable:]))))
        data_list.append(X)
        bias_values.append(B_delta)

    bias_values = np.array(bias_values)
    weights = np.exp(-bias_values / kT)
    weights /= np.sum(weights)

    if verbose:
        print("Stable bias values:", bias_values)
        print("Weights:", weights)

    # Grid
    edges = make_edges_from_union(data_list, bins, mins=cvmins, maxs=cvmaxs)
    dV = cell_volume(edges)

    # Build densities
    p_list = [hist_density(X, edges) for X in tqdm(data_list, desc="Building histograms")]

    # Overlap alignment
    deltas = pairwise_deltas_from_overlap(p_list, kT, overlap_threshold, min_overlap_bins)
    C, ok = solve_offsets(len(p_list), deltas)
    if not ok and verbose:
        print("No usable overlaps found; using C_i=0.")

    # Combine and compute F
    P_comb = combine_probabilities(p_list, C, kT, dV, weights)
    F = free_energy_from_prob(P_comb, kT, F_max)
    sigmas = sigma if isinstance(sigma, (list, tuple)) else [float(sigma)] * ndim
    F = gaussian_smooth_F(F, sigmas, F_max)

    # Save outputs
    save_gromacs_like(out_file, F, edges, cvs, periodicities, xlim, ylim)
    print(f"Saved {out_file} (+ .pdf)")

    # Scaled-down version
    if scaled_bins is not None:
        print(f"Generating scaled-down free energy ({scaled_bins} bins/dim)...")
        ctrs = [0.5 * (e[:-1] + e[1:]) for e in edges]
        scaled_edges = [np.linspace(e[0], e[-1], scaled_bins + 1) for e in edges]
        scaled_ctrs = [0.5 * (e[:-1] + e[1:]) for e in scaled_edges]
        if ndim == 1:
            xi = scaled_ctrs[0][:, None]
            F_scaled = interpn(ctrs, F, xi, method="linear", bounds_error=False, fill_value=np.nan).reshape(scaled_bins)
        else:
            Xn, Yn = np.meshgrid(scaled_ctrs[0], scaled_ctrs[1], indexing="ij")
            xi = np.stack([Xn, Yn], axis=-1).reshape(-1, 2)
            F_scaled = interpn(ctrs, F, xi, method="linear", bounds_error=False, fill_value=np.nan).reshape(scaled_bins, scaled_bins)
        F_scaled -= np.nanmin(F_scaled)
        save_gromacs_like(scaled_out, F_scaled, scaled_edges, cvs, periodicities, xlim, ylim)
        print(f"Saved scaled free energy to {scaled_out} (+ .pdf)")

    print(f"Processed {len(pairs)} trajectories. CVs: {', '.join(cvs)}")


# =========================================================
# === Entry Point ===
# =========================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Overlap-constant matching and bias reweighting")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    run_fel_estimate(config)
