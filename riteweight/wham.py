#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import interpn

KB_KCAL_PER_MOLK = 0.00198720425864083  # kcal/(mol*K)


def cell_volume(edges):
    widths = [e[1] - e[0] for e in edges]
    return float(np.prod(widths))


def make_edges(data: np.ndarray, bins, mins=None, maxs=None):
    ndim = data.shape[1]
    if isinstance(bins, int):
        bins = [bins] * ndim
    lo = np.min(data, axis=0) if mins is None else np.asarray(mins, float)
    hi = np.max(data, axis=0) if maxs is None else np.asarray(maxs, float)
    return [np.linspace(lo[d], hi[d], int(bins[d]) + 1) for d in range(ndim)]


def hist_density(samples: np.ndarray, edges, weights: np.ndarray):
    H, _ = np.histogramdd(samples, bins=edges, weights=weights, density=False)
    dV = cell_volume(edges)
    mass = H.sum()
    if mass <= 0:
        raise ValueError("Total weighted counts <= 0. Check weights.")
    # convert to density so integral P(x) dx = 1
    P = (H / mass) / dV
    return P


def free_energy_from_prob(P, kT, F_max=None):
    eps = 1e-300
    F = -kT * np.log(np.where(P > 0, P, eps))
    F -= np.nanmin(F[np.isfinite(F)])
    if F_max is not None:
        F = np.minimum(F, float(F_max))
    return F


def gaussian_smooth_F(F, sigma_bins, F_max=None):
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


def save_gromacs_like(filename, F, edges, cvs, periodicities, xlim=None, ylim=None):
    ndim = F.ndim
    ctrs = [0.5 * (e[:-1] + e[1:]) for e in edges]
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
            plt.figure()
            plt.plot(new_ctrs[0], F)
            plt.xlabel(cvs[0]); plt.ylabel("Free Energy (kcal/mol)")
            plt.tight_layout(); plt.savefig(filename + ".pdf"); plt.close()
        else:
            for i, x in enumerate(new_ctrs[0]):
                for j, y in enumerate(new_ctrs[1]):
                    f.write(f"{x: .14e} {y: .14e} {F[i, j]: .14e}\n")
                f.write("\n")
            Xp, Yp = np.meshgrid(new_ctrs[0], new_ctrs[1], indexing="ij")
            plt.figure()
            plt.contourf(Xp, Yp, F, levels=20, cmap="RdBu_r")
            plt.colorbar(label="Free Energy (kcal/mol)")
            plt.xlabel(cvs[0]); plt.ylabel(cvs[1])
            plt.tight_layout(); plt.savefig(filename + ".pdf"); plt.close()


def save_scaled(F, edges, scaled_bins, outname, cvs, periodicities, xlim=None, ylim=None):
    if scaled_bins is None:
        return
    scaled_bins = int(scaled_bins)
    if scaled_bins <= 1:
        return

    ndim = F.ndim
    print(f"Generating scaled-down free energy ({scaled_bins} bins/dim)...")

    ctrs = [0.5 * (e[:-1] + e[1:]) for e in edges]
    scaled_edges = [np.linspace(float(e[0]), float(e[-1]), scaled_bins + 1) for e in edges]
    scaled_ctrs = [0.5 * (e[:-1] + e[1:]) for e in scaled_edges]

    if ndim == 1:
        xi = scaled_ctrs[0][:, None]
        F_scaled = interpn(ctrs, F, xi, method="linear", bounds_error=False, fill_value=np.nan).reshape(scaled_bins)
    else:
        Xn, Yn = np.meshgrid(scaled_ctrs[0], scaled_ctrs[1], indexing="ij")
        xi = np.stack([Xn, Yn], axis=-1).reshape(-1, 2)
        F_scaled = interpn(ctrs, F, xi, method="linear", bounds_error=False, fill_value=np.nan).reshape(scaled_bins, scaled_bins)

    F_scaled -= np.nanmin(F_scaled[np.isfinite(F_scaled)])
    save_gromacs_like(outname, F_scaled, scaled_edges, cvs, periodicities, xlim, ylim)
    print(f"Saved scaled free energy to {outname} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser(description="Weighted histogram + Gaussian smoothing from frame_weights.csv")
    ap.add_argument("--csv", required=True, help="Path to frame_weights.csv")
    ap.add_argument("--cv", nargs="+", required=True, help="1 or 2 CV column names")
    ap.add_argument("--bins", nargs="+", required=True, help="Bins: 1D -> N ; 2D -> Nx Ny", type=int)
    ap.add_argument("--T", type=float, required=True, help="Temperature (K)")
    ap.add_argument("--weight-col", default="weight", help="Weight column name (default: weight)")
    ap.add_argument("--out", default="pmf.dat", help="Output .dat filename")
    ap.add_argument("--periodicities", nargs="+", type=int, default=None, help="0/1 per dim (default all 0)")
    ap.add_argument("--sigma", nargs="+", type=float, default=None, help="Gaussian sigma in bins (per dim)")
    ap.add_argument("--F-max", type=float, default=None, help="Cap FEL at this value (kcal/mol)")
    ap.add_argument("--xlim", nargs=2, type=float, default=None, help="xmin xmax")
    ap.add_argument("--ylim", nargs=2, type=float, default=None, help="ymin ymax")
    ap.add_argument("--scaled-bins", type=int, default=None, help="Generate scaled-down FEL with this bins/dim")
    ap.add_argument("--scaled-out", default="pmf_scaled.dat", help="Scaled output .dat filename")
    ap.add_argument("--grid-csv", default="pmf_grid.csv", help="Output grid CSV (centers + prob + F)")
    args = ap.parse_args()

    cvs = args.cv
    if len(cvs) not in (1, 2):
        raise SystemExit("--cv must have 1 or 2 column names")

    df = pd.read_csv(args.csv)
    for c in cvs:
        if c not in df.columns:
            raise SystemExit(f"CV column '{c}' not in {args.csv}")
    if args.weight_col not in df.columns:
        raise SystemExit(f"Weight column '{args.weight_col}' not in {args.csv}")

    X = df[cvs].to_numpy(dtype=float)
    w = df[args.weight_col].to_numpy(dtype=float)
    s = w.sum()
    if s <= 0:
        raise SystemExit("Sum of weights <= 0")
    w = w / s

    ndim = X.shape[1]
    bins = args.bins
    if ndim == 1 and len(bins) != 1:
        raise SystemExit("1D requires one --bins value")
    if ndim == 2 and len(bins) != 2:
        raise SystemExit("2D requires two --bins values")

    periodicities = args.periodicities if args.periodicities is not None else [0] * ndim
    if len(periodicities) != ndim:
        raise SystemExit("--periodicities length must match ndim")

    sigma = args.sigma
    if sigma is not None:
        if len(sigma) == 1 and ndim == 2:
            sigma = [sigma[0], sigma[0]]
        if len(sigma) != ndim:
            raise SystemExit("--sigma length must match ndim (or give single value for 2D)")

    kT = KB_KCAL_PER_MOLK * float(args.T)

    edges = make_edges(X, bins=bins, mins=args.xlim if ndim == 1 else args.xlim,
                       maxs=args.ylim if ndim == 2 else None)

    # If user provided xlim/ylim, override edges range for histogram as well
    if ndim == 1 and args.xlim is not None:
        edges = [np.linspace(args.xlim[0], args.xlim[1], bins[0] + 1)]
    if ndim == 2:
        xlo = args.xlim[0] if args.xlim is not None else float(np.min(X[:, 0]))
        xhi = args.xlim[1] if args.xlim is not None else float(np.max(X[:, 0]))
        ylo = args.ylim[0] if args.ylim is not None else float(np.min(X[:, 1]))
        yhi = args.ylim[1] if args.ylim is not None else float(np.max(X[:, 1]))
        edges = [np.linspace(xlo, xhi, bins[0] + 1), np.linspace(ylo, yhi, bins[1] + 1)]

    P = hist_density(X, edges, w)
    F = free_energy_from_prob(P, kT, F_max=args.F_max)
    F = gaussian_smooth_F(F, sigma_bins=sigma, F_max=args.F_max)

    save_gromacs_like(args.out, F, edges, cvs, periodicities, xlim=args.xlim, ylim=args.ylim)

    # grid csv
    ctrs = [0.5 * (e[:-1] + e[1:]) for e in edges]
    if ndim == 1:
        grid = np.column_stack([ctrs[0], P, F])
        np.savetxt(args.grid_csv, grid, delimiter=",", header=f"{cvs[0]}_center,prob_density,fes_kcal_per_mol", comments="")
    else:
        Xc, Yc = np.meshgrid(ctrs[0], ctrs[1], indexing="ij")
        grid = np.column_stack([Xc.ravel(), Yc.ravel(), P.ravel(), F.ravel()])
        np.savetxt(args.grid_csv, grid, delimiter=",",
                   header=f"{cvs[0]}_center,{cvs[1]}_center,prob_density,fes_kcal_per_mol", comments="")

    save_scaled(F, edges, args.scaled_bins, args.scaled_out, cvs, periodicities, xlim=args.xlim, ylim=args.ylim)

    print(f"[OK] Saved: {args.out} (+.pdf), grid: {args.grid_csv}")
    if args.scaled_bins is not None:
        print(f"[OK] Saved scaled: {args.scaled_out} (+.pdf)")


if __name__ == "__main__":
    main()