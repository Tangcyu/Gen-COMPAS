#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

KB_KCAL_PER_MOL_K = 0.00198720425864083


@dataclass
class PMFND:
    ndim: int
    F: np.ndarray
    centers: List[np.ndarray]
    edges: List[np.ndarray]
    periodic: List[bool]
    binw: List[float]
    axis_names: List[str]


@dataclass
class BasinDef:
    ranges: Dict[str, Tuple[float, float]]
    Flo: Optional[float] = None
    Fhi: Optional[float] = None


def parse_float_or_inf(s: str) -> float:
    s = s.strip().lower()
    if s in ("inf", "+inf", "infinity", "+infinity"):
        return float("inf")
    if s in ("-inf", "-infinity"):
        return float("-inf")
    return float(s)


def parse_header_line(line: str) -> Tuple[float, float, int, int]:
    parts = line.lstrip("#").split()
    if len(parts) != 4:
        raise ValueError(f"Bad PMF header line: {line.strip()}")
    return float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])


def default_axis_names(ndim: int) -> List[str]:
    base = ["x", "y", "z"]
    if ndim <= 3:
        return base[:ndim]
    return base + [f"d{i}" for i in range(4, ndim + 1)]


def read_gromacs_like_pmf(path: str) -> PMFND:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if len(lines) < 2:
        raise ValueError("PMF file too short.")

    ndim = int(lines[0].lstrip("#").strip())
    if ndim not in (2, 3):
        raise ValueError(f"This script supports only 2D or 3D PMF, got ndim={ndim}")

    mins, binw, nbins, periodic = [], [], [], []
    for i in range(ndim):
        rmin, bw, nbin, per = parse_header_line(lines[1 + i])
        mins.append(rmin)
        binw.append(bw)
        nbins.append(nbin)
        periodic.append(bool(per))

    rows = []
    for ln in lines[1 + ndim:]:
        s = ln.strip()
        if not s:
            continue
        vals = s.split()
        if len(vals) < ndim + 1:
            continue
        rows.append([float(x) for x in vals[:ndim + 1]])

    arr = np.asarray(rows, dtype=float)
    expected_rows = int(np.prod(nbins))
    if arr.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows from header, got {arr.shape[0]}")

    centers = [np.unique(arr[:, i]) for i in range(ndim)]
    for i in range(ndim):
        if len(centers[i]) != nbins[i]:
            raise ValueError(
                f"Axis {i}: unique centers {len(centers[i])} != header bins {nbins[i]}"
            )

    shape = tuple(nbins)
    F = np.full(shape, np.nan, dtype=float)

    index_maps = [{v: j for j, v in enumerate(c)} for c in centers]
    for row in arr:
        idx = tuple(index_maps[d][row[d]] for d in range(ndim))
        F[idx] = row[ndim]

    if np.isnan(F).any():
        raise ValueError("Failed to reconstruct complete PMF grid.")

    edges = [
        np.linspace(mins[d], mins[d] + nbins[d] * binw[d], nbins[d] + 1)
        for d in range(ndim)
    ]

    return PMFND(
        ndim=ndim,
        F=F,
        centers=centers,
        edges=edges,
        periodic=periodic,
        binw=binw,
        axis_names=default_axis_names(ndim),
    )


def parse_basin(s: str, ndim: int) -> BasinDef:
    axis_names = default_axis_names(ndim)
    parts = [p.strip() for p in s.split(";") if p.strip()]
    parsed_ranges: Dict[str, Tuple[float, float]] = {}
    Flo = Fhi = None

    for p in parts:
        if ":" not in p:
            raise ValueError(f"Bad basin token: {p}")
        key, rng = p.split(":", 1)
        key = key.strip().lower()
        vals = [parse_float_or_inf(x) for x in rng.split(",")]
        if len(vals) != 2:
            raise ValueError(f"Bad range in basin token: {p}")

        if key == "f":
            Flo, Fhi = vals
        else:
            parsed_ranges[key] = (vals[0], vals[1])

    missing = [ax for ax in axis_names if ax not in parsed_ranges]
    if missing:
        raise ValueError(f"Basin missing axis ranges: {missing}")

    return BasinDef(ranges=parsed_ranges, Flo=Flo, Fhi=Fhi)


def interval_mask(values: np.ndarray, lo: float, hi: float, period: Optional[float]) -> np.ndarray:
    if period is None:
        m = np.ones_like(values, dtype=bool)
        if not np.isneginf(lo):
            m &= (values >= lo)
        if not np.isposinf(hi):
            m &= (values <= hi)
        return m

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Periodic coordinates do not support +/-inf basin bounds.")

    vals = np.mod(values, period)
    lo = lo % period
    hi = hi % period
    if lo <= hi:
        return (vals >= lo) & (vals <= hi)
    return (vals >= lo) | (vals <= hi)


def make_basin_mask(pmf: PMFND, basin: BasinDef) -> np.ndarray:
    masks_1d = []
    for d, ax in enumerate(pmf.axis_names):
        lo, hi = basin.ranges[ax]
        per = (pmf.edges[d][-1] - pmf.edges[d][0]) if pmf.periodic[d] else None
        masks_1d.append(interval_mask(pmf.centers[d], lo, hi, per))

    mask = np.ones_like(pmf.F, dtype=bool)
    for d, m1 in enumerate(masks_1d):
        shape = [1] * pmf.ndim
        shape[d] = len(m1)
        mask &= m1.reshape(shape)

    if basin.Flo is not None:
        mask &= (pmf.F >= basin.Flo)
    if basin.Fhi is not None:
        mask &= (pmf.F <= basin.Fhi)

    return mask


def cell_volume(binw: List[float]) -> float:
    return float(np.prod(binw))


def partition_from_mask(F: np.ndarray, mask: np.ndarray, beta: float, dV: float) -> float:
    if not np.any(mask):
        return 0.0
    vals = F[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    fmin = np.min(vals)
    return float(np.sum(np.exp(-beta * (vals - fmin))) * math.exp(-beta * fmin) * dV)


def total_partition(F: np.ndarray, beta: float, dV: float) -> float:
    vals = F[np.isfinite(F)]
    if vals.size == 0:
        raise ValueError("No finite PMF values.")
    fmin = np.min(vals)
    return float(np.sum(np.exp(-beta * (vals - fmin))) * math.exp(-beta * fmin) * dV)


def deltaG_from_Z(ZA: float, ZB: float, kT: float) -> float:
    if ZA <= 0 or ZB <= 0:
        return float("nan")
    return float(-kT * math.log(ZB / ZA))


def bootstrap_deltaG_from_pmf(
    F: np.ndarray,
    maskA: np.ndarray,
    maskB: np.ndarray,
    beta: float,
    dV: float,
    kT: float,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, np.ndarray]:
    """
    Bootstrap deltaG by resampling PMF bins inside each basin with replacement.
    We sample the Boltzmann weights exp(-beta F_i) of bins within each basin.
    """
    rng = np.random.default_rng(seed)

    valsA = F[maskA]
    valsA = valsA[np.isfinite(valsA)]
    valsB = F[maskB]
    valsB = valsB[np.isfinite(valsB)]

    if valsA.size == 0 or valsB.size == 0:
        return float("nan"), float("nan"), np.array([], dtype=float)

    wA = np.exp(-beta * valsA)
    wB = np.exp(-beta * valsB)

    dgs = np.zeros(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampleA = rng.choice(wA, size=wA.size, replace=True)
        sampleB = rng.choice(wB, size=wB.size, replace=True)
        ZA = float(np.sum(sampleA) * dV)
        ZB = float(np.sum(sampleB) * dV)
        dgs[i] = deltaG_from_Z(ZA, ZB, kT)

    return float(np.mean(dgs)), float(np.std(dgs, ddof=1)), dgs


def save_mask_csv(path: str, pmf: PMFND, maskA: np.ndarray, maskB: np.ndarray, beta: float):
    grids = np.meshgrid(*pmf.centers, indexing="ij")
    data = {pmf.axis_names[d]: grids[d].ravel() for d in range(pmf.ndim)}
    finite = np.isfinite(pmf.F)
    boltz = np.zeros_like(pmf.F, dtype=float)
    boltz[finite] = np.exp(-beta * pmf.F[finite])

    data["F_kcal_per_mol"] = pmf.F.ravel()
    data["in_basin_A"] = maskA.ravel().astype(int)
    data["in_basin_B"] = maskB.ravel().astype(int)
    data["boltzmann_unnorm"] = boltz.ravel()

    pd.DataFrame(data).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Integrate 2D/3D PMF over two basins and compute deltaG with bootstrap std.")
    ap.add_argument("--pmf", required=True, help="Input PMF .dat file")
    ap.add_argument("--basinA", required=True, help='Example 2D: "x:20,40;y:10,25" | 3D: "x:20,40;y:10,25;z:0,8"')
    ap.add_argument("--basinB", required=True, help='Example 2D: "x:5,15;y:35,50" | 3D: "x:5,15;y:35,50;z:20,30"')
    ap.add_argument("--T", type=float, default=300.0, help="Temperature in K")
    ap.add_argument("--out", default="basin_integration", help="Output prefix or directory")
    ap.add_argument("--save-mask-csv", action="store_true", help="Write per-bin basin mask CSV")
    ap.add_argument("--bootstrap", type=int, default=500, help="Number of bootstrap replicas")
    ap.add_argument("--seed", type=int, default=2026, help="Random seed for bootstrap")
    args = ap.parse_args()

    pmf = read_gromacs_like_pmf(args.pmf)
    basinA = parse_basin(args.basinA, pmf.ndim)
    basinB = parse_basin(args.basinB, pmf.ndim)

    beta = 1.0 / (KB_KCAL_PER_MOL_K * args.T)
    kT = KB_KCAL_PER_MOL_K * args.T
    dV = cell_volume(pmf.binw)

    maskA = make_basin_mask(pmf, basinA)
    maskB = make_basin_mask(pmf, basinB)

    ZA = partition_from_mask(pmf.F, maskA, beta, dV)
    ZB = partition_from_mask(pmf.F, maskB, beta, dV)
    Ztot = total_partition(pmf.F, beta, dV)

    if ZA <= 0:
        raise SystemExit("Basin A has zero partition function. Check basin definition.")
    if ZB <= 0:
        raise SystemExit("Basin B has zero partition function. Check basin definition.")

    PA = ZA / Ztot
    PB = ZB / Ztot

    GA = -kT * math.log(ZA)
    GB = -kT * math.log(ZB)
    deltaG = GB - GA

    dg_boot_mean, dg_boot_std, dg_boot = bootstrap_deltaG_from_pmf(
        pmf.F, maskA, maskB, beta, dV, kT, args.bootstrap, args.seed
    )

    if "." not in os.path.basename(args.out):
        os.makedirs(args.out, exist_ok=True)
        txt_out = os.path.join(args.out, "deltaG_summary.txt")
        csv_out = os.path.join(args.out, "basin_summary.csv")
        mask_out = os.path.join(args.out, "basin_masks.csv")
        boot_out = os.path.join(args.out, "deltaG_bootstrap.csv")
    else:
        txt_out = f"{args.out}_summary.txt"
        csv_out = f"{args.out}_summary.csv"
        mask_out = f"{args.out}_masks.csv"
        boot_out = f"{args.out}_bootstrap.csv"

    rows = [
        ("ndim", pmf.ndim),
        ("T_K", args.T),
        ("kT_kcal_per_mol", kT),
        ("cell_volume", dV),
        ("Z_A", ZA),
        ("Z_B", ZB),
        ("P_A", PA),
        ("P_B", PB),
        ("G_A_kcal_per_mol", GA),
        ("G_B_kcal_per_mol", GB),
        ("deltaG_B_minus_A_kcal_per_mol", deltaG),
        ("deltaG_bootstrap_mean_kcal_per_mol", dg_boot_mean),
        ("deltaG_bootstrap_std_kcal_per_mol", dg_boot_std),
        ("n_bins_A", int(np.sum(maskA))),
        ("n_bins_B", int(np.sum(maskB))),
        ("n_bootstrap", args.bootstrap),
    ]
    pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(csv_out, index=False)

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(f"Input PMF: {args.pmf}\n")
        f.write(f"ndim = {pmf.ndim}\n")
        f.write(f"T = {args.T:.6f} K\n")
        f.write(f"kT = {kT:.8f} kcal/mol\n")
        f.write(f"cell volume = {dV:.12e}\n\n")

        f.write("Basin A:\n")
        for ax in pmf.axis_names:
            lo, hi = basinA.ranges[ax]
            f.write(f"  {ax} in [{lo}, {hi}]\n")
        if basinA.Flo is not None or basinA.Fhi is not None:
            f.write(f"  F in [{basinA.Flo}, {basinA.Fhi}]\n")
        f.write(f"  selected bins = {int(np.sum(maskA))}\n")
        f.write(f"  Z_A = {ZA:.12e}\n")
        f.write(f"  P_A = {PA:.12e}\n")
        f.write(f"  G_A = {GA:.8f} kcal/mol\n\n")

        f.write("Basin B:\n")
        for ax in pmf.axis_names:
            lo, hi = basinB.ranges[ax]
            f.write(f"  {ax} in [{lo}, {hi}]\n")
        if basinB.Flo is not None or basinB.Fhi is not None:
            f.write(f"  F in [{basinB.Flo}, {basinB.Fhi}]\n")
        f.write(f"  selected bins = {int(np.sum(maskB))}\n")
        f.write(f"  Z_B = {ZB:.12e}\n")
        f.write(f"  P_B = {PB:.12e}\n")
        f.write(f"  G_B = {GB:.8f} kcal/mol\n\n")

        f.write(f"deltaG = G_B - G_A = {deltaG:.8f} kcal/mol\n")
        f.write(f"deltaG bootstrap mean = {dg_boot_mean:.8f} kcal/mol\n")
        f.write(f"deltaG bootstrap std  = {dg_boot_std:.8f} kcal/mol\n")

    pd.DataFrame({"deltaG_bootstrap_kcal_per_mol": dg_boot}).to_csv(boot_out, index=False)

    if args.save_mask_csv:
        save_mask_csv(mask_out, pmf, maskA, maskB, beta)

    print(f"[OK] Wrote summary: {txt_out}")
    print(f"[OK] Wrote CSV: {csv_out}")
    print(f"[OK] Wrote bootstrap samples: {boot_out}")
    if args.save_mask_csv:
        print(f"[OK] Wrote mask CSV: {mask_out}")
    print(f"deltaG (B - A) = {deltaG:.8f} kcal/mol")
    print(f"deltaG std      = {dg_boot_std:.8f} kcal/mol")


if __name__ == "__main__":
    main()