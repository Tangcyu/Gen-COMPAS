"""Weighted free-energy projections from RiteWeight frame data."""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
import yaml

from tools.tensor_table import load_tensor_table


KB_KCAL_PER_MOL_K = 0.00198720425864083


def load_riteweight_table(path: str) -> pd.DataFrame:
    """Load RiteWeight frame data from its Torch table or CSV output."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"RiteWeight result not found: {path}")
    if path.lower().endswith(('.pt', '.pth')):
        return load_tensor_table(path)
    return pd.read_csv(path)


def _expand(value: Any, ndim: int, name: str, default):
    if value is None:
        return [default] * ndim
    if np.isscalar(value):
        return [value] * ndim
    values = list(value)
    if len(values) != ndim:
        raise ValueError(f"{name} must contain {ndim} values; got {len(values)}.")
    return values


def _projection_ranges(values: np.ndarray, configured):
    ndim = values.shape[1]
    if configured is None:
        ranges = []
        for axis in range(ndim):
            lower = float(np.min(values[:, axis]))
            upper = float(np.max(values[:, axis]))
            if not upper > lower:
                upper = lower + 1.0
            ranges.append((lower, upper))
        return ranges

    if len(configured) != ndim:
        raise ValueError(f"ranges must contain one [min, max] pair per CV ({ndim}).")
    ranges = []
    for limits in configured:
        if limits is None or len(limits) != 2:
            raise ValueError("Each projection range must be [min, max].")
        lower, upper = map(float, limits)
        if not upper > lower:
            raise ValueError(f"Invalid projection range: {limits}")
        ranges.append((lower, upper))
    return ranges


def _wrap_periodic(values: np.ndarray, ranges, periodicities):
    wrapped = values.copy()
    for axis, periodic in enumerate(periodicities):
        if periodic:
            lower, upper = ranges[axis]
            wrapped[:, axis] = (wrapped[:, axis] - lower) % (upper - lower) + lower
    return wrapped


def _free_energy_from_probability(
    probability: np.ndarray,
    projection: Mapping[str, Any],
    temperature_K: float,
    probability_floor: float,
    landscape_f_max: float,
):
    occupied = probability > 0
    free_energy = np.full_like(probability, np.nan)
    free_energy[occupied] = -KB_KCAL_PER_MOL_K * float(temperature_K) * np.log(
        np.maximum(probability[occupied], float(probability_floor))
    )
    free_energy[occupied] -= np.min(free_energy[occupied])

    free_energy[~occupied] = landscape_f_max
    sigma = [
        float(x)
        for x in _expand(
            projection.get("sigma_bins", 0.0),
            probability.ndim,
            "sigma_bins",
            0.0,
        )
    ]
    if any(value > 0 for value in sigma):
        free_energy = ndimage.gaussian_filter(
            free_energy, sigma=sigma, mode="nearest"
        )
        free_energy -= np.min(free_energy)
    free_energy = np.minimum(free_energy, landscape_f_max)
    return free_energy, occupied


def compute_weighted_projection(
    dataframe: pd.DataFrame,
    projection: dict,
    *,
    weight_column: str,
    temperature_K: float,
    probability_floor: float,
    landscape_f_max: float = 10.0,
):
    """Compute a weighted 1D or 2D probability and free-energy projection."""
    cvs = projection.get("cvs", [])
    if isinstance(cvs, str):
        cvs = [cvs]
    if len(cvs) not in (1, 2):
        raise ValueError("Each FEL projection must define one or two CV columns.")
    if not np.isfinite(landscape_f_max) or landscape_f_max <= 0:
        raise ValueError("landscape_F_max must be finite and positive.")
    plot_f_max = projection.get("F_max")
    if plot_f_max is not None:
        plot_f_max = float(plot_f_max)
        if not np.isfinite(plot_f_max) or plot_f_max <= 0:
            raise ValueError("Projection F_max must be finite and positive.")
        if plot_f_max > landscape_f_max:
            raise ValueError(
                "Projection F_max cannot exceed FEL_estimate.landscape_F_max."
            )

    required = [weight_column, *cvs]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise KeyError(f"RiteWeight table is missing projection columns: {missing}")

    values = dataframe[cvs].to_numpy(dtype=np.float64)
    weights = dataframe[weight_column].to_numpy(dtype=np.float64)
    valid = np.isfinite(weights) & (weights > 0.0) & np.all(np.isfinite(values), axis=1)
    if not np.any(valid):
        raise ValueError("No rows have finite CV values and positive RiteWeight weights.")
    values = values[valid]
    weights = weights[valid]
    weights /= weights.sum()

    ndim = len(cvs)
    bins = [int(x) for x in _expand(projection.get("bins", 200), ndim, "bins", 200)]
    if any(value < 2 for value in bins):
        raise ValueError("Every projection dimension needs at least two bins.")
    ranges = _projection_ranges(values, projection.get("ranges"))
    periodicities = [bool(x) for x in _expand(
        projection.get("periodicities", False), ndim, "periodicities", False
    )]
    values = _wrap_periodic(values, ranges, periodicities)

    probability, edges = np.histogramdd(values, bins=bins, range=ranges, weights=weights)
    probability = probability.astype(np.float64)
    total = probability.sum()
    if total <= 0:
        raise ValueError("Weighted histogram is empty; check projection ranges.")
    probability /= total

    free_energy, occupied = _free_energy_from_probability(
        probability,
        projection,
        temperature_K,
        probability_floor,
        float(landscape_f_max),
    )

    centers = [0.5 * (edge[:-1] + edge[1:]) for edge in edges]
    return {
        "cvs": cvs,
        "probability": probability,
        "free_energy": free_energy,
        "edges": edges,
        "centers": centers,
        "occupied": occupied,
        "periodicities": periodicities,
        "n_samples": int(np.sum(valid)),
        "temperature_K": float(temperature_K),
        "landscape_f_max": float(landscape_f_max),
        "plot_f_max": plot_f_max,
    }


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return cleaned or "projection"


def _free_energy_for_plot(free_energy: np.ndarray, plot_f_max):
    """Mask energies above the display cap so they render as white."""
    values = np.asarray(free_energy)
    if plot_f_max is None:
        return values
    return np.ma.masked_greater(values, float(plot_f_max))


def save_projection(result: dict, output_dir: str, name: str):
    """Write a projection as text, compressed NumPy data, and PNG."""
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, _safe_name(name))
    cvs = result["cvs"]
    probability = result["probability"]
    free_energy = result["free_energy"]
    centers = result["centers"]
    plot_f_max = result.get("plot_f_max")
    plotted_free_energy = _free_energy_for_plot(free_energy, plot_f_max)

    if len(cvs) == 1:
        table = np.column_stack([centers[0], probability, free_energy])
        np.savetxt(
            stem + ".dat",
            table,
            header=f"{cvs[0]} probability free_energy_kcal_per_mol",
        )
        fig, axis = plt.subplots()
        axis.plot(centers[0], plotted_free_energy)
        axis.set_facecolor("white")
        axis.set_xlabel(cvs[0])
        axis.set_ylabel("Free energy (kcal/mol)")
        if plot_f_max is not None:
            axis.set_ylim(0.0, plot_f_max)
    else:
        grid_x, grid_y = np.meshgrid(centers[0], centers[1], indexing="ij")
        table = np.column_stack([
            grid_x.ravel(), grid_y.ravel(), probability.ravel(), free_energy.ravel()
        ])
        np.savetxt(
            stem + ".dat",
            table,
            header=f"{cvs[0]} {cvs[1]} probability free_energy_kcal_per_mol",
        )
        fig, axis = plt.subplots()
        levels = (
            np.linspace(0.0, plot_f_max, 21)
            if plot_f_max is not None
            else 20
        )
        colormap = plt.get_cmap("turbo").copy()
        colormap.set_bad("white")
        contour = axis.contourf(
            grid_x,
            grid_y,
            plotted_free_energy,
            levels=levels,
            cmap=colormap,
        )
        axis.set_facecolor("white")
        fig.colorbar(contour, ax=axis, label="Free energy (kcal/mol)")
        axis.set_xlabel(cvs[0])
        axis.set_ylabel(cvs[1])

    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=200)
    plt.close(fig)

    arrays = {
        "probability": probability,
        "free_energy": free_energy,
        "occupied": np.asarray(result.get("occupied", probability > 0), dtype=bool),
        "temperature_K": np.asarray(result.get("temperature_K", np.nan)),
        "landscape_F_max": np.asarray(result.get("landscape_f_max", np.nan)),
        "plot_F_max": np.asarray(
            np.nan if plot_f_max is None else plot_f_max
        ),
        **{f"edge_{i}": edge for i, edge in enumerate(result["edges"])},
    }
    np.savez_compressed(stem + ".npz", **arrays)
    return {"dat": stem + ".dat", "png": stem + ".png", "npz": stem + ".npz"}


def run_fel_estimate(config: dict):
    """Run all configured weighted projections from a RiteWeight result table."""
    dataframe = load_riteweight_table(config["input"])
    projections = config.get("projections", [])
    if not projections:
        raise ValueError("FEL_estimate.projections must contain at least one projection.")

    output_dir = config.get("output_dir", "./fel")
    weight_column = config.get("weight_column", "fel_weight")
    if weight_column not in dataframe.columns and weight_column == "fel_weight":
        if "weight" not in dataframe.columns:
            raise KeyError(
                "RiteWeight table has neither the requested 'fel_weight' column "
                "nor the legacy 'weight' column."
            )
        print(
            "[WARN] RiteWeight table predates symmetric FEL weights; "
            "falling back from 'fel_weight' to legacy 'weight'."
        )
        weight_column = "weight"
    temperature_K = float(config.get("temperature_K", 300.0))
    probability_floor = float(config.get("probability_floor", 1.0e-300))
    landscape_f_max = float(config.get("landscape_F_max", 10.0))
    if not np.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("FEL_estimate.temperature_K must be finite and positive.")
    if not np.isfinite(probability_floor) or not 0 < probability_floor < 1:
        raise ValueError("FEL_estimate.probability_floor must be between 0 and 1.")
    if not np.isfinite(landscape_f_max) or landscape_f_max <= 0:
        raise ValueError("FEL_estimate.landscape_F_max must be finite and positive.")

    outputs = []
    for index, projection in enumerate(projections):
        name = projection.get("name") or "_".join(projection.get("cvs", [])) or f"projection_{index}"
        result = compute_weighted_projection(
            dataframe,
            projection,
            weight_column=weight_column,
            temperature_K=temperature_K,
            probability_floor=probability_floor,
            landscape_f_max=landscape_f_max,
        )
        paths = save_projection(result, output_dir, name)
        outputs.append(paths)
        print(f"[OK] Saved weighted FEL projection '{name}' to {paths['dat']}")
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Gen-COMPAS YAML configuration")
    args = parser.parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_fel_estimate(config.get("FEL_estimate", config))


if __name__ == "__main__":
    main()
