"""Calibration of reverse-diffusion noise for the autonoise_diffusion step.

The workflow handles stage scheduling and persistence of the selected noise.
Default scoring selects intermediate proposals for subsequent CA-targeted MD.
All-atom geometry is diagnostic by default; strict filtering is opt-in.
Intermediate labels are geometric proposals, not committor or kinetic estimates.
"""
from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Mapping

import mdtraj as md
import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.special import ndtr

from common.config import deep_merge
from configs import load_defaults
from common.diffusion_sample import setup_device, setup_model_and_diffusion
from utils.coordinate_contract import load_coordinate_contract, validate_topology
from utils.diffusion import Diffusion
from utils.mdtraj_io import filter_dcdplugin_messages, load_topology

logger = logging.getLogger(__name__)

DEFAULT_AUTONOISE = load_defaults("Generative.autonoise")


def autonoise_config(config: Mapping) -> dict:
    """Merge packaged defaults with Generative.autonoise without caller mutation."""
    if "AutoNoise" in config:
        raise ValueError("Move the top-level AutoNoise section to Generative.autonoise.")
    supplied = config.get("Generative", {}).get("autonoise", {})
    if not isinstance(supplied, Mapping):
        raise ValueError("Generative.autonoise must be a mapping.")
    cfg = deep_merge(DEFAULT_AUTONOISE, supplied)
    if cfg["geometry"]["mode"] not in ("diagnostic", "strict"):
        raise ValueError("Generative.autonoise.geometry.mode must be diagnostic or strict.")
    for key in ("enabled", "save_dcd"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"Generative.autonoise.{key} must be a boolean.")
    for section, keys in {
        "reference": ("frames_per_state", "max_feature_atoms"),
        "search": ("pilot_samples", "verify_top_k", "verify_samples", "max_samples",
                   "max_batch_size", "min_basin_samples", "recommendations"),
    }.items():
        for key in keys:
            value = cfg[section][key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Generative.autonoise.{section}.{key} must be a positive integer.")
    rounds = cfg["search"]["refine_rounds"]
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
        raise ValueError("refine_rounds must be a nonnegative integer.")
    if not isinstance(cfg["seed"], int) or isinstance(cfg["seed"], bool) or cfg["seed"] < 0:
        raise ValueError("Generative.autonoise.seed must be a nonnegative integer.")
    bounds = np.asarray(cfg["search"]["bounds"], dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 <= bounds[0] < bounds[1]:
        raise ValueError("Generative.autonoise.search.bounds must contain two increasing nonnegative values.")
    cfg["search"]["bounds"] = bounds.tolist()
    search, ref = cfg["search"], cfg["reference"]
    if ref["frames_per_state"] < 8 or ref["max_feature_atoms"] < 2:
        raise ValueError("Use at least 8 reference frames/state and 2 feature atoms.")
    if search["verify_samples"] < search["pilot_samples"]:
        raise ValueError("verify_samples must be at least pilot_samples.")
    if search["max_samples"] < 3 * search["pilot_samples"]:
        raise ValueError("max_samples must cover the three initial pilot candidates.")
    for section, key, lo, hi in [
        ("search", "min_valid_fraction", 0, 1),
        ("region", "basin_quantile", 0.5, 1),
        ("terminal_check", "train_fraction", 0, 1),
        ("geometry", "reference_min_valid_fraction", 0, 1),
    ]:
        value = float(cfg[section][key])
        if not np.isfinite(value) or not lo < value < hi:
            raise ValueError(f"Generative.autonoise.{section}.{key} must be between {lo} and {hi}.")
    for section, keys in {
        "geometry": ("bond_relative_tolerance", "max_bond_relative_tolerance", "angle_cosine_tolerance", "max_angle_cosine_tolerance", "reference_slack", "clash_distance_nm"),
        "region": ("corridor_factor", "corridor_floor_nm", "progress_margin", "cluster_rms_nm"),
    }.items():
        for key in keys:
            value = float(cfg[section][key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Generative.autonoise.{section}.{key} must be positive and finite.")
    if cfg["geometry"]["reference_slack"] < 1 or cfg["region"]["corridor_factor"] < 1:
        raise ValueError("reference_slack and corridor_factor must be at least 1.")
    if not np.isfinite(cfg["region"]["basin_margin"]) or cfg["region"]["basin_margin"] < 0:
        raise ValueError("Generative.autonoise.region.basin_margin must be finite and nonnegative.")
    bond_range = np.asarray(cfg["geometry"]["bond_median_range_nm"], dtype=float)
    if bond_range.shape != (2,) or not np.isfinite(bond_range).all() or not 0 < bond_range[0] < bond_range[1]:
        raise ValueError("geometry.bond_median_range_nm needs two increasing positive lengths.")
    for key in ("bond_relative_tolerance", "angle_cosine_tolerance"):
        if cfg["geometry"][key] > cfg["geometry"]["max_" + key]:
            raise ValueError(f"geometry.{key} exceeds its hard maximum.")
    return cfg


def sample_reference(path: str, topology, count: int):
    """Read bounded, evenly spaced DCD frames without loading the full trajectory.

    Low-level DCD coordinates are Angstrom; MDTraj trajectories use nanometers.
    Full-complex atom order must match the model topology.
    """
    with filter_dcdplugin_messages(), md.formats.DCDTrajectoryFile(str(path)) as handle:
        n_frames = len(handle)
        if n_frames < 8:
            raise ValueError(f"Reference trajectory needs at least 8 frames: {path}")
        indices = np.unique(np.linspace(0, n_frames - 1, min(count, n_frames), dtype=int))
        frames = []
        for index in indices:
            handle.seek(int(index))
            xyz, _, _ = handle.read(n_frames=1)
            if xyz.shape != (1, topology.n_atoms, 3):
                raise ValueError(f"Reference atom count does not match topology: {path}")
            frames.append(xyz[0] * 0.1)
    coordinates = np.asarray(frames, dtype=np.float32)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Reference contains nonfinite coordinates: {path}")
    return md.Trajectory(coordinates, topology), indices.tolist()


def terminal_separability(state_a, state_b, alpha_bar: float, train_fraction: float) -> dict:
    """Held-out linear discrimination after the *training* terminal corruption.

    Inputs are aligned, normalized, per-frame centered coordinates. A direction
    is fitted only on the earlier reference block. Gaussian projection permits
    exact expectation over forward noise without generating full noisy arrays.
    This is a diagnostic lower bound, not proof of distribution matching.
    """
    if not 0 <= alpha_bar <= 1 or not 0 < train_fraction < 1:
        raise ValueError("Invalid alpha_bar or train_fraction.")
    a, b = (np.asarray(x, dtype=np.float64).reshape(len(x), -1) for x in (state_a, state_b))
    if min(len(a), len(b)) < 4 or a.shape[1] != b.shape[1]:
        raise ValueError("Terminal check needs matching coordinates and at least four frames/state.")
    na, nb = (max(2, min(len(x) - 2, int(len(x) * train_fraction))) for x in (a, b))
    ma, mb = a[:na].mean(0), b[:nb].mean(0)
    direction = mb - ma
    separation = float(np.linalg.norm(direction))
    if separation < 1e-12:
        return {"status": "indistinct_reference_means", "alpha_bar_terminal": alpha_bar}
    direction /= separation
    threshold = float((ma + mb) @ direction / 2)
    pa, pb = a[na:] @ direction, b[nb:] @ direction
    clean_accuracy = float(((pa < threshold).mean() + (pb >= threshold).mean()) / 2)
    if alpha_bar == 1:
        terminal_accuracy = clean_accuracy
    else:
        ratio = np.sqrt(alpha_bar / (1 - alpha_bar))
        terminal_accuracy = float((ndtr(ratio * (threshold - pa)).mean()
                                   + ndtr(ratio * (pb - threshold)).mean()) / 2)
    return {
        "status": "ok", "alpha_bar_terminal": float(alpha_bar),
        "signal_coefficient": float(np.sqrt(alpha_bar)),
        "noise_coefficient": float(np.sqrt(1 - alpha_bar)),
        "clean_balanced_accuracy": clean_accuracy,
        "terminal_expected_balanced_accuracy": terminal_accuracy,
        "fit_frames": [na, nb], "held_out_frames": [len(pa), len(pb)],
        "interpretation": "Above-chance discrimination suggests retained state information; chance accuracy does not prove a Gaussian terminal distribution. Reference frames can be temporally correlated.",
    }


def _noise_bank(seed, sample_ids, timesteps, num_atoms, device):
    # A stream belongs to the sample ID, not its noise scale or chunk. Coupling
    # candidates reduces comparison variance and makes OOM retries reproducible.
    arrays = [np.random.default_rng(np.random.SeedSequence([seed, int(i)])).standard_normal(
        (timesteps + 1, num_atoms, 3), dtype=np.float32) for i in sample_ids]
    return torch.from_numpy(np.stack(arrays, axis=1)).to(device)


@torch.inference_mode()
def generate_mixed_noise(model, diffusion, requests, num_atoms, device, max_batch_size=32, seed=42):
    """Generate (reverse_noise_scale, sample_id) requests in mixed batches.

    Exactly the existing DDPM posterior coefficients, all T steps, and initial
    N(0,I) are retained. Only the scalar reverse multiplier becomes per-sample.
    Private, paired random streams do not alter global NumPy/PyTorch RNG state.
    """
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive.")
    if not requests:
        return np.empty((0, num_atoms, 3), dtype=np.float32)
    if any(not np.isfinite(s) or s < 0 or i < 0 or int(i) != i for s, i in requests):
        raise ValueError("Requests need finite nonnegative scales and nonnegative integer IDs.")
    model.eval()
    device = torch.device(device)
    coefficients = [x.to(device) for x in (diffusion.posterior_mean_coef1,
                    diffusion.posterior_mean_coef2, diffusion.posterior_log_variance_clipped)]

    def chunk_sample(chunk):
        ids, inverse = np.unique([i for _, i in chunk], return_inverse=True)
        bank = _noise_bank(seed, ids, diffusion.timesteps, num_atoms, device)
        index = torch.as_tensor(inverse, device=device)
        scale = torch.tensor([s for s, _ in chunk], device=device, dtype=torch.float32).view(-1, 1, 1)
        x = bank[0, index]
        t = torch.empty(len(chunk), device=device, dtype=torch.long)
        for step in reversed(range(diffusion.timesteps)):
            t.fill_(step)
            predicted = model(x, t)
            x = coefficients[0][step] * predicted + coefficients[1][step] * x
            if step:
                x = x + torch.exp(0.5 * coefficients[2][step]) * scale * bank[step + 1, index]
        return x.float().cpu().numpy()

    outputs, offset, batch_size = [], 0, max_batch_size
    while offset < len(requests):
        chunk = requests[offset:offset + batch_size]
        try:
            result = chunk_sample(chunk)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            logger.warning("CUDA OOM: retrying the same requests with batch size %d", batch_size)
            continue
        outputs.append(result)
        offset += len(chunk)
        logger.info("Denoised %d/%d requests (mixed-noise batch %d)", offset, len(requests), len(chunk))
    return np.concatenate(outputs)


def _check_reference_bonds(topology, reference, cfg):
    bonds = np.asarray([(x.index, y.index) for x, y in topology.bonds], dtype=int).reshape(-1, 2)
    if not len(bonds):
        raise ValueError("Geometry validation requires a topology with covalent bonds.")
    lengths = np.linalg.norm(reference[:, bonds[:, 0]] - reference[:, bonds[:, 1]], axis=-1)
    center = np.median(lengths, axis=0)
    lower, upper = cfg["geometry"]["bond_median_range_nm"]
    bad = np.flatnonzero((center < lower) | (center > upper))
    if len(bad):
        index = bad[0]
        left, right = (topology.atom(int(i)) for i in bonds[index])
        raise ValueError(
            f"Reference covalent geometry is incompatible with the topology: {left}--{right} "
            f"median length {center[index]:.4f} nm outside [{lower}, {upper}]. "
            "Check trajectory atom order/selection and PBC wrapping; do not calibrate from broken references."
        )
    return center


class StructureEvaluator:
    """A--B feature corridor with optional all-atom geometry filtering."""

    def __init__(self, topology, a, b, cfg):
        self.cfg = cfg
        selected = topology.select(cfg["reference"]["atomselect"])
        cap = cfg["reference"]["max_feature_atoms"]
        if len(selected) < 2:
            raise ValueError("AutoNoise reference.atomselect must select at least two atoms.")
        if len(selected) > cap:
            # Preserve at least one selected atom per chain, including small chains.
            first = list(dict.fromkeys(next(i for i in selected if topology.atom(int(i)).residue.chain.index == c)
                                       for c in sorted({topology.atom(int(i)).residue.chain.index for i in selected})))
            if len(first) > cap:
                raise ValueError("max_feature_atoms cannot represent every selected chain.")
            for i in selected[np.linspace(0, len(selected) - 1, cap, dtype=int)]:
                if i not in first and len(first) < cap:
                    first.append(int(i))
            selected = np.asarray(sorted(first), dtype=int)
        self.feature_atoms = selected
        i, j = np.triu_indices(len(selected), 1)
        self.pairs = np.column_stack((selected[i], selected[j]))
        self.bonds = np.asarray([(x.index, y.index) for x, y in topology.bonds], dtype=int).reshape(-1, 2)
        if not len(self.bonds):
            raise ValueError("Geometry validation requires a topology with covalent bonds.")
        neighbors = [set() for _ in topology.atoms]
        for i, j in self.bonds:
            neighbors[i].add(j)
            neighbors[j].add(i)
        self.excluded = {tuple(sorted((int(i), int(j)))) for i, j in self.bonds}
        angles = []
        for center, linked in enumerate(neighbors):
            linked = sorted(linked)
            for k, left in enumerate(linked):
                for right in linked[k + 1:]:
                    self.excluded.add((left, right))
                    angles.append((left, center, right))
        self.angles = np.asarray(angles, dtype=int).reshape(-1, 3)
        self.heavy = np.asarray([x.index for x in topology.atoms if x.element != md.element.hydrogen])
        chiral = []
        for residue in topology.residues:
            atoms = {x.name: x.index for x in residue.atoms}
            if all(name in atoms for name in ("CA", "N", "C", "CB")):
                chiral.append([atoms[name] for name in ("CA", "N", "C", "CB")])
        self.chiral = np.asarray(chiral, dtype=int).reshape(-1, 4)
        reference = np.concatenate((a, b))
        self.bond_center = _check_reference_bonds(topology, reference, cfg)
        geo = cfg["geometry"]
        angles = self.angle_cosines(reference)
        self.angle_center = np.median(angles, axis=0)
        volumes = self.volumes(reference)
        self.chiral_sign = np.sign(np.median(volumes, axis=0))
        if volumes.size and np.any(volumes * self.chiral_sign <= 1e-8):
            raise ValueError("Reference has inconsistent or degenerate CA chirality.")
        bond_error, angle_error = self.geometry_errors(reference)
        self.bond_limit = min(geo["max_bond_relative_tolerance"], max(geo["bond_relative_tolerance"], float(np.quantile(bond_error, .99)) * geo["reference_slack"]))
        self.angle_limit = min(geo["max_angle_cosine_tolerance"], max(geo["angle_cosine_tolerance"], float(np.quantile(angle_error, .99)) * geo["reference_slack"]))
        self.clash_limit = int(np.ceil(np.quantile(self.clashes(reference), .99)))
        reference_valid = (bond_error <= self.bond_limit) & (angle_error <= self.angle_limit)
        for name, mask in (("A", reference_valid[:len(a)]), ("B", reference_valid[len(a):])):
            if mask.mean() < geo["reference_min_valid_fraction"]:
                raise ValueError(f"Reference state {name} fails geometry validation for {1 - mask.mean():.1%} of frames; check topology/PBC/preparation.")
        fa, fb = self.features(a), self.features(b)
        self.origin = fa.mean(0)
        self.delta = fb.mean(0) - self.origin
        self.delta2 = float(self.delta @ self.delta)
        if self.delta2 < 1e-12:
            raise ValueError("A/B references cannot be separated by the selected distance features.")
        ua, ra = self.project(fa)
        ub, rb = self.project(fb)
        region = cfg["region"]
        q = region["basin_quantile"]
        # Padding prevents tiny reference-tail fluctuations from being rewarded
        # as novel intermediates just beyond an estimated basin quantile.
        self.a_upper = float(np.quantile(ua, q) + region["basin_margin"])
        self.b_lower = float(np.quantile(ub, 1 - q) - region["basin_margin"])
        if self.a_upper >= self.b_lower:
            raise ValueError("A/B reference regions overlap; supply better reference basins or features.")
        self.progress_lower = float(min(ua.min(), ub.min()) - region["progress_margin"])
        self.progress_upper = float(max(ua.max(), ub.max()) + region["progress_margin"])
        self.corridor_limit = max(region["corridor_floor_nm"], float(np.quantile(np.r_[ra, rb], .99)) * region["corridor_factor"])

    @staticmethod
    def distances(x, pairs):
        return np.linalg.norm(x[:, pairs[:, 0]] - x[:, pairs[:, 1]], axis=-1)

    def features(self, x):
        return self.distances(x, self.pairs)

    def angle_cosines(self, x):
        a = x[:, self.angles[:, 0]] - x[:, self.angles[:, 1]]
        b = x[:, self.angles[:, 2]] - x[:, self.angles[:, 1]]
        return np.sum(a * b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)

    def volumes(self, x):
        center = x[:, self.chiral[:, 0]]
        return np.sum(np.cross(x[:, self.chiral[:, 1]] - center, x[:, self.chiral[:, 2]] - center)
                      * (x[:, self.chiral[:, 3]] - center), axis=-1)

    def geometry_errors(self, x):
        bond = np.max(np.abs(self.distances(x, self.bonds) / self.bond_center - 1), axis=1)
        angle = np.max(np.abs(self.angle_cosines(x) - self.angle_center), axis=1) if len(self.angles) else np.zeros(len(x))
        return bond, angle

    def clashes(self, x):
        counts = []
        for frame in x:
            if not np.isfinite(frame).all():
                counts.append(len(self.heavy) ** 2)
                continue
            pairs = cKDTree(frame[self.heavy]).query_pairs(self.cfg["geometry"]["clash_distance_nm"])
            counts.append(sum(tuple(sorted((int(self.heavy[i]), int(self.heavy[j])))) not in self.excluded for i, j in pairs))
        return np.asarray(counts)

    def project(self, features):
        delta = features - self.origin
        progress = delta @ self.delta / self.delta2
        residual = np.sqrt(np.mean((delta - progress[:, None] * self.delta) ** 2, axis=1))
        return progress, residual

    def evaluate(self, x):
        bond, angle = self.geometry_errors(x)
        clash = self.clashes(x)
        failures = dict(nonfinite_failed=~np.isfinite(x).all(axis=(1, 2)),
                        bond_failed=~(bond <= self.bond_limit), angle_failed=~(angle <= self.angle_limit),
                        clash_failed=clash > self.clash_limit, chirality_failed=np.zeros(len(x), dtype=bool))
        if len(self.chiral):
            failures["chirality_failed"] = ~(self.volumes(x) * self.chiral_sign > 1e-8).all(axis=1)
        geometry_valid = ~np.logical_or.reduce(list(failures.values()))
        eligible = (~failures["nonfinite_failed"] if self.cfg["geometry"]["mode"] == "diagnostic"
                    else geometry_valid)
        features = self.features(x)
        progress, residual = self.project(features)
        inside = (eligible & (residual <= self.corridor_limit) & (progress >= self.progress_lower)
                  & (progress <= self.progress_upper))
        labels = np.full(len(x), "invalid", dtype="U12")
        labels[eligible] = "outlier"
        labels[inside & (progress <= self.a_upper)] = "A"
        labels[inside & (progress >= self.b_lower)] = "B"
        labels[inside & (progress > self.a_upper) & (progress < self.b_lower)] = "intermediate"
        # "valid" means usable for this calibration, not force-field readiness.
        # Off-corridor outliers must not inflate the usable fraction.
        return dict(valid=inside, geometry_valid=geometry_valid, labels=labels, progress=progress, residual_nm=residual,
                    features=features, bond_error=bond, angle_error=angle, clashes=clash, **failures)

    def describe(self):
        return dict(geometry_mode=self.cfg["geometry"]["mode"],
                    valid_definition="Finite coordinates inside the A/B feature corridor; strict mode also requires all-atom geometry.",
                    feature_atoms=self.feature_atoms.tolist(), feature_pair_count=len(self.pairs),
                    bond_relative_limit=self.bond_limit, angle_cosine_limit=self.angle_limit,
                    clash_count_limit=self.clash_limit, chirality_checks=len(self.chiral),
                    a_progress_upper=self.a_upper, b_progress_lower=self.b_lower,
                    progress_range=[self.progress_lower, self.progress_upper],
                    corridor_rms_nm=self.corridor_limit)


def _wilson(successes, n):
    if not n:
        return 0.0, 1.0
    p, z = successes / n, 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(center - half), float(center + half)


def summarize_candidates(records, cfg):
    """Cluster all valid intermediates together, using a fixed physical scale."""
    mid = [(scale, i, feature) for scale, record in sorted(records.items())
           for i, (label, feature) in enumerate(zip(record["labels"], record["features"])) if label == "intermediate"]
    clusters = {scale: set() for scale in records}
    if mid:
        features = np.asarray([x[2] for x in mid])
        labels = (fcluster(linkage(pdist(features) / np.sqrt(features.shape[1]), method="complete"),
                           cfg["region"]["cluster_rms_nm"], criterion="distance") if len(mid) > 1 else [1])
        for (scale, _, _), label in zip(mid, labels):
            clusters[scale].add(int(label))
    rows = []
    for scale, record in sorted(records.items()):
        labels, n = record["labels"], len(record["labels"])
        a, b = int(np.sum(labels == "A")), int(np.sum(labels == "B"))
        valid = int(record["valid"].sum())
        intermediate = int(np.sum(labels == "intermediate"))
        ci = _wilson(b, a + b)
        feasible = valid / n >= cfg["search"]["min_valid_fraction"]
        bias = 0
        if feasible and a + b >= cfg["search"]["min_basin_samples"]:
            bias = -1 if ci[1] < .5 else (1 if ci[0] > .5 else 0)
        rows.append(dict(noise_scale=float(scale), samples=n, valid_fraction=valid / n,
                         valid_ci_low=_wilson(valid, n)[0], a_count=a, b_count=b,
                         geometry_valid_fraction=float(np.mean(record["geometry_valid"])) if "geometry_valid" in record else None,
                         intermediate_count=intermediate, intermediate_fraction=intermediate / n,
                         outlier_count=int(np.sum(labels == "outlier")),
                         invalid_count=int(np.sum(labels == "invalid")),
                         b_fraction=b / (a + b) if a + b else None,
                         b_fraction_ci_low=ci[0], b_fraction_ci_high=ci[1], bias=bias,
                         feasible=feasible, intermediate_clusters=len(clusters[scale]),
                         diversity_score=len(clusters[scale]) / n,
                         score=intermediate / n if feasible else 0.0))
        for key in ("nonfinite", "bond", "angle", "clash", "chirality"):
            rows[-1][key + "_failure_count"] = int(np.sum(record.get(key + "_failed", [])))
    return rows, clusters


def find_brackets(rows):
    biased = [row for row in sorted(rows, key=lambda x: x["noise_scale"]) if row["bias"]]
    return [(a["noise_scale"], b["noise_scale"]) for a, b in zip(biased, biased[1:]) if a["bias"] != b["bias"]]


def next_noise(rows, bounds):
    """Refine the widest gap in a verified bracket, or explore the input interval.

    Multiple opposite-bias brackets are retained; no global monotonicity assumed.
    """
    brackets = find_brackets(rows)
    scales = sorted(row["noise_scale"] for row in rows)
    gaps = [(b - a, a, b) for a, b in zip(scales, scales[1:])
            if not brackets or any(lo <= a < b <= hi for lo, hi in brackets)]
    if not gaps:
        return None
    _, lo, hi = max(gaps)
    value = (lo + hi) / 2
    return value if bounds[0] <= value <= bounds[1] and lo < value < hi else None


def _rank(row):
    balance = abs(row["b_fraction"] - .5) if row["b_fraction"] is not None else 1.0
    return (row["feasible"], row["score"], row["diversity_score"], -balance, row["valid_fraction"])


def choose_recommendations(rows, clusters, cfg):
    """Require confirmed samples, feasibility, a bracket, and intermediate yield."""
    if not any(row["feasible"] for row in rows):
        return "no_valid_noise", []
    brackets = find_brackets(rows)
    if not brackets:
        return "unbracketed", []
    candidates = [row for row in rows if row["feasible"] and row["intermediate_clusters"]
                  and row["samples"] >= cfg["search"]["verify_samples"]
                  and any(lo <= row["noise_scale"] <= hi for lo, hi in brackets)]
    if not candidates:
        return "no_verified_intermediate", []
    chosen, covered = [], set()
    while candidates and len(chosen) < cfg["search"]["recommendations"]:
        # First maximize intermediate yield; additional scales add new coverage.
        best = (max(candidates, key=_rank) if not chosen else
                max(candidates, key=lambda row: (len(clusters[row["noise_scale"]] - covered) / row["samples"], _rank(row))))
        novel = clusters[best["noise_scale"]] - covered
        if not novel:
            break
        chosen.append(best["noise_scale"])
        covered.update(novel)
        candidates.remove(best)
    return "ok", chosen


def _distribution_figure(rows, recommendations, status):
    """One stacked bar per noise; fractions include excluded samples in the total."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    rows = sorted(rows, key=lambda row: row["noise_scale"])
    figure = Figure(figsize=(max(7, len(rows) * 1.1), 4.8))
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    x = np.arange(len(rows))
    totals = np.asarray([row["samples"] for row in rows])
    counts = np.asarray([[row["a_count"], row["intermediate_count"], row["b_count"]] for row in rows])
    excluded = totals - counts.sum(axis=1)
    counts = np.column_stack((counts, excluded))
    bottom = np.zeros(len(rows))
    for column, (label, color) in enumerate(zip(
            ("A", "I (intermediate)", "B", "Excluded"), ("#3969AC", "#E6A23C", "#20A387", "#BDC3C7"))):
        if column == 3 and not excluded.any():
            continue
        fraction = counts[:, column] / totals
        ax.bar(x, fraction, bottom=bottom, width=.66, label=label, color=color)
        bottom += fraction
    for i, row in enumerate(rows):
        ax.text(i, 1.025, f"n={row['samples']}\nI={row['intermediate_count']}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['noise_scale']:g}" + (" *" if row["noise_scale"] in recommendations else "")
                        for row in rows])
    ax.set_xlabel("Reverse noise scale (* recommended)")
    ax.set_ylabel("Fraction of all generated samples")
    ax.set_ylim(0, 1.17)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    selected = ", ".join(f"{value:g}" for value in recommendations)
    subtitle = f"Recommended: {selected}" if recommendations else f"No recommendation: {status.replace('_', ' ')}"
    ax.set_title(f"A / I / B distribution for TMD proposals\n{subtitle}", pad=14)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.18), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def _save_outputs(out, topology, coordinates, rows, recommendations, status, save_dcd):
    """Write only the distribution PNG and, optionally, one full-atom DCD per noise."""
    figure = _distribution_figure(rows, recommendations, status)
    plot_path = out / "noise_distribution.png"
    figure.savefig(str(plot_path), dpi=180)
    figure.clear()
    paths = {}
    if save_dcd:
        for scale, xyz in sorted(coordinates.items()):
            finite = np.isfinite(xyz).all(axis=(1, 2))
            if not finite.all():
                logger.warning("Noise %g: omitting %d nonfinite samples from DCD", scale, int((~finite).sum()))
            if finite.any():
                path = out / f"noise_{float(scale)}.dcd"
                md.Trajectory(xyz[finite], topology).save_dcd(str(path))
                paths[str(float(scale))] = str(path.resolve())
    return {"distribution_plot": str(plot_path.resolve()), "dcd_files": paths}


def run_autonoise(config: Mapping, *, dry_run=False, diagnostic_only=False) -> dict:
    """Calibrate a resolved Gen-COMPAS config without mutating caller settings."""
    cfg = autonoise_config(config)
    if not cfg["enabled"]:
        return {"status": "disabled", "message": "Set Generative.autonoise.enabled: true to run calibration."}
    generative = deepcopy(config["Generative"])
    checkpoint = Path(generative["inference"]["checkpoint"] or "")
    topology_path = Path(generative["data"].get("topology_path") or generative["data"].get("psf_path") or "")
    contract_path = checkpoint.parent / generative.get("coordinate_contract", {}).get("filename", "coordinate_contract.pt")
    files = {"checkpoint": checkpoint, "topology": topology_path, "coordinate_contract": contract_path}
    for state in ("state_a", "state_b"):
        files[state] = Path(cfg["reference"][state] or "")
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"AutoNoise {name} file not found: {path}")
    report = {"status": "dry_run", "paths": {k: str(v.resolve()) for k, v in files.items()},
              "output_dir": str(Path(cfg["output_dir"]).resolve())}
    if dry_run:
        return report
    out = Path(cfg["output_dir"])
    if not diagnostic_only and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"AutoNoise output directory is not empty; choose a new output_dir: {out}")
    topology = load_topology(str(topology_path))
    contract = load_coordinate_contract(str(contract_path))
    validate_topology(topology, contract["topology"], context="AutoNoise model topology")
    mean = np.asarray(contract["coord_mean"], dtype=np.float64).reshape(1, 1, 3)
    std = np.asarray(contract["coord_std"], dtype=np.float64).reshape(1, 1, 3)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid coordinate-contract normalization.")
    reference = md.Trajectory(np.asarray(contract["reference_xyz"])[None], topology)
    alignment = np.asarray(contract["alignment_atom_indices"], dtype=int)
    if len(alignment) < 3 or np.any(alignment < 0) or np.any(alignment >= topology.n_atoms):
        raise ValueError("Invalid coordinate-contract alignment atom indices.")
    states, normalized, frame_indices = [], [], {}
    for state in ("state_a", "state_b"):
        traj, indices = sample_reference(str(files[state]), topology, cfg["reference"]["frames_per_state"])
        traj.superpose(reference, atom_indices=alignment, ref_atom_indices=alignment)
        states.append(traj.xyz)
        coords = (traj.xyz - mean) / std
        normalized.append(coords - coords.mean(axis=1, keepdims=True))
        frame_indices[state] = indices
    report["reference_frame_indices"] = frame_indices
    # Also guard diagnostic-only calls: a DCD's atom count cannot establish
    # that its coordinates have the same atom order as the supplied topology.
    _check_reference_bonds(topology, np.concatenate(states), cfg)
    if cfg["terminal_check"]["enabled"] or diagnostic_only:
        diffusion = Diffusion(**generative["diffusion"], device="cpu")
        report["terminal_check"] = terminal_separability(*normalized, float(diffusion.alphas_cumprod[-1]), cfg["terminal_check"]["train_fraction"])
    if diagnostic_only:
        report["status"] = "diagnostic_only"
        return report
    evaluator = StructureEvaluator(topology, *states, cfg)
    report["evaluator"] = evaluator.describe()
    out.mkdir(parents=True, exist_ok=True)
    device = setup_device(generative.get("device"))
    model, diffusion, _, _, _ = setup_model_and_diffusion(generative, device)
    records, coordinates = {}, {}
    search = cfg["search"]

    def acquire(targets, stage):
        remaining = search["max_samples"] - sum(len(x["labels"]) for x in records.values())
        jobs = [(float(scale), i) for scale, total in targets.items()
                for i in range(len(records[scale]["labels"]) if scale in records else 0, total)]
        # Interleave candidates to keep small pilot batches full.
        jobs.sort(key=lambda x: (x[1], x[0]))
        jobs = jobs[:remaining]
        if not jobs:
            return
        logger.info("AutoNoise %s: %d requests at %s", stage, len(jobs), sorted({x[0] for x in jobs}))
        generated = generate_mixed_noise(model, diffusion, jobs, topology.n_atoms, device,
                                         search["max_batch_size"], cfg["seed"])
        physical = (generated * std + mean).astype(np.float32)
        for scale in sorted({x[0] for x in jobs}):
            mask = np.asarray([s == scale for s, _ in jobs])
            chunk = physical[mask]
            result = evaluator.evaluate(chunk)
            if scale in records:
                records[scale] = {k: np.concatenate((records[scale][k], v)) for k, v in result.items()}
                coordinates[scale] = np.concatenate((coordinates[scale], chunk))
            else:
                records[scale], coordinates[scale] = result, chunk

    lo, hi = search["bounds"]
    acquire({s: search["pilot_samples"] for s in (lo, (lo + hi) / 2, hi)}, "pilot")
    for iteration in range(search["refine_rounds"]):
        rows, _ = summarize_candidates(records, cfg)
        candidate = next_noise(rows, search["bounds"])
        if candidate is None or sum(row["samples"] for row in rows) >= search["max_samples"]:
            break
        acquire({candidate: search["pilot_samples"]}, f"refine_{iteration + 1}")
    rows, _ = summarize_candidates(records, cfg)
    finalists = sorted(rows, key=_rank, reverse=True)[:search["verify_top_k"]]
    acquire({row["noise_scale"]: search["verify_samples"] for row in finalists}, "verify")
    rows, clusters = summarize_candidates(records, cfg)
    status, recommendations = choose_recommendations(rows, clusters, cfg)
    report.update(status=status, scores=rows, brackets=find_brackets(rows),
                  recommended_noise_scales=recommendations,
                  generated_samples=sum(row["samples"] for row in rows),
                  interpretation="Geometric intermediates only. This does not establish transition paths, committors, equilibrium weights, or kinetics.")
    report.update(_save_outputs(out, topology, coordinates, rows, recommendations, status, cfg["save_dcd"]))
    return report
