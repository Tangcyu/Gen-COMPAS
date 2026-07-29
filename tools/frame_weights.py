"""Pure NumPy helpers for mapping RiteWeight segments onto trajectory frames."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def segment_weights_to_frame_weights(
    n_frames: int,
    seg_start_idx: np.ndarray,
    seg_end_idx: np.ndarray,
    w_segment: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build transition-origin and symmetric FEL weights for all frames.

    Transition weights retain the original RiteWeight convention: a segment's
    weight belongs to its origin frame. That convention is appropriate for
    lagged transition/VCN samples, but leaves the final ``lag`` frames of every
    trajectory at zero because those frames have no outgoing segment.

    FEL weights use the time-symmetric endpoint marginal instead. Half of each
    segment weight is assigned to its origin and half to its endpoint. This
    includes the equilibrated tail without introducing transitions between
    trajectories; ``seg_start_idx`` and ``seg_end_idx`` have already been built
    independently inside each trajectory.
    """
    starts = np.asarray(seg_start_idx, dtype=np.int64)
    ends = np.asarray(seg_end_idx, dtype=np.int64)
    weights = np.asarray(w_segment, dtype=np.float64)
    if starts.shape != ends.shape or starts.shape != weights.shape:
        raise ValueError("Segment starts, ends, and weights must have the same shape.")
    if n_frames < 1:
        raise ValueError("n_frames must be positive.")
    if starts.size == 0:
        raise ValueError("At least one lagged segment is required.")
    if (
        np.any(starts < 0)
        or np.any(ends < 0)
        or np.any(starts >= n_frames)
        or np.any(ends >= n_frames)
    ):
        raise ValueError("Segment frame indices fall outside the trajectory table.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Segment weights must be finite and non-negative.")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Segment weights must have a positive sum.")
    weights = weights / total

    transition = np.zeros(n_frames, dtype=np.float64)
    np.add.at(transition, starts, weights)
    transition /= transition.sum()

    fel = np.zeros(n_frames, dtype=np.float64)
    np.add.at(fel, starts, 0.5 * weights)
    np.add.at(fel, ends, 0.5 * weights)
    fel /= fel.sum()
    return transition, fel
