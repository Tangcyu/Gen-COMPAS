import numpy as np


Q_CENTER = 0.5
DEFAULT_Q_VARIANCE = 0.1


def committor_slice_bounds(q_variance=DEFAULT_Q_VARIANCE):
    """Return the q=0.5 slice bounds for a validated half-width."""
    if isinstance(q_variance, bool):
        raise ValueError("VCN.q_variance must be a number between 0 and 0.5.")
    try:
        q_variance = float(q_variance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "VCN.q_variance must be a number between 0 and 0.5."
        ) from exc
    if not np.isfinite(q_variance) or not 0 <= q_variance <= 0.5:
        raise ValueError("VCN.q_variance must be a number between 0 and 0.5.")
    return Q_CENTER - q_variance, Q_CENTER + q_variance


def select_slice_targets(
    candidate_indices,
    n_targets,
    *,
    require_n_targets,
    q_bounds=None,
):
    """Return the first requested targets from the configured committor slice."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    q_min, q_max = q_bounds or committor_slice_bounds()
    if n_targets < 1:
        raise ValueError("VCN.n_targets must be at least 1.")
    if len(candidate_indices) < n_targets:
        if require_n_targets:
            raise ValueError(
                f"Only {len(candidate_indices)} frames lie in the configured "
                f"{q_min:g}--{q_max:g} committor range; "
                f"{n_targets} structural targets were required. Generate more samples."
            )
        return candidate_indices
    return candidate_indices[:n_targets]
