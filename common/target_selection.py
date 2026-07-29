import numpy as np

from common.config import committor_slice_bounds


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
