import numpy as np
import pytest

pytest.importorskip("mdtraj")

from common.vcn_slice import (
    Q_SLICE_MAX,
    Q_SLICE_MIN,
    committor_projection_plotting_enabled,
    select_structural_representatives,
)


def test_fixed_slice_and_structural_representative_count():
    assert (Q_SLICE_MIN, Q_SLICE_MAX) == (0.4, 0.6)

    features = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [10.0, 10.0],
        [10.1, 10.0],
    ])
    frame_indices = np.array([4, 8, 15, 16])
    selected = select_structural_representatives(
        features,
        frame_indices,
        2,
        random_seed=42,
        require_n_targets=True,
    )

    assert len(selected) == 2
    assert set(selected).issubset(set(frame_indices))
    assert len(set(selected) & {4, 8}) == 1
    assert len(set(selected) & {15, 16}) == 1


def test_committor_projection_plotting_is_opt_in():
    assert committor_projection_plotting_enabled({}) is False
    assert committor_projection_plotting_enabled(
        {"plot_committor_projections": True}
    ) is True
    with pytest.raises(TypeError, match="must be true or false"):
        committor_projection_plotting_enabled(
            {"plot_committor_projections": "false"}
        )
