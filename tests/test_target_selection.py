import numpy as np
import pytest

from common.target_selection import committor_slice_bounds, select_slice_targets


def test_fixed_slice_selects_first_targets_without_clustering():
    assert committor_slice_bounds() == (0.4, 0.6)

    frame_indices = np.array([4, 8, 15, 16])
    selected = select_slice_targets(
        frame_indices,
        2,
        require_n_targets=True,
    )

    np.testing.assert_array_equal(selected, [4, 8])


def test_slice_target_shortfall_policy():
    frame_indices = np.array([4, 8])
    np.testing.assert_array_equal(
        select_slice_targets(frame_indices, 3, require_n_targets=False),
        frame_indices,
    )
    with pytest.raises(ValueError, match="3 structural targets were required"):
        select_slice_targets(frame_indices, 3, require_n_targets=True)


def test_target_count_must_be_positive():
    with pytest.raises(ValueError, match="must be at least 1"):
        select_slice_targets([4, 8], 0, require_n_targets=True)


def test_q_variance_controls_slice_half_width():
    assert committor_slice_bounds(0.2) == pytest.approx((0.3, 0.7))


@pytest.mark.parametrize("value", [-0.1, 0.6, float("inf"), "invalid", True])
def test_q_variance_is_validated(value):
    with pytest.raises(ValueError, match="VCN.q_variance"):
        committor_slice_bounds(value)
