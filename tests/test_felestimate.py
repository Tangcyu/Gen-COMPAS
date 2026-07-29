import numpy as np
import pandas as pd
import pytest

from tools.felestimate import (
    _free_energy_for_plot,
    compute_weighted_projection,
    save_projection,
)
from tools.frame_weights import segment_weights_to_frame_weights


def test_symmetric_fel_weights_include_trajectory_endpoints():
    # Two independent four-frame trajectories at lag one.
    starts = np.array([0, 1, 2, 4, 5, 6])
    ends = np.array([1, 2, 3, 5, 6, 7])
    segment_weights = np.full(6, 1.0 / 6.0)

    transition, fel = segment_weights_to_frame_weights(
        8, starts, ends, segment_weights
    )

    assert transition.sum() == pytest.approx(1.0)
    assert fel.sum() == pytest.approx(1.0)
    assert transition[3] == 0.0
    assert transition[7] == 0.0
    assert fel[3] > 0.0
    assert fel[7] > 0.0
    # No endpoint contribution can jump across the trajectory boundary.
    assert fel[3] == pytest.approx(segment_weights[2] / 2.0)
    assert fel[4] == pytest.approx(segment_weights[3] / 2.0)


def test_fel_projection_uses_one_global_weighted_histogram():
    dataframe = pd.DataFrame(
        {
            "CV": [0.5, 1.5],
            "fel_weight": [0.75, 0.25],
        }
    )
    result = compute_weighted_projection(
        dataframe,
        {
            "cvs": ["CV"],
            "bins": 2,
            "ranges": [[0.0, 2.0]],
            "sigma_bins": 0.0,
        },
        weight_column="fel_weight",
        temperature_K=300.0,
        probability_floor=1.0e-300,
    )

    assert result["probability"] == pytest.approx([0.75, 0.25])


def test_plot_f_max_does_not_clip_saved_landscape(tmp_path):
    dataframe = pd.DataFrame(
        {
            "CV": [0.25, 1.75],
            "fel_weight": [0.5, 0.5],
        }
    )
    result = compute_weighted_projection(
        dataframe,
        {
            "cvs": ["CV"],
            "bins": 3,
            "ranges": [[0.0, 2.0]],
            "sigma_bins": 0.0,
            "F_max": 5.0,
        },
        weight_column="fel_weight",
        temperature_K=300.0,
        probability_floor=1.0e-300,
        landscape_f_max=50.0,
    )

    # The empty center bin keeps the high landscape cap even though the plot
    # is configured to show only 0--5 kcal/mol.
    assert result["free_energy"][1] == pytest.approx(50.0)
    assert result["plot_f_max"] == pytest.approx(5.0)

    paths = save_projection(result, str(tmp_path), "separate_caps")
    saved_table = np.loadtxt(paths["dat"])
    with np.load(paths["npz"]) as saved:
        assert saved_table[:, 2].max() == pytest.approx(50.0)
        assert saved["free_energy"].max() == pytest.approx(50.0)
        assert saved["landscape_F_max"] == pytest.approx(50.0)
        assert saved["plot_F_max"] == pytest.approx(5.0)


def test_plot_f_max_cannot_exceed_landscape_cap():
    dataframe = pd.DataFrame({"CV": [0.5], "fel_weight": [1.0]})
    with pytest.raises(ValueError, match="cannot exceed"):
        compute_weighted_projection(
            dataframe,
            {
                "cvs": ["CV"],
                "bins": 2,
                "ranges": [[0.0, 1.0]],
                "F_max": 60.0,
            },
            weight_column="fel_weight",
            temperature_K=300.0,
            probability_floor=1.0e-300,
            landscape_f_max=50.0,
        )


def test_energies_above_plot_f_max_are_masked_white():
    plotted = _free_energy_for_plot(np.array([0.0, 5.0, 5.1, 10.0]), 5.0)

    assert np.ma.getmaskarray(plotted).tolist() == [False, False, True, True]
