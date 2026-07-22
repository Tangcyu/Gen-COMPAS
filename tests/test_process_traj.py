import pandas as pd

from vcn.process_traj import preprocess_traj


def test_lagged_pairs_do_not_cross_trajectory_boundaries():
    data = pd.DataFrame(
        {
            "trajectory_id": [0, 0, 0, 1, 1, 1],
            "value": [0, 1, 2, 100, 101, 102],
            "weight": [1.0] * 6,
        }
    )
    lagged, _, _ = preprocess_traj(
        data,
        val_ratio=0.25,
        time_shift=1,
        trajectory_column="trajectory_id",
        random_seed=1,
    )
    assert len(lagged) == 4
    assert (lagged["trajectory_id_origin"] == lagged["trajectory_id_target"]).all()
    assert set(zip(lagged["value_origin"], lagged["value_target"])) == {
        (0, 1),
        (1, 2),
        (100, 101),
        (101, 102),
    }
