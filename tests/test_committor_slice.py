import pytest

pytest.importorskip("mdtraj")

from common.vcn_slice import (
    committor_projection_plotting_enabled,
)


def test_committor_projection_plotting_is_opt_in():
    assert committor_projection_plotting_enabled({}) is False
    assert committor_projection_plotting_enabled(
        {"plot_committor_projections": True}
    ) is True
    with pytest.raises(TypeError, match="must be true or false"):
        committor_projection_plotting_enabled(
            {"plot_committor_projections": "false"}
        )
