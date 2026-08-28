from pathlib import Path

import pytest

from utils.mdtraj_io import load_topology as quiet_md_load_topology


PROJECT_ROOT = Path(__file__).parents[1]
SYSTEMS = (
    ("1.NANMA", "alad.psf", 22, 10),
    ("2.Trialanine", "triala.psf", 42, 20),
)


@pytest.mark.parametrize("example,topology_name,total,heavy", SYSTEMS)
def test_mdtraj_heavy_atom_selection(example, topology_name, total, heavy):
    md = pytest.importorskip("mdtraj")
    topology_path = (
        PROJECT_ROOT / "examples" / example / "NAMD_inputs" / "TOP_files"
        / topology_name
    )
    topology = quiet_md_load_topology(topology_path)

    assert topology.n_atoms == total
    assert len(topology.select("element != H")) == heavy


@pytest.mark.parametrize("example,topology_name,total,heavy", SYSTEMS)
def test_mdanalysis_heavy_atom_selection(example, topology_name, total, heavy):
    mda = pytest.importorskip("MDAnalysis")
    topology_path = (
        PROJECT_ROOT / "examples" / example / "NAMD_inputs" / "TOP_files"
        / topology_name
    )
    universe = mda.Universe(topology_path)

    assert universe.atoms.n_atoms == total
    assert universe.select_atoms("not name H*").n_atoms == heavy
