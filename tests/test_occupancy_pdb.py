from pathlib import Path

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

from tools.occupancy import (
    hydrogenate_and_set_occupancy,
    write_pdb_with_custom_occupancy,
)


def test_tmd_pdb_keeps_all_atoms_and_normalizes_alad_name(tmp_path: Path):
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("ALAD", chain, resSeq=1)
    for index in range(10):
        topology.add_atom(f"C{index + 1}", md.element.carbon, residue)
    for index in range(12):
        topology.add_atom(f"H{index + 1}", md.element.hydrogen, residue)

    trajectory = md.Trajectory(
        np.zeros((1, 22, 3), dtype=np.float32), topology
    )
    occupancies = np.zeros((1, 22), dtype=np.float32)
    occupancies[:, topology.select("element != H")] = 1.0
    output = tmp_path / "tmd_target.pdb"

    write_pdb_with_custom_occupancy(trajectory, occupancies, str(output))

    atom_lines = [
        line for line in output.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    ]
    assert len(atom_lines) == 22
    assert all(line[17:20] == "ALA" for line in atom_lines)
    assert all(line[21] == "A" for line in atom_lines)
    assert [float(line[54:60]) for line in atom_lines] == [1.0] * 10 + [0.0] * 12


def test_nanma_occupancy_pipeline_reconstructs_full_22_atom_tmd_pdb(tmp_path: Path):
    project_root = Path(__file__).parents[1]
    top_dir = project_root / "examples" / "1.NANMA" / "NAMD_inputs" / "TOP_files"
    full = md.load(top_dir / "alad.pdb", top=top_dir / "alad.psf")
    heavy = full.atom_slice(full.topology.select("element != H"))

    target_dir = tmp_path / "heavy_targets"
    output_dir = tmp_path / "tmd_targets"
    target_dir.mkdir()
    heavy.save_pdb(target_dir / "target.pdb")

    hydrogenate_and_set_occupancy(
        str(target_dir),
        str(top_dir / "alad.psf"),
        str(top_dir / "alad.pdb"),
        str(output_dir),
        "element != H",
    )

    atom_lines = [
        line
        for line in (output_dir / "target.pdb").read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    ]
    assert len(atom_lines) == 22
    assert all(line[17:20] == "ALA" for line in atom_lines)
    assert [float(line[54:60]) for line in atom_lines].count(1.0) == 10
    assert [float(line[54:60]) for line in atom_lines].count(0.0) == 12
