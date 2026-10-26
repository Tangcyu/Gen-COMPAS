import os

import pytest

from utils.mdtraj_io import filter_dcdplugin_messages, save_psf


def test_filter_dcdplugin_messages_hides_only_known_lines(capfd):
    with filter_dcdplugin_messages():
        os.write(
            1,
            b"dcdplugin) detected standard 32-bit DCD file of native endianness\n",
        )
        os.write(
            2,
            b"dcdplugin) CHARMM format DCD file (also NAMD 2.1 and later)\n",
        )
        os.write(1, b"dcdplugin) retained diagnostic\n")

    captured = capfd.readouterr()
    combined = captured.out + captured.err
    assert "detected standard 32-bit DCD" not in combined
    assert "CHARMM format DCD file" not in combined
    assert "dcdplugin) retained diagnostic" in combined


def test_save_psf_round_trips_long_residue_names_and_bonds(tmp_path):
    md = pytest.importorskip("mdtraj")
    topology = md.Topology()
    protein = topology.add_chain()
    cysp = topology.add_residue("CYSP", protein, resSeq=25, segment_id="PROA")
    sg = topology.add_atom("SG", md.element.sulfur, cysp)
    c1 = topology.add_atom("C1", md.element.carbon, cysp)
    topology.add_bond(sg, c1)
    ligand = topology.add_chain()
    agonist = topology.add_residue("AGO1", ligand, resSeq=1, segment_id="HETA")
    n1 = topology.add_atom("N1", md.element.nitrogen, agonist)
    topology.add_bond(c1, n1)

    path = tmp_path / "topology.psf"
    save_psf(topology, path)
    reloaded = md.load_topology(path)

    assert [
        (atom.residue.chain.index, atom.residue.segment_id, atom.residue.resSeq,
         atom.residue.name, atom.name, atom.element)
        for atom in reloaded.atoms
    ] == [
        (0, "PROA", 25, "CYSP", "SG", md.element.sulfur),
        (0, "PROA", 25, "CYSP", "C1", md.element.carbon),
        (1, "HETA", 1, "AGO1", "N1", md.element.nitrogen),
    ]
    assert [(bond[0].index, bond[1].index) for bond in reloaded.bonds] == [(0, 1), (1, 2)]
