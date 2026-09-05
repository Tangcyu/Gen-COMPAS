from pathlib import Path

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

from tools.riteweight import write_diffusion_training_data


def test_diffusion_export_rmsd_aligns_all_frames(tmp_path: Path):
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("MOL", chain)
    for index in range(4):
        topology.add_atom(f"C{index + 1}", md.element.carbon, residue)

    reference = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.4]],
        dtype=np.float32,
    )
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    moved = reference @ rotation.T + np.array([1.5, -0.7, 0.4])
    source = md.Trajectory(np.stack([reference, moved]), topology)

    topology_path = tmp_path / "source.pdb"
    source_path = tmp_path / "source.dcd"
    output_path = tmp_path / "aligned.dcd"
    output_topology = tmp_path / "aligned.pdb"
    source[0].save_pdb(topology_path)
    source.save_dcd(source_path)

    write_diffusion_training_data(
        used_pairs=[(str(source_path), "unused.colvars.traj")],
        expected_frames=[2],
        top_path=str(topology_path),
        stride=1,
        atomselect="all",
        alignment_atomselect="all",
        dcd_path=str(output_path),
        topology_path=str(output_topology),
        chunk_size=1,
    )

    aligned = md.load(output_path, top=output_topology)
    np.testing.assert_allclose(aligned.xyz[0], aligned.xyz[1], atol=2.0e-5)


def test_diffusion_export_uses_prior_canonical_reference(tmp_path: Path):
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("MOL", chain, resSeq=7)
    atoms = [
        topology.add_atom(f"C{index + 1}", md.element.carbon, residue)
        for index in range(4)
    ]
    for left, right in zip(atoms, atoms[1:]):
        topology.add_bond(left, right)

    canonical = np.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.4]],
        dtype=np.float32,
    )
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    current = canonical @ rotation.T + np.array([2.0, -1.0, 0.5])

    topology_path = tmp_path / "source.pdb"
    canonical_path = tmp_path / "canonical_reference.pdb"
    source_path = tmp_path / "current_iteration.dcd"
    output_path = tmp_path / "aligned.dcd"
    output_topology = tmp_path / "aligned.pdb"
    md.Trajectory(canonical[None, :, :], topology).save_pdb(topology_path)
    md.Trajectory(canonical[None, :, :], topology).save_pdb(canonical_path)
    md.Trajectory(np.stack([current, current]), topology).save_dcd(source_path)

    write_diffusion_training_data(
        used_pairs=[(str(source_path), "unused.colvars.traj")],
        expected_frames=[2],
        top_path=str(topology_path),
        stride=1,
        atomselect="all",
        alignment_atomselect="all",
        reference_path=str(canonical_path),
        dcd_path=str(output_path),
        topology_path=str(output_topology),
        chunk_size=1,
    )

    aligned = md.load(output_path, top=output_topology)
    np.testing.assert_allclose(
        aligned.xyz, np.repeat(canonical[None, :, :], 2, axis=0), atol=2.0e-5
    )
