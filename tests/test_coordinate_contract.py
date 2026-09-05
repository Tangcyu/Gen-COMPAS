from pathlib import Path

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")
torch = pytest.importorskip("torch")

from utils.coordinate_contract import (
    create_coordinate_contract,
    load_coordinate_contract,
    save_coordinate_contract,
    topology_signature,
    validate_topology,
)
from utils.data_loader import ProteinDataset


def _topology(atom_names=("C1", "C2", "C3", "C4")):
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("MOL", chain, resSeq=3)
    atoms = [
        topology.add_atom(name, md.element.carbon, residue)
        for name in atom_names
    ]
    for left, right in zip(atoms, atoms[1:]):
        topology.add_bond(left, right)
    return topology


def test_warm_start_reuses_reference_and_normalization(tmp_path: Path):
    topology = _topology()
    reference = np.array(
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.5]],
        dtype=np.float32,
    )
    initial_frames = np.stack(
        [reference, reference + np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.04]])]
    ).astype(np.float32)
    topology_path = tmp_path / "topology.pdb"
    initial_dcd = tmp_path / "initial.dcd"
    md.Trajectory(reference[None, :, :], topology).save_pdb(topology_path)
    md.Trajectory(initial_frames, topology).save_dcd(initial_dcd)

    initial = ProteinDataset(
        str(topology_path),
        str(initial_dcd),
        alignment_atomselect="all",
    )
    mean, std = initial.get_normalization_constants()
    contract = create_coordinate_contract(
        topology=initial.topology,
        reference_xyz=initial.reference_xyz,
        alignment_atom_indices=initial.alignment_atom_indices,
        coord_mean=mean,
        coord_std=std,
    )
    contract_path = tmp_path / "coordinate_contract.pt"
    canonical_path = tmp_path / "canonical_reference.pdb"
    save_coordinate_contract(
        contract, str(contract_path), str(canonical_path), initial.topology
    )
    contract = load_coordinate_contract(str(contract_path))
    assert canonical_path.is_file()

    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    later_frames = (
        (initial_frames * 1.4) @ rotation.T
        + np.array([4.0, -2.0, 1.0], dtype=np.float32)
    )
    later_dcd = tmp_path / "later.dcd"
    md.Trajectory(later_frames, topology).save_dcd(later_dcd)
    later = ProteinDataset(
        str(topology_path),
        str(later_dcd),
        coordinate_contract=contract,
    )

    later_mean, later_std = later.get_normalization_constants()
    torch.testing.assert_close(later_mean, mean)
    torch.testing.assert_close(later_std, std)
    # The first later frame is fitted into the original canonical orientation.
    later_raw = later.unnormalize(later.coords)[0]
    expected = (
        (reference - reference.mean(axis=0, keepdims=True)) * 1.4
        + reference.mean(axis=0, keepdims=True)
    )
    torch.testing.assert_close(
        later_raw,
        torch.as_tensor(expected),
        atol=2.0e-5,
        rtol=0.0,
    )


def test_topology_validation_rejects_semantic_reordering():
    expected = topology_signature(_topology())
    reordered = _topology(("C2", "C1", "C3", "C4"))

    with pytest.raises(ValueError, match="atom ordering/identity"):
        validate_topology(reordered, expected)
