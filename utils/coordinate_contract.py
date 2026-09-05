"""Persistent coordinate-system and topology contract for diffusion models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import mdtraj as md
import torch


CONTRACT_VERSION = 1


def atom_identity(atom) -> list[Any]:
    """Return the stable ordered identity used across trajectory topologies."""
    element = getattr(atom, "element", None)
    return [
        int(atom.residue.chain.index),
        int(atom.residue.index),
        int(getattr(atom.residue, "resSeq", atom.residue.index)),
        str(atom.residue.name),
        str(atom.name),
        None if element is None else str(element.symbol),
    ]


def topology_signature(topology: md.Topology) -> dict[str, Any]:
    """Describe atom order, residue membership, segments, and covalent bonds."""
    atoms = [atom_identity(atom) for atom in topology.atoms]
    bonds = sorted(
        sorted((int(bond.atom1.index), int(bond.atom2.index)))
        for bond in topology.bonds
    )
    return {
        "atoms": atoms,
        "bonds": bonds,
        "num_atoms": int(topology.n_atoms),
        "num_residues": int(topology.n_residues),
        "num_segments": len({identity[0] for identity in atoms}),
    }


def validate_topology(
    topology: md.Topology,
    expected: Mapping[str, Any],
    *,
    context: str = "diffusion topology",
) -> None:
    """Raise a useful error if topology semantics differ from a contract."""
    actual = topology_signature(topology)
    expected_atoms = list(expected.get("atoms", []))
    actual_atoms = actual["atoms"]
    if actual_atoms != expected_atoms:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(actual_atoms, expected_atoms)
                )
                if left != right
            ),
            min(len(actual_atoms), len(expected_atoms)),
        )
        actual_value = actual_atoms[mismatch] if mismatch < len(actual_atoms) else None
        expected_value = (
            expected_atoms[mismatch] if mismatch < len(expected_atoms) else None
        )
        raise ValueError(
            f"{context} atom ordering/identity does not match the warm-start "
            f"coordinate contract at atom {mismatch}: expected "
            f"{expected_value!r}, got {actual_value!r}."
        )

    expected_bonds = sorted(list(pair) for pair in expected.get("bonds", []))
    if actual["bonds"] != expected_bonds:
        raise ValueError(
            f"{context} covalent bonds do not match the warm-start coordinate "
            f"contract: expected {len(expected_bonds)} bonds, got "
            f"{len(actual['bonds'])}."
        )


def create_coordinate_contract(
    *,
    topology: md.Topology,
    reference_xyz: torch.Tensor,
    alignment_atom_indices: torch.Tensor,
    coord_mean: torch.Tensor,
    coord_std: torch.Tensor,
) -> dict[str, Any]:
    """Build a serializable contract from a canonicalized training dataset."""
    return {
        "version": CONTRACT_VERSION,
        "topology": topology_signature(topology),
        "reference_xyz": reference_xyz.detach().cpu().to(torch.float32),
        "alignment_atom_indices": alignment_atom_indices.detach()
        .cpu()
        .to(torch.long),
        "coord_mean": coord_mean.detach().cpu().to(torch.float32),
        "coord_std": coord_std.detach().cpu().to(torch.float32),
    }


def save_coordinate_contract(
    contract: Mapping[str, Any],
    contract_path: str,
    reference_path: str,
    topology: md.Topology,
) -> None:
    """Save contract metadata and its human-readable canonical PDB."""
    contract_file = Path(contract_path)
    reference_file = Path(reference_path)
    contract_file.parent.mkdir(parents=True, exist_ok=True)
    reference_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(contract), contract_file)
    reference_xyz = torch.as_tensor(contract["reference_xyz"]).numpy()
    md.Trajectory(reference_xyz[None, :, :], topology).save_pdb(
        str(reference_file), force_overwrite=True
    )


def load_coordinate_contract(path: Optional[str]) -> Optional[dict[str, Any]]:
    """Load and minimally validate a coordinate contract."""
    if not path:
        return None
    contract_path = Path(path)
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Warm-start coordinate contract not found: {contract_path}"
        )
    try:
        contract = torch.load(contract_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        contract = torch.load(contract_path, map_location="cpu")
    if not isinstance(contract, dict):
        raise ValueError(f"Invalid coordinate contract in {contract_path}.")
    if int(contract.get("version", -1)) != CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported coordinate contract version "
            f"{contract.get('version')!r} in {contract_path}."
        )
    required = {
        "topology",
        "reference_xyz",
        "alignment_atom_indices",
        "coord_mean",
        "coord_std",
    }
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError(
            f"Coordinate contract {contract_path} is missing: {', '.join(missing)}."
        )
    return contract
