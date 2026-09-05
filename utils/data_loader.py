import torch
from torch.utils.data import Dataset
import os
import numpy as np
import mdtraj as md

from utils.coordinate_contract import validate_topology
from utils.mdtraj_io import load_dcd as quiet_md_load_dcd


class ProteinDataset(Dataset):
    def __init__(
        self,
        topology_path: str,
        dcd_path: str,
        alignment_atomselect: str = "all",
        *,
        coordinate_contract: dict = None,
    ):
        super().__init__()
        if not os.path.exists(topology_path):
            raise FileNotFoundError(f"Topology file not found: {topology_path}")
        if not os.path.exists(dcd_path):
            raise FileNotFoundError(f"DCD file not found: {dcd_path}")

        traj = quiet_md_load_dcd(dcd_path, top=topology_path)
        if traj is None or traj.n_frames == 0:
            raise ValueError(f"Could not load trajectory: {dcd_path}")

        if coordinate_contract is not None:
            validate_topology(
                traj.topology,
                coordinate_contract["topology"],
                context="diffusion training topology",
            )
            reference_xyz = torch.as_tensor(
                coordinate_contract["reference_xyz"], dtype=torch.float32
            )
            alignment_indices = torch.as_tensor(
                coordinate_contract["alignment_atom_indices"], dtype=torch.long
            )
        else:
            selected = traj.topology.select(alignment_atomselect or "all")
            if len(selected) < 3:
                raise ValueError(
                    "Diffusion coordinate alignment needs at least three atoms; "
                    f"{alignment_atomselect!r} matched {len(selected)}."
                )
            reference_xyz = torch.tensor(traj.xyz[0], dtype=torch.float32)
            alignment_indices = torch.tensor(selected, dtype=torch.long)

        if reference_xyz.shape != (traj.n_atoms, 3):
            raise ValueError(
                "Coordinate-contract reference shape does not match the training "
                f"trajectory: expected {(traj.n_atoms, 3)}, got "
                f"{tuple(reference_xyz.shape)}."
            )
        if (
            alignment_indices.ndim != 1
            or alignment_indices.numel() < 3
            or torch.any(alignment_indices < 0)
            or torch.any(alignment_indices >= traj.n_atoms)
        ):
            raise ValueError(
                "Coordinate-contract alignment indices must contain at least "
                "three valid atom indices."
            )
        reference = md.Trajectory(
            reference_xyz.numpy()[None, :, :], traj.topology
        )
        align = alignment_indices.numpy().astype(np.int64, copy=False)
        traj.superpose(reference, atom_indices=align, ref_atom_indices=align)

        self.num_atoms = traj.n_atoms
        self.num_samples = traj.n_frames
        self.topology = traj.topology
        self.reference_xyz = reference_xyz
        self.alignment_atom_indices = alignment_indices
        self.coords = torch.tensor(traj.xyz, dtype=torch.float32)

        # Normalize coordinates
        if coordinate_contract is None:
            self.coord_mean = self.coords.mean(dim=(0, 1), keepdim=True)
            self.coord_std = self.coords.std(dim=(0, 1), keepdim=True) + 1e-8
        else:
            self.coord_mean = torch.as_tensor(
                coordinate_contract["coord_mean"], dtype=torch.float32
            ).view(1, 1, 3)
            self.coord_std = torch.as_tensor(
                coordinate_contract["coord_std"], dtype=torch.float32
            ).view(1, 1, 3)
            if not torch.isfinite(self.coord_mean).all():
                raise ValueError("Coordinate-contract mean contains non-finite values.")
            if (
                not torch.isfinite(self.coord_std).all()
                or torch.any(self.coord_std <= 0)
            ):
                raise ValueError(
                    "Coordinate-contract standard deviation must be finite and positive."
                )
        self.coords = (self.coords - self.coord_mean) / self.coord_std
        print(
            f"Dataset loaded and RMSD-aligned to canonical reference: "
            f"{self.num_samples} frames, {self.num_atoms} atoms "
            f"({len(alignment_indices)} fit atoms from "
            f"{alignment_atomselect!r})."
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict:
        return {'coords': self.coords[index]}

    def get_normalization_constants(self) -> tuple:
        return self.coord_mean.squeeze(), self.coord_std.squeeze()

    def unnormalize(self, coords_norm: torch.Tensor) -> torch.Tensor:
        mean = self.coord_mean.to(coords_norm.device)
        std = self.coord_std.to(coords_norm.device)
        return coords_norm * std + mean
