import torch
from torch.utils.data import Dataset
import os

from utils.mdtraj_io import load_dcd as quiet_md_load_dcd


class ProteinDataset(Dataset):
    def __init__(
        self,
        topology_path: str,
        dcd_path: str,
        alignment_atomselect: str = "all",
    ):
        super().__init__()
        if not os.path.exists(topology_path):
            raise FileNotFoundError(f"Topology file not found: {topology_path}")
        if not os.path.exists(dcd_path):
            raise FileNotFoundError(f"DCD file not found: {dcd_path}")

        traj = quiet_md_load_dcd(dcd_path, top=topology_path)
        if traj is None or traj.n_frames == 0:
            raise ValueError(f"Could not load trajectory: {dcd_path}")

        # Remove global translations and rotations before deriving any dataset
        # statistics. The fit can use a structural subset, but MDTraj applies
        # the resulting rigid transform to every atom in every frame.
        alignment_atomselect = alignment_atomselect or "all"
        alignment_indices = traj.topology.select(alignment_atomselect)
        if len(alignment_indices) < 3:
            raise ValueError(
                "Diffusion RMSD alignment needs at least three atoms; "
                f"{alignment_atomselect!r} matched {len(alignment_indices)}."
            )
        traj.superpose(traj[0], frame=0, atom_indices=alignment_indices)

        self.num_atoms = traj.n_atoms
        self.num_samples = traj.n_frames
        self.topology = traj.topology
        self.coords = torch.tensor(traj.xyz, dtype=torch.float32)

        # Normalize coordinates
        self.coord_mean = self.coords.mean(dim=(0, 1), keepdim=True)
        self.coord_std = self.coords.std(dim=(0, 1), keepdim=True) + 1e-8
        self.coords = (self.coords - self.coord_mean) / self.coord_std
        print(
            f"Dataset loaded and RMSD-aligned to frame 0: "
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
