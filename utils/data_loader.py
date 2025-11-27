import torch
from torch.utils.data import Dataset
import mdtraj as md
import os

class ProteinDataset(Dataset):
    def __init__(self, psf_path: str, dcd_path: str):
        super().__init__()
        if not os.path.exists(psf_path):
            raise FileNotFoundError(f"PSF file not found: {psf_path}")
        if not os.path.exists(dcd_path):
            raise FileNotFoundError(f"DCD file not found: {dcd_path}")

        traj = md.load_dcd(dcd_path, top=psf_path)
        if traj is None or traj.n_frames == 0:
            raise ValueError(f"Could not load trajectory: {dcd_path}")

        self.num_atoms = traj.n_atoms
        self.num_samples = traj.n_frames
        self.topology = traj.topology
        self.coords = torch.tensor(traj.xyz, dtype=torch.float32)  

        # Normalize coordinates
        self.coord_mean = self.coords.mean(dim=(0, 1), keepdim=True)
        self.coord_std = self.coords.std(dim=(0, 1), keepdim=True) + 1e-8
        self.coords = (self.coords - self.coord_mean) / self.coord_std
        print(f"Dataset loaded: {self.num_samples} frames, {self.num_atoms} atoms.")

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