import math
import torch
import torch.nn as nn
from typing import Optional


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional encoding for time conditioning in diffusion models."""
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {embedding_dim}")
        self.embedding_dim = embedding_dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal embeddings for timesteps.
        
        Args:
            timesteps: Timestep values [B]
        
        Returns:
            Sinusoidal embeddings [B, embedding_dim]
        """
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        embeddings = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return embeddings


class AtomTypeEmbedding(nn.Module):
    """Learnable embedding layer for atom types."""
    
    def __init__(self, num_atom_types: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, embedding_dim)
        print(f"Initialized AtomTypeEmbedding with {num_atom_types} types and dim {embedding_dim}")
    
    def forward(self, atom_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_types: Integer atom type indices [B, N]
        
        Returns:
            Embedded atom types [B, N, embedding_dim]
        """
        return self.embedding(atom_types)