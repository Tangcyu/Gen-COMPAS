# utils/model.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import math 
from .embedding import SinusoidalEmbedding 
from typing import List


def scatter_mean_torch(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int | None = None):
    """Compute scatter-mean with native PyTorch ops to avoid torch-scatter."""
    if dim != 0:
        raise NotImplementedError("scatter_mean_torch currently supports only dim=0.")

    if src.ndim == 0:
        raise ValueError("src must have at least one dimension.")

    index = index.to(device=src.device, dtype=torch.long)
    if index.ndim != 1 or index.shape[0] != src.shape[0]:
        raise ValueError("index must be a 1D tensor with the same length as src along dim 0.")

    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    out_shape = (dim_size,) + tuple(src.shape[1:])
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    counts = torch.zeros(dim_size, dtype=src.dtype, device=src.device)

    if index.numel() == 0:
        return out

    expand_shape = (index.shape[0],) + (1,) * (src.ndim - 1)
    expanded_index = index.view(expand_shape).expand_as(src)
    out.scatter_add_(0, expanded_index, src)
    counts.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))

    counts = counts.clamp_min(1)
    view_shape = (dim_size,) + (1,) * (src.ndim - 1)
    return out / counts.view(view_shape)

def knn_graph_pytorch(x: torch.Tensor, k: int, batch: torch.Tensor = None, loop: bool = False, flow: str = 'source_to_target'):
    """
    Builds a k-nearest neighbor graph from node positions using PyTorch operations.

    Args:
        x: Node positions [num_nodes_total, num_dimensions]
        k: Number of neighbors per node
        batch: Batch assignment for each node [num_nodes_total]
        loop: Whether to include self-loops
        flow: Edge direction ('source_to_target' or 'target_to_source')

    Returns:
        Edge indices [2, num_edges]
    """
    assert flow in ['source_to_target', 'target_to_source']

    if batch is None:
        num_nodes = x.shape[0]
        dist = torch.cdist(x, x) 

        if not loop:
            dist.fill_diagonal_(float('inf'))

        effective_k = min(k, num_nodes - (1 if not loop else 0))
        if effective_k <= 0:
             return torch.empty((2,0), dtype=torch.long, device=x.device)

        _, col = torch.topk(dist, effective_k, dim=1, largest=False) 
        row = torch.arange(num_nodes, device=x.device).view(-1, 1).repeat(1, effective_k) 

        row = row.flatten() 
        col = col.flatten() 

    else:
        num_nodes_total = x.shape[0]

        # Mask prevents edges between different batch items
        batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1)

        dist_full = torch.cdist(x, x) 
        dist_full[~batch_mask] = float('inf') 

        if not loop:
            dist_full.fill_diagonal_(float('inf'))

        effective_k = min(k, num_nodes_total - (1 if not loop else 0))
        if effective_k <= 0:
             return torch.empty((2,0), dtype=torch.long, device=x.device)

        _, col = torch.topk(dist_full, effective_k, dim=1, largest=False) 
        row = torch.arange(num_nodes_total, device=x.device).view(-1, 1).repeat(1, effective_k) 

        row = row.flatten() 
        col = col.flatten() 

        if not loop:
            mask = row != col
            row = row[mask]
            col = col[mask]


    if flow == 'source_to_target':
        edge_index = torch.stack([row, col], dim=0)
    else: 
        edge_index = torch.stack([col, row], dim=0)

    return edge_index.to(torch.long)


class SchNetLayer(nn.Module):
    """
    Continuous-filter convolution layer for local atom interactions.
    Uses distance-based edge filters to modulate message passing.
    """
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        # Distance-dependent filter network
        self.filter_net = nn.Sequential(
            nn.Linear(1, hidden_dim), 
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim) 
        )
        # Node update network
        self.update_net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim) 
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, edge_indices: torch.Tensor, batch_size: int):
        """
        Args:
            x: Node coordinates [B, N, 3]
            h: Node features [B, N, node_dim]
            edge_indices: Global edge indices [2, E_total]
            batch_size: Number of graphs in batch

        Returns:
            Updated node features [B, N, node_dim]
        """
        B, N, node_dim_runtime = h.shape
        device = h.device
        E_total = edge_indices.shape[1]

        row, col = edge_indices

        batch_idx_row = row // N
        node_idx_row = row % N
        batch_idx_col = col // N
        node_idx_col = col % N

        x_row = x[batch_idx_row, node_idx_row, :]
        x_col = x[batch_idx_col, node_idx_col, :]
        h_col = h[batch_idx_col, node_idx_col, :] 

        dist = torch.norm(x_row - x_col, dim=-1, keepdim=True) 

        # Compute distance-dependent edge weights
        edge_filters = self.filter_net(dist)

        # Build messages by modulating neighbor features
        messages = h_col * edge_filters

        # Aggregate messages to target nodes
        num_nodes_total = B * N
        agg_messages_flat = scatter_mean_torch(messages, row.long(), dim=0, dim_size=num_nodes_total) # Shape [B*N, node_dim]
        agg_messages = agg_messages_flat.view(B, N, -1)

        # Update node features with aggregated information
        update_input = torch.cat([h, agg_messages], dim=-1)
        h_update = self.update_net(update_input) # Shape [B, N, node_dim]

        return h_update



class DiffusionModel(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        topology, 
        atom_types: List[str],
        node_feature_dim: int = 64, 
        time_embedding_dim: int = 128, 
        hidden_dim: int = 128,        
        num_schnet_layers: int = 3,   
        num_gat_layers: int = 2,       
        residue_attn_heads: int = 4,   
        k_neighbors: int = 16        
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.k_neighbors = k_neighbors
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim

        # Process atom types and build lookup table
        self.unique_atom_types = sorted(list(set(atom_types)))
        self.atom_type_map = {name: i for i, name in enumerate(self.unique_atom_types)}
        print(f"Unique atom types found: {len(self.unique_atom_types)}")
        self.num_atom_types_unique = len(self.unique_atom_types)

        self.set_topology(topology)

        # Initial feature embeddings
        self.residue_embedding = nn.Embedding(self.num_residues, node_feature_dim)
        self.atom_type_embedding = nn.Embedding(self.num_atom_types_unique, node_feature_dim)
        self.coord_encoder = nn.Linear(3, node_feature_dim)
        self.initial_embed_norm = nn.LayerNorm(node_feature_dim)

        # Time embedding
        self.time_embed_module = SinusoidalEmbedding(time_embedding_dim)
        self.time_embed_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_feature_dim) 
        )

        # Atom-level Processing 
        self.schnet_layers = nn.ModuleList([
            SchNetLayer(node_feature_dim, hidden_dim)
            for _ in range(num_schnet_layers)
        ])
        self.schnet_norms = nn.ModuleList([nn.LayerNorm(node_feature_dim) for _ in range(num_schnet_layers)])

        #  Residue-level Attention
        self.residue_input_proj = nn.Linear(node_feature_dim, hidden_dim) 
        self.residue_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=residue_attn_heads, batch_first=True, dropout=0.1)
            for _ in range(num_gat_layers)
        ])
        self.residue_attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)])
        self.residue_output_proj = nn.Linear(hidden_dim, node_feature_dim) 
        self.residue_final_norm = nn.LayerNorm(node_feature_dim) 

        # Output layers
        self.final_mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, node_feature_dim) 
        )
        self.output_mlp = nn.Linear(node_feature_dim, 3) 


    def set_topology(self, topology):
        """Sets topology information and precomputes indices."""
        self.topology = topology
        self.num_residues = topology.n_residues
        print(f"Topology: {self.num_residues} residues, {topology.n_atoms} atoms.")

        # Map each atom to its residue
        self.register_buffer('residue_indices', torch.tensor([atom.residue.index for atom in topology.atoms], dtype=torch.long))

        try:
             atom_type_indices = [self.atom_type_map[atom.name] for atom in topology.atoms]
        except KeyError as e:
             raise ValueError(f"Atom name '{e}' found in topology but not in the initial atom_types list used for mapping.")
        self.register_buffer('atom_types_mapped', torch.tensor(atom_type_indices, dtype=torch.long))

        # Extract covalent bonds as base edges
        if hasattr(topology, 'bonds') and topology.bonds:
             bond_list = [[b.atom1.index, b.atom2.index] for b in topology.bonds]
             if not bond_list: 
                  base_edges_tensor = torch.empty((2,0), dtype=torch.long)
             else:
                  base_edges_tensor = torch.tensor(bond_list, dtype=torch.long).t().contiguous()
                  base_edges_tensor = torch.cat([base_edges_tensor, base_edges_tensor.flip(0)], dim=1)
        else:
             base_edges_tensor = torch.empty((2,0), dtype=torch.long) 

        num_bonds = topology.n_bonds if hasattr(topology, 'n_bonds') else base_edges_tensor.shape[1] // 2
        print(f"Found {num_bonds} covalent bonds.")

        self.register_buffer('base_edges', base_edges_tensor)


    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor):
        """
        Predict clean coordinates from noisy input.

        Args:
            x_noisy: Noisy atom coordinates (centered) [B, N, 3]
            t: Diffusion timesteps [B]

        Returns:
            Predicted clean coordinates (centered) [B, N, 3]
        """
        B, N, _ = x_noisy.shape
        assert N == self.num_atoms, f"Input N ({N}) does not match model's num_atoms ({self.num_atoms})"
        assert x_noisy.ndim == 3 and x_noisy.shape[-1] == 3, f"Expected x_noisy shape [B, N, 3], got {x_noisy.shape}"
        assert t.ndim == 1 and t.shape[0] == B, f"Expected t shape [B], got {t.shape}"
        device = x_noisy.device

        residue_indices = self.residue_indices
        atom_types_mapped = self.atom_types_mapped
        base_edges = self.base_edges

        # Build dynamic k-NN graph based on current positions
        x_flat = x_noisy.view(B * N, 3) # Shape [B*N, 3]
        batch_vector = torch.arange(B, device=device).repeat_interleave(N) # Shape [B*N]
        spatial_edges = knn_graph_pytorch(x_flat, k=self.k_neighbors, batch=batch_vector, loop=False)

        # Combine covalent bonds with spatial neighbors
        if base_edges.numel() > 0:
            batch_offsets = torch.arange(B, device=device) * N
            expanded_base_edges = torch.cat([base_edges + offset for offset in batch_offsets], dim=1)
            edge_indices = torch.cat([expanded_base_edges, spatial_edges], dim=1)
        else:
            edge_indices = spatial_edges

        edge_indices = torch.unique(edge_indices, dim=1)

        use_amp = x_noisy.is_cuda and x_noisy.dtype == torch.float32
        with autocast(enabled=use_amp):
            # Build initial node features from embeddings
            res_emb = self.residue_embedding(residue_indices).unsqueeze(0).expand(B, -1, -1)
            atom_emb = self.atom_type_embedding(atom_types_mapped).unsqueeze(0).expand(B, -1, -1)
            coord_emb = self.coord_encoder(x_noisy) 

            h = res_emb + atom_emb + coord_emb 
            h = self.initial_embed_norm(h)

            # --- Time Embedding ---
            t_emb = self.time_embed_module(t) 
            t_proj = self.time_embed_mlp(t_emb).unsqueeze(1) 
            h = h + t_proj 

            # Local atom-level processing
            for i, layer in enumerate(self.schnet_layers):
                h_residual = h
                h_update = layer(x_noisy, h, edge_indices, B)
                h = self.schnet_norms[i](h_residual + h_update)

            # Aggregate atom features to residue level
            residue_indices_expanded = residue_indices.repeat(B)
            batch_offsets_res = torch.arange(B, device=device).repeat_interleave(N) * self.num_residues 
            residue_indices_global = residue_indices_expanded + batch_offsets_res

            h_flat_for_scatter = h.view(B * N, -1) 
            num_residues_total = B * self.num_residues

            # Average atom features per residue globally
            residue_h_flat = scatter_mean_torch(h_flat_for_scatter, residue_indices_global.long(), dim=0, dim_size=num_residues_total)
            residue_h = residue_h_flat.view(B, self.num_residues, -1)

            # Residue-level attention
            residue_h_proj = self.residue_input_proj(residue_h) 

            for i, attn_layer in enumerate(self.residue_attn_layers):
                residue_residual = residue_h_proj
                attn_out, _ = attn_layer(residue_h_proj, residue_h_proj, residue_h_proj) 
                residue_h_proj = self.residue_attn_norms[i](residue_residual + attn_out)

            # Broadcast residue features back to atoms
            residue_h_updated = self.residue_output_proj(residue_h_proj) 
            residue_h_updated_flat = residue_h_updated.view(B * self.num_residues, -1) 
            residue_context_gathered_flat = residue_h_updated_flat[residue_indices_global.long()] 
            residue_context_gathered = residue_context_gathered_flat.view(B, N, -1) 

            h = self.residue_final_norm(h + residue_context_gathered)

            # --- Final Output Prediction ---
            h = self.final_mlp(h) 
            x0_pred = self.output_mlp(h) 

        return x0_pred.float()