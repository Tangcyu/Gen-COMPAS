import pytest

torch = pytest.importorskip("torch")
md = pytest.importorskip("mdtraj")

from utils.model import DiffusionModel, knn_graph_pytorch


def test_knn_never_connects_different_batch_items():
    coordinates = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.1, 0.0], [1.1, 0.0]]
    )
    batch = torch.tensor([0, 0, 1, 1])
    edges = knn_graph_pytorch(coordinates, k=4, batch=batch)
    assert edges.shape == (2, 4)
    assert torch.equal(batch[edges[0]], batch[edges[1]])


def _two_segment_topology():
    topology = md.Topology()
    for chain_index in range(2):
        chain = topology.add_chain()
        residue = topology.add_residue(f"S{chain_index}", chain)
        left = topology.add_atom("C1", md.element.carbon, residue)
        right = topology.add_atom("C2", md.element.carbon, residue)
        topology.add_bond(left, right)
    return topology


def test_hierarchical_path_keeps_local_edges_and_couples_distant_segments():
    topology = _two_segment_topology()
    model = DiffusionModel(
        num_atoms=4,
        topology=topology,
        atom_types=[atom.name for atom in topology.atoms],
        node_feature_dim=8,
        time_embedding_dim=8,
        hidden_dim=8,
        num_schnet_layers=1,
        num_gat_layers=1,
        residue_attn_heads=2,
        k_neighbors=1,
        num_segment_layers=1,
        segment_distance_rbf=4,
    )
    model.eval()
    coordinates = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [20.2, 0.0, 0.0],
            ]
        ],
        requires_grad=True,
    )
    captured = {}

    def capture_edges(_module, inputs):
        captured["edges"] = inputs[2].detach()

    handle = model.schnet_layers[0].register_forward_pre_hook(capture_edges)
    output = model(coordinates, torch.tensor([3]))
    handle.remove()

    assert torch.isfinite(output).all()
    row, col = captured["edges"]
    atom_segments = model.atom_segment_indices
    assert torch.equal(atom_segments[row % 4], atom_segments[col % 4])

    # Segment 0 output depends on segment 1 even though they are far beyond its
    # atom-level nearest-neighbor graph.
    output[:, :2].sum().backward()
    assert coordinates.grad[:, 2:].abs().sum() > 0


def test_residue_attention_is_local_and_complex_batches_are_independent():
    torch.manual_seed(17)
    topology = md.Topology()
    for _ in range(2):
        chain = topology.add_chain()
        for _ in range(2):
            residue = topology.add_residue("ALA", chain)
            topology.add_atom("CA", md.element.carbon, residue)
    model = DiffusionModel(
        4, topology, ["CA"] * 4, node_feature_dim=8,
        time_embedding_dim=8, hidden_dim=8, num_schnet_layers=1,
        num_gat_layers=1, residue_attn_heads=2, k_neighbors=1,
        num_segment_layers=1, segment_distance_rbf=4,
    ).eval()
    captured = {}

    def capture_mask(_module, _args, kwargs):
        captured["mask"] = kwargs["attn_mask"]

    handle = model.residue_attn_layers[0].register_forward_pre_hook(
        capture_mask, with_kwargs=True
    )
    coordinates = torch.randn(2, 4, 3, requires_grad=True)
    output = model(coordinates, torch.tensor([2, 7]))
    handle.remove()
    assert not captured["mask"][0, 1]  # different residues, same chain
    assert captured["mask"][0, 2]      # different chains
    alone = model(coordinates[:1], torch.tensor([2]))
    torch.testing.assert_close(output[:1], alone, atol=1e-6, rtol=1e-5)
    output[0, :2].square().sum().backward()
    assert coordinates.grad[0, 2:].abs().sum() > 0
    assert coordinates.grad[1].abs().sum() == 0
