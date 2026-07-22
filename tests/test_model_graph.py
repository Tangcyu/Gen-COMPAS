import pytest

torch = pytest.importorskip("torch")

from utils.model import knn_graph_pytorch


def test_knn_never_connects_different_batch_items():
    coordinates = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.1, 0.0], [1.1, 0.0]]
    )
    batch = torch.tensor([0, 0, 1, 1])
    edges = knn_graph_pytorch(coordinates, k=4, batch=batch)
    assert edges.shape == (2, 4)
    assert torch.equal(batch[edges[0]], batch[edges[1]])
