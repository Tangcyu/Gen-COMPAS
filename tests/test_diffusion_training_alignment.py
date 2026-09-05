from pathlib import Path

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

from utils.data_loader import ProteinDataset


def test_dataset_aligns_frames_before_computing_normalization(tmp_path: Path):
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
    moved = reference @ rotation.T + np.array([1.5, -0.7, 0.4], dtype=np.float32)
    trajectory = md.Trajectory(np.stack([reference, moved]), topology)

    topology_path = tmp_path / "training.pdb"
    trajectory_path = tmp_path / "training.dcd"
    trajectory[0].save_pdb(topology_path)
    trajectory.save_dcd(trajectory_path)

    dataset = ProteinDataset(str(topology_path), str(trajectory_path))
    aligned = dataset.unnormalize(dataset.coords).numpy()

    np.testing.assert_allclose(aligned[0], aligned[1], atol=2.0e-5)
    np.testing.assert_allclose(
        dataset.get_normalization_constants()[0].numpy(),
        aligned.mean(axis=(0, 1)),
        atol=1.0e-7,
    )
