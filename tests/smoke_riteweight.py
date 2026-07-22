"""Standalone small-data smoke test for the RiteWeight output contract."""

from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mdtraj as md
import numpy as np

from common.config import deep_merge, DEFAULT_CONFIG
from tools.riteweight import run_riteweight
from tools.tensor_table import load_tensor_table


def _topology():
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("ALA", chain)
    atoms = [
        topology.add_atom(name, md.element.carbon, residue)
        for name in ("CA", "CB", "C", "N")
    ]
    for left, right in zip(atoms, atoms[1:]):
        topology.add_bond(left, right)
    return topology


def main():
    with tempfile.TemporaryDirectory(prefix="gen_compas_rw_") as directory:
        root = Path(directory)
        topology = _topology()
        base = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
             [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            dtype=np.float32,
        )
        top_path = root / "topology.pdb"
        md.Trajectory(base[None, :, :], topology).save_pdb(str(top_path))

        for state_index, state in enumerate(("A", "B")):
            job = root / f"job_{state}"
            job.mkdir()
            rng = np.random.default_rng(state_index)
            xyz = base[None, :, :] + rng.normal(0, 0.005, size=(8, 4, 3))
            md.Trajectory(xyz.astype(np.float32), topology).save_dcd(
                str(job / f"Unbiased.{state}.dcd")
            )
            cv = np.linspace(state_index, 1 - state_index, 8)
            table = np.column_stack([np.arange(8), cv, cv])
            np.savetxt(
                job / f"Unbiased.{state}.colvars.traj",
                table,
                header="step CV1 CV2",
            )

        config = deep_merge(
            DEFAULT_CONFIG["RiteWeight"],
            {
                "folders": [str(root)],
                "io": {"top": str(top_path), "out": str(root / "out")},
                "features": {
                    "internal_zmat": {
                        "atomselect": None,
                        "atom_order": [0, 1, 2, 3],
                    },
                    "cache": {"enabled": False},
                },
                "riteweight": {
                    "n_clusters": 2,
                    "n_iter": 5,
                    "lag": 1,
                    "avg_last": 2,
                },
                "committor_labels": {
                    "basin_A": [0.0, 0.0],
                    "basin_B": [1.0, 1.0],
                    "basin_size": [0.2, 0.2],
                },
                "outputs": {"diffusion": {"atomselect": "all"}},
            },
        )
        outputs = run_riteweight(config)
        table = load_tensor_table(outputs["vcn_pt"])
        trajectory = md.load_dcd(
            outputs["diffusion_dcd"], top=outputs["diffusion_topology"]
        )
        assert len(table) == trajectory.n_frames == 16
        assert set(table["trajectory_id"]) == {0, 1}
        assert list(table.groupby("trajectory_id").size()) == [8, 8]
        print("RiteWeight smoke test passed")


if __name__ == "__main__":
    main()
