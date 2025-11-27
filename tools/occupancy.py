import os
import yaml
import mdtraj as md
import numpy as np
from typing import Optional
from tqdm import tqdm


# =========================================================
# === Utility Functions ===
# =========================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing YAML file: {e}")


def write_pdb_with_custom_occupancy(traj, occupancies, out_path: str):
    """Write PDB file with custom occupancy values."""
    with open(out_path, "w") as f:
        for i, atom in enumerate(traj.topology.atoms):
            res = atom.residue
            coord = traj.xyz[0, i] * 10.0  # nm → Å
            occupancy = occupancies[0, i]  # assumes single frame
            f.write(
                "ATOM  {:5d} {:>4s} {:>3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f} {:4.2f}  0.00           {:>2s}\n".format(
                    atom.index + 1,
                    atom.name,
                    " A",
                    " ",
                    res.index + 1,
                    coord[0],
                    coord[1],
                    coord[2],
                    occupancy,
                    atom.element.symbol.rjust(2),
                )
            )
        f.write("END\n")


def load_reference(topology_file: str, pdb_file: str):
    """Load reference structure and topology."""
    ref_traj = md.load(pdb_file, top=topology_file)
    return ref_traj, ref_traj.topology, ref_traj.xyz


def get_atom_groups(topology):
    """Identify hydrogen and heavy atoms, and build hydrogen-heavy atom bonds."""
    hydrogen_atoms = [a.index for a in topology.atoms if a.name.startswith("H")]
    heavy_atoms = [a.index for a in topology.atoms if a.index not in hydrogen_atoms]

    bonds = {}
    for bond in topology.bonds:
        a1, a2 = bond[0], bond[1]
        if a1.index in hydrogen_atoms and a2.index in heavy_atoms:
            bonds[a1.index] = a2.index
        elif a2.index in hydrogen_atoms and a1.index in heavy_atoms:
            bonds[a2.index] = a1.index

    missing = [h for h in hydrogen_atoms if h not in bonds]
    if missing:
        print(f"Warning: {len(missing)} hydrogens without bonded heavy atom: {missing}")

    atom_names = {atom.index: atom.name for atom in topology.atoms}
    return hydrogen_atoms, heavy_atoms, bonds, atom_names


# =========================================================
# === Core Processing ===
# =========================================================

def hydrogenate_and_set_occupancy(
    pdb_dir: str,
    topology_file: str,
    pdb_file: str,
    output_dir: str,
    selection: Optional[str] = None,
):
    """Add hydrogens to PDBs and set occupancy values for selected atoms."""
    os.makedirs(output_dir, exist_ok=True)

    # Load reference
    ref_traj, ref_top, ref_xyz = load_reference(topology_file, pdb_file)
    hydrogen_atoms, heavy_atoms, bonds, atom_names = get_atom_groups(ref_top)

    selected_indices = ref_top.select(selection) if selection else []

    print(f"Selection: {len(selected_indices)} atoms will have occupancy = 1")

    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    for pdb_file in tqdm(pdb_files, desc="Processing PDBs"):
        input_path = os.path.join(pdb_dir, pdb_file)
        output_path = os.path.join(output_dir, pdb_file)

        traj = md.load(input_path)
        num_frames = traj.n_frames
        xyz = traj.xyz
        top = traj.topology

        heavy_atoms_in_traj = [a.index for a in top.atoms if a.element.symbol != "H"]
        heavy_xyz = xyz[:, heavy_atoms_in_traj, :]

        full_xyz = np.zeros((num_frames, ref_top.n_atoms, 3))
        full_xyz[:, heavy_atoms, :] = heavy_xyz

        # Add hydrogens based on reference geometry
        for h_idx in hydrogen_atoms:
            if h_idx in bonds:
                heavy_idx = bonds[h_idx]
                full_xyz[:, h_idx, :] = (
                    full_xyz[:, heavy_idx, :] + (ref_xyz[:, h_idx, :] - ref_xyz[:, heavy_idx, :])
                )

        new_traj = md.Trajectory(full_xyz, ref_top)
        for atom in new_traj.topology.atoms:
            atom.name = atom_names[atom.index]

        occupancies = np.zeros((num_frames, ref_top.n_atoms))
        occupancies[:, selected_indices] = 1.0

        write_pdb_with_custom_occupancy(new_traj, occupancies, output_path)
        print(f"Saved hydrogenated structure to: {output_path}")


def set_occupancy_only(
    pdb_dir: str,
    topology_file: str,
    pdb_file: str,
    output_dir: str,
    selection: Optional[str] = None,
):
    """Set occupancy values for selected atoms without hydrogenation."""
    os.makedirs(output_dir, exist_ok=True)

    ref_traj, ref_top, _ = load_reference(topology_file, pdb_file)
    atom_names = {atom.index: atom.name for atom in ref_top.atoms}
    selected_indices = ref_top.select(selection) if selection else []

    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    for pdb_file in tqdm(pdb_files, desc="Setting occupancy"):
        input_path = os.path.join(pdb_dir, pdb_file)
        output_path = os.path.join(output_dir, pdb_file)

        traj = md.load(input_path, top=topology_file)
        num_frames = traj.n_frames

        for atom in traj.topology.atoms:
            atom.name = atom_names[atom.index]

        occupancies = np.zeros((num_frames, ref_top.n_atoms))
        occupancies[:, selected_indices] = 1.0

        write_pdb_with_custom_occupancy(traj, occupancies, output_path)
        print(f"Saved occupancy-adjusted structure to: {output_path}")


# =========================================================
# === Main Routine ===
# =========================================================

def add_occupancy(config):
    """Run hydrogenation or occupancy pipeline based on config['Occupancy']."""

    pdb_dir = config["pdb_dir"]
    topology_file = config["topology_file"]
    pdb_file =config["pdb_file"]
    output_dir = config.get("output_dir", "./occupancy_output")
    add_h = config.get("add_hydrogens", True)
    selection = config.get("selection", None)

    if add_h:
        print("Running hydrogenation + occupancy assignment...")
        hydrogenate_and_set_occupancy(pdb_dir, topology_file, pdb_file, output_dir, selection)
    else:
        print("Running occupancy-only assignment...")
        set_occupancy_only(pdb_dir, topology_file, pdb_file, output_dir, selection)

    print("Pipeline completed successfully.")


# =========================================================
# === Entry Point ===
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hydrogenate or set occupancy in PDB files.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    run_occupancy_pipeline(args.config)
