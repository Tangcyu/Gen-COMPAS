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
    if traj.n_frames != 1:
        raise ValueError("Each occupancy target must contain exactly one frame.")
    if occupancies.shape != (1, traj.n_atoms):
        raise ValueError("Occupancy array does not match the target structure.")
    with open(out_path, "w") as f:
        for i, atom in enumerate(traj.topology.atoms):
            res = atom.residue
            chain = getattr(res.chain, "chain_id", None) or chr(65 + res.chain.index % 26)
            residue_number = getattr(res, "resSeq", None) or res.index + 1
            # PDB residue names occupy exactly three columns. Truncating names
            # such as ALAD to ALA keeps all later fixed-width fields where
            # NAMD's TMD reader expects them.
            residue_name = str(res.name)[:3]
            coord = traj.xyz[0, i] * 10.0  # nm → Å
            occupancy = occupancies[0, i]  # assumes single frame
            element = atom.element.symbol if atom.element is not None else atom.name[:1]
            f.write(
                "ATOM  {:5d} {:>4s} {:>3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(
                    atom.index + 1,
                    atom.name,
                    residue_name,
                    chain[:1],
                    residue_number,
                    coord[0],
                    coord[1],
                    coord[2],
                    occupancy,
                    0.0,
                    element,
                )
            )
        f.write("END\n")


def load_reference(topology_file: str, pdb_file: str):
    """Load reference structure and topology."""
    ref_traj = md.load(pdb_file, top=topology_file)
    if ref_traj.n_frames != 1:
        raise ValueError("Occupancy reference PDB must contain exactly one frame.")
    return ref_traj, ref_traj.topology, ref_traj.xyz


def get_atom_groups(topology):
    """Identify hydrogen and heavy atoms, and build hydrogen-heavy atom bonds."""
    hydrogen_atoms = [
        atom.index
        for atom in topology.atoms
        if atom.element is not None and atom.element.symbol == "H"
    ]
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
        raise ValueError(
            f"Reference topology has {len(missing)} hydrogens without a bonded "
            f"heavy atom: {missing}"
        )

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

    pdb_files = sorted(f for f in os.listdir(pdb_dir) if f.lower().endswith(".pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB targets found in {pdb_dir}.")
    for pdb_file in tqdm(pdb_files, desc="Processing PDBs"):
        input_path = os.path.join(pdb_dir, pdb_file)
        output_path = os.path.join(output_dir, pdb_file)

        traj = md.load(input_path)
        num_frames = traj.n_frames
        xyz = traj.xyz
        top = traj.topology

        heavy_atoms_in_traj = [a.index for a in top.atoms if a.element.symbol != "H"]
        heavy_xyz = xyz[:, heavy_atoms_in_traj, :]
        if len(heavy_atoms_in_traj) != len(heavy_atoms):
            raise ValueError(
                f"Heavy-atom count mismatch for {input_path}: "
                f"target={len(heavy_atoms_in_traj)}, reference={len(heavy_atoms)}."
            )
        target_names = [top.atom(index).name for index in heavy_atoms_in_traj]
        reference_names = [ref_top.atom(index).name for index in heavy_atoms]
        if target_names != reference_names:
            raise ValueError(f"Heavy-atom order/name mismatch for {input_path}.")

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

    pdb_files = sorted(f for f in os.listdir(pdb_dir) if f.lower().endswith(".pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB targets found in {pdb_dir}.")
    for pdb_file in tqdm(pdb_files, desc="Setting occupancy"):
        input_path = os.path.join(pdb_dir, pdb_file)
        output_path = os.path.join(output_dir, pdb_file)

        traj = md.load(input_path, top=topology_file)
        num_frames = traj.n_frames
        if traj.n_atoms != ref_top.n_atoms:
            raise ValueError(
                f"Atom-count mismatch for {input_path}: "
                f"target={traj.n_atoms}, reference={ref_top.n_atoms}."
            )

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
    return sorted(
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.lower().endswith(".pdb")
    )


# =========================================================
# === Entry Point ===
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hydrogenate or set occupancy in PDB files.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    args = parser.parse_args()

    from common.config import load_config as load_shared_config
    add_occupancy(load_shared_config(args.config)["Occupancy"])
