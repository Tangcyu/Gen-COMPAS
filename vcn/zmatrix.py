import mdtraj as md
import numpy as np
from itertools import combinations


def get_internal_coordinates(traj, selected_atoms):
    """
    Compute bond lengths, bond angles, and dihedral angles between the
    selected atoms, and return the corresponding labels and values.

    Parameters:
    - traj: mdtraj.Trajectory object containing the loaded trajectory.
    - selected_atoms: list of selected atom indices (for example [1, 2, 3, 4]).

    Returns:
    - labels: list of internal-coordinate labels
      (for example ["Bond_1-2", "Angle_1-2-3", "Dihedral_1-2-3-4"]).
    - values: numpy.ndarray containing internal-coordinate values with shape
      [n_frames, n_internals].
    """
    labels = []
    values = []
    selected_atoms = [i - 1 for i in selected_atoms]

    # Compute bond lengths in angstroms.
    bonds = list(combinations(selected_atoms, 2))
    if bonds:
        bond_labels = [f"Bond_{a+1}-{b+1}" for a, b in bonds]
        bond_values = md.compute_distances(traj, bonds) * 10  # nm -> A
        labels.extend(bond_labels)
        values.append(bond_values)

    # Compute bond angles in degrees.
    angles = list(combinations(selected_atoms, 3))
    if angles:
        angle_labels = [f"Angle_{a+1}-{b+1}-{c+1}" for a, b, c in angles]
        angle_values = np.degrees(md.compute_angles(traj, angles))  # rad -> deg
        labels.extend(angle_labels)
        values.append(angle_values)

    # Compute dihedral angles in degrees and wrap them to [-180, 180].
    dihedrals = list(combinations(selected_atoms, 4))
    if dihedrals:
        dihedral_labels = [f"Dihedral_{a+1}-{b+1}-{c+1}-{d+1}" for a, b, c, d in dihedrals]
        dihedral_values = np.degrees(md.compute_dihedrals(traj, dihedrals))  # rad -> deg
        dihedral_values = (dihedral_values + 180) % 360 - 180
        labels.extend(dihedral_labels)
        values.append(dihedral_values)

    # Combine the values into a single array before returning.
    if values:
        values = np.hstack(values)
    else:
        values = np.empty((traj.n_frames, 0))

    return labels, values


def get_minimal_internal_coordinates(traj, selected_atoms):
    """
    Compute the canonical RiteWeight/VCN fixed-anchor internal coordinates.

    Distances are in MDTraj's native nanometers and angles/dihedrals are in
    radians.  For ordered atoms ``a0, a1, ...``, the first three features are
    ``r(a0,a1)``, ``r(a0,a2)``, and ``angle(a1,a0,a2)``.  Every later atom
    contributes its distance and angle from the same anchors plus one
    dihedral, for a total of 3N-6 features.

    Parameters:
    - traj: mdtraj.Trajectory
    - selected_atoms: list of 1-based atom indices (e.g., [1, 2, 3, 4])

    Returns:
    - labels: list of coordinate labels
    - values: np.ndarray of shape (n_frames, 3N-6)
    """
    selected_atoms = [i - 1 for i in selected_atoms]  # convert to 0-based
    num_atoms = len(selected_atoms)

    if num_atoms < 4:
        raise ValueError("Must select at least 4 atoms for RiteWeight features.")

    a0, a1, a2 = selected_atoms[:3]
    later_atoms = selected_atoms[3:]

    distances = [(a0, a1), (a0, a2)] + [(a0, atom) for atom in later_atoms]
    angles = [(a1, a0, a2)] + [(a1, a0, atom) for atom in later_atoms]
    dihedrals = [(a2, a1, a0, atom) for atom in later_atoms]

    distance_values = md.compute_distances(traj, distances)
    angle_values = md.compute_angles(traj, angles)
    dihedral_values = md.compute_dihedrals(traj, dihedrals)

    labels = [
        f"Distance_{a0+1}-{a1+1}",
        f"Distance_{a0+1}-{a2+1}",
        f"Angle_{a1+1}-{a0+1}-{a2+1}",
    ]
    labels.extend(f"Distance_{a0+1}-{atom+1}" for atom in later_atoms)
    labels.extend(f"Angle_{a1+1}-{a0+1}-{atom+1}" for atom in later_atoms)
    labels.extend(
        f"Dihedral_{a2+1}-{a1+1}-{a0+1}-{atom+1}" for atom in later_atoms
    )

    values = np.concatenate(
        [
            distance_values[:, [0]],
            distance_values[:, [1]],
            angle_values[:, [0]],
            distance_values[:, 2:],
            angle_values[:, 1:],
            dihedral_values,
        ],
        axis=1,
    )

    dof_expected = 3 * num_atoms - 6
    if values.shape[1] != dof_expected:
        raise ValueError(f"Expected {dof_expected} features but got {values.shape[1]}.")

    return labels, values



def get_pair_distances(traj, selected_atoms):
    """
    Compute all pairwise distances between the selected atoms.

    Parameters:
    - traj: mdtraj.Trajectory
    - selected_atoms: list of 1-based atom indices (e.g., [1, 2, 3, 4])

    Returns:
    - labels: list of pair-distance labels
    - values: np.ndarray of shape (n_frames, n_pairs)
    """
    labels = []
    values = []
    selected_atoms = [i - 1 for i in selected_atoms]  # convert to 0-based
    num_atoms = len(selected_atoms)

    if num_atoms < 2:
        raise ValueError("Must select at least 2 atoms to compute internal coordinates.")

    # Compute pairwise distances in angstroms.
    bonds = list(combinations(selected_atoms, 2))
    if bonds:
        bond_labels = [f"Bond_{a+1}-{b+1}" for a, b in bonds]
        bond_values = md.compute_distances(traj, bonds) * 10  # nm -> A
        labels.extend(bond_labels)
        values.append(bond_values)

    # Combine into a single array.
    values = np.hstack(values) if values else np.empty((traj.n_frames, 0))
    return labels, values


def get_internal_bins(labels, values, bins=10):
    num_dimensions = values.shape[1]  # Number of internal-coordinate dimensions.
    if num_dimensions == 0:
        return labels, values, np.empty((0, bins))

    # Compute the discretization centers for each dimension using vectorized operations.
    min_vals = np.min(values, axis=0)  # [N]
    max_vals = np.max(values, axis=0)  # [N]

    bin_edges = np.linspace(min_vals[:, None], max_vals[:, None], bins + 1, axis=1)  # shape [N, bins+1]
    bin_centers = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2  # shape [N, bins]

    # Build the N-dimensional grid from the per-dimension bin centers.
    meshgrid = np.meshgrid(*bin_centers, indexing="ij")
    discrete_grid = np.column_stack([grid.ravel() for grid in meshgrid])
    return discrete_grid

# # # # Example usage
# from sys import argv
# dcdtraj = md.load(argv[1], top=argv[2])

# # # Define the selected atom indices
# selected_atoms = [1, 2, 3, 5, 9, 13, 14, 15, 17, 19]

# labels, values = get_internal_coordinates(dcdtraj, selected_atoms)

# print("Labels:", labels)
# print("Values:", values)
# print("Values shape:", values[0])

# import pandas as pd

# traj = pd.read_csv("../2D-RMSD-5ns-k-1.csv.gz")[:10000]
# data = pd.DataFrame({label: value for label, value in zip(labels, values.T)})
# traj_2 = traj.join(data)
