import mdtraj as md
import numpy as np
from itertools import combinations

def get_internal_coordinates(traj, selected_atoms):
    """
    计算选定原子之间的键长、键角和二面角，并返回对应的标签和数值。

    参数:
    - traj: mdtraj.Trajectory 对象，已加载的轨迹数据。
    - selected_atoms: list，选中的原子索引列表（如 [0, 1, 2, 3]）。

    返回:
    - labels: list，内坐标标签（如 ["Bond_0-1", "Angle_0-1-2", "Dihedral_0-1-2-3"]）。
    - values: numpy.ndarray，内坐标数值（形状为 [n_frames, n_internals]）。
    """
    labels = []
    values = []
    selected_atoms = [ i-1 for i in selected_atoms]
    # 计算键长（Bond，单位：Å）
    bonds = list(combinations(selected_atoms, 2))
    if bonds:
        bond_labels = [f"Bond_{a+1}-{b+1}" for a, b in bonds]
        bond_values = md.compute_distances(traj, bonds) * 10  # nm → Å
        labels.extend(bond_labels)
        values.append(bond_values)

    # 计算角度（Angle，单位：°）
    angles = list(combinations(selected_atoms, 3))
    if angles:
        angle_labels = [f"Angle_{a+1}-{b+1}-{c+1}" for a, b, c in angles]
        angle_values = np.degrees(md.compute_angles(traj, angles))  # rad → °
        labels.extend(angle_labels)
        values.append(angle_values)

    # 计算二面角（Dihedral，单位：°，范围：-180° 到 180°）
    dihedrals = list(combinations(selected_atoms, 4))
    if dihedrals:
        dihedral_labels = [f"Dihedral_{a+1}-{b+1}-{c+1}-{d+1}" for a, b, c, d in dihedrals]
        dihedral_values = np.degrees(md.compute_dihedrals(traj, dihedrals))  # rad → °
        dihedral_values = (dihedral_values + 180) % 360 - 180  # 规范到 [-180, 180]
        labels.extend(dihedral_labels)
        values.append(dihedral_values)

    # 合并数值并返回
    if values:
        values = np.hstack(values)
    else:
        values = np.empty((traj.n_frames, 0))

    return labels, values

def get_minimal_internal_coordinates(traj, selected_atoms):
    """
    Compute minimal internal coordinates (3N-6 DOF) for selected atoms, with
    periodic handling of angles and torsions using sin/cos transforms.

    Parameters:
    - traj: mdtraj.Trajectory
    - selected_atoms: list of 1-based atom indices (e.g., [1, 2, 3, 4])

    Returns:
    - labels: list of coordinate labels
    - values: np.ndarray of shape (n_frames, 3N-6)
    """
    labels = []
    values = []
    selected_atoms = [i - 1 for i in selected_atoms]  # convert to 0-based
    num_atoms = len(selected_atoms)

    if num_atoms < 2:
        raise ValueError("Must select at least 2 atoms to compute internal coordinates.")

    # 1. Bond lengths
    bonds = [(selected_atoms[i], selected_atoms[i + 1]) for i in range(num_atoms - 1)]
    bond_labels = [f"Bond_{a+1}-{b+1}" for a, b in bonds]
    bond_values = md.compute_distances(traj, bonds) * 10  # nm → Å
    labels.extend(bond_labels)
    values.append(bond_values)

    # 2. Bond angles (converted to sin/cos)
    angles = [(selected_atoms[i], selected_atoms[i + 1], selected_atoms[i + 2]) for i in range(num_atoms - 2)]
    angle_labels = [f"Angle_{a+1}-{b+1}-{c+1}_cos" for a, b, c in angles] + \
                   [f"Angle_{a+1}-{b+1}-{c+1}_sin" for a, b, c in angles]
    angle_radians = md.compute_angles(traj, angles)
    angle_cos = np.cos(angle_radians)
    angle_sin = np.sin(angle_radians)
    labels.extend(angle_labels)
    values.append(angle_cos)
    values.append(angle_sin)

    # 3. Dihedrals (converted to sin/cos)
    dihedrals = [(selected_atoms[i], selected_atoms[i + 1], selected_atoms[i + 2], selected_atoms[i + 3]) 
                 for i in range(num_atoms - 3)]
    dihedral_labels = [f"Dihedral_{a+1}-{b+1}-{c+1}-{d+1}_cos" for a, b, c, d in dihedrals] + \
                      [f"Dihedral_{a+1}-{b+1}-{c+1}-{d+1}_sin" for a, b, c, d in dihedrals]
    dihedral_radians = md.compute_dihedrals(traj, dihedrals)
    dihedral_cos = np.cos(dihedral_radians)
    dihedral_sin = np.sin(dihedral_radians)
    labels.extend(dihedral_labels)
    values.append(dihedral_cos)
    values.append(dihedral_sin)

    # Combine into a single array
    values = np.hstack(values) if values else np.empty((traj.n_frames, 0))

    # Corrected degrees of freedom: 1 bond → 1D, 1 angle → 2D, 1 torsion → 2D
    dof_expected = (num_atoms - 1) + 2 * (num_atoms - 2) + 2 * (num_atoms - 3)  # 3N - 6
    if values.shape[1] != dof_expected:
        raise ValueError(f"Expected {dof_expected} DOF but got {values.shape[1]}.")

    return labels, values


def get_pair_distances(traj, selected_atoms):
    """
    Compute minimal internal coordinates (3N-6 DOF) for selected atoms, with
    periodic handling of angles and torsions using sin/cos transforms.

    Parameters:
    - traj: mdtraj.Trajectory
    - selected_atoms: list of 1-based atom indices (e.g., [1, 2, 3, 4])

    Returns:
    - labels: list of coordinate labels
    - values: np.ndarray of shape (n_frames, 3N-6)
    """
    labels = []
    values = []
    selected_atoms = [i - 1 for i in selected_atoms]  # convert to 0-based
    num_atoms = len(selected_atoms)

    if num_atoms < 2:
        raise ValueError("Must select at least 2 atoms to compute internal coordinates.")

    # 计算键长（Bond，单位：Å）
    bonds = list(combinations(selected_atoms, 2))
    if bonds:
        bond_labels = [f"Bond_{a+1}-{b+1}" for a, b in bonds]
        bond_values = md.compute_distances(traj, bonds) * 10  # nm → Å
        labels.extend(bond_labels)
        values.append(bond_values)
        
    # Combine into a single array
    values = np.hstack(values) if values else np.empty((traj.n_frames, 0))
    return labels, values

def get_internal_bins(labels, values, bins=10):
    N = values.shape[1]  # 内坐标维度数
    if N == 0:
        return labels, values, np.empty((0, bins))

    # 计算每个维度的离散化中心点（矢量化计算）
    min_vals = np.min(values, axis=0)  # [N]
    max_vals = np.max(values, axis=0)  # [N]
    
    bin_edges = np.linspace(min_vals[:, None], max_vals[:, None], bins + 1, axis=1)  # 形状 [N, bins+1]
    bin_centers = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2  # 计算中心点，形状 [N, bins]

    # 生成 N 维格子（矢量化处理）
    meshgrid = np.meshgrid(*bin_centers, indexing="ij")  # 创建 N 维网格
    discrete_grid = np.column_stack([grid.ravel() for grid in meshgrid])  # 形状 [bins^N, N]
    return discrete_grid

# # # # 使用示例
# from sys import argv
# dcdtraj = md.load(argv[1], top=argv[2])

# # # 定义选中的原子索引
# selected_atoms = [1, 2, 3, 5, 9, 13, 14, 15, 17, 19]

# labels, values = get_internal_coordinates(dcdtraj, selected_atoms)

# print("Labels:", labels)
# print("Values:", values)
# print("Values shape:", values[0])

# import pandas as pd

# traj = pd.read_csv("../2D-RMSD-5ns-k-1.csv.gz")[:10000]
# data = pd.DataFrame({label: value for label, value in zip(labels,values.T)})
# traj_2 = traj.join(data)