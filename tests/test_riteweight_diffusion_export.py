from pathlib import Path

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

from tools.riteweight import write_diffusion_training_data


# A lipidated cysteine (CYSP) and a four-character agonist (AGO1) are not
# MDTraj "protein"; the water and hydrogens are absent from the model topology.
SYSTEM_ATOMS = [
    ("PROA", 1, "ALA", "N", 14.007),
    ("PROA", 1, "ALA", "HN", 1.008),
    ("PROA", 1, "ALA", "CA", 12.011),
    ("PROA", 1, "ALA", "CB", 12.011),
    ("PROA", 1, "ALA", "C", 12.011),
    ("PROA", 1, "ALA", "O", 15.999),
    ("PROA", 2, "CYSP", "N", 14.007),
    ("PROA", 2, "CYSP", "HN", 1.008),
    ("PROA", 2, "CYSP", "CA", 12.011),
    ("PROA", 2, "CYSP", "CB", 12.011),
    ("PROA", 2, "CYSP", "SG", 32.06),
    ("PROA", 2, "CYSP", "C1", 12.011),
    ("PROA", 2, "CYSP", "O1", 15.999),
    ("PROA", 2, "CYSP", "C2", 12.011),
    ("PROA", 2, "CYSP", "C", 12.011),
    ("PROA", 2, "CYSP", "O", 15.999),
    ("HETA", 1, "AGO1", "C1", 12.011),
    ("HETA", 1, "AGO1", "N1", 14.007),
    ("HETA", 1, "AGO1", "C2", 12.011),
    ("HETA", 1, "AGO1", "H1", 1.008),
    ("SOLV", 1, "TIP3", "OH2", 15.999),
    ("SOLV", 1, "TIP3", "H1", 1.008),
    ("SOLV", 1, "TIP3", "H2", 1.008),
]
SYSTEM_BONDS = [
    (1, 2), (1, 3), (3, 4), (3, 5), (5, 6), (5, 7), (7, 8), (7, 9),
    (9, 10), (10, 11), (11, 12), (12, 13), (12, 14), (9, 15), (15, 16),
    (17, 18), (18, 19), (17, 20), (21, 22), (21, 23),
]


def _write_psf(path: Path, atoms, bonds):
    """Write a standard-format PSF like VMD writepsf."""
    lines = ["PSF", "", "       1 !NTITLE", " REMARKS test system", ""]
    lines.append(f"{len(atoms):8d} !NATOM")
    for index, (segid, resid, resname, name, mass) in enumerate(atoms, start=1):
        lines.append(
            f"{index:8d} {segid:<4s} {resid:<4d} {resname:<4s} {name:<4s} "
            f"{name[:1]:<4s} {0.0:10.6f} {mass:13.4f} {0:11d}"
        )
    lines += ["", f"{len(bonds):8d} !NBOND: bonds"]
    flat = [index for bond in bonds for index in bond]
    for start in range(0, len(flat), 8):
        lines.append("".join(f"{index:8d}" for index in flat[start:start + 8]))
    path.write_text("\n".join(lines) + "\n\n", encoding="utf-8")


def _write_system(tmp_path: Path):
    """Write a full-system PSF/DCD and the matching heavy-atom model PSF."""
    full_psf = tmp_path / "system.psf"
    _write_psf(full_psf, SYSTEM_ATOMS, SYSTEM_BONDS)
    heavy = [
        number
        for number, atom in enumerate(SYSTEM_ATOMS, start=1)
        if atom[4] > 2.0 and atom[0] != "SOLV"
    ]
    renumber = {old: new for new, old in enumerate(heavy, start=1)}
    reference_psf = tmp_path / "model_noh.psf"
    _write_psf(
        reference_psf,
        [SYSTEM_ATOMS[number - 1] for number in heavy],
        [
            (renumber[left], renumber[right])
            for left, right in SYSTEM_BONDS
            if left in renumber and right in renumber
        ],
    )
    rng = np.random.default_rng(0)
    xyz = rng.normal(scale=0.3, size=(2, len(SYSTEM_ATOMS), 3)).astype(np.float32)
    source_dcd = tmp_path / "Unbiased.A.dcd"
    md.Trajectory(xyz, md.load_topology(full_psf)).save_dcd(source_dcd)
    return full_psf, source_dcd, reference_psf


def _identity(topology):
    atoms = [
        (
            atom.residue.chain.index,
            atom.residue.segment_id,
            atom.residue.resSeq,
            atom.residue.name,
            atom.name,
            atom.element.symbol,
        )
        for atom in topology.atoms
    ]
    bonds = sorted(
        tuple(sorted((bond[0].index, bond[1].index))) for bond in topology.bonds
    )
    return atoms, bonds


def _export(tmp_path: Path, full_psf, source_dcd, **kwargs):
    options = dict(
        used_pairs=[(str(source_dcd), "unused.colvars.traj")],
        expected_frames=[2],
        top_path=str(full_psf),
        stride=1,
        atomselect=None,
        alignment_atomselect="element != H",
        dcd_path=str(tmp_path / "out" / "diffusion_training.dcd"),
        topology_path=str(tmp_path / "out" / "diffusion_training.psf"),
        chunk_size=1,
    )
    options.update(kwargs)
    write_diffusion_training_data(**options)
    return Path(options["dcd_path"]), Path(options["topology_path"])


def test_diffusion_export_rmsd_aligns_all_frames(tmp_path: Path):
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
    moved = reference @ rotation.T + np.array([1.5, -0.7, 0.4])
    source = md.Trajectory(np.stack([reference, moved]), topology)

    topology_path = tmp_path / "source.pdb"
    source_path = tmp_path / "source.dcd"
    output_path = tmp_path / "aligned.dcd"
    output_topology = tmp_path / "aligned.psf"
    source[0].save_pdb(topology_path)
    source.save_dcd(source_path)

    write_diffusion_training_data(
        used_pairs=[(str(source_path), "unused.colvars.traj")],
        expected_frames=[2],
        top_path=str(topology_path),
        stride=1,
        atomselect="all",
        alignment_atomselect="all",
        dcd_path=str(output_path),
        topology_path=str(output_topology),
        chunk_size=1,
    )

    aligned = md.load(output_path, top=output_topology)
    np.testing.assert_allclose(aligned.xyz[0], aligned.xyz[1], atol=2.0e-5)


def test_default_export_reproduces_model_atoms_names_and_bonds(tmp_path: Path):
    full_psf, source_dcd, reference_psf = _write_system(tmp_path)

    dcd_path, topology_path = _export(
        tmp_path, full_psf, source_dcd, reference_topology_path=str(reference_psf)
    )

    exported = md.load_topology(topology_path)
    assert _identity(exported) == _identity(md.load_topology(reference_psf))
    assert [residue.name for residue in exported.residues] == ["ALA", "CYSP", "AGO1"]
    bonded_names = {
        frozenset((bond[0].name, bond[1].name))
        for bond in exported.bonds
        if bond[0].residue.name == "CYSP" and bond[1].residue.name == "CYSP"
    }
    assert frozenset(("SG", "C1")) in bonded_names
    assert md.load(dcd_path, top=topology_path).xyz.shape == (2, 17, 3)


def test_export_keeps_model_residue_names(tmp_path: Path):
    """A model topology truncating AGO1 to AGO (as in examples/1.NANMA, where
    ALAD became ALA) still matches by segment, residue number and atom name,
    and the export keeps the model's own names."""
    full_psf, source_dcd, reference_psf = _write_system(tmp_path)
    reference_psf.write_text(
        reference_psf.read_text(encoding="utf-8").replace(" AGO1 ", " AGO  "),
        encoding="utf-8",
    )

    _, topology_path = _export(
        tmp_path, full_psf, source_dcd, reference_topology_path=str(reference_psf)
    )

    assert _identity(md.load_topology(topology_path)) == _identity(
        md.load_topology(reference_psf)
    )


def test_explicit_selection_matching_the_model_is_accepted(tmp_path: Path):
    full_psf, source_dcd, reference_psf = _write_system(tmp_path)

    _, topology_path = _export(
        tmp_path,
        full_psf,
        source_dcd,
        atomselect="not element H and not water",
        reference_topology_path=str(reference_psf),
    )

    assert _identity(md.load_topology(topology_path)) == _identity(
        md.load_topology(reference_psf)
    )


def test_protein_only_selection_dropping_model_atoms_is_rejected(tmp_path: Path):
    full_psf, source_dcd, reference_psf = _write_system(tmp_path)

    with pytest.raises(ValueError, match="selects 5 atoms, but the reference has 17"):
        _export(
            tmp_path,
            full_psf,
            source_dcd,
            atomselect="protein and not element H",
            reference_topology_path=str(reference_psf),
        )


def test_reference_atoms_missing_from_the_system_are_rejected(tmp_path: Path):
    full_psf, source_dcd, _ = _write_system(tmp_path)
    other_model = tmp_path / "other_noh.psf"
    _write_psf(other_model, [("PROA", 3, "ALA", "CA", 12.011)], [])

    with pytest.raises(ValueError, match="PROA:ALA3:CA.*no match"):
        _export(
            tmp_path, full_psf, source_dcd, reference_topology_path=str(other_model)
        )


def test_pdb_diffusion_topology_is_rejected(tmp_path: Path):
    full_psf, source_dcd, reference_psf = _write_system(tmp_path)

    with pytest.raises(ValueError, match=r"must end in \.psf"):
        _export(
            tmp_path,
            full_psf,
            source_dcd,
            reference_topology_path=str(reference_psf),
            topology_path=str(tmp_path / "out" / "diffusion_training.pdb"),
        )
