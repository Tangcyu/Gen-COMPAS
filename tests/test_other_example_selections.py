from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parents[1]
CARBON_TMD_SELECTION = (
    "protein and (name CA or name CB or name CC or name CD or "
    "name CG or name CY or name CZ)"
)
AAC_HOLO_FEATURE_SELECTION = (
    "(protein and name CA) or "
    "(resname ADP and (name N1 or name C4 or name PA or name PB))"
)


def _load_example(relative_path: str) -> dict:
    path = PROJECT_ROOT / relative_path
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_multiname_tmd_selections_use_explicit_mdtraj_booleans():
    trpcage = _load_example("examples/0.DEMO_Trp-cage/trpcage.workflow.yaml")
    vo_domain = _load_example("examples/5.Vo-domain/vo-domain.workflow.yaml")

    assert trpcage["Occupancy"]["selection"] == CARBON_TMD_SELECTION
    assert vo_domain["Occupancy"]["selection"] == CARBON_TMD_SELECTION


def test_aac_holo_includes_adp_anchors_and_exports_the_full_heavy_ligand():
    config = _load_example("examples/4.AAC/2.holo/aac_3states.workflow.yaml")

    assert config["VCN"]["atomselect"] == AAC_HOLO_FEATURE_SELECTION
    assert config["Clustering"]["atom_selection"] == AAC_HOLO_FEATURE_SELECTION
    assert config["Occupancy"]["selection"] == AAC_HOLO_FEATURE_SELECTION
    assert (
        config["RiteWeight"]["features"]["internal_zmat"]["atomselect"]
        == AAC_HOLO_FEATURE_SELECTION
    )

    diffusion = config["RiteWeight"]["outputs"]["diffusion"]
    assert diffusion["atomselect"] == "(protein or resname ADP) and element != H"
    assert diffusion["alignment_atomselect"] == AAC_HOLO_FEATURE_SELECTION


def test_aac_holo_adp_anchors_exist_once_and_ligand_has_27_heavy_atoms():
    topology_path = (
        PROJECT_ROOT
        / "examples"
        / "4.AAC"
        / "2.holo"
        / "NAMD_inputs"
        / "TOP_files"
        / "ligand_protein.psf"
    )
    lines = topology_path.read_text(encoding="utf-8").splitlines()
    natom_index = next(index for index, line in enumerate(lines) if "!NATOM" in line)
    atom_count = int(lines[natom_index].split()[0])
    atoms = [line.split() for line in lines[natom_index + 1:natom_index + 1 + atom_count]]
    adp_heavy_names = [
        atom[4] for atom in atoms if atom[3] == "ADP" and float(atom[7]) >= 2.0
    ]

    assert len(adp_heavy_names) == 27
    assert {name: adp_heavy_names.count(name) for name in ("N1", "C4", "PA", "PB")} == {
        "N1": 1,
        "C4": 1,
        "PA": 1,
        "PB": 1,
    }
