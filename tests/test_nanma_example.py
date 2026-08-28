from pathlib import Path

import yaml


def test_nanma_uses_ten_heavy_atoms_consistently():
    project_root = Path(__file__).parents[1]
    config_path = project_root / "examples" / "1.NANMA" / "nanma.workflow.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    initial = config["Workflow"]["initial_diffusion_data"]
    assert initial["dcd_path"].endswith("alad_unbiased_heavy.dcd")
    assert initial["topology_path"].endswith("alad_heavy.psf")
    assert (project_root / "examples" / "1.NANMA" / initial["topology_path"]).is_file()

    assert config["VCN"]["atomselect"] == "element != H"
    assert config["Clustering"]["atom_selection"] == "not name H*"
    assert config["Occupancy"]["selection"] == "element != H"
    assert (
        config["RiteWeight"]["features"]["internal_zmat"]["atomselect"]
        == "element != H"
    )
    diffusion = config["RiteWeight"]["outputs"]["diffusion"]
    assert diffusion["atomselect"] == "element != H"
    assert diffusion["alignment_atomselect"] == "element != H"


def test_nanma_heavy_topology_has_expected_atoms():
    topology_path = (
        Path(__file__).parents[1]
        / "examples"
        / "1.NANMA"
        / "NAMD_inputs"
        / "TOP_files"
        / "alad_heavy.psf"
    )
    lines = topology_path.read_text(encoding="utf-8").splitlines()
    natom_index = next(index for index, line in enumerate(lines) if "!NATOM" in line)
    assert int(lines[natom_index].split()[0]) == 10
    atom_lines = lines[natom_index + 1:natom_index + 11]
    assert all(" ALA " in line for line in atom_lines)
    assert all(float(line.split()[7]) > 10.0 for line in atom_lines)
