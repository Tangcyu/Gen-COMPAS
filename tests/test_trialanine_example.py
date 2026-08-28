from pathlib import Path

import yaml


def test_trialanine_uses_all_twenty_heavy_atoms_consistently():
    project_root = Path(__file__).parents[1]
    config_path = (
        project_root / "examples" / "2.Trialanine" / "trialanine.workflow.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["VCN"]["atomselect"] == "element != H"
    assert config["Clustering"]["atom_selection"] == "not name H*"
    assert config["Occupancy"]["selection"] == "element != H"
    assert config["RiteWeight"]["io"]["topology"].endswith("triala.psf")
    assert (
        config["RiteWeight"]["features"]["internal_zmat"]["atomselect"]
        == "element != H"
    )
    diffusion = config["RiteWeight"]["outputs"]["diffusion"]
    assert diffusion["atomselect"] == "element != H"
    assert diffusion["alignment_atomselect"] == "element != H"
