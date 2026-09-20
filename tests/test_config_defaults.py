"""Configuration resource loading and run override boundaries."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import configs
from common.config import DEFAULT_CONFIG, load_config, resolve_iteration_config
from helper.config_helper import complete_config


def test_full_template_is_not_loaded_as_a_default_section():
    root = Path(__file__).resolve().parents[1]
    full = yaml.safe_load((root / "configs" / "Full.yaml").read_text(encoding="utf-8"))
    assert "Full" not in DEFAULT_CONFIG
    assert full == load_config(root / "config.yaml")
    assert load_config(root / "configs" / "Full.yaml") == full


def test_run_and_helper_share_nested_defaults_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    supplied = {
        "Workflow": {
            "initial_diffusion_data": {"dcd_path": "initial.dcd", "topology_path": "top.psf"},
        },
        "Generative": {
            "model": {"num_segment_layers": 3},
            "autonoise": {"search": {"max_samples": 1024}},
        },
        "NAMD": {"execution": {"gpu": {"devices": ["2"]}}},
        "RiteWeight": {"io": {"top": "system.psf"}},
        "Custom": {"keep": True},
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(supplied), encoding="utf-8")
    loaded = load_config(path)
    assert loaded == complete_config(supplied)
    assert loaded["Generative"]["model"]["num_segment_layers"] == 3
    assert loaded["Generative"]["model"]["hidden_dim"] == 128
    assert loaded["Generative"]["autonoise"]["search"]["max_samples"] == 1024
    assert loaded["Generative"]["autonoise"]["search"]["pilot_samples"] == 32
    assert loaded["NAMD"]["execution"]["gpu"]["devices"] == ["2"]
    assert loaded["NAMD"]["execution"]["cpu"]["threads_per_job"] == 1
    assert loaded["RiteWeight"]["io"]["topology"] == "system.psf"
    assert "top" not in loaded["RiteWeight"]["io"]
    assert loaded["Custom"] == {"keep": True}
    assert loaded["VCN"]["val_ratio"] == 0.1
    assert loaded["RiteWeight"]["pmf_output"]["enabled"] is False
    before = deepcopy(loaded)
    resolved = resolve_iteration_config(loaded, 0)
    assert resolved["Generative"]["autonoise"]["search"]["max_samples"] == 1024
    assert loaded == before
    loaded["Generative"]["autonoise"]["search"]["bounds"][0] = 999
    assert load_config(path) == before


def test_packaged_defaults_are_independent_and_preserve_yaml_types():
    first = configs.load_workflow_defaults()
    assert first == DEFAULT_CONFIG
    assert isinstance(first["Generative"]["training"]["lr"], float)
    assert isinstance(first["FEL_estimate"]["probability_floor"], float)
    assert first["Generative"]["init_checkpoint_path"] is None
    assert first["Workflow"]["iteration_noise_scales"] == {}
    assert first["RiteWeight"]["tag_regex"] == r"\.([AB])(?:\.|$)"
    first["NAMD"]["protocols"][0]["name"] = "changed"
    first["Generative"]["autonoise"]["search"]["bounds"][0] = 999
    assert configs.load_workflow_defaults() == DEFAULT_CONFIG


def test_new_dotted_resource_is_used_without_a_python_registry(tmp_path, monkeypatch):
    (tmp_path / "Module.yaml").write_text("enabled: true\n", encoding="utf-8")
    (tmp_path / "Module.model.yaml").write_text("width: 32\n", encoding="utf-8")
    (tmp_path / "Module.model.attention.yaml").write_text("heads: 4\n", encoding="utf-8")
    monkeypatch.setattr(configs, "files", lambda package: tmp_path)
    assert configs.load_workflow_defaults() == {
        "Module": {"enabled": True, "model": {"width": 32, "attention": {"heads": 4}}},
    }
    (tmp_path / "Module.yaml").write_text("model: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate packaged defaults"):
        configs.load_workflow_defaults()


def test_invalid_resource_has_an_actionable_error(tmp_path, monkeypatch):
    (tmp_path / "Module.yaml").write_text("- wrong shape\n", encoding="utf-8")
    monkeypatch.setattr(configs, "files", lambda package: tmp_path)
    with pytest.raises(TypeError, match="Module.yaml must contain a mapping"):
        configs.load_workflow_defaults()
