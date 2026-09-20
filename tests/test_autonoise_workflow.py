"""Workflow handoff, manual-noise precedence, retries and calibration provenance."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import types

import pytest

from common.autonoise_state import accept_autonoise_result, apply_autonoise_selection, calibration_fingerprint
from common.config import DEFAULT_CONFIG, deep_merge, load_config, resolve_iteration_config
from common import runner
import workflow


def make_config(tmp_path, iteration=0, isolated=True):
    config = deep_merge(DEFAULT_CONFIG, {
        "Workflow": {"root_dir": str(tmp_path / "iterations"), "isolate_steps": isolated,
                     "initial_diffusion_data": {"dcd_path": str(tmp_path / "initial.dcd"),
                                                "topology_path": str(tmp_path / "top.psf")}},
        "Generative": {"autonoise": {"enabled": True,
                       "reference": {"state_a": str(tmp_path / "a.dcd"), "state_b": str(tmp_path / "b.dcd")}}},
    })
    resolved = resolve_iteration_config(config, iteration)
    checkpoint = Path(resolved["Generative"]["inference"]["checkpoint"])
    for path in (checkpoint, checkpoint.parent / "coordinate_contract.pt",
                 Path(resolved["Generative"]["data"]["topology_path"]),
                 tmp_path / "a.dcd", tmp_path / "b.dcd"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test input")
    return config


def manifest_at(config, iteration=0):
    resolved = resolve_iteration_config(config, iteration)
    return Path(resolved["Workflow"]["runtime"]["iteration_dir"]) / "workflow_manifest.json"


def install_stages(monkeypatch, *, noise=2.75, status="ok"):
    calls = []

    def calibrate(config):
        out = Path(config["Generative"]["autonoise"]["output_dir"])
        assert not out.exists()  # Retries get a fresh attempt directory.
        out.mkdir(parents=True)
        (out / "noise_distribution.png").write_bytes(b"test plot")
        calls.append(("autonoise_diffusion", str(out)))
        return {"status": status, "recommended_noise_scales": [noise] if status == "ok" else [],
                "output_dir": str(out)}

    def sample(generative):
        calls.append(("sample_diffusion", generative["inference"]["noise_scale"]))

    monkeypatch.setitem(sys.modules, "common.diffusion_autonoise", types.SimpleNamespace(run_autonoise=calibrate))
    monkeypatch.setitem(sys.modules, "common.diffusion_sample", types.SimpleNamespace(run_diffusion_inference=sample))
    monkeypatch.setattr(workflow, "_run_isolated_step",
                        lambda step, path: runner.main(["--step", step, "--config", str(path)]))
    return calls


@pytest.mark.parametrize("iteration", [0, 1, 3])
def test_optional_stage_order_and_manual_override_warning(tmp_path, iteration):
    base = make_config(tmp_path, iteration)
    base["Workflow"]["iteration_noise_scales"] = {0: 99, "2": 88}
    before = deepcopy(base)
    with pytest.warns(UserWarning, match="iteration_noise_scales.*ignored"):
        config = resolve_iteration_config(base, iteration)
    steps = workflow.iteration_steps(config, iteration)
    assert steps[steps.index("sample_diffusion") - 1] == "autonoise_diffusion"
    assert config["Generative"]["inference"]["noise_scale"] is None
    assert config["Generative"]["autonoise"]["output_dir"].endswith("/autonoise")
    assert base == before
    base["Generative"]["autonoise"]["enabled"] = False
    assert "autonoise_diffusion" not in workflow.iteration_steps(base, iteration)
    with pytest.raises(ValueError, match="not part of this iteration"):
        workflow.run_iteration(base, iteration, run_step_name="autonoise_diffusion", dry_run=True)


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("iteration", [0, 1])
def test_selection_survives_stage_isolation_and_new_sampling_invocation(tmp_path, monkeypatch, isolated, iteration):
    base = make_config(tmp_path, iteration, isolated)
    calls = install_stages(monkeypatch)
    workflow.run_iteration(base, iteration, run_step_name="autonoise_diffusion")
    manifest_path = manifest_at(base, iteration)
    first = json.loads(manifest_path.read_text())
    assert first["steps"]["autonoise_diffusion"]["selection"]["noise_scale"] == 2.75
    # A separate invocation resolves the base config again, then restores the selection.
    workflow.run_iteration(base, iteration, run_step_name="sample_diffusion")
    assert calls[-1] == ("sample_diffusion", 2.75)
    effective = load_config(manifest_path.parent / "effective_config.yaml")
    assert effective["Generative"]["inference"]["noise_scale"] == 2.75
    workflow.run_iteration(base, iteration, start_at="autonoise_diffusion", stop_after="sample_diffusion", resume=True)
    assert len(calls) == 2
    assert json.loads(manifest_path.read_text())["steps"]["sample_diffusion"]["attempts"] == 1
    base["Workflow"]["isolate_steps"] = True
    monkeypatch.setattr(workflow, "_run_isolated_step", lambda *args: None)
    workflow.run_iteration(base, iteration, run_step_name="occupancy")
    effective = load_config(manifest_path.parent / "effective_config.yaml")
    assert effective["Generative"]["inference"]["noise_scale"] == 2.75


def test_sampling_without_calibration_fails_without_using_manual_noise(tmp_path, monkeypatch):
    base = make_config(tmp_path)
    calls = install_stages(monkeypatch)
    with pytest.raises(RuntimeError, match="autonoise_diffusion.*before sample_diffusion"):
        workflow.run_iteration(base, 0, run_step_name="sample_diffusion")
    with pytest.raises(RuntimeError, match="No completed AutoNoise selection"):
        runner.run_step("sample_diffusion", resolve_iteration_config(base, 0))
    assert calls == []
    assert json.loads(manifest_at(base).read_text())["steps"]["sample_diffusion"]["status"] == "failed"


def test_failed_calibration_stops_sampling_and_retry_preserves_outputs(tmp_path, monkeypatch):
    base = make_config(tmp_path)
    failed = install_stages(monkeypatch, status="no_verified_intermediate")
    with pytest.raises(RuntimeError, match="no_verified_intermediate"):
        workflow.run_iteration(base, 0, start_at="autonoise_diffusion", stop_after="sample_diffusion")
    first_output = Path(failed[0][1])
    state = json.loads(manifest_at(base).read_text())
    assert state["steps"]["autonoise_diffusion"]["status"] == "failed"
    assert "sample_diffusion" not in state["steps"]
    calls = install_stages(monkeypatch)
    workflow.run_iteration(base, 0, start_at="autonoise_diffusion", stop_after="sample_diffusion", resume=True)
    assert first_output.is_dir()
    assert calls[0][1].endswith("attempt_002")
    assert calls[-1] == ("sample_diffusion", 2.75)


def test_rerun_invalidates_downstream_then_resume_uses_new_noise(tmp_path, monkeypatch):
    base = make_config(tmp_path)
    install_stages(monkeypatch)
    workflow.run_iteration(base, 0, start_at="autonoise_diffusion", stop_after="sample_diffusion")
    calls = install_stages(monkeypatch, noise=3.)
    workflow.run_iteration(base, 0, rerun_step="autonoise_diffusion")
    state = json.loads(manifest_at(base).read_text())
    assert len(calls) == 1
    assert state["steps"]["sample_diffusion"]["status"] == "pending"
    workflow.run_iteration(base, 0, start_at="autonoise_diffusion", stop_after="sample_diffusion", resume=True)
    assert calls[-1] == ("sample_diffusion", 3.)
    state = json.loads(manifest_at(base).read_text())
    assert state["steps"]["sample_diffusion"]["attempts"] == 2


@pytest.mark.parametrize("changed", ["checkpoint", "reference", "settings"])
def test_stale_selection_is_rejected_for_sampling_and_recalibrated_on_resume(tmp_path, monkeypatch, changed):
    base = make_config(tmp_path)
    install_stages(monkeypatch)
    workflow.run_iteration(base, 0, start_at="autonoise_diffusion", stop_after="sample_diffusion")
    if changed == "checkpoint":
        Path(resolve_iteration_config(base, 0)["Generative"]["inference"]["checkpoint"]).write_bytes(b"new model")
    elif changed == "reference":
        (tmp_path / "a.dcd").write_bytes(b"new reference")
    else:
        base["Generative"]["autonoise"]["search"] = {"refine_rounds": 4}
    calls = install_stages(monkeypatch, noise=3.)
    with pytest.raises(RuntimeError, match="inputs changed"):
        workflow.run_iteration(base, 0, rerun_step="sample_diffusion")
    assert calls == []
    workflow.run_iteration(base, 0, start_at="autonoise_diffusion", stop_after="sample_diffusion", resume=True)
    assert calls[0][1].endswith("attempt_002")
    assert calls[-1] == ("sample_diffusion", 3.)


def test_training_invalidates_existing_calibration(tmp_path, monkeypatch):
    base = make_config(tmp_path)
    install_stages(monkeypatch)
    workflow.run_iteration(base, 0, start_at="autonoise_diffusion", stop_after="sample_diffusion")
    monkeypatch.setattr(workflow, "_run_isolated_step", lambda *args: None)
    workflow.run_iteration(base, 0, rerun_step="train_diffusion")
    state = json.loads(manifest_at(base).read_text())
    assert state["steps"]["autonoise_diffusion"]["status"] == "pending"
    assert state["steps"]["sample_diffusion"]["status"] == "pending"


def test_selection_cannot_cross_iterations_or_mutated_inputs(tmp_path):
    base = make_config(tmp_path)
    config = resolve_iteration_config(base, 0)
    fingerprint = calibration_fingerprint(config)
    report = {"status": "ok", "recommended_noise_scales": [2.75], "output_dir": "out"}
    selection = accept_autonoise_result(config, report, fingerprint)
    config["Workflow"]["runtime"]["iteration"] = 1
    with pytest.raises(RuntimeError, match="inputs changed"):
        apply_autonoise_selection(config, selection)
    config["Workflow"]["runtime"]["iteration"] = 0
    Path(config["Generative"]["inference"]["checkpoint"]).write_bytes(b"updated during generation")
    with pytest.raises(RuntimeError, match="inputs changed"):
        accept_autonoise_result(config, report, fingerprint)
