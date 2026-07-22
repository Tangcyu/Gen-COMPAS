import json
from pathlib import Path

import pytest

from common.config import deep_merge, DEFAULT_CONFIG, resolve_iteration_config
import workflow
from workflow import iteration_steps


def minimal_config(tmp_path: Path):
    return deep_merge(
        DEFAULT_CONFIG,
        {
            "Workflow": {
                "root_dir": str(tmp_path / "iterations"),
                "initial_diffusion_data": {
                    "dcd_path": str(tmp_path / "initial.dcd"),
                    "topology_path": str(tmp_path / "initial.pdb"),
                },
            },
            "RiteWeight": {"io": {"top": str(tmp_path / "system.psf")}},
        },
    )


def test_iteration_zero_has_distinct_bootstrap_schedule(tmp_path):
    config = resolve_iteration_config(minimal_config(tmp_path), 0)
    assert iteration_steps(config, 0) == [
        "train_diffusion",
        "sample_diffusion",
        "clustering",
        "occupancy",
        "namd",
        "riteweight",
        "fel_estimate",
    ]
    assert config["Generative"]["data"]["dcd_path"].endswith(
        "initial.dcd"
    )
    assert config["Workflow"]["runtime"]["training_riteweight_dir"] is None
    assert config["RiteWeight"]["folders"] == [
        str(tmp_path / "iterations" / "0th.Iteration" / "namd")
    ]


def test_later_iteration_uses_previous_cumulative_data(tmp_path):
    config = resolve_iteration_config(minimal_config(tmp_path), 2)
    assert "train_committor" in iteration_steps(config, 2)
    assert "clustering" not in iteration_steps(config, 2)
    assert config["Generative"]["data"]["dcd_path"].endswith(
        "1st.Iteration/riteweight/diffusion_training.dcd"
    )
    assert config["RiteWeight"]["folders"][-1].endswith("2nd.Iteration/namd")
    assert config["Generative"]["init_checkpoint_path"].endswith(
        "1st.Iteration/models/diffusion/best_model.pt"
    )


def test_iteration_noise_scale_override_and_fallback(tmp_path):
    base = minimal_config(tmp_path)
    base["Generative"]["inference"]["noise_scale"] = 1.25
    base["Workflow"]["iteration_noise_scales"] = {0: 20, "2": 3.5}

    assert resolve_iteration_config(base, 0)["Generative"]["inference"]["noise_scale"] == 20.0
    assert resolve_iteration_config(base, 1)["Generative"]["inference"]["noise_scale"] == 1.25
    assert resolve_iteration_config(base, 2)["Generative"]["inference"]["noise_scale"] == 3.5


@pytest.mark.parametrize("value", [-1, float("inf"), "not-a-number"])
def test_iteration_noise_scale_is_validated(tmp_path, value):
    base = minimal_config(tmp_path)
    base["Workflow"]["iteration_noise_scales"] = {0: value}

    with pytest.raises(ValueError, match="Sampling noise for iteration 0"):
        resolve_iteration_config(base, 0)


def test_iteration_diffusion_epochs_override_and_fallback(tmp_path):
    base = minimal_config(tmp_path)
    base["Generative"]["training"]["epochs"] = 25
    base["Workflow"]["iteration_diffusion_epochs"] = {0: 50, "1": 10, 2: 10}

    assert resolve_iteration_config(base, 0)["Generative"]["training"]["epochs"] == 50
    assert resolve_iteration_config(base, 1)["Generative"]["training"]["epochs"] == 10
    assert resolve_iteration_config(base, 2)["Generative"]["training"]["epochs"] == 10
    assert resolve_iteration_config(base, 3)["Generative"]["training"]["epochs"] == 25


@pytest.mark.parametrize("value", [0, -1, 1.5, "not-an-integer", True])
def test_iteration_diffusion_epochs_is_validated(tmp_path, value):
    base = minimal_config(tmp_path)
    base["Workflow"]["iteration_diffusion_epochs"] = {0: value}

    with pytest.raises(ValueError, match="Diffusion epochs for iteration 0"):
        resolve_iteration_config(base, 0)


def test_warm_start_diffusion_can_be_disabled(tmp_path):
    base = minimal_config(tmp_path)
    base["Workflow"]["warm_start_diffusion"] = False

    assert resolve_iteration_config(base, 1)["Generative"]["init_checkpoint_path"] is None


def test_resume_restarts_interrupted_step_and_skips_completed_steps(tmp_path, monkeypatch):
    base = minimal_config(tmp_path)
    calls = []

    def interrupt_sample(step, effective_config):
        calls.append(step)
        if step == "sample_diffusion":
            raise KeyboardInterrupt()

    monkeypatch.setattr(workflow, "_run_isolated_step", interrupt_sample)
    with pytest.raises(KeyboardInterrupt):
        workflow.run_iteration(base, 0, stop_after="sample_diffusion")

    manifest_path = tmp_path / "iterations" / "0th.Iteration" / "workflow_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["steps"]["train_diffusion"]["status"] == "completed"
    assert manifest["steps"]["sample_diffusion"]["status"] == "interrupted"

    monkeypatch.setattr(
        workflow,
        "_run_isolated_step",
        lambda step, effective_config: calls.append(step),
    )
    workflow.run_iteration(base, 0, stop_after="sample_diffusion", resume=True)

    assert calls == ["train_diffusion", "sample_diffusion", "sample_diffusion"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["steps"]["sample_diffusion"]["status"] == "completed"
    assert manifest["steps"]["sample_diffusion"]["attempts"] == 2


def test_stepwise_stop_preserves_completed_step_for_resume(tmp_path, monkeypatch):
    base = minimal_config(tmp_path)
    calls = []
    monkeypatch.setattr(
        workflow,
        "_run_isolated_step",
        lambda step, effective_config: calls.append(step),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "q")

    with pytest.raises(workflow.WorkflowPaused, match="--resume --stepwise"):
        workflow.run_iteration(
            base,
            0,
            stop_after="sample_diffusion",
            stepwise=True,
        )

    manifest_path = tmp_path / "iterations" / "0th.Iteration" / "workflow_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert calls == ["train_diffusion"]
    assert manifest["steps"]["train_diffusion"]["status"] == "completed"
    assert "sample_diffusion" not in manifest["steps"]

    monkeypatch.setattr("builtins.input", lambda prompt: "")
    workflow.run_iteration(
        base,
        0,
        stop_after="sample_diffusion",
        resume=True,
        stepwise=True,
    )
    assert calls == ["train_diffusion", "sample_diffusion"]


def test_stepwise_requires_interactive_input(tmp_path, monkeypatch):
    base = minimal_config(tmp_path)
    monkeypatch.setattr(workflow, "_run_isolated_step", lambda *args: None)

    def no_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_input)
    with pytest.raises(workflow.WorkflowPaused, match="interactive input is unavailable"):
        workflow.run_iteration(
            base,
            0,
            stop_after="train_diffusion",
            stepwise=True,
        )


def test_run_step_runs_only_requested_incomplete_step(tmp_path, monkeypatch):
    base = minimal_config(tmp_path)
    calls = []
    monkeypatch.setattr(
        workflow,
        "_run_isolated_step",
        lambda step, effective_config: calls.append(step),
    )

    workflow.run_iteration(base, 1, stop_after="train_committor")
    workflow.run_iteration(base, 1, run_step_name="train_diffusion")

    assert calls == ["train_diffusion", "train_committor"]


def test_rerun_step_forces_only_requested_completed_step(tmp_path, monkeypatch):
    base = minimal_config(tmp_path)
    calls = []
    monkeypatch.setattr(
        workflow,
        "_run_isolated_step",
        lambda step, effective_config: calls.append(step),
    )

    workflow.run_iteration(base, 1, stop_after="train_committor")
    workflow.run_iteration(base, 1, resume=True, rerun_step="train_diffusion")

    assert calls == ["train_diffusion", "train_committor", "train_diffusion"]
    manifest_path = tmp_path / "iterations" / "1st.Iteration" / "workflow_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["steps"]["train_diffusion"]["status"] == "completed"
    assert manifest["steps"]["train_diffusion"]["attempts"] == 2
    assert manifest["steps"]["train_committor"]["attempts"] == 1


@pytest.mark.parametrize("step", ["train_committor", "committor_slice"])
@pytest.mark.parametrize("option", ["run_step_name", "rerun_step"])
def test_iteration_zero_rejects_committor_single_steps(tmp_path, step, option):
    with pytest.raises(ValueError, match="committor steps.*iterations 1 and later"):
        workflow.run_iteration(
            minimal_config(tmp_path), 0, dry_run=True, **{option: step}
        )


def test_rerun_step_rejects_step_outside_iteration_schedule(tmp_path):
    with pytest.raises(ValueError, match="not part of this iteration's workflow"):
        workflow.run_iteration(
            minimal_config(tmp_path), 1, rerun_step="clustering", dry_run=True
        )


def test_single_step_rejects_start_and_stop_filters(tmp_path):
    with pytest.raises(ValueError, match="cannot be combined"):
        workflow.run_iteration(
            minimal_config(tmp_path),
            1,
            run_step_name="sample_diffusion",
            start_at="train_diffusion",
            dry_run=True,
        )


def test_run_step_and_rerun_step_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="cannot be combined"):
        workflow.run_iteration(
            minimal_config(tmp_path),
            1,
            run_step_name="train_diffusion",
            rerun_step="train_diffusion",
            dry_run=True,
        )


def test_help_lists_single_step_options_and_all_step_names():
    help_text = workflow.build_parser().format_help()
    assert "--run_step STEP" in help_text
    assert "--rerun_step STEP" in help_text
    assert "--redo" not in help_text
    for step in workflow.STEP_NAMES:
        assert step in help_text


def test_isolated_step_uses_packaged_runner(tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, check, env):
        captured["command"] = command
        captured["check"] = check
        captured["env"] = env
        return workflow.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    effective_config = tmp_path / "effective_config.yaml"
    workflow._run_isolated_step("sample_diffusion", effective_config)

    assert captured["command"][1:3] == ["-m", "common.runner"]
    assert captured["command"][-3:] == [
        "sample_diffusion",
        "--config",
        str(effective_config),
    ]
    assert captured["check"] is False
    assert captured["env"]["PYTHONPATH"].split(workflow.os.pathsep)[0] == str(
        Path(workflow.__file__).resolve().parent
    )


def test_packaged_runner_is_discoverable_from_an_unrelated_cwd(tmp_path):
    module_root = str(Path(workflow.__file__).resolve().parent)
    env = workflow.os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        module_root
        if not existing_pythonpath
        else workflow.os.pathsep.join((module_root, existing_pythonpath))
    )

    completed = workflow.subprocess.run(
        [workflow.sys.executable, "-m", "common.runner", "--help"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Internal isolated-stage runner" in completed.stdout
