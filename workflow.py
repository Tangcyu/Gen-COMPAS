"""Iteration-aware orchestration for the Gen-COMPAS sampling loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

from common.config import load_config, resolve_iteration_config, write_effective_config
from common.runner import STEP_NAMES, run_step


BOOTSTRAP_STEPS = (
    "train_diffusion",
    "sample_diffusion",
    "clustering",
    "occupancy",
    "namd",
    "riteweight",
    "fel_estimate",
)

ITERATIVE_STEPS = (
    "train_diffusion",
    "train_committor",
    "sample_diffusion",
    "committor_slice",
    "occupancy",
    "namd",
    "riteweight",
    "fel_estimate",
)


class WorkflowPaused(RuntimeError):
    """Raised when a user deliberately stops an interactive stepwise run."""


def iteration_steps(config: Mapping[str, Any], iteration: int) -> list[str]:
    """Return the ordered stages for bootstrap iteration 0 or iterations 1+."""
    steps = list(BOOTSTRAP_STEPS if iteration == 0 else ITERATIVE_STEPS)
    workflow = config["Workflow"]
    if iteration == 0 and workflow.get("run_initial_unbiased", False):
        steps.insert(0, "initial_unbiased")
    if not workflow.get("run_fel", True):
        steps.remove("fel_estimate")
    return steps


def _selected_steps(
    steps: Sequence[str],
    start_at: Optional[str],
    stop_after: Optional[str],
) -> list[str]:
    if start_at is not None and start_at not in steps:
        raise ValueError(f"start-at step {start_at!r} is not in this iteration: {steps}")
    if stop_after is not None and stop_after not in steps:
        raise ValueError(f"stop-after step {stop_after!r} is not in this iteration: {steps}")
    start = steps.index(start_at) if start_at is not None else 0
    stop = steps.index(stop_after) + 1 if stop_after is not None else len(steps)
    if start >= stop:
        raise ValueError("--start-at must occur before or equal to --stop-after.")
    return list(steps[start:stop])


def _validate_single_step(
    step: str,
    all_steps: Sequence[str],
    iteration: int,
    option: str,
) -> None:
    """Validate a run/rerun request against the resolved iteration schedule."""
    if step not in STEP_NAMES:
        raise ValueError(
            f"Unknown {option} step {step!r}; choose from {', '.join(STEP_NAMES)}."
        )
    if step in all_steps:
        return
    if iteration == 0 and step in ("train_committor", "committor_slice"):
        raise ValueError(
            f"Cannot {option} {step!r} in iteration 0: committor steps are "
            "only part of iterations 1 and later."
        )
    raise ValueError(
        f"Cannot {option} {step!r} in iteration {iteration}: the step is not "
        f"part of this iteration's workflow ({' -> '.join(all_steps)})."
    )


def _read_manifest(path: Path, iteration: int) -> dict:
    if not path.exists():
        return {"iteration": iteration, "steps": {}}
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("iteration") != iteration:
        raise ValueError(f"Manifest iteration mismatch in {path}.")
    manifest.setdefault("steps", {})
    return manifest


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically persist workflow state so an interruption cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_special_step(step: str, config: Mapping[str, Any]):
    runtime = config["Workflow"]["runtime"]
    if step == "initial_unbiased":
        from tools.namd import run_namd_workflow
        return run_namd_workflow(runtime["initial_namd"])
    return run_step(step, config)


def _run_isolated_step(step: str, effective_config: Path) -> None:
    """Run a regular stage in a clean runtime and propagate any failure.

    Prepending this source/install directory to PYTHONPATH keeps direct
    ``python /path/to/workflow.py`` execution working from any caller cwd.
    The cwd itself is deliberately unchanged because user configuration may
    contain paths relative to the directory where the workflow was launched.
    """
    module_root = str(Path(__file__).resolve().parent)
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        module_root
        if not existing_pythonpath
        else os.pathsep.join((module_root, existing_pythonpath))
    )
    command = [
        sys.executable,
        "-m",
        "common.runner",
        "--step",
        step,
        "--config",
        str(effective_config),
    ]
    completed = subprocess.run(command, check=False, env=child_env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Workflow step {step!r} exited with status {completed.returncode}."
        )


def _stepwise_pause(iteration: int, step: str, iteration_dir: Path) -> None:
    """Wait for explicit confirmation after a completed workflow stage."""
    prompt = (
        f"\n[stepwise] Iteration {iteration} step {step!r} completed.\n"
        f"Inspect outputs under {iteration_dir}.\n"
        "Press Enter to continue, or enter q to stop: "
    )
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError as exc:
            raise WorkflowPaused(
                "Stepwise workflow paused because interactive input is unavailable. "
                "Rerun it in a terminal with --resume --stepwise."
            ) from exc
        if answer in ("", "c", "continue", "y", "yes"):
            return
        if answer in ("q", "quit", "stop", "n", "no"):
            raise WorkflowPaused(
                f"Stepwise workflow stopped after iteration {iteration} step {step!r}. "
                "Rerun the same iteration list with --resume --stepwise to continue."
            )
        print("Please press Enter to continue or enter q to stop.")


def run_iteration(
    base_config: Mapping[str, Any],
    iteration: int,
    *,
    start_at: Optional[str] = None,
    stop_after: Optional[str] = None,
    resume: bool = False,
    stepwise: bool = False,
    run_step_name: Optional[str] = None,
    rerun_step: Optional[str] = None,
    dry_run: bool = False,
) -> list[str]:
    """Run one complete bootstrap or committor-guided iteration."""
    config = resolve_iteration_config(base_config, iteration)
    all_steps = iteration_steps(config, iteration)
    single_steps = [step for step in (run_step_name, rerun_step) if step is not None]
    if len(single_steps) > 1:
        raise ValueError("--run_step and --rerun_step cannot be combined.")
    selected_step = single_steps[0] if single_steps else None
    if selected_step is not None:
        if start_at is not None or stop_after is not None:
            raise ValueError(
                "--run_step/--rerun_step cannot be combined with "
                "--start-at or --stop-after."
            )
        option = "rerun" if rerun_step is not None else "run"
        _validate_single_step(selected_step, all_steps, iteration, option)
        steps = [selected_step]
    else:
        steps = _selected_steps(all_steps, start_at, stop_after)
    iteration_dir = Path(config["Workflow"]["runtime"]["iteration_dir"])

    print(f"Iteration {iteration}: {' -> '.join(steps)}")
    if run_step_name is not None:
        print(f"[run_step] Selected step: {run_step_name}")
    if rerun_step is not None:
        print(f"[rerun_step] Forcing step to run again: {rerun_step}")
    if dry_run:
        print(f"Resolved iteration directory: {iteration_dir}")
        print(
            "Resolved sampling noise: "
            f"{config['Generative']['inference']['noise_scale']}"
        )
        print(
            "Resolved diffusion epochs: "
            f"{config['Generative']['training']['epochs']}"
        )
        init_checkpoint = config["Generative"].get("init_checkpoint_path")
        print(
            "Diffusion initialization: "
            f"{init_checkpoint if init_checkpoint else 'new model'}"
        )
        return steps

    iteration_dir.mkdir(parents=True, exist_ok=True)
    effective_config = iteration_dir / "effective_config.yaml"
    write_effective_config(config, effective_config)
    manifest_path = iteration_dir / "workflow_manifest.json"
    manifest = _read_manifest(manifest_path, iteration)

    for step in steps:
        previous = manifest["steps"].get(step, {})
        if run_step_name is not None and previous.get("status") == "completed":
            print(
                f"[run_step] Step is already completed: {step}. "
                f"Use --rerun_step {step} to force it to run again."
            )
            continue
        if resume and rerun_step is None and previous.get("status") == "completed":
            print(f"[resume] Skipping completed step: {step}")
            continue
        if resume and rerun_step is None and previous:
            print(
                f"[resume] Restarting incomplete step: {step} "
                f"(previous status: {previous.get('status', 'unknown')})"
            )

        started = datetime.now(timezone.utc).isoformat()
        attempts = int(previous.get("attempts", 0)) + 1
        manifest["steps"][step] = {
            "status": "running",
            "started": started,
            "attempts": attempts,
        }
        _write_manifest(manifest_path, manifest)
        print(f"\n=== Iteration {iteration}: {step} ===")
        try:
            if (
                config["Workflow"].get("isolate_steps", True)
                and step != "initial_unbiased"
            ):
                _run_isolated_step(step, effective_config)
            else:
                _run_special_step(step, config)
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["steps"][step].update(
                status=status,
                finished=datetime.now(timezone.utc).isoformat(),
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_manifest(manifest_path, manifest)
            raise
        manifest["steps"][step].update(
            status="completed", finished=datetime.now(timezone.utc).isoformat()
        )
        _write_manifest(manifest_path, manifest)
        if stepwise:
            _stepwise_pause(iteration, step, iteration_dir)

    print(f"Iteration {iteration} completed. Manifest: {manifest_path}")
    return steps


def build_parser() -> argparse.ArgumentParser:
    step_names = ", ".join(STEP_NAMES)
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            f"Available step names:\n  {step_names}\n\n"
            f"Iteration 0:\n  {' -> '.join(BOOTSTRAP_STEPS)}\n\n"
            f"Iterations 1+:\n  {' -> '.join(ITERATIVE_STEPS)}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Minimal or full YAML config")
    parser.add_argument(
        "--iteration",
        type=int,
        nargs="+",
        required=True,
        help="One or more iterations to run in order, for example: 0 1 2",
    )
    parser.add_argument("--start-at", help="Start at this step")
    parser.add_argument("--stop-after", help="Stop after this step")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed steps and restart the last interrupted or failed step",
    )
    parser.add_argument(
        "--stepwise",
        action="store_true",
        help="Pause after every completed step for output inspection",
    )
    single_step_group = parser.add_mutually_exclusive_group()
    single_step_group.add_argument(
        "--run_step",
        choices=STEP_NAMES,
        metavar="STEP",
        help="Run one incomplete workflow step for each requested iteration",
    )
    single_step_group.add_argument(
        "--rerun_step",
        choices=STEP_NAMES,
        metavar="STEP",
        help="Force one workflow step to run again for each requested iteration",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print steps and derived paths only")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    selected_step = args.run_step or args.rerun_step
    if selected_step is not None and (
        args.start_at is not None or args.stop_after is not None
    ):
        parser.error(
            "--run_step/--rerun_step cannot be combined with "
            "--start-at or --stop-after"
        )
    config = load_config(args.config)
    try:
        for iteration in args.iteration:
            try:
                run_iteration(
                    config,
                    iteration,
                    start_at=args.start_at,
                    stop_after=args.stop_after,
                    resume=args.resume,
                    stepwise=args.stepwise,
                    run_step_name=args.run_step,
                    rerun_step=args.rerun_step,
                    dry_run=args.dry_run,
                )
            except ValueError as exc:
                if selected_step is not None:
                    parser.error(str(exc))
                raise
    except WorkflowPaused as exc:
        print(f"\n{exc}")


if __name__ == "__main__":
    main()
