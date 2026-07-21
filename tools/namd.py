#!/usr/bin/env python3
"""Run isolated, parallel TMD -> unbiased NAMD workflows."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml


@dataclass(frozen=True)
class Protocol:
    """The TMD and unbiased templates for one starting-state protocol."""

    name: str
    tmd_template: str
    unbiased_template: str


@dataclass(frozen=True)
class NAMDJob:
    """A prepared target/protocol simulation directory."""

    name: str
    target: Path
    protocol: Protocol
    work_dir: Path


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least 1, got {value!r}.")
    return result


def _resolve_executable(value: str) -> str:
    """Resolve either an explicit NAMD path or a command available on PATH."""
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    has_path_separator = os.sep in expanded or (os.altsep and os.altsep in expanded)
    if has_path_separator:
        executable = Path(expanded).resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"NAMD executable does not exist: {executable}")
        if not os.access(executable, os.X_OK):
            raise PermissionError(f"NAMD executable is not executable: {executable}")
        return str(executable)

    executable = shutil.which(expanded)
    if executable is None:
        raise FileNotFoundError(f"NAMD command was not found on PATH: {expanded}")
    return executable


def _configured_path(value: Any) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    return Path(expanded).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _load_protocols(config: Mapping[str, Any], template_dir: Path) -> List[Protocol]:
    raw_protocols = config.get("protocols")
    if not isinstance(raw_protocols, list) or not raw_protocols:
        raise ValueError("NAMD.protocols must contain at least one protocol.")

    protocols = []
    names = set()
    for raw in raw_protocols:
        if not isinstance(raw, Mapping):
            raise TypeError("Each NAMD protocol must be a mapping.")
        protocol = Protocol(
            name=str(raw["name"]),
            tmd_template=str(raw["tmd_template"]),
            unbiased_template=str(raw["unbiased_template"]),
        )
        if protocol.name in names:
            raise ValueError(f"Duplicate NAMD protocol name: {protocol.name!r}")
        names.add(protocol.name)

        for relative_name in (protocol.tmd_template, protocol.unbiased_template):
            source = (template_dir / relative_name).resolve()
            if not _is_within(source, template_dir):
                raise ValueError(f"NAMD template escapes template_path: {relative_name}")
            if not source.is_file():
                raise FileNotFoundError(f"NAMD template file does not exist: {source}")
        protocols.append(protocol)
    return protocols


def _discover_targets(config: Mapping[str, Any]) -> tuple[Path, List[Path], str]:
    target_config = config.get("targets", {})
    target_dir = _configured_path(target_config["path"])
    if not target_dir.is_dir():
        raise NotADirectoryError(f"NAMD target path is not a directory: {target_dir}")

    pattern = str(target_config.get("pattern", "*.pdb"))
    recursive = bool(target_config.get("recursive", False))
    candidates = target_dir.rglob(pattern) if recursive else target_dir.glob(pattern)
    targets = sorted(path.resolve() for path in candidates if path.is_file())
    if not targets:
        raise FileNotFoundError(
            f"No NAMD targets matched {pattern!r} under {target_dir}."
        )

    target_filename = str(target_config.get("target_filename", "output.pdb"))
    if Path(target_filename).name != target_filename or not target_filename:
        raise ValueError("NAMD.targets.target_filename must be a plain filename.")
    if any(character.isspace() for character in target_filename):
        raise ValueError("NAMD.targets.target_filename cannot contain whitespace.")
    return target_dir, targets, target_filename


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not result:
        raise ValueError(f"Cannot derive a safe NAMD job name from {value!r}.")
    return result


def _replace_directive(text: str, directive: str, value: str) -> str:
    pattern = re.compile(
        rf"^([ \t]*{re.escape(directive)}[ \t]+)(\S+)([^\r\n]*)$",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if len(pattern.findall(text)) != 1:
        raise ValueError(
            f"TMD template must contain exactly one active {directive} directive "
            f"or its corresponding placeholder."
        )
    return pattern.sub(rf"\g<1>{value}\g<3>", text, count=1)


def _render_template(
    source: Path,
    destination: Path,
    replacements: Mapping[str, str],
    *,
    is_tmd: bool,
) -> None:
    text = source.read_text(encoding="utf-8")
    for name, value in replacements.items():
        text = text.replace("{{" + name + "}}", value)

    if is_tmd:
        text = _replace_directive(
            text, "TMDk", replacements["TMD_FORCE_CONSTANT"]
        )
        text = _replace_directive(text, "TMDFile", replacements["TARGET_PDB"])

    unresolved = sorted(set(re.findall(r"\{\{([A-Z][A-Z0-9_]*)\}\}", text)))
    if unresolved:
        raise ValueError(
            f"Unresolved placeholders in {source}: {', '.join(unresolved)}"
        )
    destination.write_text(text, encoding="utf-8")


def _prepare_jobs(
    *,
    template_dir: Path,
    output_dir: Path,
    target_dir: Path,
    targets: Sequence[Path],
    target_filename: str,
    protocols: Sequence[Protocol],
    force_constant: float,
    existing_policy: str,
) -> tuple[List[NAMDJob], List[Dict[str, Any]]]:
    specifications = []
    names = set()
    for target in targets:
        relative_stem = target.relative_to(target_dir).with_suffix("").as_posix()
        for protocol in protocols:
            name = _safe_name(f"{relative_stem}__{protocol.name}")
            if name in names:
                raise ValueError(f"NAMD job-name collision: {name}")
            names.add(name)
            specifications.append((name, target, protocol, output_dir / name))

    if existing_policy not in ("error", "skip"):
        raise ValueError("NAMD.existing_job_policy must be 'error' or 'skip'.")
    existing = [work_dir for _, _, _, work_dir in specifications if work_dir.exists()]
    if existing and existing_policy == "error":
        preview = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(
            f"NAMD job directories already exist ({preview}). "
            "Choose a new output_dir or set existing_job_policy: skip."
        )

    jobs = []
    skipped = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, target, protocol, work_dir in specifications:
        if work_dir.exists():
            skipped.append(
                {"job": name, "status": "skipped_existing", "work_dir": str(work_dir)}
            )
            continue

        shutil.copytree(template_dir, work_dir)
        shutil.copy2(target, work_dir / target_filename)
        replacements = {
            "TMD_FORCE_CONSTANT": f"{force_constant:g}",
            "TARGET_PDB": target_filename,
            "TARGET_SOURCE": str(target),
            "JOB_NAME": name,
            "JOB_DIR": str(work_dir),
            "PROTOCOL": protocol.name,
        }
        _render_template(
            template_dir / protocol.tmd_template,
            (work_dir / protocol.tmd_template).resolve(),
            replacements,
            is_tmd=True,
        )
        _render_template(
            template_dir / protocol.unbiased_template,
            (work_dir / protocol.unbiased_template).resolve(),
            replacements,
            is_tmd=False,
        )
        jobs.append(NAMDJob(name, target, protocol, work_dir))
    return jobs, skipped


def _build_command(
    executable: str,
    execution: Mapping[str, Any],
    mode: str,
    config_filename: str,
    device: Optional[str],
) -> List[str]:
    mode_config = execution.get(mode, {})
    threads = _positive_int(mode_config.get("threads_per_job", 1), "threads_per_job")
    default = (
        ["{namd}", "+p{threads}", "{config}"]
        if mode == "cpu"
        else ["{namd}", "+p{threads}", "+devices", "{device}", "{config}"]
    )
    command_template = mode_config.get("command", default)
    if not isinstance(command_template, list) or not command_template:
        raise ValueError(f"NAMD.execution.{mode}.command must be a non-empty list.")
    command_text = "\0".join(str(token) for token in command_template)
    required_placeholders = ["{namd}", "{config}"]
    if mode == "gpu":
        required_placeholders.append("{device}")
    missing = [item for item in required_placeholders if item not in command_text]
    if missing:
        raise ValueError(
            f"NAMD.execution.{mode}.command is missing required placeholders: "
            + ", ".join(missing)
        )

    values = {
        "namd": executable,
        "threads": str(threads),
        "device": "" if device is None else device,
        "config": config_filename,
    }
    try:
        return [str(token).format_map(values) for token in command_template]
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder in NAMD.execution.{mode}.command: {exc.args[0]}"
        ) from exc


def _run_phase(
    job: NAMDJob,
    phase: str,
    command: Sequence[str],
    dry_run: bool,
) -> int:
    log_path = job.work_dir / f"{phase}.log"
    command_text = shlex.join(command)
    print(f"[{job.name}] {phase}: {command_text}")
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# command: {command_text}\n")
        log.write(f"# working_directory: {job.work_dir}\n")
        if dry_run:
            log.write("# dry run: command was not executed\n")
            return 0
        completed = subprocess.run(
            list(command),
            cwd=job.work_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def _run_job(
    job: NAMDJob,
    executable: str,
    execution: Mapping[str, Any],
    mode: str,
    device: Optional[str],
    run_tmd: bool,
    run_unbiased: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "job": job.name,
        "protocol": job.protocol.name,
        "target": str(job.target),
        "work_dir": str(job.work_dir),
        "mode": mode,
        "device": device,
    }
    phases = []
    if run_tmd:
        phases.append(("tmd", job.protocol.tmd_template))
    if run_unbiased:
        phases.append(("unbiased", job.protocol.unbiased_template))

    try:
        for phase, config_filename in phases:
            command = _build_command(
                executable, execution, mode, config_filename, device
            )
            return_code = _run_phase(job, phase, command, dry_run)
            if return_code != 0:
                result.update(
                    status="failed", phase=phase, return_code=return_code
                )
                return result
    except OSError as exc:
        result.update(status="failed", phase="launch", error=str(exc))
        return result

    result["status"] = "dry_run" if dry_run else "completed"
    return result


def run_namd_workflow(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Prepare and run all configured TMD -> unbiased NAMD jobs."""
    executable = _resolve_executable(str(config["namd_path"]))
    template_dir = _configured_path(config["template_path"])
    if not template_dir.is_dir():
        raise NotADirectoryError(f"NAMD template_path is not a directory: {template_dir}")

    output_dir = _configured_path(config.get("output_dir", "./output_namd"))
    if _is_within(output_dir, template_dir):
        raise ValueError("NAMD.output_dir cannot be inside NAMD.template_path.")

    force_constant = float(config["tmd_force_constant"])
    if not math.isfinite(force_constant) or force_constant <= 0:
        raise ValueError("NAMD.tmd_force_constant must be finite and positive.")

    execution = config.get("execution", {})
    mode = str(execution.get("device", "cpu")).lower()
    if mode not in ("cpu", "gpu"):
        raise ValueError("NAMD.execution.device must be 'cpu' or 'gpu'.")
    parallel_jobs = _positive_int(
        execution.get("parallel_jobs", 1), "NAMD.execution.parallel_jobs"
    )
    devices: List[Optional[str]] = [None]
    if mode == "gpu":
        raw_devices = execution.get("gpu", {}).get("devices", [])
        if not isinstance(raw_devices, list) or not raw_devices:
            raise ValueError("NAMD.execution.gpu.devices must be a non-empty list.")
        devices = [str(device) for device in raw_devices]
        if parallel_jobs > len(devices):
            raise ValueError(
                "NAMD.execution.parallel_jobs cannot exceed the number of GPU device slots."
            )

    protocols = _load_protocols(config, template_dir)
    target_dir, targets, target_filename = _discover_targets(config)
    phase_config = config.get("phases", {})
    run_tmd = bool(phase_config.get("tmd", True))
    run_unbiased = bool(phase_config.get("unbiased", True))
    if not run_tmd and not run_unbiased:
        raise ValueError("At least one NAMD phase must be enabled.")
    dry_run = bool(execution.get("dry_run", False))

    # Validate command templates before creating any job directories.
    validation_device = devices[0] if mode == "gpu" else None
    for protocol in protocols:
        if run_tmd:
            _build_command(
                executable, execution, mode, protocol.tmd_template, validation_device
            )
        if run_unbiased:
            _build_command(
                executable,
                execution,
                mode,
                protocol.unbiased_template,
                validation_device,
            )

    jobs, results = _prepare_jobs(
        template_dir=template_dir,
        output_dir=output_dir,
        target_dir=target_dir,
        targets=targets,
        target_filename=target_filename,
        protocols=protocols,
        force_constant=force_constant,
        existing_policy=str(config.get("existing_job_policy", "error")).lower(),
    )

    device_locks = {
        device: threading.Lock() for device in devices if device is not None
    }

    def execute(index: int, job: NAMDJob) -> Dict[str, Any]:
        device = devices[index % len(devices)] if mode == "gpu" else None
        if device is None:
            return _run_job(
                job, executable, execution, mode, device,
                run_tmd, run_unbiased, dry_run,
            )
        with device_locks[device]:
            return _run_job(
                job, executable, execution, mode, device,
                run_tmd, run_unbiased, dry_run,
            )

    with ThreadPoolExecutor(max_workers=parallel_jobs) as pool:
        future_jobs = {
            pool.submit(execute, index, job): job for index, job in enumerate(jobs)
        }
        for future in as_completed(future_jobs):
            results.append(future.result())

    results.sort(key=lambda item: item["job"])
    summary_path = output_dir / "namd_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    failed = [result for result in results if result["status"] == "failed"]
    completed = [result for result in results if result["status"] in ("completed", "dry_run")]
    print(
        f"NAMD workflow finished: {len(completed)} completed, "
        f"{len(failed)} failed, {len(results) - len(completed) - len(failed)} skipped."
    )
    print(f"Summary: {summary_path}")
    if failed:
        names = ", ".join(result["job"] for result in failed)
        raise RuntimeError(f"NAMD jobs failed: {names}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parallel TMD and unbiased NAMD jobs")
    parser.add_argument("--config", required=True, help="Path to Gen-COMPAS YAML config")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_namd_workflow(config["NAMD"])


if __name__ == "__main__":
    main()
