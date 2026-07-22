"""Legacy standalone single-stage CLI, retained for reproducibility.

The supported installed interface is now ``gen-compas``, which dispatches to
``workflow.py`` and provides ``--run_step`` and ``--rerun_step``.
"""

from __future__ import annotations

import argparse

from common.config import load_config, resolve_iteration_config
from common.runner import STEP_NAMES, run_step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one step of the Gen-COMPAS workflow."
    )
    parser.add_argument("--step", required=True, choices=STEP_NAMES)
    parser.add_argument("--config", required=True, help="Full or minimal YAML config")
    parser.add_argument(
        "--iteration",
        type=int,
        help="Resolve iteration-managed input/output paths before running the step",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.iteration is not None:
        config = resolve_iteration_config(config, args.iteration)
    return run_step(args.step, config)


if __name__ == "__main__":
    main()
