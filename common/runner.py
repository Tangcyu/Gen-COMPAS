"""Lazy step dispatcher and internal isolated-stage command."""

from __future__ import annotations

import argparse
from typing import Any, Mapping


STEP_NAMES = (
    "train_diffusion",
    "sample_diffusion",
    "train_committor",
    "committor_slice",
    "clustering",
    "occupancy",
    "namd",
    "riteweight",
    "fel_estimate",
)


def run_step(step: str, config: Mapping[str, Any]):
    """Run one configured Gen-COMPAS step and return its outputs."""
    if step == "train_diffusion":
        from common.diffusion_train import train_diffusion_model
        return train_diffusion_model(config["Generative"])
    if step == "sample_diffusion":
        from common.diffusion_sample import run_diffusion_inference
        return run_diffusion_inference(config["Generative"])
    if step == "train_committor":
        from common.vcn_train import train_committor_model
        return train_committor_model(config["VCN"], config["RiteWeight"])
    if step == "committor_slice":
        from common.vcn_slice import run_committor_slice
        return run_committor_slice(config["VCN"], config["RiteWeight"])
    if step == "clustering":
        from tools.clustering import run_clustering
        return run_clustering(config["Clustering"])
    if step == "occupancy":
        from tools.occupancy import add_occupancy
        return add_occupancy(config["Occupancy"])
    if step == "namd":
        from tools.namd import run_namd_workflow
        return run_namd_workflow(config["NAMD"])
    if step == "riteweight":
        from tools.riteweight import run_riteweight
        return run_riteweight(config["RiteWeight"])
    if step == "fel_estimate":
        from tools.felestimate import run_fel_estimate
        return run_fel_estimate(config["FEL_estimate"])
    raise ValueError(f"Unknown step {step!r}; choose from {', '.join(STEP_NAMES)}.")


def main(argv=None):
    """Run one stage from an already resolved effective configuration."""
    parser = argparse.ArgumentParser(
        description="Internal isolated-stage runner for the Gen-COMPAS workflow."
    )
    parser.add_argument("--step", required=True, choices=STEP_NAMES)
    parser.add_argument("--config", required=True, help="Resolved YAML configuration")
    args = parser.parse_args(argv)

    from common.config import load_config

    return run_step(args.step, load_config(args.config))


if __name__ == "__main__":
    main()
