#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcn2.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="VCN 2.0 workflow")
    parser.add_argument("command", choices=["train", "path", "cluster", "analyze", "plot", "all"])
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    results = {}
    if args.command in ("train", "all"):
        from vcn2.train import train_from_config

        results["train"] = train_from_config(config)
    if args.command in ("path", "all"):
        from vcn2.path import find_paths_from_config

        results["path"] = find_paths_from_config(config)
    if args.command in ("cluster", "all"):
        from vcn2.cluster import cluster_from_config

        results["cluster"] = cluster_from_config(config)
    if args.command in ("analyze", "all"):
        from vcn2.analysis import analyze_gradients_from_config

        results["analysis"] = analyze_gradients_from_config(config)
    if args.command in ("plot", "all"):
        from vcn2.plot import plot_projection_from_config

        results["plot"] = plot_projection_from_config(config)

    out_dir = Path(config.get("output", {}).get("out_dir", "./vcn2_output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.command}_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
