import argparse
import os

import yaml


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate protein structures using the Gen-COMPAS workflow."
    )
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        help=(
            'Step to run: "train_diffusion", "sample_diffusion", '
            '"train_committor", "committor_slice", "clustering", '
            '"occupancy", "namd", "riteweight", "fel_estimate"'
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.step == "train_diffusion":
        from common.diffusion_train import train_diffusion_model
        train_diffusion_model(config["Generative"])
    elif args.step == "sample_diffusion":
        from common.diffusion_sample import run_diffusion_inference
        run_diffusion_inference(config["Generative"])
    elif args.step == "train_committor":
        from common.vcn_train import train_committor_model
        train_committor_model(config["VCN"], config["RiteWeight"])
    elif args.step == "committor_slice":
        from common.vcn_slice import run_committor_slice
        run_committor_slice(config["VCN"], config["RiteWeight"])
    elif args.step == "clustering":
        from tools.clustering import run_clustering
        run_clustering(config["Clustering"])
    elif args.step == "occupancy":
        from tools.occupancy import add_occupancy
        add_occupancy(config["Occupancy"])
    elif args.step == "namd":
        from tools.namd import run_namd_workflow
        run_namd_workflow(config["NAMD"])
    elif args.step == "riteweight":
        from tools.riteweight import run_riteweight
        # Prefer the unified Gen-COMPAS config, while still accepting a
        # RiteWeight-only mapping for programmatic use.
        riteweight_config = config.get("RiteWeight")
        if riteweight_config is None:
            required_keys = {"folders", "io", "colvars", "features", "riteweight"}
            if required_keys.issubset(config):
                riteweight_config = config
            else:
                raise KeyError(
                    "RiteWeight step requires a 'RiteWeight' config section."
                )
        run_riteweight(riteweight_config)
    elif args.step == "fel_estimate":
        from tools.felestimate import run_fel_estimate
        run_fel_estimate(config["FEL_estimate"])

    else:
        raise ValueError(
            f"Unknown step: {args.step}. Choose from "
            "'train_diffusion', 'sample_diffusion', 'train_committor', "
            "'committor_slice', 'clustering', 'occupancy', 'namd', 'riteweight', "
            "'fel_estimate'."
        )


if __name__ == "__main__":
    main()
