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
            '"train_committor", "committor_analysis", "clustering", '
            '"occupancy", "reweighting"'
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
        train_committor_model(config["VCN"])
    elif args.step == "committor_analysis":
        from common.vcn_slice import run_committor_analysis
        run_committor_analysis(config["VCN"])
    elif args.step == "clustering":
        from tools.clustering import run_clustering
        run_clustering(config["Clustering"])
    elif args.step == "occupancy":
        from tools.occupancy import add_occupancy
        add_occupancy(config["Occupancy"])
    elif args.step == "reweighting":
        from tools.reweighting import run_reweighting
        run_reweighting(config["Reweighting"])

    # elif args.step == "fel_estimate":
    #     from tools.felestimate import run_fel_estimate
    #     run_fel_estimate(config["FEL_estimate"])

    else:
        raise ValueError(
            f"Unknown step: {args.step}. Choose from "
            "'train_diffusion', 'sample_diffusion', 'train_committor', "
            "'committor_analysis', 'clustering', 'occupancy', 'reweighting'."
        )


if __name__ == "__main__":
    main()
