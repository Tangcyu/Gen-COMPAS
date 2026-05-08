import argparse
import os

import yaml

DIFFUSION_STEPS = ("train_diffusion", "sample_diffusion")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and sample the Gen-COMPAS diffusion model."
    )
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=DIFFUSION_STEPS,
        help='Step to run: "train_diffusion" or "sample_diffusion".',
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

    generative_config = config["Generative"]

    if args.step == "train_diffusion":
        from common.diffusion_train import train_diffusion_model

        train_diffusion_model(generative_config)
    elif args.step == "sample_diffusion":
        from common.diffusion_sample import run_diffusion_inference

        run_diffusion_inference(generative_config)


if __name__ == "__main__":
    main()
