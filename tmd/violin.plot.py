#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load_csvs_sequential(csv_files):

    dfs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        system_name = Path(csv_file).stem
        df["system"] = system_name

        dfs.append(df)

    return dfs


def plot_violin(dfs, output, title):

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = []
    data = []

    pos = 1
    spacing_within = 0.8
    spacing_between = 1.6

    xticks = []
    xticklabels = []

    # Colors
    color_initial = "#4C72B0"
    color_final = "#DD8452"

    colors = []

    for df in dfs:

        system = df["system"].iloc[0]

        first_vals = df[df["segment"] == "first"]["rmsd_to_target"].values
        last_vals = df[df["segment"] == "last"]["rmsd_to_target"].values

        pos_first = pos
        pos_last = pos + spacing_within

        if len(first_vals) > 0:
            data.append(first_vals)
            positions.append(pos_first)
            colors.append(color_initial)

        if len(last_vals) > 0:
            data.append(last_vals)
            positions.append(pos_last)
            colors.append(color_final)

        xticks.append((pos_first + pos_last) / 2)
        xticklabels.append(system)

        pos += spacing_within + spacing_between

    vp = ax.violinplot(
        data,
        positions=positions,
        widths=0.6,
        showmedians=True,
    )

    # Color the violins
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.8)

    # Format axes
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    ax.set_ylabel("RMSD to target (Å)")
    ax.set_title(title)

    # Legend
    legend_elements = [
        Patch(facecolor=color_initial, edgecolor="black", label="Initial (first 2%)"),
        Patch(facecolor=color_final, edgecolor="black", label="Final (last 2%)"),
    ]

    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(output, dpi=300)

    print(f"Saved plot to {output}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv_files",
        nargs="+",
        help="CSV files (order preserved)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="tmd_rmsd_violin.png",
    )

    parser.add_argument(
        "--title",
        default="TMD RMSD distributions: first vs last 2%",
    )

    args = parser.parse_args()

    dfs = load_csvs_sequential(args.csv_files)

    plot_violin(dfs, args.output, args.title)


if __name__ == "__main__":
    main()