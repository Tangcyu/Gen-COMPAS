#!/usr/bin/env python3
import os
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_paired_csvs(root_dir):
    """
    Recursively find folders containing both TMD.A.csv and TMD.B.csv.
    Returns list of tuples: (folder, a_csv, b_csv)
    """
    a_files = glob.glob(os.path.join(root_dir, "**", "TMD_noh.A.csv"), recursive=True)
    pairs = []

    for a_csv in sorted(a_files):
        folder = os.path.dirname(os.path.abspath(a_csv))
        b_csv = os.path.join(folder, "TMD_noh.B.csv")
        if os.path.exists(b_csv):
            pairs.append((folder, a_csv, b_csv))

    return pairs


def get_edge_fraction(arr, fraction=0.02, side="first"):
    """
    Return first or last fraction of an array.
    Always return at least 1 frame if the array is non-empty.
    """
    n = len(arr)
    if n == 0:
        return np.array([])

    m = max(1, int(math.ceil(fraction * n)))

    if side == "first":
        return arr[:m]
    elif side == "last":
        return arr[-m:]
    else:
        raise ValueError("side must be 'first' or 'last'")


def compute_deltaq_for_pair(a_csv, b_csv, fraction=0.02):
    """
    Compute:
      delta_q_start = q_A(first 2%) - q_B(first 2%)
      delta_q_end   = q_A(last 2%)  - q_B(last 2%)

    If A and B edge segments differ in length, truncate to the smaller length.
    """
    df_a = pd.read_csv(a_csv)
    df_b = pd.read_csv(b_csv)

    if "q" not in df_a.columns:
        raise ValueError(f"'q' column not found in {a_csv}")
    if "q" not in df_b.columns:
        raise ValueError(f"'q' column not found in {b_csv}")

    q_a = df_a["q"].to_numpy()
    q_b = df_b["q"].to_numpy()

    # First 2%
    q_a_first = get_edge_fraction(q_a, fraction=fraction, side="first")
    q_b_first = get_edge_fraction(q_b, fraction=fraction, side="first")
    n_first = min(len(q_a_first), len(q_b_first))
    if n_first == 0:
        delta_q_start = np.array([])
    else:
        delta_q_start = q_a_first[:n_first] - q_b_first[:n_first]

    # Last 2%
    q_a_last = get_edge_fraction(q_a, fraction=fraction, side="last")
    q_b_last = get_edge_fraction(q_b, fraction=fraction, side="last")
    n_last = min(len(q_a_last), len(q_b_last))
    if n_last == 0:
        delta_q_end = np.array([])
    else:
        delta_q_end = q_a_last[-n_last:] - q_b_last[-n_last:]
    summary = {
        "n_A_total": len(q_a),
        "n_B_total": len(q_b),
        "n_first_used": n_first,
        "n_last_used": n_last,
        "mean_deltaq_start": float(np.mean(delta_q_start)) if n_first > 0 else np.nan,
        "std_deltaq_start": float(np.std(delta_q_start, ddof=1)) if n_first > 1 else np.nan,
        "mean_deltaq_end": float(np.mean(delta_q_end)) if n_last > 0 else np.nan,
        "std_deltaq_end": float(np.std(delta_q_end, ddof=1)) if n_last > 1 else np.nan,
    }

    return delta_q_start, delta_q_end, summary


def save_pair_deltaq_csv(folder, delta_q_start, delta_q_end):
    """
    Save per-folder delta q arrays.
    """
    max_len = max(len(delta_q_start), len(delta_q_end))
    out_df = pd.DataFrame({
        "delta_q_first_2pct": pd.Series(delta_q_start),
        "delta_q_last_2pct": pd.Series(delta_q_end),
    })
    out_csv = os.path.join(folder, "deltaq_first_last_2pct.csv")
    out_df.to_csv(out_csv, index=False)
    return out_csv


def plot_box_deltaq(deltaq_start_all, deltaq_end_all, out_png):

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(6,6))

    data = [deltaq_start_all, deltaq_end_all]
    colors = ["tab:blue", "tab:orange"]

    box = ax.boxplot(
        data,
        widths=0.5,
        patch_artist=True,
        showfliers=False
    )

    # colour boxes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # compute mean and median
    means = [np.mean(d) for d in data]
    medians = [np.median(d) for d in data]

    x = [1,2]

    # mean markers (diamond)
    ax.scatter(
        x,
        means,
        marker='D',
        color='black',
        s=80,
        zorder=3
    )

    # median markers (square)
    ax.scatter(
        x,
        medians,
        marker='s',
        color='white',
        edgecolor='black',
        s=80,
        zorder=3
    )

    ax.set_xticks([1,2])
    ax.set_xticklabels(["First 2%", "Last 2%"])

    ax.set_ylabel(r"$\delta q = q_A - q_B$")
    ax.set_title(r"Distribution of $\delta q$")

    ax.axhline(0, linestyle="--", color="gray")

    ax.grid(alpha=0.3)

    # legend
    legend_elements = [
        Line2D([0],[0], marker='D', color='w', label='Mean',
               markerfacecolor='black', markersize=8),
        Line2D([0],[0], marker='s', color='w', label='Median',
               markerfacecolor='white', markeredgecolor='black', markersize=8)
    ]

    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main(root_dir, fraction=0.02):
    pairs = find_paired_csvs(root_dir)

    if not pairs:
        print(f"No paired TMD.A.csv / TMD.B.csv found under: {root_dir}")
        return

    deltaq_start_all = []
    deltaq_end_all = []
    summary_rows = []

    for folder, a_csv, b_csv in pairs:
        try:
            delta_q_start, delta_q_end, summary = compute_deltaq_for_pair(
                a_csv, b_csv, fraction=fraction
            )

            if len(delta_q_start) > 0 or len(delta_q_end) > 0:
                save_pair_deltaq_csv(folder, delta_q_start, delta_q_end)

            deltaq_start_all.extend(delta_q_start.tolist())
            deltaq_end_all.extend(delta_q_end.tolist())

            row = {
                "folder": folder,
                "a_csv": a_csv,
                "b_csv": b_csv,
            }
            row.update(summary)
            summary_rows.append(row)

            print(
                f"[OK] {folder} | "
                f"n_first={summary['n_first_used']}, "
                f"n_last={summary['n_last_used']}, "
                f"mean_start={summary['mean_deltaq_start']:.6e}, "
                f"mean_end={summary['mean_deltaq_end']:.6e}"
            )

        except Exception as e:
            print(f"[ERROR] Failed on {folder}")
            print(f"        {type(e).__name__}: {e}")

    if len(deltaq_start_all) == 0 and len(deltaq_end_all) == 0:
        print("No valid delta_q values were computed.")
        return

    # Save pairwise summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(root_dir, "deltaq_first_last_2pct_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Save accumulated distributions
    dist_df = pd.DataFrame({
        "delta_q_first_2pct_distribution": pd.Series(deltaq_start_all),
        "delta_q_last_2pct_distribution": pd.Series(deltaq_end_all),
    })
    dist_csv = os.path.join(root_dir, "deltaq_first_last_2pct_distributions.csv")
    dist_df.to_csv(dist_csv, index=False)

    # Plot violin
    violin_png = os.path.join(root_dir, "deltaq_first_last_2pct_violin.png")
    plot_box_deltaq(deltaq_start_all, deltaq_end_all, violin_png)

    print("\n=== Accumulated distributions ===")
    print(f"N(delta_q first 2%) = {len(deltaq_start_all)}")
    print(f"N(delta_q last 2%)  = {len(deltaq_end_all)}")
    if len(deltaq_start_all) > 0:
        print(f"Mean delta_q first 2% = {np.mean(deltaq_start_all):.6e}")
    if len(deltaq_end_all) > 0:
        print(f"Mean delta_q last 2%  = {np.mean(deltaq_end_all):.6e}")

    print("\nSaved:")
    print(f"  {summary_csv}")
    print(f"  {dist_csv}")
    print(f"  {violin_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Accumulate delta q = q_A - q_B for first and last 2% of paired TMD.A.csv / TMD.B.csv"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root folder to search recursively",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.02,
        help="Fraction of frames used from beginning and end",
    )
    args = parser.parse_args()

    main(os.path.abspath(args.root), fraction=args.fraction)