#!/usr/bin/env python3
"""
Recursively scan a folder tree for log.TMD.* files, parse TMD step and target RMSD,
extract the first 5% and last 5% of each trajectory, and save the data to a CSV.

Example
-------
python extract_tmd_rmsd_distribution.py /path/to/root -o tmd_rmsd_distribution.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def find_log_files(root: Path, pattern: str = "log.TMD.*") -> List[Path]:
    """Find all matching log files under root."""
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def parse_tmd_line(line: str) -> Optional[Tuple[int, float]]:
    """
    Parse a TMD log line.

    Expected example:
        TMD  48200 Domain: 0 0.0748686 1.00012

    Returns
    -------
    (step, rmsd) if the line matches, otherwise None
    """
    line = line.strip()
    if not line.startswith("TMD"):
        return None

    parts = line.split()
    if len(parts) < 3:
        return None

    # Step is expected to be the second token
    try:
        step = int(parts[1])
    except ValueError:
        return None

    # RMSD is expected to be the last token; remove trailing punctuation if present
    last_token = parts[-1].rstrip(".,;:")
    try:
        rmsd = float(last_token)
    except ValueError:
        return None

    return step, rmsd


def read_tmd_series(logfile: Path) -> List[Tuple[int, float]]:
    """Read all (step, rmsd) pairs from one log file."""
    series: List[Tuple[int, float]] = []
    with logfile.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_tmd_line(line)
            if parsed is not None:
                series.append(parsed)
    return series


def first_last_percent(
    series: List[Tuple[int, float]],
    fraction: float = 0.05,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Get first and last fraction of a trajectory.

    Ensures at least 1 point in each segment if the series is non-empty.
    """
    n = len(series)
    if n == 0:
        return [], []

    k = max(1, math.ceil(n * fraction))
    first = series[:k]
    last = series[-k:]
    return first, last


def infer_system_name(root: Path, logfile: Path) -> str:
    """
    Infer a system name from the path.

    By default, use the first directory level under root if possible.
    Otherwise use the parent directory name.
    """
    try:
        rel = logfile.relative_to(root)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except ValueError:
        pass

    return logfile.parent.name


def write_csv(
    rows: Iterable[dict],
    output_csv: Path,
) -> None:
    """Write extracted data to CSV."""
    fieldnames = [
        "system",
        "logfile",
        "segment",
        "fraction",
        "point_index_within_segment",
        "global_index",
        "step",
        "rmsd_to_target",
        "n_total_points",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract first/last 5%% RMSD distributions from TMD log files."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to search recursively.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("tmd_rmsd_distribution.csv"),
        help="Output CSV file.",
    )
    parser.add_argument(
        "--pattern",
        default="log.TMD.*",
        help="Glob pattern for log files (default: log.TMD.*).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.01,
        help="Fraction for first/last segment (default: 0.01).",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    logfiles = find_log_files(root, args.pattern)

    if not logfiles:
        print(f"No files matching '{args.pattern}' found under {root}")
        return

    output_rows = []
    n_files_with_data = 0

    for logfile in logfiles:
        series = read_tmd_series(logfile)
        if not series:
            print(f"Skipping {logfile}: no parsable TMD lines found")
            continue

        n_files_with_data += 1
        system = infer_system_name(root, logfile)
        first, last = first_last_percent(series, args.fraction)
        n_total = len(series)

        for i, (step, rmsd) in enumerate(first):
            output_rows.append(
                {
                    "system": system,
                    "logfile": str(logfile),
                    "segment": "first",
                    "fraction": args.fraction,
                    "point_index_within_segment": i,
                    "global_index": i,
                    "step": step,
                    "rmsd_to_target": rmsd,
                    "n_total_points": n_total,
                }
            )

        first_len = len(first)
        for i, (step, rmsd) in enumerate(last):
            global_index = n_total - len(last) + i
            output_rows.append(
                {
                    "system": system,
                    "logfile": str(logfile),
                    "segment": "last",
                    "fraction": args.fraction,
                    "point_index_within_segment": i,
                    "global_index": global_index,
                    "step": step,
                    "rmsd_to_target": rmsd,
                    "n_total_points": n_total,
                }
            )

    write_csv(output_rows, args.output)

    print(f"Found {len(logfiles)} matching log files")
    print(f"Parsed data from {n_files_with_data} files")
    print(f"Wrote CSV: {args.output}")


if __name__ == "__main__":
    main()