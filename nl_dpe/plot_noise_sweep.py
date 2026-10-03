"""Plot mean and standard deviation from a weight-noise sweep summary."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--title", default="Weight-noise sweep")
    parser.add_argument("--ylabel", default="Accuracy (%)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.summary.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            grouped[row["model"]].append(row)

    colors = {"Clean": "#2864dc", "NAF-Full": "#d94b40"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    all_values: list[float] = []
    for label, rows in grouped.items():
        rows.sort(key=lambda row: float(row["noise_percent"]))
        x = np.asarray([float(row["noise_percent"]) for row in rows])
        y = 100 * np.asarray([float(row[f"{args.metric}_mean"]) for row in rows])
        std = 100 * np.asarray([float(row[f"{args.metric}_std"]) for row in rows])
        color = colors.get(label)
        ax.plot(x, y, marker="o", linewidth=2, markersize=4.5, label=label, color=color)
        ax.fill_between(x, y - std, y + std, alpha=0.16, color=color)
        all_values.extend((y - std).tolist())
        all_values.extend((y + std).tolist())

    ax.set_xlabel(r"Gaussian noise std / clean max $|W|$ (%)")
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.set_xticks(sorted({float(row["noise_percent"]) for rows in grouped.values() for row in rows}))
    if all_values:
        lower, upper = min(all_values), max(all_values)
        margin = max(0.4, 0.15 * (upper - lower))
        ax.set_ylim(lower - margin, upper + margin)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.output_prefix.with_suffix(f".{suffix}"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
