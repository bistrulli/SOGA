"""
M4.2 — Plot resilience curves: MC (solid) vs SOGA (dashed) for Step 1 sweep.

HONESTY DISCLAIMER: Input-side fault model. NOT register-level injection.
Strada Q discipline.

Reads results/step1_combined.csv and saves figures/resilience_vs_input_value.png.

Usage:
    python3 experiments/lishan_resilience_2026-05-25/plot_step1.py
    python3 experiments/lishan_resilience_2026-05-25/plot_step1.py --csv path/to/step1_combined.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)


def load_csv(csv_path: str) -> dict:
    """Load step1_combined.csv into dict of lists."""
    data: dict[str, list] = {
        "v": [], "MC_MSK": [], "MC_SDC": [], "MC_OTR": [],
        "SOGA_MSK": [], "SOGA_SDC": [], "SOGA_OTR": [],
    }
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))
    return data


def plot_step1(data: dict, out_path: str, disclaimer: str) -> None:
    """Three-panel plot: MSK / SDC / OTR vs |v|."""
    v_vals = np.array(data["v"])
    abs_v = np.abs(v_vals)
    sort_idx = np.argsort(abs_v)

    abs_v_sorted = abs_v[sort_idx]

    # Use unique |v| — symmetric sweep so |v| gives 5 unique points
    # If symmetric, average duplicate |v| pairs for cleaner plot
    unique_abs_v = []
    seen: dict = {}
    for idx_orig, av in zip(sort_idx, abs_v_sorted):
        key = round(av, 12)
        if key not in seen:
            seen[key] = {"count": 0, "sums": {k: 0.0 for k in data if k != "v"}}
        seen[key]["count"] += 1
        for k in data:
            if k != "v":
                seen[key]["sums"][k] += data[k][idx_orig]

    xs = sorted(seen.keys())
    mc_msk = [seen[x]["sums"]["MC_MSK"] / seen[x]["count"] for x in xs]
    mc_sdc = [seen[x]["sums"]["MC_SDC"] / seen[x]["count"] for x in xs]
    mc_otr = [seen[x]["sums"]["MC_OTR"] / seen[x]["count"] for x in xs]
    sg_msk = [seen[x]["sums"]["SOGA_MSK"] / seen[x]["count"] for x in xs]
    sg_sdc = [seen[x]["sums"]["SOGA_SDC"] / seen[x]["count"] for x in xs]
    sg_otr = [seen[x]["sums"]["SOGA_OTR"] / seen[x]["count"] for x in xs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    colors = {"MSK": "#1f77b4", "SDC": "#d62728", "OTR": "#ff7f0e"}

    panels = [
        ("MSK", mc_msk, sg_msk, axes[0]),
        ("SDC", mc_sdc, sg_sdc, axes[1]),
        ("OTR", mc_otr, sg_otr, axes[2]),
    ]

    for label, mc_y, sg_y, ax in panels:
        ax.semilogx(xs, mc_y, "o-", color=colors[label], linewidth=2,
                    markersize=6, label=f"MC (N=1000)")
        ax.semilogx(xs, sg_y, "s--", color=colors[label], linewidth=2,
                    markersize=6, alpha=0.85, label="SOGA analytical")
        ax.set_xlabel("|v| (input scale)", fontsize=11)
        ax.set_ylabel(f"P({label})", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(-0.02, 1.05)

    fig.suptitle(
        "Step 1: Resilience vs input value scale — A=I_32, p_fault=0.01\n"
        f"[{disclaimer}]",
        fontsize=9,
        y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="M4.2 Step 1 resilience plot")
    parser.add_argument("--csv", default=None,
                        help="Path to step1_combined.csv (default: results/step1_combined.csv)")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: figures/resilience_vs_input_value.png)")
    args = parser.parse_args()

    results_dir = os.path.join(EXP_DIR, "results")
    figures_dir = os.path.join(EXP_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = args.csv if args.csv else os.path.join(results_dir, "step1_combined.csv")
    out_path = args.out if args.out else os.path.join(figures_dir, "resilience_vs_input_value.png")

    import json
    config_path = os.path.join(EXP_DIR, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    disclaimer = cfg["honesty_disclaimer"][:80] + "..."

    data = load_csv(csv_path)
    plot_step1(data, out_path, disclaimer)


if __name__ == "__main__":
    main()
