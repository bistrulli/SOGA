"""
R3.2 — Three-model comparison figure: MC | 5-class | bit-exact.

Generates figures/resilience_vs_input_value_3panel.png.
Reads:
  results/mc_step1.csv              (Monte Carlo reference)
  results/soga_step1_5class.csv     (historical 5-class model)
  results/soga_step1_bitexact.csv   (bit-exact primary predictor)

Usage:
    python3 experiments/lishan_resilience_2026-05-25/plot_step1_3panel_comparison.py
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))


def load_simple_csv(path: str) -> dict[float, dict[str, float]]:
    """Load v,MSK,SDC,OTR CSV into dict keyed by v."""
    rows: dict[float, dict[str, float]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = float(row["v"])
            rows[v] = {k: float(row[k]) for k in ("MSK", "SDC", "OTR")}
    return rows


def main() -> None:
    results_dir = os.path.join(EXP_DIR, "results")
    figures_dir = os.path.join(EXP_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    mc = load_simple_csv(os.path.join(results_dir, "mc_step1.csv"))
    fc = load_simple_csv(os.path.join(results_dir, "soga_step1_5class.csv"))
    be = load_simple_csv(os.path.join(results_dir, "soga_step1_bitexact.csv"))

    # Use absolute value of v; average symmetric pairs
    def aggregate(data: dict[float, dict[str, float]]) -> tuple[list, list, list, list]:
        seen: dict[float, dict[str, list]] = {}
        for v, vals in data.items():
            key = round(abs(v), 12)
            if key not in seen:
                seen[key] = {"MSK": [], "SDC": [], "OTR": []}
            for k in ("MSK", "SDC", "OTR"):
                seen[key][k].append(vals[k])
        xs = sorted(seen.keys())
        msk = [float(np.mean(seen[x]["MSK"])) for x in xs]
        sdc = [float(np.mean(seen[x]["SDC"])) for x in xs]
        otr = [float(np.mean(seen[x]["OTR"])) for x in xs]
        return xs, msk, sdc, otr

    xs_mc, mc_msk, mc_sdc, mc_otr = aggregate(mc)
    xs_fc, fc_msk, fc_sdc, fc_otr = aggregate(fc)
    xs_be, be_msk, be_sdc, be_otr = aggregate(be)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    colors = {
        "MC": "#333333",
        "5class": "#ff7f0e",
        "bitexact": "#1f77b4",
    }

    panels = [
        ("MSK", mc_msk, fc_msk, be_msk),
        ("SDC", mc_sdc, fc_sdc, be_sdc),
        ("OTR", mc_otr, fc_otr, be_otr),
    ]

    for (label, mc_y, fc_y, be_y), ax in zip(panels, axes):
        ax.plot(xs_mc, mc_y, "o-", color=colors["MC"], linewidth=2,
                markersize=6, label="MC (N=1000)", zorder=3)
        ax.plot(xs_fc, fc_y, "^:", color=colors["5class"], linewidth=2,
                markersize=6, alpha=0.85, label="SOGA 5-class (legacy)")
        ax.plot(xs_be, be_y, "s--", color=colors["bitexact"], linewidth=2,
                markersize=6, alpha=0.90, label="SOGA bit-exact (primary)")
        ax.set_xscale("log")
        ax.set_xlabel("|v| (input scale)", fontsize=11)
        ax.set_ylabel(f"P({label})", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    # Separate y-limits for SDC (very small)
    axes[1].set_ylim(-5e-6, 5e-4)

    fig.suptitle(
        "Step 1: Resilience vs input scale — A=I_32, p_fault=0.01, eps=0.001\n"
        "Three-model comparison: MC reference | SOGA 5-class (legacy) | SOGA bit-exact (primary)",
        fontsize=9,
        y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(figures_dir, "resilience_vs_input_value_3panel.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print accuracy summary
    print("\n--- Accuracy summary vs MC ---")
    print(f"{'|v|':>8}  {'MC_SDC':>10}  {'5cls_SDC':>12}  {'be_SDC':>10}  {'5cls_rel':>10}  {'be_rel':>10}")
    for xv, mc_s, fc_s, be_s in zip(xs_mc, mc_sdc, fc_sdc, be_sdc):
        if mc_s > 1e-9:
            fc_rel = abs(fc_s - mc_s) / mc_s
            be_rel = abs(be_s - mc_s) / mc_s
        else:
            fc_rel = be_rel = float("nan")
        print(f"{xv:>8.4e}  {mc_s:>10.4e}  {fc_s:>12.4e}  {be_s:>10.4e}  {fc_rel:>10.4f}  {be_rel:>10.4f}")


if __name__ == "__main__":
    main()
