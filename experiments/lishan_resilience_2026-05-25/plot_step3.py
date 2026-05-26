"""
M5.6 — Plot Step 3 resilience vs bimodal mixing weight p.

HONESTY DISCLAIMER: Input-side fault model. NOT register-level injection.
Strada Q discipline.

Reads results/step3_soga.csv (SOGA) and runs MC at 4 validation points.
Saves figures/resilience_vs_bimodal_p.png.

Usage:
    python3 experiments/lishan_resilience_2026-05-25/plot_step3.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)

from simulate_fi_mc import simulate_p_sweep


def load_step3_csv(csv_path: str) -> dict:
    """Load step3_soga.csv → dict of lists."""
    data: dict = {"p": [], "SOGA_MSK": [], "SOGA_SDC": [], "SOGA_OTR": []}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["p"].append(float(row["p"]))
            data["SOGA_MSK"].append(float(row["SOGA_MSK"]))
            data["SOGA_SDC"].append(float(row["SOGA_SDC"]))
            data["SOGA_OTR"].append(float(row["SOGA_OTR"]))
    return data


def main() -> None:
    config_path = os.path.join(EXP_DIR, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    results_dir = os.path.join(EXP_DIR, "results")
    figures_dir = os.path.join(EXP_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "step3_soga.csv")
    soga_data = load_step3_csv(csv_path)

    A = np.load(os.path.join(results_dir, "A_kernel.npz"))["A"]
    V_low = cfg["step3"]["V_low"]
    V_high = cfg["step3"]["V_high"]
    eps = cfg["eps"]
    p_fault = cfg["fault_model"]["p_fault"]
    n_samples = cfg["step3"]["mc_n_samples"]
    seed = cfg["seed"]

    # MC at 4 validation points
    p_validate = [0.0, 0.5, 0.75, 1.0]
    print("Running MC at 4 validation points...")
    mc_results = simulate_p_sweep(
        A, p_validate, V_low, V_high,
        n_samples=n_samples, seed=seed, eps=eps, p_fault=p_fault,
    )
    mc_p = sorted(mc_results.keys())
    mc_sdc = [mc_results[p]["SDC"] for p in mc_p]
    mc_msk = [mc_results[p]["MSK"] for p in mc_p]
    mc_otr = [mc_results[p]["OTR"] for p in mc_p]

    # MC error bars: binomial std dev
    def mc_err(p_val: float, n: int) -> float:
        """Binomial standard error sqrt(p*(1-p)/n)."""
        return float(np.sqrt(max(p_val * (1.0 - p_val), 0.0) / n))

    mc_sdc_err = [mc_err(v, n_samples) for v in mc_sdc]
    mc_msk_err = [mc_err(v, n_samples) for v in mc_msk]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    colors = {"MSK": "#1f77b4", "SDC": "#d62728", "OTR": "#ff7f0e"}

    panels = [
        ("MSK", soga_data["SOGA_MSK"], mc_msk, mc_msk_err, axes[0]),
        ("SDC", soga_data["SOGA_SDC"], mc_sdc, mc_sdc_err, axes[1]),
        ("OTR", soga_data["SOGA_OTR"], mc_otr, None, axes[2]),
    ]

    disclaimer = cfg["honesty_disclaimer"][:80] + "..."

    for label, sg_y, mc_y_vals, mc_err_vals, ax in panels:
        # SOGA curve (full 21 points)
        ax.plot(soga_data["p"], sg_y, "s--", color=colors[label], linewidth=2,
                markersize=4, alpha=0.9, label="SOGA analytical")

        # MC points with error bars
        if mc_err_vals is not None:
            ax.errorbar(mc_p, mc_y_vals, yerr=mc_err_vals, fmt="o", color=colors[label],
                       linewidth=1.5, markersize=6, capsize=4, label=f"MC (N={n_samples})")
        else:
            ax.plot(mc_p, mc_y_vals, "o", color=colors[label], markersize=6,
                    label=f"MC (N={n_samples})")

        ax.set_xlabel("Mixing weight p", fontsize=11)
        ax.set_ylabel(f"P({label})", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle(
        "Step 3: Resilience vs bimodal mixing weight p — A=I_32, V_low=0, V_high=1\n"
        f"[{disclaimer}]",
        fontsize=9,
        y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(figures_dir, "resilience_vs_bimodal_p.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
