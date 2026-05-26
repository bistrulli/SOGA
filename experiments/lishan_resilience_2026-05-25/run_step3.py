"""
M5.3 — Step 3 SOGA bimodal p-sweep.

HONESTY DISCLAIMER: Input-side fault model. NOT register-level injection.
Strada Q discipline.

Sweeps p in [0,1] (21 points) for the bimodal prior B ~ p*delta(V_high) + (1-p)*delta(V_low).
For A=I_32 (sparse, m_dense=1 < 16): uses 2-component GM path.
Computes Kendall's tau for monotonicity of SDC vs p.

Usage:
    python3 experiments/lishan_resilience_2026-05-25/run_step3.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import stats as spstats

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)

from predict_resilience_soga import predict_bimodal_sweep
from lib.bimodal_prior import p_critical_sdc


def compute_kendall_tau(x: list, y: list) -> float:
    """Kendall's tau for y vs x (monotone if tau ~ 1.0)."""
    if len(x) < 2:
        return float("nan")
    tau, _ = spstats.kendalltau(x, y)
    return float(tau)


def main() -> None:
    config_path = os.path.join(EXP_DIR, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    results_dir = os.path.join(EXP_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    A = np.load(os.path.join(results_dir, "A_kernel.npz"))["A"]
    p_list = cfg["step3"]["p_sweep"]
    V_low = cfg["step3"]["V_low"]
    V_high = cfg["step3"]["V_high"]
    eps = cfg["eps"]
    p_fault = cfg["fault_model"]["p_fault"]
    tau_threshold = cfg["step3"]["monotonicity_tau_threshold"]
    counterexample_tau = cfg["step3"]["counterexample_tau_threshold"]

    print(f"Step 3 bimodal sweep: {len(p_list)} p-points")
    print(f"V_low={V_low}, V_high={V_high}, eps={eps}, p_fault={p_fault}")
    print(f"Honesty: {cfg['honesty_disclaimer']}")
    print()

    # Compute p_critical estimate
    p_crit = p_critical_sdc(V_low, V_high, eps)
    print(f"p_critical (analytical estimate): {p_crit:.4f}")

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        soga_results = predict_bimodal_sweep(
            A, p_list, V_low, V_high, eps=eps, p_fault=p_fault
        )
    elapsed = time.time() - t0
    print(f"SOGA sweep done: {elapsed:.2f}s")

    p_sorted = sorted(p_list)
    soga_msk = [soga_results[p]["MSK"] for p in p_sorted]
    soga_sdc = [soga_results[p]["SDC"] for p in p_sorted]
    soga_otr = [soga_results[p]["OTR"] for p in p_sorted]

    # Kendall's tau: SDC vs p (should be monotone increasing for higher p → more risk)
    tau_sdc = compute_kendall_tau(p_sorted, soga_sdc)
    tau_msk = compute_kendall_tau(p_sorted, [-x for x in soga_msk])  # decreasing MSK = increasing risk

    print()
    print("=" * 60)
    print("STEP 3 SOGA SWEEP REPORT")
    print("=" * 60)
    print(f"{'p':>8} | {'SOGA_MSK':>10} {'SOGA_SDC':>10} {'SOGA_OTR':>10} | {'mode':>20}")
    print("-" * 70)
    for p in p_sorted:
        r = soga_results[p]
        print(f"{p:>8.3f} | {r['MSK']:>10.6f} {r['SDC']:>10.6f} {r['OTR']:>10.6f} | {r['mode']:>20}")

    print()
    print(f"Kendall tau SDC vs p: {tau_sdc:.3f}")
    print(f"Kendall tau MSK vs p (inverted): {tau_msk:.3f}")

    is_monotone_sdc = tau_sdc >= tau_threshold
    is_counterexample = tau_sdc <= counterexample_tau

    if is_monotone_sdc:
        print(f"SDC MONOTONE: tau={tau_sdc:.3f} >= {tau_threshold} (expected behavior)")
    elif is_counterexample:
        print(f"COUNTEREXAMPLE FOUND: tau={tau_sdc:.3f} <= {counterexample_tau}")
        print("  SDC is NOT monotone in p — investigate non-monotone resilience behavior.")
    else:
        print(f"PARTIAL MONOTONE: {counterexample_tau} < tau={tau_sdc:.3f} < {tau_threshold}")
        print("  NOTE: Strada Q mantissa overestimation may affect monotonicity shape.")
    print("=" * 60)

    # Save CSV
    csv_path = os.path.join(results_dir, "step3_soga.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "SOGA_MSK", "SOGA_SDC", "SOGA_OTR", "mode"])
        for p in p_sorted:
            r = soga_results[p]
            writer.writerow([p, r["MSK"], r["SDC"], r["OTR"], r["mode"]])
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
