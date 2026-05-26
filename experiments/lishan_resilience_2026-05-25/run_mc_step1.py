"""
M2.4 — CLI script: MC Step 1 sweep.

Runs simulate_v_sweep and saves results/mc_step1.csv.

Usage:
    python3 experiments/lishan_resilience_2026-05-25/run_mc_step1.py [--n-samples N] [--seed S]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)

from simulate_fi_mc import simulate_v_sweep


def main():
    parser = argparse.ArgumentParser(description="MC Step 1 sweep for lishan resilience POC")
    parser.add_argument("--n-samples", type=int, default=None, help="MC samples per v-point")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--v-subset", nargs="+", type=float, default=None,
                        help="Subset of v values (default: all from config.json)")
    args = parser.parse_args()

    config_path = os.path.join(EXP_DIR, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    n_samples = args.n_samples if args.n_samples is not None else cfg["step1"]["mc_n_samples"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    eps = cfg["eps"]
    v_list = args.v_subset if args.v_subset is not None else cfg["step1"]["v_sweep"]
    p_fault = cfg["fault_model"]["p_fault"]

    # Load A matrix
    results_dir = os.path.join(EXP_DIR, "results")
    A = np.load(os.path.join(results_dir, "A_kernel.npz"))["A"]

    print(f"MC Step 1 sweep: {len(v_list)} v-points × {n_samples} samples each")
    print(f"A shape: {A.shape}, seed: {seed}, eps: {eps}, p_fault: {p_fault}")

    t0 = time.time()
    results = simulate_v_sweep(A, v_list, n_samples=n_samples, seed=seed, eps=eps, p_fault=p_fault)
    total_time = time.time() - t0

    # Save CSV
    csv_path = os.path.join(results_dir, "mc_step1.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["v", "MSK", "SDC", "OTR", "n_samples", "time_s"])
        for v in sorted(results.keys()):
            r = results[v]
            writer.writerow([v, r["MSK"], r["SDC"], r["OTR"], r["n_samples"], f"{r['time_s']:.3f}"])

    print(f"Saved: {csv_path} ({len(results)} rows, total_time={total_time:.1f}s)")

    # Print summary
    print("\nv          MSK     SDC     OTR")
    for v in sorted(results.keys()):
        r = results[v]
        print(f"{v:+10.3e}  {r['MSK']:.4f}  {r['SDC']:.4f}  {r['OTR']:.4f}")


if __name__ == "__main__":
    main()
