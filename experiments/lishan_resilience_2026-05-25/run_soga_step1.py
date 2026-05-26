"""
M3.6 — CLI script: SOGA analytical Step 1 sweep.

Runs predict_v_sweep and saves results/soga_step1.csv.

Usage:
    python3 experiments/lishan_resilience_2026-05-25/run_soga_step1.py
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

from predict_resilience_soga import predict_v_sweep


def main():
    parser = argparse.ArgumentParser(description="SOGA analytical Step 1 sweep")
    parser.add_argument("--v-subset", nargs="+", type=float, default=None)
    args = parser.parse_args()

    config_path = os.path.join(EXP_DIR, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    eps = cfg["eps"]
    v_list = args.v_subset if args.v_subset is not None else cfg["step1"]["v_sweep"]
    p_fault = cfg["fault_model"]["p_fault"]

    results_dir = os.path.join(EXP_DIR, "results")
    A = np.load(os.path.join(results_dir, "A_kernel.npz"))["A"]

    print(f"SOGA Step 1 sweep: {len(v_list)} v-points, eps={eps}, p_fault={p_fault}")

    t0 = time.time()
    results = predict_v_sweep(A, 0.0, v_list, eps=eps, p_fault=p_fault)
    total_time = time.time() - t0

    csv_path = os.path.join(results_dir, "soga_step1.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["v", "MSK", "SDC", "OTR"])
        for v in sorted(results.keys()):
            r = results[v]
            writer.writerow([v, r["MSK"], r["SDC"], r["OTR"]])

    print(f"Saved: {csv_path} ({len(results)} rows, total_time={total_time:.1f}s)")

    print("\nv          MSK     SDC     OTR")
    for v in sorted(results.keys()):
        r = results[v]
        print(f"{v:+10.3e}  {r['MSK']:.4f}  {r['SDC']:.4f}  {r['OTR']:.4f}")


if __name__ == "__main__":
    main()
