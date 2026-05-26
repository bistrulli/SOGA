"""
M3.6 / R2.5 — CLI script: SOGA analytical Step 1 sweep.

Runs predict_v_sweep and saves results/soga_step1.csv.

Usage:
    python3 experiments/lishan_resilience_2026-05-25/run_soga_step1.py [--mode {bit_exact,5_class}]

BEHAVIORAL CHANGE (config_version 1->2): default mode is now 'bit_exact' (refinement primary).
Use --mode 5_class for legacy 5-class moment-matched model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)


def main():
    parser = argparse.ArgumentParser(description="SOGA analytical Step 1 sweep")
    parser.add_argument("--v-subset", nargs="+", type=float, default=None)
    parser.add_argument(
        "--mode", choices=["bit_exact", "5_class"], default="bit_exact",
        help="Fault model mode: bit_exact (default, primary) or 5_class (legacy)",
    )
    args = parser.parse_args()

    # Runtime banner (R2.5 backward-compat notice)
    if args.mode == "bit_exact":
        print("[MODE] Using bit_exact fault model (refinement primary; --mode 5_class for legacy)")
        from predict_resilience_soga import predict_v_sweep
    else:
        warnings.warn(
            "[DEPRECATED] 5_class mode selected. predict_resilience_soga_5class.py is a "
            "historical reference model with known L1/L2 limitations. Prefer bit_exact.",
            DeprecationWarning, stacklevel=1,
        )
        from predict_resilience_soga_5class import predict_v_sweep

    config_path = os.path.join(EXP_DIR, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    eps = cfg["eps"]
    v_list = args.v_subset if args.v_subset is not None else cfg["step1"]["v_sweep"]
    p_fault = cfg["fault_model"]["p_fault"]

    results_dir = os.path.join(EXP_DIR, "results")
    A = np.load(os.path.join(results_dir, "A_kernel.npz"))["A"]

    print(f"SOGA Step 1 sweep ({args.mode}): {len(v_list)} v-points, eps={eps}, p_fault={p_fault}")

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
