"""Register-level Monte Carlo reference for the int32 PolyBench 2MM kernel.

GROUND TRUTH for the experiment. A single bit-flip is injected into one of the
2*N^3 int32 accumulator FMA sites (uniform site, uniform bit 0..31); the kernel
runs step-by-step and the final D is compared cell-by-cell against the no-fault
baseline. eps = 0 (int32): any change is SDC; wraparound is SDC, never OTR.

HONESTY DISCLAIMER: register-level Python MC is NOT an NVBit-FI proxy
(see POLYBENCH_REFERENCE.md). Comparison with Lishan Yang's GPU data is shape-only.

Usage:
    python3 mc_register.py --n-samples 10000 --seed 42 --ns 4
    python3 mc_register.py --n-samples 10000 --seed 42 --ns 4 8
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from kernel2mm import (
    kernel_2mm,
    matrices_equal,
    n_fault_sites,
    polybench_AC,
    wilson_ci,
)

EXP_DIR = Path(__file__).parent
V_GRID = [-1, 0, 1, 6]


def simulate(n: int, v: int, n_samples: int, seed: int) -> dict:
    """Run n_samples register-level FI and return per-kernel SDC stats."""
    rng = np.random.default_rng(seed)
    A, C = polybench_AC(n)
    n_sites = n_fault_sites(n)

    baseline = kernel_2mm(A, C, v, n)

    sites = rng.integers(0, n_sites, size=n_samples)
    bits = rng.integers(0, 32, size=n_samples)

    t0 = time.time()
    sdc = 0
    for s, b in zip(sites.tolist(), bits.tolist()):
        d = kernel_2mm(A, C, v, n, fault_site=s, fault_bit=b)
        if not matrices_equal(d, baseline):
            sdc += 1
    elapsed = time.time() - t0

    pr_sdc = sdc / n_samples
    lo, hi = wilson_ci(sdc, n_samples)
    return {
        "N": n,
        "v": v,
        "model": "register",
        "n_sites": n_sites,
        "n_samples": n_samples,
        "seed": seed,
        "n_sdc": sdc,
        "Pr_SDC": pr_sdc,
        "Pr_SDC_lo": lo,
        "Pr_SDC_hi": hi,
        "MSK": 1.0 - pr_sdc,
        "time_s": elapsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Register-level MC for int32 2MM")
    ap.add_argument("--n-samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ns", type=int, nargs="+", default=[4],
                    help="Matrix sizes N to sweep")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rows = []
    for n in args.ns:
        out_path = Path(args.out) if args.out else EXP_DIR / "results" / f"mc_register_n{n}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_rows = []
        print(f"\n=== Register-level MC, N={n} ({n_fault_sites(n)} FMA sites) ===")
        print(f"{'v':>4} | {'Pr_SDC':>8} | {'95% CI':>20} | {'MSK':>8} | {'t(s)':>6}")
        print("-" * 60)
        for v in V_GRID:
            # Distinct seed per (N,v) for independence, derived from base seed.
            r = simulate(n, v, args.n_samples, args.seed + 1000 * n + (v + 2))
            rows.append(r)
            n_rows.append(r)
            print(f"{v:>4} | {r['Pr_SDC']:>8.4f} | "
                  f"({r['Pr_SDC_lo']:.4f}, {r['Pr_SDC_hi']:.4f}) | "
                  f"{r['MSK']:>8.4f} | {r['time_s']:>6.1f}")
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=n_rows[0].keys())
            w.writeheader()
            w.writerows(n_rows)
        print(f"Saved: {out_path}")

    cfg_path = EXP_DIR / "results" / "mc_register_config.json"
    with open(cfg_path, "w") as f:
        json.dump({"n_samples": args.n_samples, "seed": args.seed,
                   "ns": args.ns, "v_grid": V_GRID, "eps": 0}, f, indent=2)
    print(f"Saved: {cfg_path}")


if __name__ == "__main__":
    main()
