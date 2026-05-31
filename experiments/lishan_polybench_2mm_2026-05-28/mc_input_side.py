"""Input-side Monte Carlo reference for the int32 PolyBench 2MM kernel.

Same kernel as mc_register.py, but the single bit-flip is injected into one cell
of the input matrix B BEFORE the kernel runs (uniform cell over N^2, uniform bit
0..31). This is the "current state" baseline (cf. existing simulate_fi_mc.py):
it quantifies how much weaker input-side FI is than register-level FI.

Per-kernel SDC = (output != baseline) exactly (eps = 0 for int32).

Usage:
    python3 mc_input_side.py --n-samples 10000 --seed 42 --ns 4
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
    kernel_2mm_input_fault,
    matrices_equal,
    polybench_AC,
    wilson_ci,
)

EXP_DIR = Path(__file__).parent
V_GRID = [-1, 0, 1, 6]


def simulate(n: int, v: int, n_samples: int, seed: int) -> dict:
    """Run n_samples input-side FI (one B cell) and return per-kernel SDC stats."""
    rng = np.random.default_rng(seed)
    A, C = polybench_AC(n)

    baseline = kernel_2mm(A, C, v, n)

    cells = rng.integers(0, n * n, size=n_samples)
    bits = rng.integers(0, 32, size=n_samples)

    t0 = time.time()
    sdc = 0
    for cell, b in zip(cells.tolist(), bits.tolist()):
        r, c = divmod(cell, n)
        d = kernel_2mm_input_fault(A, C, v, n, fault_cell=(r, c), fault_bit=b)
        if not matrices_equal(d, baseline):
            sdc += 1
    elapsed = time.time() - t0

    pr_sdc = sdc / n_samples
    lo, hi = wilson_ci(sdc, n_samples)
    return {
        "N": n,
        "v": v,
        "model": "input_side",
        "n_sites": n * n,
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
    ap = argparse.ArgumentParser(description="Input-side MC for int32 2MM")
    ap.add_argument("--n-samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ns", type=int, nargs="+", default=[4])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rows = []
    for n in args.ns:
        out_path = Path(args.out) if args.out else EXP_DIR / "results" / f"mc_input_side_n{n}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_rows = []
        print(f"\n=== Input-side MC, N={n} ({n*n} B cells) ===")
        print(f"{'v':>4} | {'Pr_SDC':>8} | {'95% CI':>20} | {'MSK':>8} | {'t(s)':>6}")
        print("-" * 60)
        for v in V_GRID:
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

    cfg_path = EXP_DIR / "results" / "mc_input_side_config.json"
    with open(cfg_path, "w") as f:
        json.dump({"n_samples": args.n_samples, "seed": args.seed,
                   "ns": args.ns, "v_grid": V_GRID, "eps": 0}, f, indent=2)
    print(f"Saved: {cfg_path}")


if __name__ == "__main__":
    main()
