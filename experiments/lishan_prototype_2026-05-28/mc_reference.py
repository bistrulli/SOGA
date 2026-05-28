"""Monte Carlo reference simulator for the 2x2 scalar-decomposed matmul prototype.

Mirrors EXACTLY the algorithm in programs/Example/lishan_prototype_2x2_scalar.soga:
- B[0,0] and B[1,0] iid N(v, sigma_b^2)
- acc = 0
- acc = acc + b00            (FMA #1)
- fault ~ Bern(p); if fault==1: acc = -acc
- acc = acc + b10            (FMA #2)
- d00 = acc

Outputs E[d00], Var[d00], Pr(SDC) with Wilson 95% CI.

Independent implementation — does NOT reuse simulate_fi_mc.py (which models
input-side faults with different semantics).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


SIGMA_B_SQ = 1e-12  # input variance (matches .soga: gauss(v, 1e-12))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for k successes out of n binomial trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - rad), min(1.0, centre + rad))


def simulate(v: float, p_fault: float, n_samples: int, eps: float, seed: int) -> dict:
    """Run n_samples of the prototype algorithm and return aggregated stats."""
    rng = np.random.default_rng(seed)
    sigma_b = math.sqrt(SIGMA_B_SQ)

    b00 = rng.normal(v, sigma_b, size=n_samples)
    b10 = rng.normal(v, sigma_b, size=n_samples)
    fault = rng.random(size=n_samples) < p_fault

    # Step-by-step accumulator with sign flip on fault path
    acc = b00.copy()                       # acc = 0 + b00
    acc[fault] = -acc[fault]                # intermediate sign flip
    acc = acc + b10                        # acc += b10
    d00 = acc

    # Aggregates
    mean = float(np.mean(d00))
    var = float(np.var(d00, ddof=1))

    # SDC: |d00 - baseline| > eps * |baseline|, baseline = 2v (no-fault expected)
    baseline = 2.0 * v
    if abs(baseline) > 1e-10:
        threshold = eps * abs(baseline)
    else:
        threshold = eps  # absolute fallback for zero-crossing
    sdc_count = int(np.sum(np.abs(d00 - baseline) > threshold))
    sdc_p = sdc_count / n_samples
    sdc_lo, sdc_hi = wilson_ci(sdc_count, n_samples)

    return {
        "v": v,
        "p_fault": p_fault,
        "n_samples": n_samples,
        "seed": seed,
        "eps": eps,
        "E_d00": mean,
        "Var_d00": var,
        "Pr_SDC": sdc_p,
        "Pr_SDC_lo": sdc_lo,
        "Pr_SDC_hi": sdc_hi,
        "n_sdc_events": sdc_count,
    }


def main():
    parser = argparse.ArgumentParser(description="MC reference for lishan prototype 2x2")
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: results/mc_reference.csv)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    out_path = Path(args.out) if args.out else out_dir / "results" / "mc_reference.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    v_grid = [0.5, 1.0, 2.0]
    p_grid = [0.001, 0.01, 0.05]

    rows = []
    for v in v_grid:
        for p in p_grid:
            r = simulate(v, p, args.n_samples, args.eps, args.seed)
            rows.append(r)
            print(f"v={v:>4} p={p:>6} | E={r['E_d00']:>10.6f} Var={r['Var_d00']:>10.6f} "
                  f"Pr_SDC={r['Pr_SDC']:>8.4f} ({r['Pr_SDC_lo']:.4f}, {r['Pr_SDC_hi']:.4f})")

    # Save CSV
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_path}")

    # Save config used
    cfg = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "eps": args.eps,
        "sigma_b_sq": SIGMA_B_SQ,
        "v_grid": v_grid,
        "p_grid": p_grid,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
