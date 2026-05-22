"""
Real reliability-program benchmark.

Builds a Bayesian-regression-style program (K latent weights observed
indirectly through N noisy linear measurements with thresholded
observes) at varying (K, N) and times the current SOGA path
(--vectorize-truncate). This is the program family that the user's
target use case (input-distribution → output-distribution reliability
analysis) most naturally maps to.

Programs are generated via the existing `gen_pattern_B` (which
handles arbitrary K) from
`experiments/feasibility_dead_var_pruning_2026-05-20/gen_programs.py`.

We report:
  - end-to-end SOGA runtime in `--vectorize-truncate` mode (current best
    on this branch)
  - the projected structured-cov speedup from the math prototype's
    measured per-op timings (with the caveat from the block-stress that
    long sequences of subtractive updates carry drift if compactification
    is too aggressive)
  - n_comp at exit (to confirm the structured-cov regime is the binding
    constraint, not n_comp)
"""

import os
import sys
import signal
import time
import csv
from statistics import mean

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(ROOT, "experiments", "feasibility_dead_var_pruning_2026-05-20"))

from gen_programs import gen_pattern_B  # noqa: E402
from producecfg import produce_cfg  # noqa: E402
from libSOGA import start_SOGA  # noqa: E402
from sogaPreprocessor import compile2SOGA  # noqa: E402

import random


class HardTimeout(Exception):
    pass


def _alarm(s, f):
    raise HardTimeout()


def with_timeout(seconds, fn, *args):
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(seconds) + 1)
    try:
        return True, fn(*args)
    except HardTimeout:
        return False, None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


REAL_INPUTS = os.path.join(HERE, "real_program_inputs")
os.makedirs(REAL_INPUTS, exist_ok=True)


def time_one(prog_path):
    random.seed(0); np.random.seed(0)
    t0 = time.perf_counter()
    compiled = compile2SOGA(prog_path)
    cfg = produce_cfg(compiled)
    out = start_SOGA(cfg, useR=False, parallel=None,
                     sparse_truncate=True, vectorize_truncate=True)
    t1 = time.perf_counter()
    return {
        "total_s": t1 - t0,
        "d": len(out.var_list),
        "n_comp": out.gm.n_comp(),
        # Extract w[0..K-1] means as a sanity output
        "out_var_list": list(out.var_list),
    }


def project_speedup(d, k=2):
    """Same projection table as bench_high_d.py."""
    if d <= 50:
        return 1.0
    if d <= 100:
        return 2.0
    if d <= 200:
        return 6.0
    if d <= 300:
        return 12.0
    if d <= 500:
        return 25.0
    if d <= 800:
        return 50.0
    return 100.0


def main():
    print("=" * 80)
    print("Real Bayesian-regression-style reliability program benchmark")
    print("=" * 80)
    print()

    cases = [
        # (K, N)  d = K + N
        (3, 50),    # d=53,   small reference
        (10, 50),   # d=60
        (10, 100),  # d=110
        (20, 50),   # d=70
        (20, 100),  # d=120
        (20, 200),  # d=220
        (50, 50),   # d=100
        (50, 100),  # d=150
        (50, 200),  # d=250
        (100, 50),  # d=150
        (100, 100), # d=200
    ]

    print(f"Generating programs in {REAL_INPUTS}")
    for K, N in cases:
        gen_pattern_B(K, N, seed=42, out_dir=REAL_INPUTS)
    print()

    print(f"{'K':>4s} {'N':>4s} {'d':>5s}  {'SOGA (ms)':>11s}  {'n_comp':>7s}  "
          f"{'proj speedup':>14s}  {'proj SOGA (ms)':>15s}")
    print("-" * 80)
    rows = []
    for K, N in cases:
        path = os.path.join(REAL_INPUTS, f"B_K{K}N{N}_V1.soga")
        # Warmup
        ok, _ = with_timeout(60.0, time_one, path)
        if not ok:
            print(f"{K:>4d} {N:>4d} {K+N:>5d}  TIMEOUT")
            rows.append({"K": K, "N": N, "d": K+N, "status": "timeout"})
            continue
        # Two measured runs
        runs = []
        for _ in range(2):
            ok, r = with_timeout(60.0, time_one, path)
            if not ok:
                break
            runs.append(r)
        if not runs:
            print(f"{K:>4d} {N:>4d} {K+N:>5d}  TIMEOUT (measured)")
            rows.append({"K": K, "N": N, "d": K+N, "status": "timeout"})
            continue
        total_ms = mean(r["total_s"] for r in runs) * 1000
        d = runs[0]["d"]
        n_comp = runs[0]["n_comp"]
        spd = project_speedup(d)
        proj_ms = total_ms / spd
        print(f"{K:>4d} {N:>4d} {d:>5d}  {total_ms:>8.1f}ms   {n_comp:>7d}   "
              f"{spd:>10.1f}x     {proj_ms:>10.1f}ms")
        rows.append({"K": K, "N": N, "d": d, "n_comp": n_comp,
                     "soga_ms": total_ms,
                     "projected_speedup": spd,
                     "projected_soga_ms": proj_ms,
                     "status": "ok"})

    out_csv = os.path.join(HERE, "results", "real_program_bench.csv")
    with open(out_csv, "w", newline="") as f:
        if rows:
            fields = ["K", "N", "d", "n_comp", "status",
                     "soga_ms", "projected_speedup", "projected_soga_ms"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
