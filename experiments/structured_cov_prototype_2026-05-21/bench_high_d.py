"""
High-dimensional benchmark: pushes SOGA (current dense vectorize-truncate)
toward its scalability limit and projects the structured-cov speedup
based on the prototype's per-op measurements.

Generates two families:
  - Pattern B (BayesPointMachine-like, K=3 latents + N observations)
    at N in {25, 50, 100, 200, 300, 500} -> d in {28, 53, 103, 203, 303, 503}
  - Pattern C (Markov random walk + observe at every step)
    at T in {50, 100, 200, 400, 800} -> d in {51, 101, 201, 401, 801}

For each:
  - Runs SOGA in vectorize mode (current best dense path)
  - Times preprocess + CFG + soga separately
  - Captures n_comp and d
  - Hard SIGALRM timeout per run (180 s)

Then projects the predicted structured-cov runtime by re-using the
per-operation speedups from `prototype.py`:
  - matvec/quad: ~d/(few) at high d
  - inv_apply: ~d/k at high d (k expected = 2 from the sigma analysis)
"""

import os
import sys
import signal
import time
import csv
from statistics import mean

import numpy as np


class HardTimeout(Exception):
    pass


def _alarm_handler(s, f):
    raise HardTimeout()


def with_timeout(seconds, fn, *args, **kwargs):
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(seconds) + 1)
    try:
        return True, fn(*args, **kwargs)
    except HardTimeout:
        return False, "HardTimeout"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

# Reuse the existing generators
DEAD_VAR_INPUTS = os.path.join(ROOT, "experiments", "feasibility_dead_var_pruning_2026-05-20", "inputs")
sys.path.insert(0, os.path.join(ROOT, "experiments", "feasibility_dead_var_pruning_2026-05-20"))
from gen_programs import gen_pattern_B, gen_pattern_C  # noqa: E402

from producecfg import produce_cfg  # noqa: E402
from libSOGA import start_SOGA  # noqa: E402
from sogaPreprocessor import compile2SOGA  # noqa: E402

import random


HIGH_D_INPUTS = os.path.join(HERE, "high_d_inputs")
os.makedirs(HIGH_D_INPUTS, exist_ok=True)

TIMEOUT_S = 180.0
N_RUNS = 2  # high-d runs are slow; 2 measured + 1 warmup


def time_one_run(prog_path):
    random.seed(0); np.random.seed(0)
    t0 = time.perf_counter()
    compiled = compile2SOGA(prog_path)
    t1 = time.perf_counter()
    cfg = produce_cfg(compiled)
    t2 = time.perf_counter()
    out = start_SOGA(cfg, useR=False, parallel=None,
                     sparse_truncate=True, vectorize_truncate=True)
    t3 = time.perf_counter()
    return {
        "preproc_s": t1 - t0,
        "cfg_s": t2 - t1,
        "soga_s": t3 - t2,
        "total_s": t3 - t0,
        "d": len(out.var_list),
        "n_comp": out.gm.n_comp(),
    }


def project_structured_speedup(d, k=2):
    """Use prototype timing data to project structured-cov speedup at this d.

    inv_apply scaling (from prototype.py timing output):
      d=100, k=1: 11.5x ; k=2: 9.9x
      d=300, k=1: 106x  ; k=2: 84x
      d=1000, k=1: 1250x ; k=2: 810x

    matvec scaling:
      d=100: ~0.7-1x (overhead bound)
      d=300: ~3x
      d=1000: ~20-25x

    SOGA truncate is dominated by inv_apply-equivalent ops; matvec
    contributes too. Take a geometric blend.
    """
    # Interpolate inv_apply speedup vs d (log-log)
    table_d = [100, 300, 1000]
    table_inv = [9.9, 84, 810]  # k=2
    table_matvec = [0.73, 3.2, 20.3]
    if d <= 100:
        return 1.0  # below threshold
    if d >= 1000:
        idx = 2
    elif d >= 300:
        # linear in log d between 300 and 1000
        f = (np.log(d) - np.log(300)) / (np.log(1000) - np.log(300))
        return float(np.exp(np.log(table_inv[1])*(1-f) + np.log(table_inv[2])*f) **0.4
                     * np.exp(np.log(table_matvec[1])*(1-f) + np.log(table_matvec[2])*f)**0.6)
    else:
        f = (np.log(d) - np.log(100)) / (np.log(300) - np.log(100))
        return float(np.exp(np.log(table_inv[0])*(1-f) + np.log(table_inv[1])*f)**0.4
                     * np.exp(np.log(table_matvec[0])*(1-f) + np.log(table_matvec[1])*f)**0.6)


def main():
    print("=" * 90)
    print("High-d benchmark — current SOGA (vectorize-truncate) vs projected structured-cov")
    print("=" * 90)

    # Generate Pattern B at N = {25, 50, 100, 200, 300, 500}
    print("\nGenerating Pattern B (K=3 + N observations) family...")
    pat_B = []
    for N in (25, 50, 100, 200, 300, 500):
        v1, _ = gen_pattern_B(K=3, N=N, seed=42, out_dir=HIGH_D_INPUTS)
        pat_B.append(("B", N, v1, 3 + N))
        print(f"  B_K3N{N}.soga  d={3+N}")

    # Generate Pattern C at T = {50, 100, 200, 400, 800}
    print("\nGenerating Pattern C (Markov random walk) family...")
    pat_C = []
    for T in (50, 100, 200, 400, 800):
        v1, _ = gen_pattern_C(T=T, seed=42, out_dir=HIGH_D_INPUTS)
        pat_C.append(("C", T, v1, T + 1))
        print(f"  C_T{T}.soga  d={T+1}")

    all_progs = pat_B + pat_C

    print("\n" + "=" * 90)
    print(f"{'Family':>6s} {'param':>6s} {'d':>5s}  {'preproc':>9s}  {'cfg':>9s}  "
          f"{'soga':>10s}  {'total':>10s}  {'n_comp':>7s}  {'proj.':>8s}")
    print("-" * 90)

    rows = []
    for fam, param, path, d_expected in all_progs:
        # Warmup
        try:
            ok, _ = with_timeout(TIMEOUT_S, time_one_run, path)
            if not ok:
                print(f"{fam:>6s} {param:>6d} {d_expected:>5d}  TIMEOUT (warmup)")
                rows.append({"family": fam, "param": param, "d": d_expected, "status": "timeout"})
                continue
        except Exception as e:
            print(f"{fam:>6s} {param:>6d} {d_expected:>5d}  ERROR: {type(e).__name__}")
            rows.append({"family": fam, "param": param, "d": d_expected, "status": "error"})
            continue
        # Measured
        runs = []
        timed_out = False
        for _ in range(N_RUNS):
            ok, res = with_timeout(TIMEOUT_S, time_one_run, path)
            if not ok:
                timed_out = True
                break
            runs.append(res)
        if timed_out:
            print(f"{fam:>6s} {param:>6d} {d_expected:>5d}  TIMEOUT (measured)")
            rows.append({"family": fam, "param": param, "d": d_expected, "status": "timeout"})
            continue
        r = {k: mean([r[k] for r in runs]) for k in ["preproc_s", "cfg_s", "soga_s", "total_s"]}
        r["d"] = runs[0]["d"]
        r["n_comp"] = runs[0]["n_comp"]
        spd = project_structured_speedup(r["d"], k=2)
        projected_soga_s = r["soga_s"] / spd if spd > 1 else r["soga_s"]
        print(f"{fam:>6s} {param:>6d} {r['d']:>5d}  "
              f"{r['preproc_s']*1000:>7.1f}ms  {r['cfg_s']*1000:>7.1f}ms  "
              f"{r['soga_s']*1000:>8.1f}ms  {r['total_s']*1000:>8.1f}ms  "
              f"{r['n_comp']:>7d}  {spd:>6.1f}x")
        rows.append({
            "family": fam, "param": param, "d": r["d"], "n_comp": r["n_comp"],
            "preproc_s": r["preproc_s"], "cfg_s": r["cfg_s"],
            "soga_s": r["soga_s"], "total_s": r["total_s"],
            "projected_speedup_structured": spd,
            "projected_soga_s": projected_soga_s,
            "status": "ok",
        })

    out_csv = os.path.join(HERE, "results", "bench_high_d.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = ["family", "param", "d", "n_comp", "status",
              "preproc_s", "cfg_s", "soga_s", "total_s",
              "projected_speedup_structured", "projected_soga_s"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved CSV: {out_csv}")


if __name__ == "__main__":
    main()
