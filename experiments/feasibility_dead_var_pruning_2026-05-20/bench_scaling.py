"""
Scaling benchmark V1 vs V2_tmp across program sizes for Patterns B and C.

Reports:
- total_s (preprocessing + produce_cfg + start_SOGA)
- soga_s  (start_SOGA only — what auto-pruning would affect)
- cfg_s   (produce_cfg only — measures the ANTLR parsing cost difference)
"""

import sys
import os
import time
import csv
from statistics import mean, stdev

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from producecfg import produce_cfg
from libSOGA import start_SOGA
from sogaPreprocessor import compile2SOGA

import numpy as np
import random


def time_one_run(input_path):
    random.seed(0)
    np.random.seed(0)
    t0 = time.perf_counter()
    compiled = compile2SOGA(input_path)
    t1 = time.perf_counter()
    cfg = produce_cfg(compiled)
    t2 = time.perf_counter()
    out = start_SOGA(cfg, useR=False, parallel=None)
    t3 = time.perf_counter()
    return {
        "preproc_s": t1 - t0,
        "cfg_s":     t2 - t1,
        "soga_s":    t3 - t2,
        "total_s":   t3 - t0,
        "d":         len(out.var_list),
        "n_comp":    out.gm.n_comp(),
    }


def bench(input_path, N=3, warmup=1):
    for _ in range(warmup):
        time_one_run(input_path)
    runs = [time_one_run(input_path) for _ in range(N)]
    return runs


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    inputs = os.path.join(here, "inputs")
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Pattern B configurations
    B_sizes = [6, 12, 25, 50, 75]
    # Pattern C configurations
    C_sizes = [5, 10, 20, 40, 60]

    rows = []
    print(f"\n{'pattern':9s} {'size':>5s} {'variant':>8s} {'d':>4s} "
          f"{'total ms':>10s} {'soga ms':>10s} {'cfg ms':>10s}")
    print("-" * 70)

    def record(pattern, size, variant, runs):
        if runs is None:
            print(f"{pattern:9s} {size:>5d} {variant:>8s}   -  TIMEOUT or ERROR")
            return
        d = runs[0]["d"]
        totals = [r["total_s"] for r in runs]
        sogas  = [r["soga_s"]  for r in runs]
        cfgs   = [r["cfg_s"]   for r in runs]
        m_total = mean(totals); s_total = stdev(totals) if len(totals) > 1 else 0.0
        m_soga  = mean(sogas);  s_soga  = stdev(sogas) if len(sogas) > 1 else 0.0
        m_cfg   = mean(cfgs);   s_cfg   = stdev(cfgs) if len(cfgs) > 1 else 0.0
        print(f"{pattern:9s} {size:>5d} {variant:>8s} {d:>4d}  "
              f"{m_total*1000:>7.2f}±{s_total*1000:.2f}  "
              f"{m_soga*1000:>7.2f}±{s_soga*1000:.2f}  "
              f"{m_cfg*1000:>7.2f}±{s_cfg*1000:.2f}")
        rows.append({
            "pattern": pattern, "size": size, "variant": variant, "d": d,
            "total_mean_s": m_total, "total_std_s": s_total,
            "soga_mean_s":  m_soga,  "soga_std_s":  s_soga,
            "cfg_mean_s":   m_cfg,   "cfg_std_s":   s_cfg,
        })

    # === Pattern B ===
    print("\n=== Pattern B (BayesPointMachine-like; K=3 latents + N observations) ===")
    for N in B_sizes:
        for variant in ["V1", "V2"]:
            path = os.path.join(inputs, f"B_K{3}N{N}_{variant}.soga")
            try:
                runs = bench(path, N=3, warmup=1)
                record("B", N, variant, runs)
            except Exception as e:
                print(f"B N={N} {variant} ERROR: {e}")

    # === Pattern C ===
    print("\n=== Pattern C (Markov sequential; T-step random walk) ===")
    for T in C_sizes:
        for variant in ["V1", "V2"]:
            path = os.path.join(inputs, f"C_T{T}_{variant}.soga")
            try:
                runs = bench(path, N=3, warmup=1)
                record("C", T, variant, runs)
            except Exception as e:
                print(f"C T={T} {variant} ERROR: {e}")

    # Save CSV
    out_csv = os.path.join(results_dir, "scaling_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{out_csv} written ({len(rows)} rows)")

    # Speedups
    print("\n=== Speedups (V1/V2) ===")
    print(f"{'pattern':9s} {'size':>5s} {'d_V1':>5s} {'total':>8s} {'soga_only':>10s}")
    # group rows by (pattern, size)
    by_key = {}
    for r in rows:
        by_key.setdefault((r["pattern"], r["size"]), {})[r["variant"]] = r
    for (pat, sz), pair in sorted(by_key.items()):
        if "V1" in pair and "V2" in pair:
            v1, v2 = pair["V1"], pair["V2"]
            su_total = v1["total_mean_s"] / v2["total_mean_s"]
            su_soga  = v1["soga_mean_s"]  / v2["soga_mean_s"]
            print(f"{pat:9s} {sz:>5d} {v1['d']:>5d}  {su_total:>7.2f}x   {su_soga:>9.2f}x")


if __name__ == "__main__":
    main()
