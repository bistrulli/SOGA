"""
A/B benchmark: SOGA classic vs --sparse-truncate on real and synthetic programs.

Programs tested:
  - BayesPointMachine.soga (real, d=9)
  - B_K3N{50,100}_V1.soga, C_T{60,100,150}_V1.soga (synthetic from prior study)

For each program:
  - Run in-process N=3 times after 1 warm-up, sparse OFF and ON
  - Verify E[var] outputs match across modes
  - Report wall-clock speedup
"""

import os
import sys
import time
import random
import numpy as np
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)

from producecfg import produce_cfg
from libSOGA import start_SOGA
from sogaPreprocessor import compile2SOGA


def time_run(soga_path, sparse_truncate):
    random.seed(0)
    np.random.seed(0)
    t0 = time.perf_counter()
    compiled = compile2SOGA(soga_path)
    t1 = time.perf_counter()
    cfg = produce_cfg(compiled)
    t2 = time.perf_counter()
    out = start_SOGA(cfg, useR=False, parallel=None, sparse_truncate=sparse_truncate)
    t3 = time.perf_counter()
    return {
        "preproc_s": t1 - t0,
        "cfg_s": t2 - t1,
        "soga_s": t3 - t2,
        "total_s": t3 - t0,
        "d": len(out.var_list),
        "n_comp": out.gm.n_comp(),
        "means": dict(zip(out.var_list, [float(v) for v in out.gm.mean()])),
    }


def bench_one(label, soga_path, n_runs=3, target_vars=None):
    print(f"\n=== {label} ===")
    # Warmup once each
    time_run(soga_path, sparse_truncate=False)
    time_run(soga_path, sparse_truncate=True)
    classic = [time_run(soga_path, False) for _ in range(n_runs)]
    sparse = [time_run(soga_path, True) for _ in range(n_runs)]
    c_total = [r["total_s"] for r in classic]
    s_total = [r["total_s"] for r in sparse]
    c_soga = [r["soga_s"] for r in classic]
    s_soga = [r["soga_s"] for r in sparse]
    d = classic[0]["d"]
    n_comp = classic[0]["n_comp"]
    speedup_total = mean(c_total) / mean(s_total)
    speedup_soga = mean(c_soga) / mean(s_soga)

    # Verify equivalence
    eq_ok = True
    max_diff = 0.0
    vars_check = target_vars or list(classic[0]["means"].keys())
    for v in vars_check:
        mc = classic[0]["means"].get(v)
        ms = sparse[0]["means"].get(v)
        if mc is None or ms is None:
            continue
        diff = abs(mc - ms)
        max_diff = max(max_diff, diff)
        if diff > 1e-3 * max(abs(mc), 1.0):
            eq_ok = False
    print(f"  d={d}  n_comp={n_comp}")
    print(f"  classic total: mean={mean(c_total)*1000:.2f}ms  soga={mean(c_soga)*1000:.2f}ms")
    print(f"  sparse  total: mean={mean(s_total)*1000:.2f}ms  soga={mean(s_soga)*1000:.2f}ms")
    print(f"  speedup total: {speedup_total:.2f}x   soga-only: {speedup_soga:.2f}x")
    print(f"  equivalence: {'OK' if eq_ok else 'FAIL'}   max |delta E[var]|: {max_diff:.2e}")
    return {
        "label": label,
        "d": d,
        "n_comp": n_comp,
        "classic_total_ms": mean(c_total) * 1000,
        "classic_soga_ms": mean(c_soga) * 1000,
        "sparse_total_ms": mean(s_total) * 1000,
        "sparse_soga_ms": mean(s_soga) * 1000,
        "speedup_total": speedup_total,
        "speedup_soga": speedup_soga,
        "equivalence_ok": eq_ok,
        "max_diff": max_diff,
    }


def main():
    here = HERE
    PROGRAMS_DIR = os.path.abspath(os.path.join(SRC, "..", "programs", "SOGA"))
    SYNTH_DIR = os.path.abspath(os.path.join(here, "..", "feasibility_dead_var_pruning_2026-05-20", "inputs"))

    cases = [
        ("BayesPointMachine (d=9)", os.path.join(PROGRAMS_DIR, "BayesPointMachine.soga"), ["w[0]", "w[1]", "w[2]"]),
        ("TrueSkills (d=6)", os.path.join(PROGRAMS_DIR, "TrueSkills.soga"), ["skillA", "skillB", "skillC"]),
        ("B_K3N50 (d=53)", os.path.join(SYNTH_DIR, "B_K3N50_V1.soga"), ["w[0]", "w[1]", "w[2]"]),
        ("B_K3N100 (d=103)", os.path.join(SYNTH_DIR, "B_K3N100_V1.soga"), ["w[0]", "w[1]", "w[2]"]),
        ("C_T60 (d=61)", os.path.join(SYNTH_DIR, "C_T60_V1.soga"), None),
        ("C_T100 (d=101)", os.path.join(SYNTH_DIR, "C_T100_V1.soga"), None),
        ("C_T150 (d=151)", os.path.join(SYNTH_DIR, "C_T150_V1.soga"), None),
    ]

    rows = []
    for label, path, vars_check in cases:
        if not os.path.exists(path):
            print(f"\n!! Skipping {label}: file not found at {path}")
            continue
        try:
            rows.append(bench_one(label, path, n_runs=3, target_vars=vars_check))
        except Exception as e:
            print(f"\n!! {label} FAILED: {e}")

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Program':<28s} {'d':>4s} {'classic ms':>11s} {'sparse ms':>10s} {'spd total':>10s} {'spd soga':>10s} {'eq':>4s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['label']:<28s} {r['d']:>4d} {r['classic_total_ms']:>9.2f}   {r['sparse_total_ms']:>8.2f}   {r['speedup_total']:>8.2f}x   {r['speedup_soga']:>8.2f}x   {'OK' if r['equivalence_ok'] else 'FAIL':>4s}")

    # Save CSV
    import csv
    out_csv = os.path.join(here, "results", "bench_ab.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nSaved A/B comparison to {out_csv}")


if __name__ == "__main__":
    main()
