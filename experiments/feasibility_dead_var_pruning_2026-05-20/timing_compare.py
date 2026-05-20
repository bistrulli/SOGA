"""
In-process timing comparison V1 vs V2_inline vs V2_tmp.

Bypasses SOGA's multiprocessing.Process wrapper (which adds ~1s overhead).
Measures: preprocessing + CFG + start_SOGA, repeated N times.
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
        "E_w0":      float(out.gm.mean()[out.var_list.index("w[0]")]),
        "E_w1":      float(out.gm.mean()[out.var_list.index("w[1]")]),
        "E_w2":      float(out.gm.mean()[out.var_list.index("w[2]")]),
    }


def main():
    here = os.path.dirname(__file__)
    inputs = {
        "V1":         os.path.join(here, "inputs/BayesPointMachine_V1.soga"),
        "V2_tmp":     os.path.join(here, "inputs/BayesPointMachine_V2_tmp.soga"),
        "V2_inline":  os.path.join(here, "inputs/BayesPointMachine_V2_inline.soga"),
    }
    N = 5
    WARMUP = 1

    rows = []
    summary = {}
    for name, path in inputs.items():
        print(f"\n=== {name} ===")
        # warmup
        for _ in range(WARMUP):
            time_one_run(path)
        # measured
        runs = [time_one_run(path) for _ in range(N)]
        for r in runs:
            r["variant"] = name
            rows.append(r)

        totals  = [r["total_s"]   for r in runs]
        sogas   = [r["soga_s"]    for r in runs]
        cfgs    = [r["cfg_s"]     for r in runs]
        d       = runs[0]["d"]

        summary[name] = {
            "d": d,
            "total_mean": mean(totals),  "total_std": stdev(totals),  "total_min": min(totals),
            "soga_mean":  mean(sogas),   "soga_std":  stdev(sogas),   "soga_min":  min(sogas),
            "cfg_mean":   mean(cfgs),    "cfg_std":   stdev(cfgs),
        }
        print(f"  d={d}, n_comp={runs[0]['n_comp']}")
        print(f"  E[w] = [{runs[0]['E_w0']:.5f}, {runs[0]['E_w1']:.5f}, {runs[0]['E_w2']:.5f}]")
        print(f"  preproc+cfg+soga mean={mean(totals)*1000:.2f}ms  std={stdev(totals)*1000:.2f}ms  min={min(totals)*1000:.2f}ms")
        print(f"            soga only mean={mean(sogas)*1000:.2f}ms  std={stdev(sogas)*1000:.2f}ms  min={min(sogas)*1000:.2f}ms")

    # Save raw CSV
    out_csv = os.path.join(here, "results/timings_raw.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nRaw timings saved to {out_csv}")

    # Save summary CSV
    out_csv = os.path.join(here, "results/timings_summary.csv")
    with open(out_csv, "w", newline="") as f:
        cols = ["variant","d","total_mean","total_std","total_min","soga_mean","soga_std","soga_min","cfg_mean","cfg_std"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k, s in summary.items():
            w.writerow({"variant": k, **s})
    print(f"Summary saved to {out_csv}")

    # Print speedups
    print("\n=== Speedups vs V1 (total) ===")
    v1_t = summary["V1"]["total_mean"]
    v1_s = summary["V1"]["soga_mean"]
    for k, s in summary.items():
        if k == "V1": continue
        print(f"  V1 / {k}:  total={v1_t/s['total_mean']:.2f}x   soga_only={v1_s/s['soga_mean']:.2f}x")

if __name__ == "__main__":
    main()
