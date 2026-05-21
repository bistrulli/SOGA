"""
3-way A/B benchmark across the canonical SOGA suite:
  classic (default) vs --sparse-truncate vs --vectorize-truncate.

Same methodology as bench_all_canonical.py: compile each .soga once, run
start_SOGA N times per mode on freshly-built CFGs, hard SIGALRM timeout
per run.

Outputs:
  - results/bench_3way.csv
  - BENCH_3WAY_REPORT.md
"""

import os
import sys
import signal
import time
import csv
import random
from statistics import mean, stdev

import numpy as np


class HardTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise HardTimeout()


def with_timeout(seconds, fn, *args, **kwargs):
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(seconds) + 1)
    try:
        result = fn(*args, **kwargs)
        return True, result
    except HardTimeout:
        return False, "HardTimeout"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from producecfg import produce_cfg  # noqa: E402
from libSOGA import start_SOGA  # noqa: E402
from libSOGAtruncate import set_sparse_truncate, set_vectorize_truncate  # noqa: E402
from sogaPreprocessor import compile2SOGA  # noqa: E402


PROGRAM_TIMEOUT_S = 60.0
N_RUNS = 3
WARMUP = 1
MODES = ("classic", "sparse", "vectorize")


def discover_programs():
    out = []
    for d in (os.path.join(ROOT, "programs", "SOGA"),
              os.path.join(ROOT, "programs", "Example")):
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            full = os.path.join(d, name)
            if os.path.isfile(full) and name.endswith(".soga"):
                out.append((f"{os.path.basename(d)}/{name}", full))
    return out


def prepare(path):
    random.seed(0); np.random.seed(0)
    compiled = compile2SOGA(path)
    cfg = produce_cfg(compiled)
    return compiled


def configure_mode(mode):
    if mode == "classic":
        set_sparse_truncate(False); set_vectorize_truncate(False)
    elif mode == "sparse":
        set_sparse_truncate(True); set_vectorize_truncate(False)
    elif mode == "vectorize":
        set_sparse_truncate(True); set_vectorize_truncate(True)


def run_soga(compiled, mode):
    """Single timed start_SOGA call on a fresh CFG."""
    configure_mode(mode)
    random.seed(0); np.random.seed(0)
    cfg = produce_cfg(compiled)
    t0 = time.perf_counter()
    out = start_SOGA(cfg, useR=False, parallel=None,
                     sparse_truncate=(mode != "classic"),
                     vectorize_truncate=(mode == "vectorize"))
    return time.perf_counter() - t0, out


def bench_program(label, path):
    try:
        ok, compiled = with_timeout(PROGRAM_TIMEOUT_S, prepare, path)
        if not ok:
            return {"label": label, "status": "timeout in setup"}
    except Exception as e:
        return {"label": label, "status": f"error in setup: {type(e).__name__}: {str(e)[:80]}"}

    results = {}
    d, n_comp = None, None
    means_per_mode = {}
    for mode in MODES:
        try:
            # Warmup
            for _ in range(WARMUP):
                ok, _ = with_timeout(PROGRAM_TIMEOUT_S, run_soga, compiled, mode)
                if not ok:
                    results[mode] = {"status": "timeout warmup", "times": []}
                    break
            else:
                times = []
                last_out = None
                for _ in range(N_RUNS):
                    ok, res = with_timeout(PROGRAM_TIMEOUT_S, run_soga, compiled, mode)
                    if not ok:
                        results[mode] = {"status": "timeout measured", "times": times}
                        break
                    t, out = res
                    times.append(t)
                    last_out = out
                else:
                    results[mode] = {"status": "ok", "times": times}
                    if d is None and last_out is not None:
                        d = len(last_out.var_list)
                        n_comp = last_out.gm.n_comp()
                    if last_out is not None:
                        means_per_mode[mode] = dict(zip(last_out.var_list, [float(v) for v in last_out.gm.mean()]))
        except Exception as e:
            results[mode] = {"status": f"error: {type(e).__name__}: {str(e)[:80]}", "times": []}

    row = {"label": label, "d": d if d is not None else -1, "n_comp": n_comp if n_comp is not None else -1}
    for mode in MODES:
        r = results.get(mode, {})
        ts = r.get("times", [])
        row[f"{mode}_status"] = r.get("status", "missing")
        row[f"{mode}_mean_ms"] = mean(ts) * 1000 if ts else None
        row[f"{mode}_min_ms"] = min(ts) * 1000 if ts else None

    # Speedups (only if all modes ok)
    if all(results.get(m, {}).get("status") == "ok" for m in MODES):
        c, s, v = (mean(results[m]["times"]) for m in MODES)
        row["spd_sparse_over_classic"] = c / s
        row["spd_vec_over_classic"] = c / v
        row["spd_vec_over_sparse"] = s / v

    # Equivalence check: vectorize vs classic on E[var]
    if "classic" in means_per_mode and "vectorize" in means_per_mode:
        max_diff = 0.0
        common = set(means_per_mode["classic"].keys()) & set(means_per_mode["vectorize"].keys())
        for v in common:
            a, b = means_per_mode["classic"][v], means_per_mode["vectorize"][v]
            diff = abs(a - b) / max(abs(a), abs(b), 1.0)
            max_diff = max(max_diff, diff)
        row["max_rel_diff_E"] = max_diff
    return row


def render_table(rows):
    ok_rows = [r for r in rows if r.get("vectorize_status") == "ok" and r.get("classic_status") == "ok"]
    fail_rows = [r for r in rows if r not in ok_rows]
    ok_rows.sort(key=lambda r: -r.get("spd_vec_over_sparse", 0))
    out = []
    out.append("| Program | d | n_comp | classic ms | sparse ms | vectorize ms | sp/cl | vec/cl | vec/sp |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in ok_rows:
        out.append(
            f"| {r['label']} | {r['d']} | {r['n_comp']} "
            f"| {r['classic_mean_ms']:.2f} | {r['sparse_mean_ms']:.2f} | {r['vectorize_mean_ms']:.2f} "
            f"| {r.get('spd_sparse_over_classic', 0):.2f}x "
            f"| {r.get('spd_vec_over_classic', 0):.2f}x "
            f"| **{r.get('spd_vec_over_sparse', 0):.2f}x** |"
        )
    if fail_rows:
        out.append("")
        out.append("**Failed / partial:**")
        for r in fail_rows:
            statuses = [f"{m}={r.get(f'{m}_status','?')}" for m in MODES]
            out.append(f"- `{r['label']}`: " + ", ".join(statuses))
    return "\n".join(out)


def main():
    programs = discover_programs()
    print(f"Found {len(programs)} programs. Running N={N_RUNS} + {WARMUP} warmup per mode (3 modes), timeout {PROGRAM_TIMEOUT_S}s per run.\n")
    rows = []
    t_start = time.perf_counter()
    for i, (label, path) in enumerate(programs):
        sys.stdout.write(f"[{i+1:2d}/{len(programs)}] {label} ... ")
        sys.stdout.flush()
        t0 = time.perf_counter()
        r = bench_program(label, path)
        rows.append(r)
        elapsed = time.perf_counter() - t0
        if all(r.get(f"{m}_status") == "ok" for m in MODES):
            sys.stdout.write(
                f"d={r['d']:3d} n_comp={r['n_comp']:5d}  "
                f"c={r['classic_mean_ms']:7.1f}ms s={r['sparse_mean_ms']:7.1f}ms v={r['vectorize_mean_ms']:7.1f}ms  "
                f"-> sp/cl={r.get('spd_sparse_over_classic',0):.2f}x  vec/sp={r.get('spd_vec_over_sparse',0):.2f}x  "
                f"({elapsed:.1f}s)\n"
            )
        else:
            statuses = [f"{m}={r.get(f'{m}_status','?')[:14]}" for m in MODES]
            sys.stdout.write(" " + ", ".join(statuses) + f"  ({elapsed:.1f}s)\n")
        sys.stdout.flush()
    total = time.perf_counter() - t_start
    print(f"\nTotal wall: {total:.1f}s")

    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "bench_3way.csv")
    fields = ["label", "d", "n_comp"]
    for m in MODES:
        fields += [f"{m}_status", f"{m}_mean_ms", f"{m}_min_ms"]
    fields += ["spd_sparse_over_classic", "spd_vec_over_classic", "spd_vec_over_sparse", "max_rel_diff_E"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved CSV: {csv_path}")

    body = []
    body.append("# A/B/C benchmark — canonical SOGA suite, classic vs --sparse-truncate vs --vectorize-truncate")
    body.append("")
    body.append(f"- Programs: {len(rows)}, runs per mode: {N_RUNS}, warmup: {WARMUP}, hard timeout: {PROGRAM_TIMEOUT_S}s")
    body.append(f"- Wall: {total:.1f}s")
    body.append("")
    ok = [r for r in rows if r.get("vectorize_status") == "ok" and r.get("sparse_status") == "ok"]
    if ok:
        spd_vs = [r.get("spd_vec_over_sparse", 1.0) for r in ok]
        spd_vc = [r.get("spd_vec_over_classic", 1.0) for r in ok]
        spd_sc = [r.get("spd_sparse_over_classic", 1.0) for r in ok]
        body.append("## Aggregate (geometric mean over OK programs)")
        body.append("")
        body.append(f"- sparse over classic: **{float(np.exp(np.mean(np.log(spd_sc)))):.2f}x**")
        body.append(f"- vectorize over classic: **{float(np.exp(np.mean(np.log(spd_vc)))):.2f}x**")
        body.append(f"- vectorize over sparse: **{float(np.exp(np.mean(np.log(spd_vs)))):.2f}x**")
        body.append(f"- vectorize over sparse, median: {float(np.median(spd_vs)):.2f}x")
        body.append(f"- vectorize over sparse, max: {max(spd_vs):.2f}x")
        max_diff = max((r.get("max_rel_diff_E", 0) or 0) for r in ok)
        body.append(f"- max relative diff E[var] across modes: {max_diff:.2e} (equivalence check)")
    body.append("")
    body.append("## Per-program results (sorted by vec/sparse speedup desc)")
    body.append("")
    body.append(render_table(rows))
    md_path = os.path.join(HERE, "BENCH_3WAY_REPORT.md")
    with open(md_path, "w") as f:
        f.write("\n".join(body) + "\n")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
