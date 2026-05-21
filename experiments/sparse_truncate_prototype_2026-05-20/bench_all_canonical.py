"""
A/B benchmark across the canonical SOGA program suite, comparing the classic
and sparse-aware (--sparse-truncate) truncate paths.

Compiles each .soga once (cached as a compiled SOGA text file by
compile2SOGA) and reuses the produced CFG across runs of both modes.
Only the dispatcher loop in start_SOGA is timed — that is where
--sparse-truncate has an effect. compile2SOGA (sklearn EM fitting for
non-Gaussian primitives like uniform/beta) and produce_cfg (ANTLR) are
measured once per program, attributed to "setup_s", and excluded from
the speedup ratio.

For every program:
  - 1 warm-up + N=3 measured start_SOGA runs in each mode.
  - Compares E[var] from a representative run (last one in each mode);
    flags any mismatch above relative tolerance 1e-3.
  - Skips programs where the first classic run exceeds PROGRAM_TIMEOUT_S.

Outputs:
  - results/bench_all_canonical.csv  (raw per-program data)
  - BENCH_ALL_REPORT.md              (rendered summary table + statistics)
"""

import os
import sys
import signal
import time
import csv
import random
from copy import deepcopy
from statistics import mean, stdev

import numpy as np


class HardTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise HardTimeout()


def with_timeout(seconds, fn, *args, **kwargs):
    """Run fn with a hard SIGALRM timeout. POSIX only. Returns (ok, result_or_exc)."""
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    # signal.alarm() takes integer seconds; round up to be safe
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
from libSOGAtruncate import set_sparse_truncate  # noqa: E402
from sogaPreprocessor import compile2SOGA  # noqa: E402


PROGRAM_TIMEOUT_S = 60.0   # hard SIGALRM timeout per run
N_RUNS = 3
WARMUP = 1


def discover_programs():
    """Top-level programs/SOGA/*.soga + programs/Example/Bernoulli.soga."""
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
    """Run preprocessing + CFG construction once. Returns (cfg_factory, setup_s)."""
    random.seed(0); np.random.seed(0)
    t0 = time.perf_counter()
    compiled = compile2SOGA(path)
    cfg = produce_cfg(compiled)
    setup_s = time.perf_counter() - t0
    return compiled, cfg, setup_s


def reload_cfg(compiled):
    """Rebuild a fresh CFG (each start_SOGA mutates node state, so we cannot reuse)."""
    random.seed(0); np.random.seed(0)
    return produce_cfg(compiled)


def run_soga(compiled, sparse):
    """Single timed start_SOGA call on a fresh CFG. Returns (soga_s, dist)."""
    cfg = reload_cfg(compiled)
    random.seed(0); np.random.seed(0)
    t0 = time.perf_counter()
    out = start_SOGA(cfg, useR=False, parallel=None, sparse_truncate=sparse)
    return time.perf_counter() - t0, out


def compare_means(d1, d2, rel_tol=1e-3, abs_tol=1e-6):
    if set(d1.keys()) != set(d2.keys()):
        return False, float("inf"), "var_list mismatch"
    max_diff, bad = 0.0, None
    for v in d1:
        diff = abs(d1[v] - d2[v])
        scale = max(abs(d1[v]), abs(d2[v]), 1.0)
        if diff > max_diff:
            max_diff = diff
        if diff > max(rel_tol * scale, abs_tol) and bad is None:
            bad = v
    return bad is None, max_diff, bad


def dist_means(dist):
    return dict(zip(dist.var_list, [float(v) for v in dist.gm.mean()]))


def bench_program(label, path):
    try:
        ok, prep_res = with_timeout(PROGRAM_TIMEOUT_S, prepare, path)
        if not ok:
            return {"label": label, "status": f"timeout in compile2SOGA/produce_cfg (>{PROGRAM_TIMEOUT_S}s)"}
        compiled, cfg_dummy, setup_s = prep_res

        # Probe: one classic run for timeout detection (hard SIGALRM enforced)
        ok, probe_res = with_timeout(PROGRAM_TIMEOUT_S, run_soga, compiled, False)
        if not ok:
            return {"label": label, "status": f"timeout in classic start_SOGA (>{PROGRAM_TIMEOUT_S}s)"}
        t_probe, out_classic_probe = probe_res
        d = len(out_classic_probe.var_list)
        n_comp = out_classic_probe.gm.n_comp()

        # Warmup (also under hard timeout)
        for _ in range(WARMUP):
            ok, _ = with_timeout(PROGRAM_TIMEOUT_S, run_soga, compiled, False)
            if not ok:
                return {"label": label, "status": f"timeout in warmup (>{PROGRAM_TIMEOUT_S}s)", "d": d, "n_comp": n_comp}
            ok, _ = with_timeout(PROGRAM_TIMEOUT_S, run_soga, compiled, True)
            if not ok:
                return {"label": label, "status": f"timeout in warmup (>{PROGRAM_TIMEOUT_S}s)", "d": d, "n_comp": n_comp}

        # Measured
        c_times, s_times = [], []
        last_classic_dist, last_sparse_dist = None, None
        for _ in range(N_RUNS):
            ok, res = with_timeout(PROGRAM_TIMEOUT_S, run_soga, compiled, False)
            if not ok:
                return {"label": label, "status": f"timeout in measured classic (>{PROGRAM_TIMEOUT_S}s)", "d": d, "n_comp": n_comp}
            t, d_c = res
            c_times.append(t)
            last_classic_dist = d_c
        for _ in range(N_RUNS):
            ok, res = with_timeout(PROGRAM_TIMEOUT_S, run_soga, compiled, True)
            if not ok:
                return {"label": label, "status": f"timeout in measured sparse (>{PROGRAM_TIMEOUT_S}s)", "d": d, "n_comp": n_comp}
            t, d_s = res
            s_times.append(t)
            last_sparse_dist = d_s
    except Exception as e:
        import traceback
        return {"label": label, "status": f"error: {type(e).__name__}: {str(e)[:120]}"}

    means_c = dist_means(last_classic_dist)
    means_s = dist_means(last_sparse_dist)
    ok, mdiff, bad = compare_means(means_c, means_s)
    return {
        "label": label,
        "status": "OK" if ok else f"DIFF on {bad}",
        "d": d, "n_comp": n_comp,
        "setup_ms": setup_s * 1000,
        "classic_soga_ms": mean(c_times) * 1000,
        "classic_soga_std_ms": stdev(c_times) * 1000 if len(c_times) > 1 else 0.0,
        "sparse_soga_ms": mean(s_times) * 1000,
        "sparse_soga_std_ms": stdev(s_times) * 1000 if len(s_times) > 1 else 0.0,
        "classic_soga_min_ms": min(c_times) * 1000,
        "sparse_soga_min_ms": min(s_times) * 1000,
        "speedup_soga": mean(c_times) / mean(s_times),
        "speedup_min": min(c_times) / min(s_times),
        "max_diff": mdiff,
    }


def render_table(rows):
    ok_rows = [r for r in rows if r.get("status") == "OK"]
    bad_rows = [r for r in rows if r.get("status") != "OK"]
    ok_rows.sort(key=lambda r: -r.get("speedup_soga", 0))
    out = []
    out.append("| Program | d | n_comp | setup ms | classic ms | sparse ms | speedup | max diff |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in ok_rows:
        out.append(
            f"| {r['label']} | {r['d']} | {r['n_comp']} "
            f"| {r['setup_ms']:.1f} "
            f"| {r['classic_soga_ms']:.2f} | {r['sparse_soga_ms']:.2f} "
            f"| **{r['speedup_soga']:.2f}x** "
            f"| {r['max_diff']:.2e} |"
        )
    if bad_rows:
        out.append("")
        out.append("**Failures / skipped:**")
        for r in bad_rows:
            out.append(f"- `{r['label']}`: {r.get('status')} (d={r.get('d', '?')})")
    return "\n".join(out)


def main():
    programs = discover_programs()
    print(f"Found {len(programs)} programs. Running N={N_RUNS} + {WARMUP} warmup per mode, timeout {PROGRAM_TIMEOUT_S}s.\n")
    rows = []
    wall_start = time.perf_counter()
    for i, (label, path) in enumerate(programs):
        sys.stdout.write(f"[{i+1:2d}/{len(programs)}] {label} ... ")
        sys.stdout.flush()
        t0 = time.perf_counter()
        r = bench_program(label, path)
        elapsed = time.perf_counter() - t0
        rows.append(r)
        if r.get("status") == "OK":
            sys.stdout.write(
                f"d={r['d']:3d} ncomp={r['n_comp']:5d} "
                f"classic={r['classic_soga_ms']:7.1f}ms sparse={r['sparse_soga_ms']:7.1f}ms "
                f"-> {r['speedup_soga']:5.2f}x  ({elapsed:.1f}s wall)\n"
            )
        else:
            sys.stdout.write(f"[{r.get('status', 'unknown')}]  ({elapsed:.1f}s wall)\n")
        sys.stdout.flush()
    total_wall = time.perf_counter() - wall_start
    print(f"\nTotal wall time: {total_wall:.1f}s")

    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "bench_all_canonical.csv")
    fields = [
        "label", "status", "d", "n_comp", "setup_ms",
        "classic_soga_ms", "classic_soga_std_ms", "classic_soga_min_ms",
        "sparse_soga_ms",  "sparse_soga_std_ms",  "sparse_soga_min_ms",
        "speedup_soga", "speedup_min", "max_diff",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved CSV: {csv_path}")

    ok_rows = [r for r in rows if r.get("status") == "OK"]
    speedups = [r["speedup_soga"] for r in ok_rows]
    geomean = float(np.exp(np.mean(np.log(speedups)))) if speedups else 0.0
    median = float(np.median(speedups)) if speedups else 0.0

    body = []
    body.append("# A/B benchmark — canonical SOGA suite, classic vs --sparse-truncate")
    body.append("")
    body.append(f"- Programs evaluated: {len(rows)} ({len(ok_rows)} OK, {len(rows)-len(ok_rows)} failed/skipped)")
    body.append(f"- N runs per mode: {N_RUNS}, warmup: {WARMUP}, per-program timeout (classic probe): {PROGRAM_TIMEOUT_S}s")
    body.append(f"- Wall time of the whole benchmark: {total_wall:.1f}s")
    body.append("")
    body.append("## Aggregate speedup (SOGA-only, excludes preprocessing and CFG construction)")
    body.append("")
    if speedups:
        body.append(f"- Geometric mean: **{geomean:.2f}x**")
        body.append(f"- Median: **{median:.2f}x**")
        body.append(f"- Range: {min(speedups):.2f}x to {max(speedups):.2f}x")
        wins = sum(1 for s in speedups if s > 1.05)
        losses = sum(1 for s in speedups if s < 0.95)
        neutral = len(speedups) - wins - losses
        body.append(f"- Programs where sparse is >5% faster: {wins}")
        body.append(f"- Programs essentially even (within ±5%): {neutral}")
        body.append(f"- Programs where sparse is >5% slower: {losses}")
    body.append("")
    body.append("## Per-program results (sorted by SOGA-only speedup, descending)")
    body.append("")
    body.append(render_table(rows))
    md_path = os.path.join(HERE, "BENCH_ALL_REPORT.md")
    with open(md_path, "w") as f:
        f.write("\n".join(body) + "\n")
    print(f"Saved Markdown report: {md_path}")


if __name__ == "__main__":
    main()
