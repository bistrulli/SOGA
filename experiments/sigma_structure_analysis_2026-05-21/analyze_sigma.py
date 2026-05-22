"""
Empirical analysis of covariance-matrix structure (Sigma) at runtime in SOGA.

For each canonical benchmark, instruments truncate(), update_rule(), and
merge() in libSOGA* to snapshot every (mu, sigma, pi) tuple encountered
during execution. For each snapshot, computes:

  - effective rank (number of singular values >= 1% of the largest)
  - sparsity (fraction of off-diagonal entries |Sigma[i,j]|/max(|Sigma|) < 1e-3)
  - best-rank-k approximation error for k in {1, 2, 5, 10}
  - block-diagonal block sizes via connected components on the non-zero
    pattern (threshold 1e-6 relative to max)

Aggregates per benchmark:
  - n_components per Sigma (so we know how many matrices we are looking at)
  - mean/median effective rank vs d
  - fraction of Sigma "well-approximable" by low rank (relative err < 1%)
  - whether Sigma is block-diagonal (single block vs many)

The output is a per-benchmark JSON report + a Markdown summary that the
plan references as the empirical justification for Option 1.
"""

import os
import sys
import json
import time
import signal
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

# Hard timeout helper (reuse pattern from bench_all_canonical.py)
class HardTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
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


# ---------- Sigma structure metrics ----------

def metric_effective_rank(Sigma, rel_threshold=1e-2):
    """Number of singular values >= rel_threshold * largest."""
    if Sigma.size == 0:
        return 0
    s = np.linalg.svd(Sigma, compute_uv=False)
    if s[0] < 1e-15:
        return 0
    return int(np.sum(s >= rel_threshold * s[0]))


def metric_low_rank_err(Sigma, k):
    """Frobenius relative error of best rank-k approximation."""
    if k >= min(Sigma.shape):
        return 0.0
    if Sigma.size == 0:
        return 0.0
    U, s, Vt = np.linalg.svd(Sigma)
    if s[0] < 1e-15:
        return 0.0
    Sigma_k = (U[:, :k] * s[:k]) @ Vt[:k, :]
    return float(np.linalg.norm(Sigma - Sigma_k, "fro") / np.linalg.norm(Sigma, "fro"))


def metric_sparsity(Sigma, rel_threshold=1e-3):
    """Fraction of strictly-off-diagonal entries that are 'effectively zero'."""
    d = Sigma.shape[0]
    if d <= 1:
        return 0.0
    diag = np.eye(d, dtype=bool)
    off = ~diag
    M = np.max(np.abs(Sigma)) + 1e-30
    zero_off = np.abs(Sigma[off]) < rel_threshold * M
    return float(zero_off.sum() / off.sum())


def metric_block_count(Sigma, rel_threshold=1e-6):
    """Number of connected components in the sparsity graph (|Sigma[i,j]| > thr)."""
    d = Sigma.shape[0]
    if d == 0:
        return 0
    M = np.max(np.abs(Sigma)) + 1e-30
    adj = (np.abs(Sigma) > rel_threshold * M)
    np.fill_diagonal(adj, True)
    # BFS to find components
    visited = np.zeros(d, dtype=bool)
    n_blocks = 0
    for start in range(d):
        if visited[start]:
            continue
        n_blocks += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            neighbors = np.where(adj[node] & ~visited)[0]
            stack.extend(neighbors.tolist())
    return n_blocks


def analyze_sigma(Sigma):
    d = Sigma.shape[0]
    eff_rank = metric_effective_rank(Sigma)
    return {
        "d": d,
        "eff_rank": eff_rank,
        "eff_rank_ratio": eff_rank / d if d > 0 else 0,
        "sparsity": metric_sparsity(Sigma),
        "n_blocks": metric_block_count(Sigma),
        "lr_err_k1": metric_low_rank_err(Sigma, 1),
        "lr_err_k2": metric_low_rank_err(Sigma, 2),
        "lr_err_k5": metric_low_rank_err(Sigma, 5),
        "lr_err_k10": metric_low_rank_err(Sigma, 10),
    }


# ---------- Instrumentation hooks ----------

SNAPSHOTS = []
MAX_SNAPSHOTS_PER_PROGRAM = 50
MAX_COMPS_PER_SNAPSHOT = 20  # cap to avoid analyzing millions of components


def reset_snapshots():
    global SNAPSHOTS
    SNAPSHOTS = []


def record_dist(label, dist):
    if len(SNAPSHOTS) >= MAX_SNAPSHOTS_PER_PROGRAM:
        return
    sigmas = dist.gm.sigma
    pis = dist.gm.pi
    n_comp = len(sigmas)
    take = min(n_comp, MAX_COMPS_PER_SNAPSHOT)
    # Sample by weight: take the top-weight components first
    if n_comp > take:
        idx = np.argsort([-p for p in pis])[:take]
    else:
        idx = list(range(n_comp))
    metrics_per_comp = []
    for i in idx:
        S = np.asarray(sigmas[i], dtype=float)
        try:
            metrics_per_comp.append(analyze_sigma(S))
        except np.linalg.LinAlgError:
            continue
    SNAPSHOTS.append({
        "label": label,
        "n_comp": n_comp,
        "metrics_per_comp": metrics_per_comp,
    })


def install_hooks():
    """Monkey-patch SOGA dispatcher to record dist after each significant op."""
    from libSOGA import SOGA as soga_dispatcher_orig
    import libSOGA

    def instrumented_SOGA(node, data, parallel, exec_queue):
        soga_dispatcher_orig(node, data, parallel, exec_queue)
        # Capture the post-state on selected node types
        ntype = node.type
        if ntype in ("state", "observe", "merge", "prune"):
            child = node.children[0] if node.children else None
            target = child if child is not None else node
            if hasattr(target, "dist") and target.dist is not None:
                record_dist(f"{ntype}:{node.name}", target.dist)

    libSOGA.SOGA = instrumented_SOGA


# ---------- Benchmark runner ----------

def run_one(prog_path, timeout_s=30.0):
    """Run one .soga via in-process SOGA; capture snapshots; return summary."""
    reset_snapshots()
    import random
    random.seed(0); np.random.seed(0)
    from sogaPreprocessor import compile2SOGA
    from producecfg import produce_cfg
    from libSOGA import start_SOGA

    install_hooks()

    def do_run():
        compiled = compile2SOGA(prog_path)
        cfg = produce_cfg(compiled)
        out = start_SOGA(cfg, useR=False, parallel=None,
                         sparse_truncate=True, vectorize_truncate=True)
        return out

    ok, res = with_timeout(timeout_s, do_run)
    if not ok:
        return {"status": "timeout", "snapshots": []}
    return {"status": "ok", "n_snapshots": len(SNAPSHOTS), "snapshots": SNAPSHOTS.copy()}


def aggregate(snapshots):
    """Reduce per-snapshot per-component metrics to per-program summary."""
    all_metrics = []
    for snap in snapshots:
        for m in snap.get("metrics_per_comp", []):
            all_metrics.append(m)
    if not all_metrics:
        return None
    ds = [m["d"] for m in all_metrics]
    eff_ratios = [m["eff_rank_ratio"] for m in all_metrics]
    sparsities = [m["sparsity"] for m in all_metrics]
    n_blocks = [m["n_blocks"] for m in all_metrics]
    lr1 = [m["lr_err_k1"] for m in all_metrics]
    lr2 = [m["lr_err_k2"] for m in all_metrics]
    lr5 = [m["lr_err_k5"] for m in all_metrics]
    lr10 = [m["lr_err_k10"] for m in all_metrics]
    return {
        "n_components_analyzed": len(all_metrics),
        "d_min": min(ds), "d_max": max(ds), "d_median": float(np.median(ds)),
        "eff_rank_ratio_mean": float(np.mean(eff_ratios)),
        "eff_rank_ratio_median": float(np.median(eff_ratios)),
        "sparsity_mean": float(np.mean(sparsities)),
        "n_blocks_mean": float(np.mean(n_blocks)),
        "n_blocks_max": int(max(n_blocks)),
        "lr_err_k1_median": float(np.median(lr1)),
        "lr_err_k2_median": float(np.median(lr2)),
        "lr_err_k5_median": float(np.median(lr5)),
        "lr_err_k10_median": float(np.median(lr10)),
    }


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


def main():
    programs = discover_programs()
    print(f"Found {len(programs)} programs. Analyzing Sigma structure...\n")
    per_prog = {}
    for label, path in programs:
        sys.stdout.write(f"  {label:<40s} ")
        sys.stdout.flush()
        t0 = time.perf_counter()
        try:
            r = run_one(path, timeout_s=30.0)
        except Exception as e:
            sys.stdout.write(f"  ERROR: {type(e).__name__}: {str(e)[:80]}\n")
            per_prog[label] = {"status": f"error: {type(e).__name__}"}
            continue
        elapsed = time.perf_counter() - t0
        if r["status"] == "ok":
            summary = aggregate(r["snapshots"])
            per_prog[label] = {
                "status": "ok",
                "elapsed_s": elapsed,
                "n_snapshots": r["n_snapshots"],
                "summary": summary,
            }
            if summary:
                sys.stdout.write(
                    f"d={summary['d_min']}-{summary['d_max']}  "
                    f"eff_rank_ratio={summary['eff_rank_ratio_median']:.2f}  "
                    f"sparsity={summary['sparsity_mean']:.2f}  "
                    f"blocks_max={summary['n_blocks_max']}  "
                    f"lr_err_k1={summary['lr_err_k1_median']:.3f}  "
                    f"({elapsed:.1f}s)\n"
                )
            else:
                sys.stdout.write(f"no snapshots ({elapsed:.1f}s)\n")
        else:
            per_prog[label] = {"status": r["status"], "elapsed_s": elapsed}
            sys.stdout.write(f"[{r['status']}] ({elapsed:.1f}s)\n")

    out_json = os.path.join(HERE, "sigma_structure_results.json")
    with open(out_json, "w") as f:
        json.dump(per_prog, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Render summary
    md_lines = []
    md_lines.append("# Sigma structure analysis — per-program summary\n")
    md_lines.append("Run on the canonical SOGA suite with --sparse-truncate --vectorize-truncate active.\n")
    md_lines.append("Snapshots of (mu, Sigma, pi) captured after every state/observe/merge/prune node.\n")
    md_lines.append("Metrics computed per component (top 20 by weight):\n")
    md_lines.append("- eff_rank_ratio: effective rank / d  (fraction of singular values above 1% of the largest)")
    md_lines.append("- sparsity: fraction of off-diagonal entries effectively zero")
    md_lines.append("- blocks_max: max # connected components in the sparsity graph (1 = single dense block)")
    md_lines.append("- lr_err_k: Frobenius relative error of best rank-k approximation")
    md_lines.append("")
    md_lines.append("| Program | d range | n_snap | eff_rank/d | sparsity | blocks_max | lr_err k=1 | k=2 | k=5 |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, info in per_prog.items():
        if info.get("status") != "ok" or not info.get("summary"):
            md_lines.append(f"| {label} | (status: {info.get('status', 'unknown')}) | | | | | | | |")
            continue
        s = info["summary"]
        md_lines.append(
            f"| {label} | {s['d_min']}–{s['d_max']} | {info['n_snapshots']} "
            f"| {s['eff_rank_ratio_median']:.2f} | {s['sparsity_mean']:.2f} | {s['n_blocks_max']} "
            f"| {s['lr_err_k1_median']:.3f} | {s['lr_err_k2_median']:.3f} | {s['lr_err_k5_median']:.3f} |"
        )
    md_path = os.path.join(HERE, "SIGMA_STRUCTURE_REPORT.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
