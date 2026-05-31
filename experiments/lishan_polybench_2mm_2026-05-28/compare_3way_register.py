"""3-way comparison: SOGA (engine) vs MC_register (exact enum) vs MC_input_side.

Validates that SOGA's analytical fault propagation reproduces the register-level
susceptibility curve, and quantifies the input-side vs register-level difference.
Reports per-point absolute agreement and Spearman rank correlation of the SHAPE
(the primary deliverable: reproducing Lishan's resilience-vs-norm shape).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

EXP_DIR = Path(__file__).parent
RES = EXP_DIR / "results"


def load(path: Path) -> List[dict]:
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def key(r: dict) -> Tuple[int, float, int]:
    return (int(r["N"]), r["eps"], int(r["v"]))


def main() -> None:
    mc = {key(r): r for r in load(RES / "resilience_curves.csv")}
    soga = {key(r): r for r in load(RES / "soga_curve.csv")}

    shared = sorted(set(mc) & set(soga))
    ns = sorted({k[0] for k in shared})
    epss = sorted({k[1] for k in shared})

    print("=" * 78)
    print("SOGA (engine) vs MC_register (exact) — per-kernel & per-cell susceptibility")
    print("=" * 78)

    summary = []
    for n in ns:
        for eps in epss:
            ks = [k for k in shared if k[0] == n and k[1] == eps]
            ks.sort(key=lambda k: k[2])
            soga_k = [soga[k]["soga_S_kernel"] for k in ks]
            mc_k = [mc[k]["reg_S_kernel"] for k in ks]
            soga_c = [soga[k]["soga_S_cell"] for k in ks]
            mc_c = [mc[k]["reg_S_cell"] for k in ks]
            inp_k = [mc[k]["inp_S_kernel"] for k in ks]

            max_abs_k = max(abs(a - b) for a, b in zip(soga_k, mc_k))
            max_abs_c = max(abs(a - b) for a, b in zip(soga_c, mc_c))
            rho_k = spearmanr(soga_k, mc_k).correlation
            # normalized-shape Spearman (vs v=1 ref)
            ref_s = next(soga[k]["soga_S_kernel"] for k in ks if k[2] == 1)
            ref_m = next(mc[k]["reg_S_kernel"] for k in ks if k[2] == 1)
            norm_s = [x / ref_s for x in soga_k]
            norm_m = [x / ref_m for x in mc_k]
            rho_norm = spearmanr(norm_s, norm_m).correlation

            # input-side MINUS register (mean over v, per-kernel). Positive => input-side
            # is MORE susceptible: one B-cell fault is read by all N Phase-1 inner
            # products sharing that column, reaching N times more accumulator paths
            # than a single register fault.
            gap = float(np.mean([i - s for i, s in zip(inp_k, mc_k)]))

            print(f"\nN={n}, eps={eps}:")
            print(f"  per-kernel: max|SOGA-MC| = {max_abs_k:.2e}   Spearman = {rho_k:.4f}")
            print(f"  per-cell  : max|SOGA-MC| = {max_abs_c:.2e}")
            print(f"  normalized-shape Spearman(SOGA,MC) = {rho_norm:.4f}")
            print(f"  input-side - register (mean per-kernel) = {gap:+.4f}")
            # *_derived: these compare the SAME susceptibility functional applied to
            # SOGA's vs MC's outcomes; agreement FOLLOWS from the propagation match and
            # is NOT independent evidence (the independent check is validate_soga_vs_mc.py).
            summary.append({
                "N": n, "eps": eps, "max_abs_kernel_derived": max_abs_k,
                "max_abs_cell_derived": max_abs_c, "spearman_kernel_derived": rho_k,
                "spearman_norm_derived": rho_norm, "inp_minus_reg_mean": gap,
            })

    out = RES / "three_way_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print(f"\nSaved: {out}")

    worst = max(s["max_abs_kernel_derived"] for s in summary)
    worst_cell = max(s["max_abs_cell_derived"] for s in summary)
    min_rho = min(s["spearman_norm_derived"] for s in summary)
    print("\n" + "=" * 78)
    print("NOTE: the susceptibility metric (count of engine-propagated shifts whose")
    print("magnitude exceeds eps*|baseline|) is the SAME functional applied to SOGA's")
    print("and to MC's outcomes. The independent, threshold-free validation that the")
    print("engine's PROPAGATION matches the int32 MC outcomes is validate_soga_vs_mc.py.")
    print(f"Here: per-kernel max|SOGA-MC| = {worst:.2e}; per-cell max = {worst_cell:.2e}")
    print(f"(per-cell gap = int32-wrap cases, R5); normalized-shape Spearman >= {min_rho:.4f}.")


if __name__ == "__main__":
    main()
