"""Exact resilience curves for the int32 PolyBench 2MM under register / input-side FI.

PIVOT (2026-05-29, user decision): the goal is to reproduce the SHAPE of Lishan
Yang's resilience-vs-input-norm curves, NOT absolute MSK values. Exact-match (eps=0)
int32 SDC was proven input-independent (see iter_1_finding.md); the input-norm
dependence comes from low-order-bit errors being negligible RELATIVE to the output
magnitude. We therefore use a relative tolerance eps>0 as the (declared) masking
mechanism. The normalized shape is QUALITATIVELY consistent across eps (always
monotone decreasing in norm); it is NOT exactly eps-invariant — the normalized
per-kernel susceptibility at the largest norm spans ~0.69-0.74 across eps in {1,4,16}
(a ~5 percentage-point spread). We report this spread honestly rather than claiming
eps-invariance.

Method: EXACT enumeration over all (site, bit) injections (no Monte Carlo) — the
N in {4,8} fault spaces (4096 / 32768 register injections) are small enough.

Metrics, per input level v (flat B = v, Frobenius norm ||B||_F = |v|*N):
  - per-kernel susceptibility  S_kernel(v) = frac. injections changing ANY output cell
  - per-cell  susceptibility   S_cell(v)   = mean over injections of frac. cells changed
  with relative tolerance: cell (i,j) counts as changed iff
    |D_pert[i,j] - D_base[i,j]| > eps * |D_base[i,j]|   (threshold 0 when base = 0)

Normalized fault susceptibility (the shape): S(v) / S(v_ref).

HONESTY DISCLAIMER: register-level Python model is NOT an NVBit-FI proxy; comparison
with Lishan is shape-only (see POLYBENCH_REFERENCE.md).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from kernel2mm import (
    kernel_2mm,
    kernel_2mm_input_fault,
    n_fault_sites,
    polybench_AC,
)

EXP_DIR = Path(__file__).parent

# Norm sweep: flat input value v (>=0; sign is irrelevant since |baseline| drives
# relative masking). ||B||_F = v * N.
V_SWEEP = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]


def _cell_changed(d_pert: List[List[int]], base: List[List[int]], n: int,
                  eps: float) -> Tuple[bool, int]:
    """Return (any_cell_changed, n_cells_changed) under relative tolerance eps."""
    any_changed = False
    n_changed = 0
    for i in range(n):
        bi, di = base[i], d_pert[i]
        for j in range(n):
            bl = bi[j]
            diff = abs(di[j] - bl)
            thr = eps * abs(bl)  # threshold 0 when bl == 0 -> any change is SDC
            if diff > thr:
                n_changed += 1
                any_changed = True
    return any_changed, n_changed


def register_susceptibility(n: int, v: int, eps: float) -> Tuple[float, float]:
    """EXACT (S_kernel, S_cell) over all register injections at input level v."""
    A, C = polybench_AC(n)
    base = kernel_2mm(A, C, v, n)
    nsites = n_fault_sites(n)
    total = nsites * 32
    n_cells = n * n
    kernel_sdc = 0
    cell_sdc_sum = 0
    for s in range(nsites):
        for b in range(32):
            d = kernel_2mm(A, C, v, n, fault_site=s, fault_bit=b)
            any_c, nc = _cell_changed(d, base, n, eps)
            if any_c:
                kernel_sdc += 1
            cell_sdc_sum += nc
    return kernel_sdc / total, cell_sdc_sum / (total * n_cells)


def input_side_susceptibility(n: int, v: int, eps: float) -> Tuple[float, float]:
    """EXACT (S_kernel, S_cell) over all input-side (B-cell) injections."""
    A, C = polybench_AC(n)
    base = kernel_2mm(A, C, v, n)
    total = (n * n) * 32
    n_cells = n * n
    kernel_sdc = 0
    cell_sdc_sum = 0
    for cell in range(n * n):
        r, c = divmod(cell, n)
        for b in range(32):
            d = kernel_2mm_input_fault(A, C, v, n, fault_cell=(r, c), fault_bit=b)
            any_c, nc = _cell_changed(d, base, n, eps)
            if any_c:
                kernel_sdc += 1
            cell_sdc_sum += nc
    return kernel_sdc / total, cell_sdc_sum / (total * n_cells)


def build_curves(n: int, eps: float, v_sweep: List[int]) -> List[dict]:
    rows = []
    for v in v_sweep:
        reg_k, reg_c = register_susceptibility(n, v, eps)
        inp_k, inp_c = input_side_susceptibility(n, v, eps)
        rows.append({
            "N": n, "eps": eps, "v": v, "norm": v * n,
            "reg_S_kernel": reg_k, "reg_S_cell": reg_c,
            "inp_S_kernel": inp_k, "inp_S_cell": inp_c,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact resilience curves for int32 2MM")
    ap.add_argument("--ns", type=int, nargs="+", default=[4])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.5, 1.0, 4.0, 16.0])
    ap.add_argument("--v-ref", type=int, default=1,
                    help="reference input level for normalization (v=0 has S=1 always)")
    args = ap.parse_args()

    all_rows = []
    for n in args.ns:
        for eps in args.eps:
            print(f"\n=== N={n}, eps={eps} (EXACT enumeration) ===")
            print(f"{'v':>4} {'norm':>5} | {'reg_Sk':>8} {'reg_Sc':>8} | "
                  f"{'inp_Sk':>8} {'inp_Sc':>8} | {'norm reg_Sk':>11}")
            rows = build_curves(n, eps, V_SWEEP)
            # normalize per-kernel register susceptibility to v_ref
            ref = next(r["reg_S_kernel"] for r in rows if r["v"] == args.v_ref)
            for r in rows:
                norm_val = r["reg_S_kernel"] / ref if ref > 0 else float("nan")
                r["reg_S_kernel_norm"] = norm_val
                print(f"{r['v']:>4} {r['norm']:>5} | {r['reg_S_kernel']:>8.4f} "
                      f"{r['reg_S_cell']:>8.4f} | {r['inp_S_kernel']:>8.4f} "
                      f"{r['inp_S_cell']:>8.4f} | {norm_val:>11.4f}")
            all_rows.extend(rows)

    out = EXP_DIR / "results" / "resilience_curves.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved: {out}")
    cfg = EXP_DIR / "results" / "resilience_curves_config.json"
    with open(cfg, "w") as f:
        json.dump({"ns": args.ns, "eps_grid": args.eps, "v_sweep": V_SWEEP,
                   "v_ref": args.v_ref, "method": "exact_enumeration"}, f, indent=2)
    print(f"Saved: {cfg}")


if __name__ == "__main__":
    main()
