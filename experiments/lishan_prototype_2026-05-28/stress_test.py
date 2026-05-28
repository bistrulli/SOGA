"""Stress test for the prototype: edge cases + multi-fault scaling.

Tests:
  T1 — edge cases: v=0 (zero crossing), p=0 (no fault), p=1 (always fault)
  T2 — multi-fault scaling: programs with N=1, 2, 4 fault injection sites; verify
       component count grows as expected (K=N+1 with pruning) and runtime extrapolates

This informs the GO/NO-GO decision for scaling to 32×32 (where N=32 per output cell).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).parent
REPO_ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from producecfg import produce_cfg
from libSOGA import start_SOGA
from sogaPreprocessor import compile2SOGA


def make_n_fault_program(n_faults: int, v: float, p: float, sigma_b_sq: float = 1e-12,
                          prune_k: int | None = None) -> str:
    """Build a .soga program with n_faults Bernoulli sign-flip injection points
    embedded in a serial accumulator over n_faults+1 input cells.

    If prune_k is set, insert prune(prune_k); after each fault branch to bound
    component count.
    """
    lines = ["/* Stress test: N-fault accumulator */"]
    for k in range(n_faults + 1):
        lines.append(f"b{k} = gauss({v!r}, {sigma_b_sq!r});")
    lines.append("acc = 0;")
    for k in range(n_faults):
        lines.append(f"acc = acc + b{k};")
        lines.append(f"fault{k} = gm([{1 - p!r}, {p!r}], [0, 1], [0, 0]);")
        lines.append(f"if fault{k} == 1 {{")
        lines.append("    acc = 0 - acc;")
        lines.append("} else {")
        lines.append("    acc = acc;")
        lines.append("} end if;")
        if prune_k is not None:
            lines.append(f"prune({prune_k});")
    lines.append(f"acc = acc + b{n_faults};")
    lines.append("d00 = acc;")
    return "\n".join(lines) + "\n"


def run(program_text: str) -> dict:
    tmp = Path("/tmp/stress.soga")
    tmp.write_text(program_text)
    t0 = time.time()
    compiled = compile2SOGA(str(tmp))
    cfg = produce_cfg(compiled)
    dist = start_SOGA(cfg, useR=False)
    elapsed = time.time() - t0
    return {
        "components": len(dist.gm.pi),
        "E_d00": float(dist.gm.mean()[dist.var_list.index("d00")]),
        "Var_d00": float(dist.gm.cov()[dist.var_list.index("d00"), dist.var_list.index("d00")]),
        "time_s": elapsed,
        "var_list": dist.var_list,
    }


# ---- T1: edge cases ----
print("\n[T1] EDGE CASES")
print(f"{'case':<25} | {'E[d00]':>10} | {'Var[d00]':>12} | {'k':>3} | {'t(s)':>5}")
print("-" * 70)

# v=0 zero-crossing
prog = make_n_fault_program(1, v=0.0, p=0.01)
r = run(prog)
print(f"{'v=0, p=0.01':<25} | {r['E_d00']:>10.6f} | {r['Var_d00']:>12.6e} | {r['components']:>3} | {r['time_s']:>5.2f}")
assert abs(r["E_d00"]) < 1e-10, f"v=0 should give E=0, got {r['E_d00']}"

# p=0 no fault
prog = make_n_fault_program(1, v=1.0, p=0.0)
r = run(prog)
print(f"{'v=1, p=0  (no fault)':<25} | {r['E_d00']:>10.6f} | {r['Var_d00']:>12.6e} | {r['components']:>3} | {r['time_s']:>5.2f}")
assert abs(r["E_d00"] - 2.0) < 1e-10, f"p=0 should give E=2v=2, got {r['E_d00']}"

# p=1 always fault
prog = make_n_fault_program(1, v=1.0, p=1.0)
r = run(prog)
print(f"{'v=1, p=1  (always)':<25} | {r['E_d00']:>10.6f} | {r['Var_d00']:>12.6e} | {r['components']:>3} | {r['time_s']:>5.2f}")
# Sequence: acc=b0=1, fault: acc=-1, acc=acc+b1 = -1+1 = 0. So E=0.
assert abs(r["E_d00"]) < 1e-10, f"p=1 should give E=0, got {r['E_d00']}"

# ---- T2: multi-fault scaling ----
print("\n[T2] MULTI-FAULT SCALING (v=1.0, p=0.01)")
print(f"{'N_faults':>8} | {'k_obs':>5} | {'k_pred':>6} | {'t(s)':>6} | {'E[d00]':>10}")
print("-" * 60)
prev_t = None
for n in [1, 2, 4, 8]:
    prog = make_n_fault_program(n, v=1.0, p=0.01)
    r = run(prog)
    k_obs = r["components"]
    k_pred_max = 2 ** n  # without pruning
    print(f"{n:>8} | {k_obs:>5} | ≤{k_pred_max:>5} | {r['time_s']:>6.3f} | {r['E_d00']:>10.6f}")

# ---- T3: 32-fault scaling, WITH explicit prune(K) ----
print("\n[T3] FULL INNER-PRODUCT (32 fault sites) WITH prune(K) per branch")
print(f"{'N_faults':>8} | {'prune_K':>7} | {'k_obs':>5} | {'t(s)':>6} | {'E[d00]':>10}")
print("-" * 60)

for k_prune in [33, 16, 8]:
    prog = make_n_fault_program(32, v=1.0, p=0.01, prune_k=k_prune)
    r = run(prog)
    print(f"{32:>8} | {k_prune:>7} | {r['components']:>5} | {r['time_s']:>6.3f} | {r['E_d00']:>10.6f}")

# Per-cell extrapolation
print("\n[Extrapolation] 32×32 full matmul (1024 cells × runtime of 1 cell)")
prog = make_n_fault_program(32, v=1.0, p=0.01, prune_k=33)
r = run(prog)
per_cell_s = r["time_s"]
print(f"  per-cell (prune K=33): {per_cell_s:.3f}s × 1024 cells = {per_cell_s*1024:.1f}s = {per_cell_s*1024/60:.1f}min")
print(f"  for 2MM (×2): {per_cell_s*1024*2/60:.1f}min")
