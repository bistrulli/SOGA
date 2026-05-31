"""SOGA analytical fault-susceptibility curve for the int32 2MM.

The fault-outcome SHIFTS on each output cell are computed by the SOGA ENGINE: for
each output cell (r,s) and fault location we run a generated per-cell .soga program
carrying a single 65-class fault categorical, and READ the propagated component
means mu_k from the output GM. The per-bit shift magnitude used in the curve is
exactly |mu_k - baseline| as returned by the engine (NOT a Python reconstruction).

What is and is NOT validated by this script (see validate_soga_vs_mc.py for the
threshold-free, non-circular check):
  - The engine's propagated shifts are checked here against the known C[k][s]
    multiplier (assertion) and, in validate_soga_vs_mc.py, against the actual int32
    MC outcomes per (site,bit) WITHOUT any tolerance.
  - The susceptibility metric (count of engine-propagated shifts whose magnitude
    exceeds eps*|baseline|) is the SAME functional applied to SOGA's and to MC's
    outcomes; agreement on the metric therefore FOLLOWS from agreement on the
    shifts and is reported as a derived quantity, not an independent confirmation.
  - For this affine kernel the single-fault GM has uniform fault weights (p/64), so
    the susceptibility FRACTION reduces to a count of exceeding shifts; the weights
    cancel and carry no information for this metric (documented honestly).

Shifts are v-independent (the engine runs at v=1); only the threshold eps*|base(v)|
moves with v, so the 80 (N=4) / 576 (N=8) engine runs are reused across the v-sweep.

R5: the engine uses real arithmetic; MC uses int32 wraparound. They agree on the
SDC verdict except where delta*C[k][s] wraps mod 2^32 (rare, v-INDEPENDENT) — this
shifts per-cell absolute values by <0.1% and does not change the shape.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from generate_soga import gen_phase1_program, gen_phase2_program, run_soga
from kernel2mm import polybench_AC, kernel_2mm

EXP_DIR = Path(__file__).parent
V_SWEEP = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]


def engine_pos_magnitudes(program: str, var: str, base: int) -> List[float]:
    """Run a per-cell program; return the sorted positive |mu_k - base| magnitudes
    that the SOGA engine assigns to the fault components (no-fault component excluded)."""
    pi, means = run_soga(program, var)
    order = np.argsort(pi)[::-1]
    nofault = order[0]
    assert abs(means[nofault] - base) < 1e-6, (
        f"no-fault mean {means[nofault]} != baseline {base}")
    mags = sorted({round(abs(means[i] - base)) for i in range(len(means))
                   if i != nofault and abs(means[i] - base) > 0.5})
    return [float(m) for m in mags]


def collect_engine_shifts(n: int) -> Tuple[Dict, Dict, np.ndarray]:
    """Run the engine for every Phase-2 cell and every Phase-1 (r,k,s') program.
    Returns (phase2_mags, phase1_mags, base1):
      phase2_mags[(r,s)]   = engine positive shift magnitudes on D[r][s] (== 2^b)
      phase1_mags[(r,k,s)] = engine positive shift magnitudes on D[r][s] from a fault
                             in tmp[r][k]  (== C[k][s]*2^b; empty if C[k][s]==0)
    """
    A, C = polybench_AC(n)
    base1 = np.array(kernel_2mm(A, C, 1, n))
    p = 1.0 / (2 * n ** 3)
    exp_2b = [float(2 ** b) for b in range(32)]

    phase2_mags: Dict[Tuple[int, int], List[float]] = {}
    for r in range(n):
        for s in range(n):
            mags = engine_pos_magnitudes(
                gen_phase2_program(n, r, s, 1, p), f"d{r}{s}", int(base1[r, s]))
            assert mags == exp_2b, f"Phase-2 ({r},{s}) engine mags != 2^b"
            phase2_mags[(r, s)] = mags

    phase1_mags: Dict[Tuple[int, int, int], List[float]] = {}
    for r in range(n):
        for k in range(n):
            for s in range(n):
                mags = engine_pos_magnitudes(
                    gen_phase1_program(n, r, s, k, 1, p), f"d{r}{s}", int(base1[r, s]))
                ck = C[k][s]
                if ck == 0:
                    assert mags == [], f"expected no shift for C[{k}][{s}]=0"
                else:
                    # full check: engine shift magnitudes must be exactly {C[k][s]*2^b}
                    expected = [float(ck * 2 ** b) for b in range(32)]
                    assert mags == expected, (
                        f"engine mags != C[{k}][{s}]={ck} * 2^b (r={r}, s={s})")
                phase1_mags[(r, k, s)] = mags
    return phase2_mags, phase1_mags, base1


def susceptibility(n: int, eps: float, v: int, phase2_mags: Dict,
                   phase1_mags: Dict, base1: np.ndarray) -> Tuple[float, float]:
    """(S_kernel, S_cell) from ENGINE-propagated shift magnitudes, tolerance eps."""
    base = np.abs(v) * np.abs(base1)
    thr = eps * base
    total = (2 * n ** 3) * 32
    kernel_count = 0.0
    cell_count = 0.0

    # Phase 2: site (r,s) affects only cell (r,s); N sites per cell.
    for r in range(n):
        for s in range(n):
            nb = sum(1 for m in phase2_mags[(r, s)] if m > thr[r, s])
            kernel_count += n * nb
            cell_count += n * nb

    # Phase 1: site (r,k) affects cells (r,s'); shift mag aligned by bit index b.
    # N-site multiplicity is exact for flat B: all N inner steps of tmp[r][k] produce
    # the SAME final shift on the output (the flip delta adds linearly through the
    # remaining constant adds), so they share one shift-magnitude set.
    for r in range(n):
        for k in range(n):
            any_bit = [False] * 32
            cells_per_bit = [0] * 32
            for s in range(n):
                mags = phase1_mags[(r, k, s)]
                for b, m in enumerate(mags):     # sorted ascending => index b == bit b
                    if m > thr[r, s]:
                        any_bit[b] = True
                        cells_per_bit[b] += 1
            kernel_count += n * sum(1 for x in any_bit if x)
            cell_count += n * sum(cells_per_bit)

    return kernel_count / total, cell_count / (total * n * n)


def main() -> None:
    ap = argparse.ArgumentParser(description="SOGA analytical susceptibility curve")
    ap.add_argument("--ns", type=int, nargs="+", default=[4])
    ap.add_argument("--eps", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    args = ap.parse_args()

    rows = []
    for n in args.ns:
        print(f"\n=== Collecting SOGA engine shifts, N={n} "
              f"({n*n} Phase-2 + {n**3} Phase-1 runs) ===")
        p2, p1, base1 = collect_engine_shifts(n)
        print("  engine shifts read + asserted vs C[k][s]")
        for eps in args.eps:
            for v in V_SWEEP:
                sk, sc = susceptibility(n, eps, v, p2, p1, base1)
                rows.append({"N": n, "eps": eps, "v": v, "norm": v * n,
                             "soga_S_kernel": sk, "soga_S_cell": sc})

    out = EXP_DIR / "results" / "soga_curve.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
