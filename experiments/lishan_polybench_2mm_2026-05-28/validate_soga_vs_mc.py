"""Threshold-free validation: does the SOGA engine's propagated fault-outcome
distribution contain the actual int32 MC outcome, for every (fault site, bit)?

This is the NON-CIRCULAR core check (no eps, no shared metric). For each output
cell (r,s) and fault location we run the SOGA engine to get the predicted outcome
SET {mu_k} for D[r][s] (the 65-class categorical), then run the actual int32 kernel
with that fault and check the realized D[r][s] is a member of the engine's set.

The 65-class signed model predicts BOTH +-2^b*C; the MC realizes one of them, so a
correct propagation => mc_outcome in engine_set. Mismatches occur exactly when
delta*C[k][s] wraps mod 2^32 (engine = real, MC = int32) — these are counted and
characterized (R5), and asserted to be wrap cases only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from generate_soga import gen_phase1_program, gen_phase2_program, run_soga
from kernel2mm import kernel_2mm, polybench_AC

INT32_LIM = 2 ** 31


def engine_outcome_set(program: str, var: str) -> set:
    pi, means = run_soga(program, var)
    return {round(m) for m in means}


def phase2_site(n: int, r: int, s: int, step: int = 0) -> int:
    return n ** 3 + (r * n + s) * n + step


def phase1_site(n: int, r: int, k: int, step: int = 0) -> int:
    return (r * n + k) * n + step


def validate(n: int) -> dict:
    A, C = polybench_AC(n)
    v = 1
    base = np.array(kernel_2mm(A, C, v, n))
    p = 1.0 / (2 * n ** 3)

    checks = matches = wrap_mismatch = other_mismatch = 0

    for r in range(n):
        for s in range(n):
            # Phase-2 fault location for D[r][s]
            eset = engine_outcome_set(gen_phase2_program(n, r, s, v, p), f"d{r}{s}")
            site = phase2_site(n, r, s)
            for b in range(32):
                mc = kernel_2mm(A, C, v, n, fault_site=site, fault_bit=b)[r][s]
                checks += 1
                if mc in eset:
                    matches += 1
                else:
                    # The 65-class signed model predicts BOTH base+-2^b, and int32 wrap
                    # of base+2^b equals base-2^b (which IS in the set), so a Phase-2
                    # mismatch should be impossible -> any mismatch here is a real bug.
                    other_mismatch += 1
        for k in range(n):
            # Phase-1 fault in tmp[r][k] -> read effect on cell (r,s) for each s
            for s in range(n):
                eset = engine_outcome_set(
                    gen_phase1_program(n, r, s, k, v, p), f"d{r}{s}")
                site = phase1_site(n, r, k)
                for b in range(32):
                    mc = kernel_2mm(A, C, v, n, fault_site=site, fault_bit=b)[r][s]
                    checks += 1
                    if mc in eset:
                        matches += 1
                    else:
                        # mismatch is legitimate only if delta*C wrapped int32
                        real_shift = (2 ** b) * C[k][s]
                        if real_shift >= INT32_LIM:
                            wrap_mismatch += 1
                        else:
                            other_mismatch += 1

    return {
        "N": n, "checks": checks, "matches": matches,
        "wrap_mismatch": wrap_mismatch, "other_mismatch": other_mismatch,
        "match_frac": matches / checks,
    }


def main() -> None:
    ns = [int(x) for x in sys.argv[1:]] or [4]
    print("Threshold-free SOGA-engine vs int32-MC outcome validation")
    print("=" * 70)
    ok = True
    for n in ns:
        r = validate(n)
        print(f"N={r['N']}: {r['matches']}/{r['checks']} match "
              f"({r['match_frac']*100:.2f}%)  wrap-only mismatch={r['wrap_mismatch']}  "
              f"OTHER(bad)={r['other_mismatch']}")
        if r["other_mismatch"] != 0:
            ok = False
    print("=" * 70)
    print("VERDICT:", "SOGA propagation EXACT (all mismatches are int32-wrap, R5)"
          if ok else "UNEXPECTED MISMATCH — propagation bug")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
