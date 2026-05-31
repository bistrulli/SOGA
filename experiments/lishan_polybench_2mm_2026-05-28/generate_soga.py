"""Generate per-cell .soga programs encoding register-level FI on the int32 2MM,
and run them through the SOGA engine to extract the fault-propagated GM.

Key simplification (proven): with flat B = v and deterministic A, every tmp[r][k]
= v * rowsum(A_r) is a DETERMINISTIC constant. So the per-output-cell computation
is deterministic except for a single injected bit-flip fault. We therefore encode
exactly ONE 65-class signed fault categorical per program (single-fault model),
which keeps the component count at 65 (no explosion, no lossy pruning needed).

This is the analytical equivalent of one NVBit-FI site: SOGA propagates the fault
categorical through the affine matmul chain and returns the output GM for D[r][s].

65-class signed fault (R7, used unconditionally):
    fault = gm([1-p, p/64, ..., p/64],
               [0, +2^0, -2^0, +2^1, -2^1, ..., +2^31, -2^31],
               [0, 0, ..., 0])
The signed model is used directly (the 33-class positive-only model is skipped:
it is directionally biased per R7, and for the |shift| > eps*|baseline| SDC metric
the sign is irrelevant, so 65-class is both correct and sufficient).

Fault sites for cell D[r][s]:
  - Phase 2: the accumulator of D[r][s] itself (N inner steps). Shift on output = delta.
  - Phase 1: the accumulator of tmp[r][k] for each k with C[k][s] != 0 (N steps each).
             Shift on output = delta * C[k][s].
All N steps within one inner product yield the same final shift (the flip delta adds
linearly through the remaining adds), so we enumerate distinct SHIFT-CLASSES and
weight each by its site multiplicity.

NOTE (R5): the SOGA engine uses real arithmetic; the MC ground truth uses int32
wraparound. They agree on the |shift|>threshold SDC verdict except for the rare,
v-INDEPENDENT cases where delta*C wraps mod 2^32 (e.g. 2^31 * even). This shifts
absolute values slightly but not the susceptibility-vs-norm SHAPE.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

EXP_DIR = Path(__file__).parent
REPO_ROOT = EXP_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from kernel2mm import polybench_AC  # noqa: E402

# Signed bit-flip deltas: +2^b, -2^b for b=0..31  (64 values)
SIGNED_DELTAS: List[int] = []
for _b in range(32):
    SIGNED_DELTAS.append(2 ** _b)
    SIGNED_DELTAS.append(-(2 ** _b))


def _dec(x: float) -> str:
    """Plain positional decimal (no scientific notation; the SOGA NUM token has no
    exponent form). Weights are exact dyadic rationals so this round-trips exactly."""
    return np.format_float_positional(x, trim="-")


def fault_gm_literal(p: float) -> str:
    """65-class signed fault categorical as a gm(...) DSL literal."""
    n_cls = len(SIGNED_DELTAS)  # 64
    w0 = 1.0 - p
    wk = p / n_cls
    weights = [w0] + [wk] * n_cls
    means = [0] + SIGNED_DELTAS
    sigmas = [0] * (n_cls + 1)
    wstr = ", ".join(_dec(x) for x in weights)
    mstr = ", ".join(str(x) for x in means)
    sstr = ", ".join(str(x) for x in sigmas)
    return f"gm([{wstr}], [{mstr}], [{sstr}])"


def gen_phase2_program(n: int, r: int, s: int, v: int, p: float) -> str:
    """D[r][s] = sum_k tmp[r][k]*C[k][s], fault injected after the first add
    (Phase-2 accumulator). tmp[r][k] = v*rowsum(A_r) is a deterministic constant."""
    A, C = polybench_AC(n)
    rowsum = sum(A[r])
    tmp_rk = v * rowsum  # constant, same for all k (B flat)
    lines = [
        f"/* Phase-2 register fault on D[{r}][{s}], N={n}, v={v}, p={p} */",
        "acc = 0;",
    ]
    for k in range(n):
        cks = C[k][s]
        lines.append(f"acc = acc + {tmp_rk * cks};")
        if k == 0:
            lines.append(f"fault = {fault_gm_literal(p)};")
            lines.append("acc = acc + fault;")
    lines.append(f"d{r}{s} = acc;")
    return "\n".join(lines) + "\n"


def gen_phase1_program(n: int, r: int, s: int, k_fault: int, v: int, p: float) -> str:
    """Fault injected into the tmp[r][k_fault] accumulator (Phase 1), then D[r][s]
    is completed. Shift on output should be delta * C[k_fault][s]."""
    A, C = polybench_AC(n)
    lines = [
        f"/* Phase-1 register fault in tmp[{r}][{k_fault}] -> D[{r}][{s}], "
        f"N={n}, v={v}, p={p} */",
    ]
    # Compute tmp[r][k] for all k; inject fault in tmp[r][k_fault]'s inner product.
    for k in range(n):
        lines.append(f"t{k} = 0;")
        for l in range(n):
            term = A[r][l] * v  # B[l][k] = v
            lines.append(f"t{k} = t{k} + {term};")
            if k == k_fault and l == 0:
                lines.append(f"fault = {fault_gm_literal(p)};")
                lines.append(f"t{k} = t{k} + fault;")
    # D[r][s] = sum_k C[k][s] * t_k  (coeff-first: the grammar requires NUM * IDV)
    lines.append("acc = 0;")
    for k in range(n):
        lines.append(f"acc = acc + {C[k][s]} * t{k};")
    lines.append(f"d{r}{s} = acc;")
    return "\n".join(lines) + "\n"


def run_soga(program_text: str, var_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run a .soga program string through the engine; return (weights, per-component
    means) of var. Raises RuntimeError on ANTLR parse errors (which the parser would
    otherwise silently recover from, producing wrong results)."""
    import contextlib
    import io

    from sogaPreprocessor import compile2SOGA
    from producecfg import produce_cfg
    from libSOGA import start_SOGA

    tmp = Path("/tmp") / "soga_gen_tmp.soga"
    tmp.write_text(program_text)

    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(err):
        compiled = compile2SOGA(str(tmp))
        cfg = produce_cfg(compiled)
        out = start_SOGA(cfg, useR=False)
    log = err.getvalue()
    parse_markers = ("mismatched", "no viable alternative", "expecting",
                     "token recognition error", "extraneous input", "missing")
    if any(m in log for m in parse_markers):
        raise RuntimeError(f"SOGA parse error:\n{log}\n--- program ---\n{program_text}")
    if var_name not in out.var_list:
        raise RuntimeError(f"var {var_name} missing from output {out.var_list}\n{log}")
    idx = out.var_list.index(var_name)
    pi = np.asarray(out.gm.pi, dtype=float)
    comp_means = np.asarray([mu[idx] for mu in out.gm.mu], dtype=float)
    return pi, comp_means


if __name__ == "__main__":
    # Smoke test: D[0][0], N=4, Phase-2 fault.
    prog = gen_phase2_program(4, 0, 0, 1, p=1.0 / (2 * 4 ** 3))
    print(prog)
    pi, means = run_soga(prog, "d00")
    print("n_components:", len(pi))
    print("weights[:5]:", pi[:5])
    print("means[:5]:", means[:5])
