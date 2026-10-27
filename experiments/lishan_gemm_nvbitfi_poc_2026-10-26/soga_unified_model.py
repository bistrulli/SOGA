"""ONE SOGA program for the whole (arithmetic) NVBitFI campaign on a GEMM cell.

NVBitFI picks a uniformly random dynamic instruction, then a random bit of its destination
register. Here that choice is a random variable of the program:

    site = gm([w_0, ..., w_S], [0, ..., S], [0, ...]);     // which instruction is hit
    if site == 0 { <kernel + fault on a + gemm.cu check> } else {
    if site == 1 { <kernel + fault on b + check> } else { ... } end if; } end if;

Sites: load a, load b, FMUL alpha*a, and the FFMA accumulator at each position k of the grid.
Weights follow the per-thread instruction counts (n of each class; FFMA spread evenly over
the k grid). Each branch body is the per-class program of soga_fault_model.py, whose inner
selectors already return E[masked | site]; hence E[masked] = sum_s w_s P(masked | site=s)
and P(masked | site=s) is read back from the output components with site == s.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from gemm_model import DIST_BY_NAME, EPS_POLYBENCH
from soga_fault_model import build_program, gm_text
from soga_predict import run_soga

EXP_DIR = Path(__file__).parent


def build_unified(dist_name: str, n: int, eps: float, k_stride: int
                  ) -> tuple[str, list[tuple[str, int | None, float]]]:
    dist = DIST_BY_NAME[dist_name]
    ks = sorted(set(range(0, n, k_stride)) | {n - 1})
    sites: list[tuple[str, int | None, float]] = [("LDG.a", None, 0.25), ("LDG.b", None, 0.25),
                                                  ("FMUL.alpha", None, 0.25)]
    sites += [("FFMA", k, 0.25 / len(ks)) for k in ks]
    bodies = []
    for cls, k, _ in sites:
        prog, _ = build_program(dist, cls, n, eps, k)
        bodies.append([ln for ln in prog.splitlines() if not ln.startswith("/*")])

    def chain(i: int) -> list[str]:
        if i == len(bodies) - 1:
            return bodies[i]
        return [f"if site == {i} {{", *bodies[i], "} else {", *chain(i + 1), "} end if;"]

    head = [f"/* ONE SOGA program = one NVBitFI campaign (arithmetic sites), GEMM cell, "
            f"A ~ {dist_name}, eps={eps}, {len(sites)} sites */",
            "masked = 0;",
            f"site = {gm_text([w for _, _, w in sites], list(range(len(sites))), [0] * len(sites))};"]
    return "\n".join(head + chain(0)) + "\n", sites


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dist", default="U[0,1]")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--eps", type=float, default=EPS_POLYBENCH)
    ap.add_argument("--k-stride", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prog, sites = build_unified(args.dist, args.n, args.eps, args.k_stride)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.dist.replace("[", "").replace("]", "").replace(",", "-").replace("/", "_")
    (args.out / f"unified_{tag}_eps{args.eps:g}.soga").write_text(prog)

    t0 = time.perf_counter()
    out = run_soga(prog)
    secs = time.perf_counter() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # bytes on macOS

    V = out.var_list
    pi = np.asarray(out.gm.pi, float)
    mu = np.asarray(out.gm.mu, float)
    m, s = mu[:, V.index("masked")], np.rint(mu[:, V.index("site")]).astype(int)
    total = float(np.sum(pi * m))
    per_site = {i: float(np.sum(pi[s == i] * m[s == i]) / w) for i, (_, _, w) in enumerate(sites)}
    per_class = {c: float(np.mean([per_site[i] for i, (cc, _, _) in enumerate(sites) if cc == c]))
                 for c in ("LDG.a", "LDG.b", "FMUL.alpha", "FFMA")}
    res = {"dist": args.dist, "eps": args.eps, "n": args.n, "n_sites": len(sites),
           "lines": prog.count("\n"), "n_comp": len(pi), "mass": float(pi.sum()),
           "p_msk_total": total, "p_msk_per_class": per_class, "solve_s": secs,
           "peak_rss_gb": peak_gb}
    (args.out / f"unified_{tag}_eps{args.eps:g}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    main()
