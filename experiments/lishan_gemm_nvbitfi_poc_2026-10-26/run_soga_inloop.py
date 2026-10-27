"""Fault injector INSIDE SOGA: solve every (distribution, class, eps[, k]) .soga program,
aggregate P(MSK) per class, compare with the CPU NVBitFI oracle, time the solves.

Classes: LDG.a, LDG.b, FMUL.alpha, FFMA (k on a grid; the 2 prologue sites per thread,
weight 0.07 %, are excluded on both sides). ARITH = their mean (equal counts n per thread).
Writes results/<run_id>/{REPORT_SOGA_INLOOP.md, soga_inloop.csv, config.json, HASHES.txt,
example_*.soga}.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from gemm_model import DIST_BY_NAME, DISTS, EPS_POLYBENCH, make_inputs
from nvbitfi_emulator import rates_by_class, run_campaign

EXP_DIR = Path(__file__).parent
CLASSES = ("LDG.a", "LDG.b", "FMUL.alpha", "FFMA")


def _job(args: tuple[str, str, float, int | None, int]) -> tuple[str, str, float, int | None, float, float, int]:
    dname, cls, eps, k, n = args
    from soga_fault_model import p_masked_soga
    t0 = time.perf_counter()
    p, meta = p_masked_soga(DIST_BY_NAME[dname], cls, n, eps, k)
    return dname, cls, eps, k, p, time.perf_counter() - t0, meta["n_comp"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k-stride", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--large", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--unroll", type=int, default=4)
    ap.add_argument("--dists", nargs="+", default=[d.name for d in DISTS])
    ap.add_argument("--eps-grid", type=float, nargs="+", default=[EPS_POLYBENCH])
    args = ap.parse_args()

    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = EXP_DIR / "results" / run_id
    out.mkdir(parents=True)
    ks = sorted(set(range(0, args.n, args.k_stride)) | {args.n - 1})
    jobs = [(d.name, c, eps, (k if c == "FFMA" else None), args.n)
            for d in DISTS if d.name in args.dists for eps in args.eps_grid for c in CLASSES
            for k in (ks if c == "FFMA" else [None])]
    jobs.sort(key=lambda j: j[1] == "FFMA")          # heavy jobs first

    from soga_fault_model import build_program
    for dname, tag in (("U[0,1]", "U01"), ("Sparse10%", "Sparse")):
        prog, _ = build_program(DIST_BY_NAME[dname], "FMUL.alpha", args.n, EPS_POLYBENCH)
        (out / f"example_FMUL_{tag}.soga").write_text(prog)

    t_wall = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_job, jobs))
    t_wall = time.perf_counter() - t_wall

    soga: dict[tuple[str, str, float], list[float]] = {}
    dists = [d for d in DISTS if d.name in args.dists]
    cpu: dict[str, float] = {d.name: 0.0 for d in dists}
    rows = []
    for dname, cls, eps, k, p, secs, ncomp in res:
        soga.setdefault((dname, cls, eps), []).append(p)
        cpu[dname] += secs
        rows.append({"dist": dname, "class": cls, "eps": eps, "k": "" if k is None else k,
                     "p_msk": round(p, 6), "solve_s": round(secs, 3), "n_comp": ncomp})
    with (out / "soga_inloop.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # CPU NVBitFI oracle, same instances and seeds as run_poc.py
    emu = {}
    for di, d in enumerate(DISTS):
        if d.name not in args.dists:
            continue
        A, B, C0 = make_inputs(d, args.n, seed_a=args.seed + di, seed_fixed=args.seed)
        camp = run_campaign(A, B, C0, args.large, args.unroll, seed=args.seed + 200 + di)
        for eps in args.eps_grid:
            r = rates_by_class(camp, eps)
            emu[(d.name, eps)] = {c: (r[c]["MSK"], r[c]["n"]) for c in CLASSES}

    L = [f"# Fault injector inside SOGA: GEMM, NVBitFI G_GP FLIP_SINGLE_BIT ({run_id})", "",
         f"n={args.n}; FFMA k-grid stride {args.k_stride} ({len(ks)} programs per eps); "
         f"oracle = CPU emulator, {args.large} injections; {args.workers} worker processes.", ""]
    for eps in args.eps_grid:
        L += [f"## eps = {eps:g}", "",
              "| A | " + " | ".join(f"{c} SOGA / emu" for c in CLASSES) + " | ARITH SOGA / emu | Δ |",
              "|---|" + "---|" * (len(CLASSES) + 2)]
        for d in dists:
            ps = {c: float(np.mean(soga[(d.name, c, eps)])) for c in CLASSES}
            es = emu[(d.name, eps)]
            ar_s = float(np.mean(list(ps.values())))
            ar_e = float(np.mean([es[c][0] for c in CLASSES]))
            L.append(f"| {d.name} | " + " | ".join(f"{ps[c]:.3f} / {es[c][0]:.3f}" for c in CLASSES)
                     + f" | **{ar_s:.3f}** / {ar_e:.3f} | {100 * (ar_s - ar_e):+.1f} pp |")
        L.append("")
    L += ["## Solve time (SOGA, all eps)", "",
          "| A | CPU seconds, sum over programs | programs |", "|---|---|---|"]
    for d in dists:
        L.append(f"| {d.name} | {cpu[d.name]:.0f} | {sum(1 for j in jobs if j[0] == d.name)} |")
    L += ["", f"Wall clock, all distributions, {args.workers} processes: {t_wall:.0f} s."]
    (out / "REPORT_SOGA_INLOOP.md").write_text("\n".join(L) + "\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=EXP_DIR).stdout.strip()
    (out / "config.json").write_text(json.dumps({**vars(args), "run_id": run_id,
                                                 "git_commit": commit, "k_grid": ks}, indent=1))
    hashes = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
              for p in sorted(EXP_DIR.glob("*.py")) + [EXP_DIR / "gm_cache.json"]]
    (out / "HASHES.txt").write_text("\n".join(hashes) + "\n")
    print((out / "REPORT_SOGA_INLOOP.md").read_text())
    print(f"written {out}")


if __name__ == "__main__":
    main()
