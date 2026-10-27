"""PoC driver: SOGA prediction vs emulated NVBitFI campaign on PolyBench-GPU GEMM.

For each input distribution of A (B, C0 fixed U[0,1]):
  - SOGA predicts P(MSK) per arithmetic site class from the prior (no injections);
  - the CPU NVBitFI emulator runs (a) a Lishan-sized campaign (1000 injections,
    G_GP + FLIP_SINGLE_BIT) and (b) a large validation campaign;
  - option B: MSK(eps) for the whole eps grid from the same campaigns.
Writes results/<run_id>/{REPORT.md, results.csv, config.json, HASHES.txt}.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from gemm_model import (ARITH_CLASSES, DISTS, EPS_POLYBENCH, STRUCT_CLASSES,
                        class_counts, make_inputs)
from nvbitfi_emulator import rates_by_class, run_campaign
from soga_predict import predict

EXP_DIR = Path(__file__).parent


def ci95(p: float, n: int) -> float:
    return 1.96 * float(np.sqrt(max(p * (1 - p), 1e-12) / n)) if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--unroll", type=int, default=4)
    ap.add_argument("--small", type=int, default=1000, help="Lishan-sized campaign")
    ap.add_argument("--large", type=int, default=30000, help="validation campaign")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--eps-grid", type=float, nargs="+",
                    default=[1e-6, 1e-5, 1e-4, EPS_POLYBENCH, 1e-3, 1e-2, 1e-1])
    args = ap.parse_args()

    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = EXP_DIR / "results" / run_id
    out.mkdir(parents=True)
    counts = class_counts(args.n, args.unroll)
    arith_share = sum(counts[c] for c in ARITH_CLASSES) / sum(counts.values())

    rows: list[dict] = []
    summary: list[dict] = []
    for di, dist in enumerate(DISTS):
        A, B, C0 = make_inputs(dist, args.n, seed_a=args.seed + di, seed_fixed=args.seed)
        t0 = time.perf_counter()
        pred = predict(dist, args.n, args.unroll, args.eps_grid, seed=args.seed)
        t_soga = time.perf_counter() - t0
        t0 = time.perf_counter()
        small = run_campaign(A, B, C0, args.small, args.unroll, seed=args.seed + 100 + di)
        large = run_campaign(A, B, C0, args.large, args.unroll, seed=args.seed + 200 + di)
        t_emu = time.perf_counter() - t0
        for eps in args.eps_grid:
            rs, rl = rates_by_class(small, eps), rates_by_class(large, eps)
            for cls in ("ALL", "ARITH") + ARITH_CLASSES + STRUCT_CLASSES:
                soga = pred.arith(eps) if cls == "ARITH" else pred.msk.get(cls, {}).get(eps)
                rows.append({
                    "dist": dist.name, "eps": eps, "class": cls,
                    "soga_msk": "" if soga is None else round(soga, 5),
                    **{f"small_{o}": round(rs.get(cls, {}).get(o, float("nan")), 5)
                       for o in ("MSK", "SDC", "DUE")},
                    "small_n": rs.get(cls, {}).get("n", 0),
                    **{f"large_{o}": round(rl.get(cls, {}).get(o, float("nan")), 5)
                       for o in ("MSK", "SDC", "DUE")},
                    "large_n": rl.get(cls, {}).get("n", 0),
                })
        rs, rl = rates_by_class(small), rates_by_class(large)
        summary.append({
            "dist": dist.name,
            "frob": float(np.linalg.norm(A.astype(np.float64))),
            "soga_arith": pred.arith(EPS_POLYBENCH),
            "large_arith": rl["ARITH"]["MSK"], "large_arith_n": rl["ARITH"]["n"],
            "small_all": rs["ALL"], "small_n": rs["ALL"]["n"],
            "small_arith": rs["ARITH"]["MSK"], "small_arith_n": rs["ARITH"]["n"],
            "large_all": rl["ALL"],
            "t_soga": t_soga, "t_emu": t_emu,
            "pred": pred, "small": small, "large": large,
        })
        print(f"{dist.name:10s} |A|_F={summary[-1]['frob']:8.1f}  SOGA ARITH MSK "
              f"{pred.arith(EPS_POLYBENCH):.3f}  emu(large) {rl['ARITH']['MSK']:.3f}  "
              f"emu(1000) ALL MSK/SDC/DUE {rs['ALL']['MSK']:.3f}/{rs['ALL']['SDC']:.3f}/"
              f"{rs['ALL']['DUE']:.3f}  [soga {t_soga:.1f}s]")

    with (out / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=EXP_DIR).stdout.strip()
    (out / "config.json").write_text(json.dumps({
        **vars(args), "run_id": run_id, "git_commit": commit,
        "class_counts_per_thread": counts, "arith_share": arith_share,
        "dists": [d.name for d in DISTS], "B_C0": "U[0,1] fixed, seed_fixed=seed",
        "A_seed": "seed + dist_index"}, indent=1))
    hashes = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
              for p in sorted(EXP_DIR.glob("*.py")) + [EXP_DIR / "gm_cache.json"]]
    (out / "HASHES.txt").write_text("\n".join(hashes) + "\n")
    write_report(out, args, summary, counts, arith_share)
    print(f"\nwritten {out}")


def write_report(out: Path, args: argparse.Namespace, summary: list[dict],
                 counts: dict[str, int], arith_share: float) -> None:
    L = [f"# GEMM register-FI: SOGA prediction vs emulated NVBitFI ({out.name})", "",
         f"n={args.n}, unroll={args.unroll}, G_GP + FLIP_SINGLE_BIT, gemm.cu check "
         f"(0.05 %), small campaign={args.small}, large={args.large}, seed={args.seed}.",
         f"ASSUMED per-thread mix: {sum(counts.values())} G_GP sites, arithmetic share "
         f"{arith_share:.1%} (replace with NVBitFI profiler counts).", "",
         "## 1. Headline at the gemm.cu criterion (eps = 5e-4)", "",
         "| A distribution | ‖A‖_F | SOGA MSK (arith) | emu MSK arith (large) | "
         "emu 1000-run ALL MSK / SDC / DUE |", "|---|---|---|---|---|"]
    for s in summary:
        a = s["small_all"]
        L.append(f"| {s['dist']} | {s['frob']:.1f} | **{s['soga_arith']:.1%}** | "
                 f"{s['large_arith']:.1%} ± {ci95(s['large_arith'], s['large_arith_n']):.1%} | "
                 f"{a['MSK']:.1%} / {a['SDC']:.1%} / {a['DUE']:.1%} (±{ci95(0.5, s['small_n']):.1%}) |")
    L += ["", "## 2. Per arithmetic class (eps = 5e-4): SOGA vs large campaign", "",
          "| class | " + " | ".join(s["dist"] for s in summary) + " |",
          "|---|" + "---|" * len(summary)]
    for c in ARITH_CLASSES:
        cells = []
        for s in summary:
            r = rates_by_class(s["large"])[c]
            cells.append(f"{s['pred'].msk[c][EPS_POLYBENCH]:.3f} vs {r['MSK']:.3f} (n={r['n']})")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## 3. Option B: MSK(eps) of the arithmetic sites, SOGA vs large campaign", "",
          "| eps | " + " | ".join(s["dist"] for s in summary) + " |",
          "|---|" + "---|" * len(summary)]
    for eps in args.eps_grid:
        cells = [f"{s['pred'].arith(eps):.3f} vs {rates_by_class(s['large'], eps)['ARITH']['MSK']:.3f}"
                 for s in summary]
        L.append(f"| {eps:g} | " + " | ".join(cells) + " |")
    L += ["", "## 4. Cost", "",
          "The CPU emulator is our validation ORACLE (it needs the kernel semantics and an "
          "assumed instruction mix), not the GPU campaign. GPU campaign wall time is NOT "
          "measured here: it is (#injections) x (one instrumented run of the binary).", "",
          "| A distribution | SOGA prediction (s, all eps, 0 injections) | "
          f"CPU oracle ({args.small}+{args.large} emulated injections, s) |",
          "|---|---|---|"]
    for s in summary:
        L.append(f"| {s['dist']} | {s['t_soga']:.1f} | {s['t_emu']:.1f} |")
    L += ["", "## 5. Assumptions (stated, not verified)", "",
          "- Per-thread G_GP mix derived from gemm.cu (accumulator in a register, unroll "
          f"{args.unroll}, 64-bit pointer increments as lo/hi): replace with NVBitFI profiler "
          "counts. Per-class rates do not depend on the mix; ALL does.",
          "- Structural sites (index / address / loop counter): buffers 2^32-aligned, "
          "out-of-buffer access => DUE, index fault => SDC. Illustrative only; SOGA does not "
          "model them, the GPU campaign measures them.",
          "- SOGA: input priors as 8-component GMs (EM preprocessor), one-term law from the "
          "engine, 512-term sums moment-matched to one Gaussian, float32 flips exact on a "
          "deterministic quadrature. B and C0 treated as U[0,1] priors (fixed instance in the "
          "campaign, averaged over 262k cells)."]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
