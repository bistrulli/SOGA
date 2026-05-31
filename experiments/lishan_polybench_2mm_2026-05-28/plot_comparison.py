"""Plot resilience-vs-input-norm curves (the SHAPE we aim to reproduce).

Reads results/resilience_curves.csv (exact enumeration, register + input-side,
multiple eps) and optionally results/soga_curve.csv (SOGA analytical overlay).

Panels:
  (1) per-kernel susceptibility vs norm   (register & input-side, one eps)
  (2) per-cell  susceptibility vs norm    (register & input-side, one eps)
  (3) NORMALIZED per-kernel susceptibility vs norm across eps (shape robustness)
  (4) SOGA vs MC_register normalized (if soga_curve.csv present)

HONESTY: register-level Python model is NOT an NVBit-FI proxy; Lishan comparison
is shape-only. No absolute-value claims.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).parent
RES = EXP_DIR / "results"
FIG = EXP_DIR / "figures"


def load_curves(path: Path) -> list:
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="matrix size N to plot")
    args = ap.parse_args()

    rows = [r for r in load_curves(RES / "resilience_curves.csv") if int(r["N"]) == args.n]
    epss = sorted({r["eps"] for r in rows})
    headline_eps = epss[len(epss) // 2]  # middle eps for panels 1-2

    soga_path = RES / "soga_curve.csv"
    soga_rows = [r for r in load_curves(soga_path) if int(r["N"]) == args.n] \
        if soga_path.exists() else []

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    sel = [r for r in rows if r["eps"] == headline_eps]
    sel.sort(key=lambda r: r["norm"])
    norms = [r["norm"] for r in sel]

    # (1) per-kernel
    ax = axes[0, 0]
    ax.plot(norms, [r["reg_S_kernel"] for r in sel], "o-", label="register")
    ax.plot(norms, [r["inp_S_kernel"] for r in sel], "s--", label="input-side")
    ax.set_title(f"Per-kernel susceptibility (1-MSK) vs norm (eps={headline_eps})")
    ax.set_xlabel("Frobenius norm ||B||_F = |v|*N"); ax.set_ylabel("susceptibility")
    ax.legend(); ax.grid(alpha=0.3)

    # (2) per-cell
    ax = axes[0, 1]
    ax.plot(norms, [r["reg_S_cell"] for r in sel], "o-", label="register")
    ax.plot(norms, [r["inp_S_cell"] for r in sel], "s--", label="input-side")
    ax.set_title(f"Per-cell susceptibility vs norm (eps={headline_eps})")
    ax.set_xlabel("Frobenius norm"); ax.set_ylabel("mean frac. cells changed")
    ax.legend(); ax.grid(alpha=0.3)

    # (3) normalized shape across eps (register per-kernel, ref = v with norm>0 min)
    ax = axes[1, 0]
    for eps in epss:
        e = [r for r in rows if r["eps"] == eps]
        e.sort(key=lambda r: r["norm"])
        ref = next(r["reg_S_kernel"] for r in e if r["v"] == 1)
        ax.plot([r["norm"] for r in e],
                [r["reg_S_kernel"] / ref for r in e], "o-", label=f"eps={eps}")
    ax.set_title("Normalized shape robustness across eps (register, per-kernel)")
    ax.set_xlabel("Frobenius norm"); ax.set_ylabel("S(v) / S(v=1)")
    ax.legend(); ax.grid(alpha=0.3)

    # (4) SOGA overlay (if available)
    ax = axes[1, 1]
    if soga_rows:
        soga_rows.sort(key=lambda r: r["norm"])
        mc = [r for r in rows if r["eps"] == headline_eps]
        mc.sort(key=lambda r: r["norm"])
        mc_ref = next(r["reg_S_kernel"] for r in mc if r["v"] == 1)
        soga_ref = next(r["soga_S_kernel"] for r in soga_rows if r["v"] == 1)
        ax.plot([r["norm"] for r in mc], [r["reg_S_kernel"] / mc_ref for r in mc],
                "o-", label="MC_register (exact)")
        ax.plot([r["norm"] for r in soga_rows],
                [r["soga_S_kernel"] / soga_ref for r in soga_rows],
                "x--", label="SOGA analytical")
        ax.set_title("SOGA vs MC_register — normalized shape")
        ax.set_xlabel("Frobenius norm"); ax.set_ylabel("S(v) / S(v=1)")
        ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "SOGA overlay pending (iter 2)", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle(f"PolyBench 2MM int32 (N={args.n}) — fault susceptibility vs input "
                 "norm (shape reproduction of Lishan's resilience curves)", fontsize=13)
    fig.tight_layout()
    out = FIG / f"soga_vs_mc_register_vs_input_side_N{args.n}.png"
    fig.savefig(out, dpi=130)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
