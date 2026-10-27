"""Figures for the note to Lishan.

Fig 1: (a) masked rate on arithmetic sites per input distribution, SOGA (fault injector
inside the .soga program) vs CPU NVBitFI oracle with 95% CI; (b) per site class.
Fig 2: MSK(eps) of arithmetic sites, SOGA program (fault injector inside) at every eps vs oracle.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gemm_model import DISTS, make_inputs  # noqa: E402
from nvbitfi_emulator import rates_by_class, run_campaign  # noqa: E402

EXP = Path(__file__).parent
INLOOP = [EXP / "results/2026-10-26_16-06-40/soga_inloop.csv",
          EXP / "results/2026-10-26_16-15-23/soga_inloop.csv",
          EXP / "results/2026-10-26_17-07-11/soga_inloop.csv"]
EPS0 = 5e-4
OUT = EXP / "figures"
CLASSES = ("LDG.a", "LDG.b", "FMUL.alpha", "FFMA")
LABEL = {"LDG.a": "load a", "LDG.b": "load b", "FMUL.alpha": "FMUL α·a", "FFMA": "FFMA acc"}
# colour = input distribution (validated categorical slots 1-4); filled = SOGA, open = FI
DIST_C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
SEED, N, UNROLL, LARGE = 20261026, 512, 4, 30000

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True})


def soga_inloop() -> dict[tuple[str, str, float], float]:
    acc: dict[tuple[str, str, float], list[float]] = {}
    for f in INLOOP:
        for r in csv.DictReader(f.open()):
            acc.setdefault((r["dist"], r["class"], float(r["eps"])), []).append(float(r["p_msk"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def oracle() -> dict[str, dict]:
    out = {}
    for di, d in enumerate(DISTS):
        A, B, C0 = make_inputs(d, N, seed_a=SEED + di, seed_fixed=SEED)
        c = run_campaign(A, B, C0, LARGE, UNROLL, seed=SEED + 200 + di)
        r = {}
        for cl in CLASSES + ("ARITH4",):
            m = np.isin(c["cls"], list(CLASSES)) if cl == "ARITH4" else c["cls"] == cl
            p = float((c["outcome"][m] == "MSK").mean())
            r[cl] = (p, 1.96 * np.sqrt(p * (1 - p) / m.sum()), int(m.sum()))
        r["frob"] = float(np.linalg.norm(A.astype(np.float64)))
        r["campaign"] = c
        out[d.name] = r
    return out


def fig1(soga: dict, emu: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    names = [d.name for d in DISTS]
    x = np.arange(len(names))
    s = [np.mean([soga[(n, c, EPS0)] for c in CLASSES]) for n in names]
    e = [emu[n]["ARITH4"][0] for n in names]
    ci = [emu[n]["ARITH4"][1] for n in names]
    w = 0.36
    ax1.bar(x - w / 2 - 0.01, s, w, color=DIST_C[:len(names)])
    ax1.bar(x + w / 2 + 0.01, e, w, facecolor="white", edgecolor=DIST_C[:len(names)], linewidth=1.4)
    ax1.errorbar(x + w / 2 + 0.01, e, yerr=ci, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax1.bar([0], [0], color=MUTED, label="SOGA, 0 injections (filled)")
    ax1.bar([0], [0], facecolor="white", edgecolor=MUTED, linewidth=1.4,
            label=f"FI oracle, {LARGE // 1000}k inj. ± 95% CI (open)")
    for xi, si in zip(x, s):
        ax1.text(xi - w / 2, si + 0.015, f"{si:.0%}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax1.set_xticks(x, [f"{n}\n‖A‖F={emu[n]['frob']:.0f}" for n in names], fontsize=7.5)
    ax1.set_ylim(0, 1.2)
    ax1.set_yticks(np.linspace(0, 1, 6))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_ylabel("masked (arithmetic sites)")
    ax1.set_title("(a) Same norm, different resilience", loc="left", fontsize=9, color=INK)
    ax1.legend(frameon=False, fontsize=7, loc="upper left", borderaxespad=0.2)
    ax1.grid(axis="x", visible=False)

    xc = np.arange(len(CLASSES))
    off = np.linspace(-0.27, 0.27, len(names))
    for i, n in enumerate(names):
        ys = [soga[(n, c, EPS0)] for c in CLASSES]
        ye = [emu[n][c][0] for c in CLASSES]
        ax2.scatter(xc + off[i], ys, s=34, color=DIST_C[i], zorder=3, label=n,
                    edgecolors="white", linewidths=1.2)
        ax2.scatter(xc + off[i], ye, s=34, facecolors="none", edgecolors=DIST_C[i], linewidths=1.2,
                    zorder=2, marker="D")
    ax2.scatter([], [], s=34, color=MUTED, label="SOGA (filled)")
    ax2.scatter([], [], s=34, facecolors="none", edgecolors=MUTED, marker="D", label="FI oracle (open)")
    ax2.set_xticks(xc, [LABEL[c] for c in CLASSES])
    ax2.set_ylim(0.3, 1.0)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_title("(b) Per destination register", loc="left", fontsize=9, color=INK)
    ax2.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_resilience.{ext}", dpi=200)


def fig2(soga: dict, emu: dict) -> None:
    eps_grid = sorted({k[2] for k in soga})
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for i, d in enumerate(DISTS):
        ys = [np.mean([soga[(d.name, c, e)] for c in CLASSES]) for e in eps_grid]
        ye = []
        for e in eps_grid:
            r = rates_by_class(emu[d.name]["campaign"], e)
            ye.append(np.mean([r[c]["MSK"] for c in CLASSES]))
        ax.plot(eps_grid, ys, color=DIST_C[i], lw=2, marker="o", ms=3.5, label=d.name)
        ax.scatter(eps_grid, ye, s=30, facecolors="none", edgecolors=DIST_C[i], linewidths=1.2,
                   marker="D", zorder=3)
    ax.axvline(EPS0, color=MUTED, lw=0.8, ls="--")
    ax.text(5.5e-4, 0.27, "gemm.cu check\n(0.05 %)", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_ylim(0.2, 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("relative error tolerance ε")
    ax.set_ylabel("masked (arithmetic sites)")
    ax.set_title("Masked rate vs tolerance (● SOGA program, ◇ FI oracle)", loc="left",
                 fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_eps_curve.{ext}", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    emu = oracle()
    soga = soga_inloop()
    fig1(soga, emu)
    fig2(soga, emu)
    for n, r in emu.items():
        print(n, {k: (round(v[0], 3), round(v[1], 3), v[2]) if isinstance(v, tuple) else round(v, 1)
                  for k, v in r.items() if k != "campaign"},
              "SOGA ARITH4", round(np.mean([soga[(n, c, EPS0)] for c in CLASSES]), 3))
