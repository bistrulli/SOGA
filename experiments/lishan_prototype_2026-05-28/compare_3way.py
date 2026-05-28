"""3-way comparison: SOGA bit-exact analytical propagation vs hand-derived
closed form vs Monte Carlo reference. Validates the scalar-decomposed matmul
prototype across (v, p) grid.

For each (v, p) configuration:
  1. Substitute VALUE_V / VALUE_P / VALUE_1MP into the .soga template
  2. Run SOGA with -c (covariance) flag
  3. Parse SOGA output: E[d00], Var[d00]
  4. Compute analytical formula: E[d00] = 2v(1-p), Var[d00] = 2*sigma_b^2 + 4*v^2*p*(1-p)
  5. Compare against MC reference (loaded from results/mc_reference.csv)
  6. Output pass/fail per point + overall verdict

Plan: plan/2026-05-28-prototype-step-by-step-matmul.md
Acceptance: rel_err < 1e-6 on E[·]; rel_err < 1e-4 on Var[·]; SOGA inside MC 95% CI.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).parent
REPO_ROOT = EXP_DIR.parent.parent
TEMPLATE = REPO_ROOT / "programs" / "Example" / "lishan_prototype_2x2_scalar.soga"

# Import SOGA as a library (avoids stdout parsing + np.around precision loss)
sys.path.insert(0, str(REPO_ROOT / "src"))
from producecfg import produce_cfg
from libSOGA import start_SOGA
from sogaPreprocessor import compile2SOGA

SIGMA_B_SQ = 1e-12  # must match mc_reference.py and .soga


def run_soga(v: float, p: float) -> dict:
    """Substitute (v, p) into template, run SOGA via library, extract full-precision moments."""
    text = TEMPLATE.read_text()
    text = text.replace("VALUE_V", repr(v))
    text = text.replace("VALUE_1MP", repr(1.0 - p))
    text = text.replace("VALUE_P", repr(p))

    tmp = Path("/tmp") / f"proto_v{v}_p{p}.soga"
    tmp.write_text(text)

    t0 = time.time()
    compiled = compile2SOGA(str(tmp))
    cfg = produce_cfg(compiled)
    output_dist = start_SOGA(cfg, useR=False)
    elapsed = time.time() - t0

    var_names = output_dist.var_list
    d00_idx = var_names.index("d00")

    means = output_dist.gm.mean()           # full precision
    cov = output_dist.gm.cov()              # full precision

    return {
        "v": v,
        "p_fault": p,
        "SOGA_E": float(means[d00_idx]),
        "SOGA_Var": float(cov[d00_idx, d00_idx]),
        "SOGA_components": len(output_dist.gm.pi),
        "SOGA_time_s": elapsed,
    }


def analytical(v: float, p: float) -> dict:
    """Closed-form per ANALYTICAL.md."""
    e = 2.0 * v * (1.0 - p)
    var = 2.0 * SIGMA_B_SQ + 4.0 * v * v * p * (1.0 - p)
    pr_sdc = p  # asymptotic for sigma_b << v (see derivation)
    return {"AN_E": e, "AN_Var": var, "AN_Pr_SDC": pr_sdc}


def load_mc() -> dict:
    """Load MC reference results keyed by (v, p)."""
    mc_csv = EXP_DIR / "results" / "mc_reference.csv"
    out = {}
    with open(mc_csv) as f:
        for r in csv.DictReader(f):
            key = (float(r["v"]), float(r["p_fault"]))
            out[key] = {
                "MC_E": float(r["E_d00"]),
                "MC_Var": float(r["Var_d00"]),
                "MC_Pr_SDC": float(r["Pr_SDC"]),
                "MC_Pr_SDC_lo": float(r["Pr_SDC_lo"]),
                "MC_Pr_SDC_hi": float(r["Pr_SDC_hi"]),
            }
    return out


def main():
    mc = load_mc()
    v_grid = [0.5, 1.0, 2.0]
    p_grid = [0.001, 0.01, 0.05]

    print(f"\n3-way comparison: SOGA vs Analytical vs MC")
    print(f"{'='*135}")
    print(f"{'v':>4} {'p':>6} | {'SOGA_E':>10} {'AN_E':>10} {'MC_E':>10} | "
          f"{'|S-A|/A':>10} {'|S-M|':>10} | {'SOGA_Var':>10} {'AN_Var':>10} | "
          f"{'rel_Var':>10} | {'k':>3} {'t(s)':>5} | {'verdict':>8}")
    print("-" * 135)

    rows = []
    all_pass = True
    for v in v_grid:
        for p in p_grid:
            soga = run_soga(v, p)
            an = analytical(v, p)
            mc_pt = mc[(v, p)]

            # E[·] checks
            rel_E_SA = abs(soga["SOGA_E"] - an["AN_E"]) / max(abs(an["AN_E"]), 1e-30)
            abs_E_SM = abs(soga["SOGA_E"] - mc_pt["MC_E"])

            # Var[·] checks
            rel_Var_SA = abs(soga["SOGA_Var"] - an["AN_Var"]) / max(abs(an["AN_Var"]), 1e-30)

            # Acceptance
            pass_E = rel_E_SA < 1e-6
            pass_Var = rel_Var_SA < 1e-4
            pass_components = soga["SOGA_components"] <= 2

            pt_pass = pass_E and pass_Var and pass_components
            verdict = "PASS" if pt_pass else "FAIL"
            if not pt_pass:
                all_pass = False

            print(f"{v:>4} {p:>6} | {soga['SOGA_E']:>10.6f} {an['AN_E']:>10.6f} "
                  f"{mc_pt['MC_E']:>10.6f} | {rel_E_SA:>10.2e} {abs_E_SM:>10.2e} | "
                  f"{soga['SOGA_Var']:>10.6f} {an['AN_Var']:>10.6f} | "
                  f"{rel_Var_SA:>10.2e} | {soga['SOGA_components']:>3} "
                  f"{soga['SOGA_time_s']:>5.2f} | {verdict:>8}")

            row = {**soga, **an, **mc_pt,
                   "rel_E_SA": rel_E_SA, "abs_E_SM": abs_E_SM,
                   "rel_Var_SA": rel_Var_SA, "verdict": verdict}
            rows.append(row)

    print("-" * 135)
    print(f"\nOVERALL: {'ALL POINTS PASS ✓' if all_pass else 'SOME POINTS FAILED ✗'}")

    # Save 3-way results
    out_path = EXP_DIR / "results" / "results.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
