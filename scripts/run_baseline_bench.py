"""
Baseline benchmark runner for M0.8 — matrix-gm-integration branch.

Runs a set of canonical SOGA benchmark programs, records timing and basic
output statistics, writes results/baseline_matrix_gm_<date>.csv.

Usage (from repo root):
    python scripts/run_baseline_bench.py

The script imports SOGA modules directly from src/ without using the
full reproduce.py framework, which avoids jinja2/natsort dependencies.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
PROGRAMS_SOGA = REPO_ROOT / "programs" / "SOGA"
PROGRAMS_AQUA = REPO_ROOT / "programs" / "SOGA" / "AQUA"
RESULTS_DIR = REPO_ROOT / "results"

sys.path.insert(0, str(SRC_DIR))

from producecfg import produce_cfg
from libSOGA import start_SOGA
from sogaPreprocessor import compile2SOGA

# ---------------------------------------------------------------------------
# Benchmark catalog: (name, soga_path, target_var)
# ---------------------------------------------------------------------------

BENCHMARKS = [
    ("BernoulliPrune",       PROGRAMS_SOGA / "BernoulliPrune.soga",        "theta"),
    ("BayesPointMachine",    PROGRAMS_SOGA / "BayesPointMachine.soga",     "w[0]"),
    ("Burglar",              PROGRAMS_SOGA / "Burglar.soga",                "burglary"),
    ("ClickGraphPrune",      PROGRAMS_SOGA / "ClickGraphPrune.soga",        "simAll"),
    ("ClinicalTrialPrune",   PROGRAMS_SOGA / "ClinicalTrialPrune.soga",     "isEff"),
    ("CoinBias",             PROGRAMS_SOGA / "CoinBias.soga",               "bias"),
    ("DigitRecognition",     PROGRAMS_SOGA / "DigitRecognition.soga",       "y"),
    ("Grass",                PROGRAMS_SOGA / "Grass.soga",                  "rain"),
    ("MurderMistery",        PROGRAMS_SOGA / "MurderMistery.soga",          "aliceDunnit"),
    ("NoisyOr",              PROGRAMS_SOGA / "NoisyOr.soga",                "n3"),
    ("SurveyUnbias",         PROGRAMS_SOGA / "SurveyUnbias.soga",           "bias1"),
    ("TrueSkills",           PROGRAMS_SOGA / "TrueSkills.soga",             "skillA"),
    ("TwoCoins",             PROGRAMS_SOGA / "TwoCoins.soga",               "first"),
    ("Altermu",              PROGRAMS_AQUA / "Altermu.soga",                "w1"),
    ("Altermu2",             PROGRAMS_AQUA / "Altermu2.soga",               "w1"),
    ("NormalMixturesPrune",  PROGRAMS_AQUA / "NormalMixturesPrune.soga",    "theta"),
    ("RadarQuery",           PROGRAMS_AQUA / "RadarQuery.soga",             "x0"),
    ("TimeSeries",           PROGRAMS_AQUA / "TimeSeries.soga",             "a"),
]


def run_one(name: str, soga_path: Path, target_var: str) -> dict:
    """Run one benchmark and return a result dict."""
    result = {
        "name": name,
        "path": str(soga_path),
        "target_var": target_var,
        "status": "ERROR",
        "preproc_time_s": float("nan"),
        "cfg_time_s": float("nan"),
        "soga_time_s": float("nan"),
        "total_time_s": float("nan"),
        "n_comp": float("nan"),
        "target_mean": float("nan"),
        "target_var_value": float("nan"),
        "error_msg": "",
    }

    if not soga_path.exists():
        result["error_msg"] = f"File not found: {soga_path}"
        return result

    try:
        # 1. Preprocess
        t0 = time.perf_counter()
        compiled_path = compile2SOGA(str(soga_path))
        result["preproc_time_s"] = time.perf_counter() - t0

        # 2. Build CFG
        t1 = time.perf_counter()
        cfg = produce_cfg(compiled_path)
        result["cfg_time_s"] = time.perf_counter() - t1

        # 3. Run SOGA
        t2 = time.perf_counter()
        dist = start_SOGA(cfg)
        result["soga_time_s"] = time.perf_counter() - t2

        result["total_time_s"] = result["preproc_time_s"] + result["cfg_time_s"] + result["soga_time_s"]

        # 4. Extract target variable statistics
        if dist is not None:
            result["n_comp"] = dist.gm.n_comp()
            # Find target variable index
            tvar = target_var.strip()
            if tvar in dist.var_list:
                idx = dist.var_list.index(tvar)
                mean_vec = dist.gm.mean()
                cov_mat = dist.gm.cov()
                result["target_mean"] = float(mean_vec[idx])
                result["target_var_value"] = float(cov_mat[idx, idx])
            else:
                result["error_msg"] = f"target var '{tvar}' not in {dist.var_list[:5]}..."

        result["status"] = "OK"

    except Exception as exc:
        result["status"] = "ERROR"
        result["error_msg"] = str(exc)[:200]

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = time.strftime("%Y-%m-%d")
    out_path = RESULTS_DIR / f"baseline_matrix_gm_{date_str}.csv"

    print(f"Running {len(BENCHMARKS)} benchmarks...")
    print(f"Output: {out_path}")
    print()

    fields = [
        "name", "path", "target_var", "status",
        "preproc_time_s", "cfg_time_s", "soga_time_s", "total_time_s",
        "n_comp", "target_mean", "target_var_value", "error_msg",
    ]

    rows = []
    for name, path, tvar in BENCHMARKS:
        print(f"  {name:<30}", end="", flush=True)
        row = run_one(name, path, tvar)
        rows.append(row)
        if row["status"] == "OK":
            print(f"OK  {row['total_time_s']:.3f}s  (n_comp={row['n_comp']}, "
                  f"mean={row['target_mean']:.4g}, var={row['target_var_value']:.4g})")
        else:
            print(f"ERROR: {row['error_msg'][:60]}")

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for r in rows if r["status"] == "OK")
    err_count = len(rows) - ok_count
    print()
    print(f"Results: {ok_count}/{len(rows)} OK, {err_count} errors")
    print(f"Saved: {out_path}")
    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
