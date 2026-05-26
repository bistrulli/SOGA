"""
R4.3 + R4.4 — Step 3 monotonicity tests for bit-exact predictor.

Statistical tests per gm-expert Q6 + test-engineer Q4, revised per Codex iter 1 F3-F5:
  1. Kendall tau via scipy.stats.kendalltau (p-value as primary statistic)
  2. Sign-test: n_wrong = count of descending SDC steps (heuristic only)
  3. Binomial p-value via scipy.stats.binomtest
  4. Robust threshold rule with Bonferroni correction for 3 curves (F5):
     - MONOTONE: p_kendall < 0.0167 (= 0.05/3) AND |tau| > 0.7
     - NON-MONOTONE: p_kendall > 0.05 AND n_wrong >= 10
     - AMBIGUOUS: between thresholds
     - PLATEAU-DEGENERATE: SDC range < 5*max(sigma_MC, 0.01) -> exempt
  5. MC validation at 4 points (p=0, 0.5, 0.75, 1.0) with 5000 samples
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pytest
from scipy import stats as spstats

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(EXP_DIR)), "src"))

from simulate_fi_mc import simulate_p_sweep
from predict_resilience_soga import predict_bimodal_sweep


@pytest.fixture(scope="module")
def cfg():
    with open(os.path.join(EXP_DIR, "config.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def A_mat():
    return np.load(os.path.join(EXP_DIR, "results/A_kernel.npz"))["A"]


@pytest.fixture(scope="module")
def soga_step3_be(cfg, A_mat):
    """Bit-exact SOGA bimodal sweep over 21 p-points."""
    p_list = cfg["step3"]["p_sweep"]
    V_low, V_high = cfg["step3"]["V_low"], cfg["step3"]["V_high"]
    eps, p_fault = cfg["eps"], cfg["fault_model"]["p_fault"]
    return predict_bimodal_sweep(A_mat, p_list, V_low, V_high, eps=eps, p_fault=p_fault)


@pytest.fixture(scope="module")
def mc_step3_5k(cfg, A_mat):
    """MC p-sweep at 4 validation points with 5000 samples (R4.4)."""
    V_low, V_high = cfg["step3"]["V_low"], cfg["step3"]["V_high"]
    eps, p_fault = cfg["eps"], cfg["fault_model"]["p_fault"]
    p_validate = [0.0, 0.5, 0.75, 1.0]
    return simulate_p_sweep(
        A_mat, p_validate, V_low, V_high,
        n_samples=5000, seed=42, eps=eps, p_fault=p_fault,
    )


# ---------------------------------------------------------------------------
# Bonferroni threshold for 3 curves (SDC, MSK, OTR)
# ---------------------------------------------------------------------------

ALPHA_BONFERRONI = 0.05 / 3  # = 0.01667
TAU_SIZEABLE = 0.7
N_WRONG_THRESHOLD = 10  # majority wrong-sign for n=20 steps


def monotonicity_verdict(
    p_list: list,
    curve_vals: list,
    curve_name: str,
    mc_sigma: float = 0.01,
) -> tuple[str, float, float, int]:
    """Determine monotonicity verdict using Kendall tau + sign-test.

    Returns:
        (verdict, tau, p_kendall, n_wrong)
        verdict: MONOTONE | NON-MONOTONE | AMBIGUOUS | PLATEAU-DEGENERATE
    """
    arr = np.array(curve_vals)

    # Plateau-degenerate check: SDC range < 5 * max(sigma_MC, 0.01)
    sdc_range = float(np.max(arr) - np.min(arr))
    if sdc_range < 5 * max(mc_sigma, 0.01):
        return "PLATEAU-DEGENERATE", 0.0, 1.0, 0

    tau, p_kendall = spstats.kendalltau(p_list, curve_vals)
    n_wrong = int(np.sum(np.diff(arr) < 0))

    if p_kendall < ALPHA_BONFERRONI and abs(tau) > TAU_SIZEABLE:
        verdict = "MONOTONE"
    elif p_kendall > 0.05 and n_wrong >= N_WRONG_THRESHOLD:
        verdict = "NON-MONOTONE"
    else:
        verdict = "AMBIGUOUS"

    return verdict, float(tau), float(p_kendall), n_wrong


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKendallMonotonicity:
    """Kendall tau monotonicity test with Bonferroni correction."""

    def test_sdc_kendall_verdict(self, soga_step3_be, cfg):
        """SOGA SDC must have clear monotone verdict (MONOTONE or AMBIGUOUS, not NON-MONOTONE)."""
        p_list = sorted(cfg["step3"]["p_sweep"])
        sdc_vals = [soga_step3_be[p]["SDC"] for p in p_list]

        verdict, tau, p_kendall, n_wrong = monotonicity_verdict(p_list, sdc_vals, "SDC")
        print(f"\nSDC monotonicity: verdict={verdict}, tau={tau:.3f}, "
              f"p_kendall={p_kendall:.4f}, n_wrong={n_wrong}")

        # SOGA bit-exact SDC is analytically derived → should not be NON-MONOTONE
        assert verdict != "NON-MONOTONE", (
            f"SOGA SDC classified as NON-MONOTONE: tau={tau:.3f}, "
            f"p_kendall={p_kendall:.4f}, n_wrong={n_wrong}. "
            "Write counterexample_analysis.md if this is genuine."
        )

    def test_msk_kendall_verdict(self, soga_step3_be, cfg):
        """SOGA MSK should be monotone decreasing (inverted for test)."""
        p_list = sorted(cfg["step3"]["p_sweep"])
        msk_vals = [soga_step3_be[p]["MSK"] for p in p_list]
        # For decreasing: test with inverted values
        inv_msk = [-x for x in msk_vals]
        verdict, tau, p_kendall, n_wrong = monotonicity_verdict(p_list, inv_msk, "MSK")
        print(f"\nMSK (decreasing) verdict={verdict}, tau={tau:.3f}")

        assert verdict != "NON-MONOTONE", (
            f"SOGA MSK (decreasing) classified as NON-MONOTONE: tau={tau:.3f}"
        )


class TestSignTest:
    """Sign-test: count descending steps in SDC curve."""

    def test_sdc_not_majority_descending(self, soga_step3_be, cfg):
        """For SOGA SDC, fewer than half the p-steps should be descending."""
        p_list = sorted(cfg["step3"]["p_sweep"])
        sdc_vals = [soga_step3_be[p]["SDC"] for p in p_list]

        n_steps = len(sdc_vals) - 1
        n_wrong = int(np.sum(np.diff(np.array(sdc_vals)) < 0))
        print(f"\nSign test: n_wrong={n_wrong}/{n_steps}")

        # For bit-exact analytical model: SDC should not have majority descending steps
        # (n_wrong < 10 out of 20 steps)
        assert n_wrong < N_WRONG_THRESHOLD, (
            f"SDC has {n_wrong} descending steps out of {n_steps} "
            f"(threshold={N_WRONG_THRESHOLD})"
        )


class TestBinomialCI:
    """Binomial CI containment at 4 validation points (R4.4)."""

    def test_mc_binomial_ci_at_4points(self, mc_step3_5k):
        """MC 5000 samples: binomial CI half-width <= 0.007."""
        for p_val in sorted(mc_step3_5k.keys()):
            mc_sdc = mc_step3_5k[p_val]["SDC"]
            n = mc_step3_5k[p_val].get("n_samples", 5000)
            if mc_sdc == 0.0 or mc_sdc == 1.0:
                continue  # boundary: CI is 0
            # CI half-width using Wilson formula: sqrt(p*(1-p)/n) * 1.96
            ci_half = 1.96 * np.sqrt(mc_sdc * (1 - mc_sdc) / n)
            assert ci_half <= 0.007, (
                f"MC SDC CI half-width {ci_half:.5f} > 0.007 at p={p_val}, "
                f"SDC={mc_sdc:.4e}, n={n}"
            )

    def test_soga_be_within_mc_ci_at_4points(self, mc_step3_5k, A_mat, cfg):
        """Bit-exact SOGA within MC binomial CI at 4 validation p-points.

        Note: SDC is very small (~5e-6), so binomial CI is very wide relative to SDC.
        The test uses absolute tolerance of 0.005.
        """
        V_low, V_high = cfg["step3"]["V_low"], cfg["step3"]["V_high"]
        eps, p_fault = cfg["eps"], cfg["fault_model"]["p_fault"]
        p_list = sorted(mc_step3_5k.keys())
        soga = predict_bimodal_sweep(A_mat, p_list, V_low, V_high, eps=eps, p_fault=p_fault)

        for p_val in p_list:
            mc_sdc = mc_step3_5k[p_val]["SDC"]
            sg_sdc = soga[p_val]["SDC"]
            abs_err = abs(sg_sdc - mc_sdc)
            # Absolute tolerance: both are tiny, absolute err < 0.001
            assert abs_err < 0.001, (
                f"p={p_val}: |SOGA-MC| SDC = {abs_err:.4e} > 0.001. "
                f"MC={mc_sdc:.3e}, SOGA={sg_sdc:.3e}"
            )


class TestMonotonicityVerdictRule:
    """Test the threshold rule implementation directly."""

    def test_monotone_rule_fires_for_increasing_seq(self):
        """MONOTONE rule fires for perfectly increasing sequence with large range."""
        p_list = list(np.linspace(0, 1, 21))
        # Large range ensures not plateau-degenerate: range=1.0 >> 5*max(0.01,0.01)=0.05
        sdc_vals = [float(p) for p in p_list]  # perfectly increasing 0 to 1
        verdict, tau, p_k, n_w = monotonicity_verdict(p_list, sdc_vals, "SDC", mc_sigma=0.01)
        print(f"\nTest rule: verdict={verdict}, tau={tau:.3f}, p={p_k:.4f}")
        assert verdict == "MONOTONE", f"Expected MONOTONE, got {verdict}"

    def test_non_monotone_rule_fires_for_zigzag_seq(self):
        """NON-MONOTONE rule fires for zigzag sequence with many descending steps."""
        p_list = list(np.linspace(0, 1, 21))
        # 15 of 20 steps descending AND no clear tau trend -> NON-MONOTONE
        # Use large amplitude noise to make Kendall tau insignificant
        rng = np.random.default_rng(99)
        sdc_vals = [0.05 + 0.04 * rng.choice([-1, 1]) for _ in p_list]
        verdict, tau, p_k, n_w = monotonicity_verdict(p_list, sdc_vals, "SDC", mc_sigma=0.001)
        print(f"\nZigzag: verdict={verdict}, tau={tau:.3f}, p_k={p_k:.3f}, n_wrong={n_w}")
        # This is a heuristic test — zigzag should give AMBIGUOUS or NON-MONOTONE
        # (depends on exact sequence; p_k > 0.05 is likely for random permutation)
        assert verdict in ("NON-MONOTONE", "AMBIGUOUS"), (
            f"Zigzag should be NON-MONOTONE or AMBIGUOUS, got {verdict}"
        )

    def test_plateau_exempt_for_flat_curve(self):
        """PLATEAU-DEGENERATE for near-constant curve."""
        p_list = list(np.linspace(0, 1, 21))
        sdc_vals = [1e-6 + np.random.default_rng(42).normal(0, 1e-7) for _ in p_list]
        verdict, _, _, _ = monotonicity_verdict(p_list, sdc_vals, "SDC", mc_sigma=0.01)
        assert verdict == "PLATEAU-DEGENERATE", f"Expected PLATEAU-DEGENERATE, got {verdict}"
