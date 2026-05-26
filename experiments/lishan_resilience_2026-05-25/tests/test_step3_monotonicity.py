"""
M5.5 — Step 3 monotonicity test and counterexample hunt.

Tests:
1. SOGA SDC curve is perfectly monotone increasing in p (Kendall tau >= 0.9)
2. SOGA MSK is monotone decreasing in p
3. MC SDC at 4 validation points agrees with SOGA in direction (both increase with p)
4. Linear interpolation check: SDC(p) ~ p * SDC_at_p1 (since V_low=0 → SDC_low=0)

HONESTY DISCLAIMER:
    MC validation at 4 points (p=0, 0.5, 0.75, 1.0) uses N=1000 samples.
    Since SDC probabilities are ~0.0001-0.0002, MC resolution is limited.
    Monotonicity of SOGA is analytical (deterministic).
    MC monotonicity may not hold at N=1000 due to noise.
    See LIMITATIONS.md for context on mantissa approximation errors.
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

from simulate_fi_mc import simulate_p_sweep
from predict_resilience_soga import predict_bimodal_sweep

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    config_path = os.path.join(EXP_DIR, "config.json")
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def A_mat():
    results_dir = os.path.join(EXP_DIR, "results")
    return np.load(os.path.join(results_dir, "A_kernel.npz"))["A"]


@pytest.fixture(scope="module")
def soga_step3(cfg, A_mat):
    """SOGA bimodal sweep over all 21 p-points."""
    p_list = cfg["step3"]["p_sweep"]
    V_low = cfg["step3"]["V_low"]
    V_high = cfg["step3"]["V_high"]
    eps = cfg["eps"]
    p_fault = cfg["fault_model"]["p_fault"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return predict_bimodal_sweep(A_mat, p_list, V_low, V_high, eps=eps, p_fault=p_fault)


@pytest.fixture(scope="module")
def mc_step3(cfg, A_mat):
    """MC p-sweep at 4 validation points."""
    V_low = cfg["step3"]["V_low"]
    V_high = cfg["step3"]["V_high"]
    eps = cfg["eps"]
    p_fault = cfg["fault_model"]["p_fault"]
    n_samples = cfg["step3"]["mc_n_samples"]
    seed = cfg["seed"]
    # 4 validation points
    p_validate = [0.0, 0.5, 0.75, 1.0]
    return simulate_p_sweep(
        A_mat, p_validate, V_low, V_high,
        n_samples=n_samples, seed=seed, eps=eps, p_fault=p_fault,
    )


# ---------------------------------------------------------------------------
# SOGA monotonicity tests
# ---------------------------------------------------------------------------


class TestSOGAMonotonicity:
    """SOGA SDC must be perfectly monotone increasing in p."""

    def test_soga_sdc_monotone_kendall_tau(self, soga_step3, cfg):
        """Kendall tau for SOGA SDC vs p must be >= 0.9."""
        p_list = sorted(cfg["step3"]["p_sweep"])
        sdc_vals = [soga_step3[p]["SDC"] for p in p_list]
        tau, _ = spstats.kendalltau(p_list, sdc_vals)
        threshold = cfg["step3"]["monotonicity_tau_threshold"]
        assert tau >= threshold, (
            f"SOGA SDC Kendall tau = {tau:.3f} < {threshold}. "
            "SDC is not monotone in p — unexpected non-monotone resilience."
        )

    def test_soga_msk_monotone_decreasing(self, soga_step3, cfg):
        """SOGA MSK must be monotone decreasing in p."""
        p_list = sorted(cfg["step3"]["p_sweep"])
        msk_vals = [soga_step3[p]["MSK"] for p in p_list]
        tau, _ = spstats.kendalltau(p_list, [-x for x in msk_vals])  # invert for "increasing"
        threshold = cfg["step3"]["monotonicity_tau_threshold"]
        assert tau >= threshold, (
            f"SOGA MSK (decreasing) Kendall tau = {tau:.3f} < {threshold}."
        )

    def test_soga_sdc_at_p0_is_zero(self, soga_step3, cfg):
        """At p=0 (all V_low=0): SDC must be 0 (no fault delta visible at v=0)."""
        p0 = 0.0
        if p0 in soga_step3:
            sdc = soga_step3[p0]["SDC"]
            assert sdc < 1e-9, (
                f"SOGA SDC={sdc:.2e} at p=0 (V_low=0) should be 0. "
                "No faults visible when input is 0."
            )

    def test_soga_msk_at_p0_is_one(self, soga_step3, cfg):
        """At p=0 (all V_low=0): MSK must be 1.0."""
        p0 = 0.0
        if p0 in soga_step3:
            msk = soga_step3[p0]["MSK"]
            assert abs(msk - 1.0) < 1e-9, (
                f"SOGA MSK={msk:.6f} at p=0 should be 1.0."
            )

    def test_soga_sdc_linear_in_p(self, soga_step3, cfg):
        """For V_low=0: SDC(p) = p * SDC(p=1) by linearity of 2-component GM.
        Relative error of linear interpolation must be < 1e-9 (analytical).
        """
        p_list = sorted(cfg["step3"]["p_sweep"])
        sdc_at_1 = soga_step3[1.0]["SDC"] if 1.0 in soga_step3 else None
        if sdc_at_1 is None:
            pytest.skip("p=1.0 not in sweep")
        for p in p_list:
            expected = p * sdc_at_1
            actual = soga_step3[p]["SDC"]
            if expected < 1e-12:
                assert actual < 1e-9, f"SDC at p={p} should be ~0"
            else:
                rel_err = abs(actual - expected) / expected
                assert rel_err < 1e-8, (
                    f"SDC linearity broken at p={p}: actual={actual:.6e}, "
                    f"expected={expected:.6e}, rel_err={rel_err:.2e}"
                )


class TestProbabilityConsistency:
    """Sanity checks on SOGA Step 3 output."""

    def test_soga_probs_sum_to_one(self, soga_step3, cfg):
        for p in cfg["step3"]["p_sweep"]:
            r = soga_step3[p]
            total = r["MSK"] + r["SDC"] + r["OTR"]
            assert abs(total - 1.0) < 0.01, (
                f"SOGA probs sum={total:.6f} != 1.0 at p={p}"
            )

    def test_no_negative_probabilities(self, soga_step3, cfg):
        for p in cfg["step3"]["p_sweep"]:
            r = soga_step3[p]
            for k in ("MSK", "SDC", "OTR"):
                assert r[k] >= -1e-9, f"Negative {k}={r[k]:.6f} at p={p}"

    def test_two_component_gm_mode_for_sparse_a(self, soga_step3, cfg):
        """For A=I_32 (m_dense=1 < 16): all results should use 2-component GM."""
        for p, r in soga_step3.items():
            assert r["mode"] == "2-component-GM", (
                f"Expected 2-component-GM mode at p={p}, got {r['mode']}"
            )


class TestMCValidation:
    """MC validation at 4 p-points (directional check, noise-tolerant)."""

    def test_mc_sdc_increases_with_p(self, mc_step3):
        """MC SDC should weakly increase from p=0 to p=1."""
        # MC at p=0 (all V_low=0): delta=0 → SDC=0 exactly
        if 0.0 in mc_step3 and 1.0 in mc_step3:
            sdc_0 = mc_step3[0.0]["SDC"]
            sdc_1 = mc_step3[1.0]["SDC"]
            assert sdc_1 >= sdc_0 - 1e-6, (
                f"MC SDC at p=1 ({sdc_1:.6f}) < p=0 ({sdc_0:.6f}) — unexpected decrease."
            )

    def test_mc_msk_at_p0_is_one(self, mc_step3):
        """At p=0 (V_low=0): MC MSK should be 1.0 (no fault possible)."""
        if 0.0 in mc_step3:
            msk = mc_step3[0.0]["MSK"]
            assert abs(msk - 1.0) < 0.01, (
                f"MC MSK at p=0 = {msk:.6f}, expected ~1.0 "
                "(no fault visible when input=0)"
            )

    @pytest.mark.xfail(
        reason=(
            "MC monotonicity at N=1000 is noisy at SDC~5e-6 level. "
            "Kendall tau may not reach 0.9 due to statistical noise. "
            "See LIMITATIONS.md: MC SDC resolution insufficient at p_fault=0.01."
        ),
        strict=False,
    )
    def test_mc_sdc_monotone_tau(self, mc_step3, cfg):
        """Kendall tau for MC SDC vs p at 4 points (expected to fail at N=1000)."""
        p_list = sorted(mc_step3.keys())
        sdc_vals = [mc_step3[p]["SDC"] for p in p_list]
        tau, _ = spstats.kendalltau(p_list, sdc_vals)
        threshold = cfg["step3"]["monotonicity_tau_threshold"]
        assert tau >= threshold, (
            f"MC SDC Kendall tau = {tau:.3f} < {threshold} (noise at N=1000)"
        )


class TestCounterexampleHunt:
    """
    Hunt for non-monotone behavior (counterexample).
    A counterexample would mean: higher p (more high-value inputs) → LOWER SDC risk.
    This would be scientifically interesting and worth reporting to Lishan.
    """

    def test_no_counterexample_in_soga(self, soga_step3, cfg):
        """SOGA must NOT show a counterexample (tau > 0.5)."""
        p_list = sorted(cfg["step3"]["p_sweep"])
        sdc_vals = [soga_step3[p]["SDC"] for p in p_list]
        tau, _ = spstats.kendalltau(p_list, sdc_vals)
        counterexample_threshold = cfg["step3"]["counterexample_tau_threshold"]
        assert tau > counterexample_threshold, (
            f"COUNTEREXAMPLE FOUND: SOGA SDC Kendall tau = {tau:.3f} <= {counterexample_threshold}. "
            "SDC is non-monotone — see counterexample_analysis.md."
        )
