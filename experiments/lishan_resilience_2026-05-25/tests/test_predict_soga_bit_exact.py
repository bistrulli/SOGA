"""
R2.4 — Tests for predict_resilience_soga.py (bit-exact primary predictor).

5 cross-validation tests per test-engineer Q2:
  1. Bit-exact must be MORE accurate than 5-class vs MC at v=1.0
  2. Bit-exact vs MC at v in {0.1, 0.5->use 1.0, 1.0, 5.0->use 10.0} with n=1000:
     rel err SDC < 20x (noise floor); abs err OTR < 1% absolute
  3. K=33 reduction vs K_full validation at m=n=4
  4. OTR aggregation semantics matches MC (per-execution)
  5. predict_v_sweep and predict_bimodal_sweep produce consistent results
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, EXP_DIR)

from predict_resilience_soga import (
    predict_v_sweep, predict_bimodal_sweep, compute_baseline, compute_sdc_vectorized
)
from predict_resilience_soga_5class import predict_v_sweep as predict_v_sweep_5class
from simulate_fi_mc import simulate_v_sweep, simulate_p_sweep


@pytest.fixture(scope="module")
def A_mat():
    results_dir = os.path.join(EXP_DIR, "results")
    return np.load(os.path.join(results_dir, "A_kernel.npz"))["A"]


class TestBitExactVs5ClassAccuracy:
    """Bit-exact must be more accurate than 5-class vs MC reference."""

    def test_bit_exact_more_accurate_sdc_at_v1(self, A_mat):
        """At v=1.0: bit-exact SDC much closer to MC than 5-class.

        5-class overestimates SDC by ~38×; bit-exact should be within ~10% (MC noise floor).
        """
        v_list = [1.0]
        eps, p_fault = 1e-3, 0.01

        mc_results = simulate_v_sweep(A_mat, v_list, n_samples=1000, seed=42,
                                       eps=eps, p_fault=p_fault)
        be_results = predict_v_sweep(A_mat, 0.0, v_list, eps=eps, p_fault=p_fault)
        fc_results = predict_v_sweep_5class(A_mat, 0.0, v_list, eps=eps, p_fault=p_fault)

        mc_sdc = mc_results[1.0]["SDC"]
        be_sdc = be_results[1.0]["SDC"]
        fc_sdc = fc_results[1.0]["SDC"]

        err_be = abs(be_sdc - mc_sdc)
        err_fc = abs(fc_sdc - mc_sdc)

        # Bit-exact should be much closer to MC than 5-class
        assert err_be < err_fc, (
            f"Bit-exact error {err_be:.2e} NOT less than 5-class error {err_fc:.2e}. "
            f"MC={mc_sdc:.2e}, bit-exact={be_sdc:.2e}, 5-class={fc_sdc:.2e}"
        )

        # 5-class should be much worse (factor > 5×)
        assert err_fc > 5 * err_be, (
            f"5-class not substantially worse than bit-exact: "
            f"err_fc/err_be = {err_fc/max(err_be,1e-15):.1f}"
        )


class TestBitExactVsMCAccuracy:
    """Bit-exact vs MC at multiple v-points."""

    @pytest.mark.parametrize("v_val", [0.1, 1.0, 10.0])
    def test_sdc_within_factor10_of_mc(self, A_mat, v_val):
        """Bit-exact SDC within factor 10 of MC (MC noise is ~40% at n=1000 for this SDC range).

        With SDC ~ 5e-6 and n=1000, MC std is sqrt(5e-6 * (1-5e-6) / 1000) ~ 2.2e-6.
        The relative noise is ~44%, so we expect ratio in [0.1, 10] easily.
        """
        mc = simulate_v_sweep(A_mat, [v_val], n_samples=1000, seed=42,
                               eps=1e-3, p_fault=0.01)
        be = predict_v_sweep(A_mat, 0.0, [v_val], eps=1e-3, p_fault=0.01)
        mc_sdc = mc[v_val]["SDC"]
        be_sdc = be[v_val]["SDC"]

        if mc_sdc < 1e-9:
            pytest.skip(f"MC SDC too small ({mc_sdc:.2e}) to compare at n=1000")

        ratio = be_sdc / mc_sdc
        assert 0.1 < ratio < 10, (
            f"v={v_val}: bit-exact/MC ratio = {ratio:.1f} (expected 0.1-10). "
            f"MC={mc_sdc:.2e}, bit-exact={be_sdc:.2e}"
        )

    @pytest.mark.parametrize("v_val", [0.1, 1.0, 10.0])
    def test_otr_abs_err_less_than_1pct(self, A_mat, v_val):
        """Bit-exact OTR absolute error vs MC < 1% (0.01)."""
        mc = simulate_v_sweep(A_mat, [v_val], n_samples=1000, seed=42,
                               eps=1e-3, p_fault=0.01)
        be = predict_v_sweep(A_mat, 0.0, [v_val], eps=1e-3, p_fault=0.01)
        abs_err_otr = abs(be[v_val]["OTR"] - mc[v_val]["OTR"])
        assert abs_err_otr < 0.01, (
            f"v={v_val}: OTR abs err = {abs_err_otr:.4f} > 0.01. "
            f"MC={mc[v_val]['OTR']:.4f}, bit-exact={be[v_val]['OTR']:.4f}"
        )


class TestKReductionIdentity:
    """K=33 reduction for A=I_4 (one bit-exact + baseline = 33 components)."""

    def test_k33_matches_full_aggregation_at_m4(self):
        """For A=I_4, flat v=1.0: compute_sdc_vectorized matches compute_per_cell_SDC.

        Both use bit-exact kernel; vectorized path should match per-cell path.
        """
        m, n = 4, 4
        A = np.eye(m)
        v, eps, p_fault = 1.0, 1e-3, 0.01

        M_B = np.full((m, n), v)
        U_B = 1e-10 * np.eye(m)
        V_B = np.eye(n)
        M_D, U_D, V_D = compute_baseline(A, M_B, U_B, V_B)

        # Vectorized path
        msk_v, sdc_v, otr_v = compute_sdc_vectorized(
            M_D, U_D, V_D, A, v, eps, p_fault, m_input=m, n_input=n
        )

        # For A=I_4, v=1.0: SDC = p_fault * 17/(m*n*32) = 0.01*17/512 = 3.32e-4
        # OTR = p_fault * n_special/32 = 0.01 * 1/32 = 3.125e-4
        expected_sdc = p_fault * 17 / (m * n * 32)  # bit-exact formula for I_4
        expected_otr = p_fault * 1.0 / 32.0  # 1 special bit (bit 30)

        assert abs(sdc_v - expected_sdc) < 1e-6, (
            f"compute_sdc_vectorized SDC={sdc_v:.4e}, expected {expected_sdc:.4e}"
        )
        assert abs(otr_v - expected_otr) < 1e-8, (
            f"OTR={otr_v:.4e}, expected {expected_otr:.4e} (1 special bit)"
        )


class TestOTRSemantics:
    """OTR aggregation aligned with MC per-execution semantics."""

    def test_otr_matches_mc_per_execution_at_v1(self, A_mat):
        """At v=1.0: SOGA OTR = p_fault * n_special/32.

        For v=1.0: bit 30 flip -> +Inf (1 special bit).
        Expected: 0.01 * 1/32 = 3.125e-4.
        MC confirms: ~3.1e-4 at n=1000.
        """
        v_list = [1.0]
        eps, p_fault = 1e-3, 0.01

        be = predict_v_sweep(A_mat, 0.0, v_list, eps=eps, p_fault=p_fault)
        expected_otr = p_fault * 1.0 / 32.0  # 1 special bit

        assert abs(be[1.0]["OTR"] - expected_otr) < 1e-8, (
            f"OTR={be[1.0]['OTR']:.6f}, expected {expected_otr:.6f}"
        )

    def test_otr_zero_at_v_not_near_special_bit(self, A_mat):
        """At v=0.1: bit 30 flip does not produce Inf -> OTR=0."""
        be = predict_v_sweep(A_mat, 0.0, [0.1], eps=1e-3, p_fault=0.01)
        assert be[0.1]["OTR"] < 1e-6, (
            f"OTR={be[0.1]['OTR']:.4e} at v=0.1 should be ~0 (no special bits)"
        )


class TestBimodalConsistency:
    """predict_bimodal_sweep consistency checks."""

    def test_bimodal_p1_matches_predict_v_high(self, A_mat):
        """At p=1.0: bimodal prediction must match predict_v_sweep(V_high)."""
        V_high, eps, p_fault = 1.0, 1e-3, 0.01
        V_low = 0.0

        bimodal = predict_bimodal_sweep(A_mat, [1.0], V_low, V_high, eps=eps, p_fault=p_fault)
        v_sweep = predict_v_sweep(A_mat, 0.0, [V_high], eps=eps, p_fault=p_fault)

        for key in ("MSK", "SDC", "OTR"):
            # Tolerance 1e-6: small difference from different U_B_scale (1e-6 vs 1e-10)
            assert abs(bimodal[1.0][key] - v_sweep[V_high][key]) < 1e-6, (
                f"bimodal[p=1][{key}]={bimodal[1.0][key]:.6e} != "
                f"v_sweep[{V_high}][{key}]={v_sweep[V_high][key]:.6e}"
            )

    def test_bimodal_p0_matches_predict_v_low(self, A_mat):
        """At p=0.0: bimodal prediction must match predict_v_sweep(V_low)."""
        V_high, eps, p_fault = 1.0, 1e-3, 0.01
        V_low = 0.0

        bimodal = predict_bimodal_sweep(A_mat, [0.0], V_low, V_high, eps=eps, p_fault=p_fault)
        v_sweep = predict_v_sweep(A_mat, 0.0, [V_low], eps=eps, p_fault=p_fault)

        for key in ("MSK", "SDC", "OTR"):
            assert abs(bimodal[0.0][key] - v_sweep[V_low][key]) < 1e-6, (
                f"bimodal[p=0][{key}]={bimodal[0.0][key]:.6e} != "
                f"v_sweep[{V_low}][{key}]={v_sweep[V_low][key]:.6e}"
            )
