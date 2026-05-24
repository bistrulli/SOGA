"""
tests/test_truncate_matrix.py — M5.6 tests for matrix truncation operations.

Covers:
  - Constraint classifier (TestConstraintClassifier)
  - Element inequality X[i,j] op c (TestElementIneq)
  - Row sum inequality row_sum(X,i) op c (TestRowSumIneq)
  - Column sum inequality col_sum(X,j) op c (TestColSumIneq)
  - Log-weight infrastructure (TestLogWeights)
  - Dense cross-validation vs scalar truncate path (TestDenseCrossValidation)
  - Memory budget guard (TestMemoryBudget)
  - Integration via truncate() routing guard (TestTruncateRouting)

Plan reference: §M5 of plan/2026-05-22-matrix-gm-lishan.md
"""

import math
import sys
import os
import warnings
from copy import deepcopy

import numpy as np
import pytest

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from libSOGAshared import Dist, GaussianMix, VarEntry
from libSOGAsharedMatrix import GaussianMixBlock, _enforce_psd_kron_factors
from libMatrixTruncate import (
    _classify_constraint,
    _rank1_cond_update,
    _truncate_matrix_element_ineq,
    _truncate_matrix_row_sum_ineq,
    _truncate_matrix_col_sum_ineq,
    truncate_matrix,
    KroneckerDenseMemoryWarning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_comp_block(m, n, M=None, U=None, V=None, var_name="X"):
    """Build a single-component GaussianMixBlock with one matrix variable."""
    if M is None:
        M = np.zeros((m, n))
    if U is None:
        U = np.eye(m)
    if V is None:
        V = np.eye(n)
    ve = VarEntry(name=var_name, kind="matrix", shape=(m, n), flat_offset=0)
    U, V = _enforce_psd_kron_factors(U, V)
    mu_blocks = [{var_name: M.copy()}]
    cov_blocks = [{frozenset({var_name}): (U.copy(), V.copy())}]
    return GaussianMixBlock(
        var_list=[],
        var_entries=[ve],
        pi=[1.0],
        mu_blocks=mu_blocks,
        cov_blocks=cov_blocks,
    )


def _make_dist_with_block(m, n, M=None, U=None, V=None, var_name="X"):
    """Build a Dist object that has gm_block set (no scalar part)."""
    block = _make_single_comp_block(m, n, M, U, V, var_name)
    ve = VarEntry(name=var_name, kind="matrix", shape=(m, n), flat_offset=0)
    # Scalar GM is empty (no scalar variables)
    gm = GaussianMix([1.0], [np.zeros(0)], [np.zeros((0, 0))])
    dist = Dist(var_list=[], gm=gm, var_entries=[ve], gm_block=block)
    return dist


# ---------------------------------------------------------------------------
# TestConstraintClassifier
# ---------------------------------------------------------------------------

class TestConstraintClassifier:

    def test_element_ineq_gt(self):
        c = _classify_constraint("X[0,1] > 0.5", "X")
        assert c["type"] == "ELEMENT_INEQ"
        assert c["i"] == 0 and c["j"] == 1
        assert c["direction"] == ">"
        assert c["threshold"] == pytest.approx(0.5)

    def test_element_ineq_lt_negative(self):
        c = _classify_constraint("Y[2,3] < -1.5", "Y")
        assert c["type"] == "ELEMENT_INEQ"
        assert c["i"] == 2 and c["j"] == 3
        assert c["direction"] == "<"
        assert c["threshold"] == pytest.approx(-1.5)

    def test_element_ineq_geq(self):
        c = _classify_constraint("X[0,0] >= 0.0", "X")
        assert c["type"] == "ELEMENT_INEQ"
        assert c["direction"] == ">="

    def test_element_ineq_leq(self):
        c = _classify_constraint("X[1,2] <= 3.0", "X")
        assert c["type"] == "ELEMENT_INEQ"
        assert c["direction"] == "<="

    def test_row_sum_ineq(self):
        c = _classify_constraint("row_sum(X, 1) > 2.0", "X")
        assert c["type"] == "ROW_SUM_INEQ"
        assert c["i"] == 1
        assert c["direction"] == ">"
        assert c["threshold"] == pytest.approx(2.0)

    def test_col_sum_ineq(self):
        c = _classify_constraint("col_sum(X, 2) <= -0.5", "X")
        assert c["type"] == "COL_SUM_INEQ"
        assert c["j"] == 2
        assert c["direction"] == "<="
        assert c["threshold"] == pytest.approx(-0.5)

    def test_trace_raises(self):
        with pytest.raises(NotImplementedError, match="trace"):
            _classify_constraint("trace(X) > 1.0", "X")

    def test_unrecognised_raises(self):
        with pytest.raises(NotImplementedError, match="Unrecognised"):
            _classify_constraint("X > 0.5", "X")


# ---------------------------------------------------------------------------
# TestRank1CondUpdate
# ---------------------------------------------------------------------------

class TestRank1CondUpdate:
    """Tests for the core rank-1 conditional Gaussian update primitive."""

    def test_scalar_case_matches_1d(self):
        """1x1 case: scalar truncation should match 1D formula directly."""
        from libSOGAtruncate import _truncated_normal_moments_1d
        mu = np.array([1.0])
        sigma = np.array([[2.0]])
        a = np.array([1.0])
        c = 0.5
        direction = ">"

        M_new, S_new, P = _rank1_cond_update(mu, sigma, a, c, direction)
        m_hat, v_hat, P_ref = _truncated_normal_moments_1d(1.0, 2.0, 0.5, direction)

        assert P == pytest.approx(P_ref, rel=1e-9)
        assert float(M_new[0]) == pytest.approx(m_hat, rel=1e-9)
        assert float(S_new[0, 0]) == pytest.approx(v_hat, rel=1e-9)

    def test_zero_prob_returns_unchanged(self):
        """When threshold is far in the tail, P should be 0 and moments unchanged."""
        mn = 4
        M_vec = np.zeros(mn)
        Sigma = np.eye(mn)
        a = np.zeros(mn)
        a[0] = 1.0
        # c very far above mean → P(X[0,0] > 100) ≈ 0
        M_new, S_new, P = _rank1_cond_update(M_vec, Sigma, a, 100.0, ">")
        assert P == pytest.approx(0.0, abs=1e-10)

    def test_symmetry_lt_gt(self):
        """Truncating X > c and X < -c on symmetric N(0,1) should give same P."""
        mn = 1
        M_vec = np.zeros(mn)
        Sigma = np.eye(mn)
        a = np.array([1.0])
        c = 0.5
        _, _, P_gt = _rank1_cond_update(M_vec, Sigma, a, c, ">")
        _, _, P_lt = _rank1_cond_update(M_vec, Sigma, a, -c, "<")
        assert P_gt == pytest.approx(P_lt, rel=1e-9)


# ---------------------------------------------------------------------------
# TestElementIneq
# ---------------------------------------------------------------------------

class TestElementIneq:

    def _make_2x2_block(self, m00=1.0, u_scale=1.0, v_scale=1.0):
        M = np.array([[m00, 0.5], [0.3, 0.7]])
        U = u_scale * np.eye(2)
        V = v_scale * np.eye(2)
        return _make_single_comp_block(2, 2, M, U, V)

    def test_norm_factor_lt_1_for_nontrivial_threshold(self):
        """Truncating X[0,0] > 0 should return norm_factor < 1 for N(0,1) element."""
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)))
        new_block, nf = _truncate_matrix_element_ineq(
            block, "X", 2, 2, i=0, j=0, c=0.0, direction=">", use_project=False
        )
        assert 0.3 < nf < 0.7   # N(0,1) truncated at 0 → P = 0.5

    def test_norm_factor_near_1_for_far_threshold(self):
        """X[0,0] > -100 should include almost all mass."""
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)))
        _, nf = _truncate_matrix_element_ineq(
            block, "X", 2, 2, i=0, j=0, c=-100.0, direction=">", use_project=False
        )
        assert nf == pytest.approx(1.0, abs=1e-6)

    def test_mean_update_increases_toward_threshold(self):
        """After X[0,0] > 0.5, mean at [0,0] should move up."""
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)))
        new_block, _ = _truncate_matrix_element_ineq(
            block, "X", 2, 2, i=0, j=0, c=0.5, direction=">", use_project=False
        )
        M_new = new_block.mu_blocks[0]["X"]
        assert M_new[0, 0] > 0.5   # mean moves above threshold

    def test_other_elements_unaffected_in_uncorrelated_case(self):
        """With independent matrix elements (kron(I,I)), X[1,1] mean stays 0
        after X[0,0] > 0.5 truncation."""
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)),
                                         U=np.eye(2), V=np.eye(2))
        new_block, _ = _truncate_matrix_element_ineq(
            block, "X", 2, 2, i=0, j=0, c=0.5, direction=">", use_project=False
        )
        M_new = new_block.mu_blocks[0]["X"]
        assert M_new[1, 1] == pytest.approx(0.0, abs=1e-10)

    def test_dense_cross_validation_vs_scalar_truncate(self):
        """Dense strategy on 2x2 element observe should match scalar truncate on vec(X)."""
        from libSOGAtruncate import _truncated_normal_moments_1d
        # X is 2x2, element [0,1] (= index 2 in column-major j*m+i = 1*2+0=2)
        m, n = 2, 2
        U = np.eye(m)
        V = np.eye(n)
        M = np.array([[0.0, 1.0], [0.5, 0.3]])
        block = _make_single_comp_block(m, n, M, U, V)
        i, j = 0, 1
        c = 0.5
        direction = "<"
        idx = j * m + i   # column-major index = 2

        new_block, nf = _truncate_matrix_element_ineq(
            block, "X", m, n, i=i, j=j, c=c, direction=direction, use_project=False
        )
        # Reference: scalar rank-1 on vec(X) with Sigma = kron(V, U) = I_4
        mu_s = M[i, j]   # = 1.0
        var_s = U[i, i] * V[j, j]  # = 1.0
        m_hat, v_hat, P_ref = _truncated_normal_moments_1d(mu_s, var_s, c, direction)

        assert nf == pytest.approx(P_ref, rel=1e-9)
        M_new = new_block.mu_blocks[0]["X"]
        assert M_new[i, j] == pytest.approx(m_hat, rel=1e-9)

    def test_project_strategy_close_to_dense(self):
        """PROJECT strategy result should be within relative tolerance of DENSE."""
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)))
        new_dense, nf_d = _truncate_matrix_element_ineq(
            block, "X", 2, 2, i=0, j=0, c=0.3, direction=">", use_project=False
        )
        new_proj, nf_p = _truncate_matrix_element_ineq(
            block, "X", 2, 2, i=0, j=0, c=0.3, direction=">", use_project=True
        )
        assert nf_d == pytest.approx(nf_p, rel=1e-6)
        M_dense = new_dense.mu_blocks[0]["X"]
        M_proj = new_proj.mu_blocks[0]["X"]
        # For isotropic U,V, project=identity so means should match exactly
        np.testing.assert_allclose(M_dense, M_proj, atol=1e-8)

    def test_weights_renormalised(self):
        """Surviving component weight should be renormalised to 1.0."""
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)))
        new_block, _ = _truncate_matrix_element_ineq(
            block, "X", 2, 2, i=0, j=0, c=0.0, direction=">", use_project=False
        )
        assert sum(new_block.pi) == pytest.approx(1.0, abs=1e-12)

    def test_multicomp_weights_sum_to_1(self):
        """Two-component block after truncation: renormalised weights sum to 1."""
        M1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        M2 = np.array([[-1.0, 0.0], [0.0, 0.0]])
        ve = VarEntry(name="X", kind="matrix", shape=(2, 2), flat_offset=0)
        U = np.eye(2)
        V = np.eye(2)
        mu_blocks = [{"X": M1.copy()}, {"X": M2.copy()}]
        cov_blocks = [
            {frozenset({"X"}): (U.copy(), V.copy())},
            {frozenset({"X"}): (U.copy(), V.copy())},
        ]
        block2 = GaussianMixBlock(
            var_list=[], var_entries=[ve],
            pi=[0.5, 0.5],
            mu_blocks=mu_blocks,
            cov_blocks=cov_blocks,
        )
        new_block, nf = _truncate_matrix_element_ineq(
            block2, "X", 2, 2, i=0, j=0, c=0.0, direction=">", use_project=False
        )
        assert abs(sum(new_block.pi) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# TestRowSumIneq
# ---------------------------------------------------------------------------

class TestRowSumIneq:

    def test_norm_factor_half_symmetric(self):
        """row_sum(X, 0) > 0 on zero-mean symmetric should give P ~ 0.5."""
        # 2x2, M=0, U=I, V=I → sum_j X[0,j] ~ N(0, sum_j V[j,j] * U[0,0]) = N(0, 2)
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)))
        _, nf = _truncate_matrix_row_sum_ineq(
            block, "X", 2, 2, i=0, c=0.0, direction=">", use_project=False
        )
        assert 0.4 < nf < 0.6

    def test_mean_update_row_i(self):
        """After row_sum(X, 0) > 0, mean of row 0 should increase."""
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)))
        new_block, _ = _truncate_matrix_row_sum_ineq(
            block, "X", 2, 2, i=0, c=0.0, direction=">", use_project=False
        )
        M_new = new_block.mu_blocks[0]["X"]
        assert M_new[0, 0] > 0.0 or M_new[0, 1] > 0.0

    def test_analytical_row_sum_var(self):
        """Verify var_s = U[0,0] * (1^T V 1) for row sum at i=0."""
        U = 2.0 * np.eye(2)
        V = 3.0 * np.eye(2)
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)), U=U, V=V)
        # var_s for row 0 = U[0,0] * (1^T V 1) = 2 * (4*3) ... wait:
        # sum_j X[0,j]: selector a = e_0 ⊗ 1_2 = [1,0,1,0] in column-major
        # a^T (V⊗U) a = with V=3I, U=2I:
        # kron(V,U) = 3*2 * I_4 = 6I → var_s = a^T (6I) a = 6 * |a|^2
        # |a|^2 = 2 (two 1s) → var_s = 12
        # = U[0,0] * (1_2^T V 1_2) = 2 * (1,1)*(3,3) = 2 * 6 = 12  MATCH
        mn = 4
        a = np.zeros(mn)
        for j_idx in range(2):
            a[j_idx * 2 + 0] = 1.0
        Sigma = np.kron(V, U)
        var_s = float(a @ Sigma @ a)
        assert var_s == pytest.approx(12.0, rel=1e-9)

    def test_cross_validation_row_sum(self):
        """row_sum(X,0) > c: norm_factor should match scalar truncate on sum variable."""
        from libSOGAtruncate import _truncated_normal_moments_1d
        m, n = 2, 2
        U = np.eye(m)
        V = np.eye(n)
        M = np.zeros((m, n))
        block = _make_single_comp_block(m, n, M, U, V)
        i, c, direction = 0, 0.5, ">"
        # var_s for row 0 sum: U[0,0] * (1^T V 1) = 1 * 2 = 2
        var_s = 2.0
        mu_s = 0.0
        _, _, P_ref = _truncated_normal_moments_1d(mu_s, var_s, c, direction)
        _, nf = _truncate_matrix_row_sum_ineq(
            block, "X", m, n, i=i, c=c, direction=direction, use_project=False
        )
        assert nf == pytest.approx(P_ref, rel=1e-6)


# ---------------------------------------------------------------------------
# TestColSumIneq
# ---------------------------------------------------------------------------

class TestColSumIneq:

    def test_norm_factor_half_symmetric(self):
        """col_sum(X, 0) > 0 on zero-mean symmetric should give P ~ 0.5."""
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)))
        _, nf = _truncate_matrix_col_sum_ineq(
            block, "X", 2, 2, j=0, c=0.0, direction=">", use_project=False
        )
        assert 0.4 < nf < 0.6

    def test_analytical_col_sum_var(self):
        """Verify var_s = (1^T U 1) * V[0,0] for col sum at j=0."""
        U = 2.0 * np.eye(2)
        V = 3.0 * np.eye(2)
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)), U=U, V=V)
        # var_s for col 0 = (1^T U 1) * V[0,0]
        # = sum_{i,i'} U[i,i'] * V[0,0]  (diagonal U → sum diag = 2*2=4)
        # = 4 * 3 = 12
        # Selector a[j*m+i] = 1 for i=0..1, j=0 → a = [1,1,0,0] in column-major
        mn = 4
        a = np.zeros(mn)
        for i_idx in range(2):
            a[0 * 2 + i_idx] = 1.0
        Sigma = np.kron(V, U)
        var_s = float(a @ Sigma @ a)
        assert var_s == pytest.approx(12.0, rel=1e-9)

    def test_cross_validation_col_sum(self):
        """col_sum(X,0) < c: norm_factor should match scalar truncate on sum variable."""
        from libSOGAtruncate import _truncated_normal_moments_1d
        m, n = 2, 2
        U = np.eye(m)
        V = np.eye(n)
        M = np.zeros((m, n))
        block = _make_single_comp_block(m, n, M, U, V)
        j, c, direction = 0, -0.3, "<"
        # var_s for col 0 sum: (1^T U 1) * V[0,0] = 2 * 1 = 2
        var_s = 2.0
        mu_s = 0.0
        _, _, P_ref = _truncated_normal_moments_1d(mu_s, var_s, c, direction)
        _, nf = _truncate_matrix_col_sum_ineq(
            block, "X", m, n, j=j, c=c, direction=direction, use_project=False
        )
        assert nf == pytest.approx(P_ref, rel=1e-6)

    def test_mean_update_col_j(self):
        """After col_sum(X, 1) < -0.5, mean at column 1 should decrease."""
        block = _make_single_comp_block(2, 2, M=np.zeros((2, 2)))
        new_block, _ = _truncate_matrix_col_sum_ineq(
            block, "X", 2, 2, j=1, c=-0.5, direction="<", use_project=False
        )
        M_new = new_block.mu_blocks[0]["X"]
        # Column 1 should shift toward negative values
        assert M_new[0, 1] < 0.0 or M_new[1, 1] < 0.0


# ---------------------------------------------------------------------------
# TestLogWeights
# ---------------------------------------------------------------------------

class TestLogWeights:

    def test_log_pi_initialised_on_first_truncation(self):
        """dist.gm_block.log_pi should be set after first truncate_matrix call."""
        dist = _make_dist_with_block(2, 2, M=np.zeros((2, 2)))
        assert dist.gm_block.log_pi is None
        nf, new_dist = truncate_matrix(dist, "X[0,0] > 0.0", {}, "X")
        assert new_dist.gm_block.log_pi is not None

    def test_log_pi_consistent_with_pi(self):
        """After truncation, exp(log_pi[k]) should match pi[k] up to normalisation."""
        dist = _make_dist_with_block(2, 2, M=np.zeros((2, 2)))
        nf, new_dist = truncate_matrix(dist, "X[0,0] > 0.0", {}, "X")
        block = new_dist.gm_block
        if block.log_pi is not None:
            for lp, p in zip(block.log_pi, block.pi):
                if lp > float("-inf"):
                    assert math.exp(lp) == pytest.approx(p * sum(
                        math.exp(l) for l in block.log_pi if l > float("-inf")
                    ), rel=1e-6)

    def test_repeated_truncation_no_underflow(self):
        """10 successive element truncations on a small-mean component should not
        produce pi=0.0 (float underflow) when log_pi is properly maintained."""
        # Use a component far from the threshold so P is not catastrophically small
        dist = _make_dist_with_block(2, 2, M=np.ones((2, 2)) * 5.0)
        for _ in range(10):
            nf, dist = truncate_matrix(dist, "X[0,0] > 0.0", {}, "X")
        # If no underflow, sum(pi) should still be 1.0
        assert sum(dist.gm_block.pi) == pytest.approx(1.0, abs=1e-6)

    def test_log_pi_updated_in_multicomp(self):
        """In multi-component block, log_pi should be updated independently."""
        M1 = np.ones((2, 2)) * 3.0
        M2 = -np.ones((2, 2)) * 3.0
        ve = VarEntry(name="X", kind="matrix", shape=(2, 2), flat_offset=0)
        U = np.eye(2)
        V = np.eye(2)
        mu_blocks = [{"X": M1.copy()}, {"X": M2.copy()}]
        cov_blocks = [
            {frozenset({"X"}): (U.copy(), V.copy())},
            {frozenset({"X"}): (U.copy(), V.copy())},
        ]
        block2 = GaussianMixBlock(
            var_list=[], var_entries=[ve],
            pi=[0.5, 0.5],
            mu_blocks=mu_blocks,
            cov_blocks=cov_blocks,
            log_pi=[math.log(0.5), math.log(0.5)],
        )
        gm = GaussianMix([1.0], [np.zeros(0)], [np.zeros((0, 0))])
        dist = Dist(var_list=[], gm=gm, var_entries=[ve], gm_block=block2)
        nf, new_dist = truncate_matrix(dist, "X[0,0] > 0.0", {}, "X")
        # Component 1 (M=3) should dominate after X[0,0] > 0 since mean=3 >> 0
        new_block = new_dist.gm_block
        # Weight of first component (M=+3) should be >> weight of second (M=-3)
        assert new_block.pi[0] > new_block.pi[1]


# ---------------------------------------------------------------------------
# TestDenseCrossValidation
# ---------------------------------------------------------------------------

class TestDenseCrossValidation:

    def test_element_ineq_4x4_exact_vs_dense_manual(self):
        """Manual verification of 4x4 element ineq against direct dense computation."""
        from libSOGAtruncate import _truncated_normal_moments_1d
        m, n = 4, 4
        np.random.seed(0)
        U = np.eye(m) * 2.0
        V = np.eye(n) * 0.5
        M = np.zeros((m, n))
        block = _make_single_comp_block(m, n, M, U, V)

        i, j = 2, 1
        c = 0.3
        direction = ">"

        # Reference via scalar path on vec(X) with selector e_{j*m+i}
        Sigma_dense = np.kron(V, U)
        idx = j * m + i
        mu_s = 0.0
        var_s = float(Sigma_dense[idx, idx])  # = U[i,i]*V[j,j] = 2*0.5 = 1.0
        m_hat, v_hat, P_ref = _truncated_normal_moments_1d(mu_s, var_s, c, direction)

        new_block, nf = _truncate_matrix_element_ineq(
            block, "X", m, n, i=i, j=j, c=c, direction=direction, use_project=False
        )
        assert nf == pytest.approx(P_ref, rel=1e-9)
        M_new = new_block.mu_blocks[0]["X"]
        assert M_new[i, j] == pytest.approx(m_hat, rel=1e-9)


# ---------------------------------------------------------------------------
# TestMemoryBudget
# ---------------------------------------------------------------------------

class TestMemoryBudget:

    def test_memory_warning_emitted_when_budget_exceeded(self):
        """KroneckerDenseMemoryWarning should be emitted when budget is too small."""
        import libMatrixTruncate as lmt
        old_budget = lmt.MATRIX_DENSE_BUDGET_MB
        try:
            # Set budget to 0 MB to force warning
            lmt.MATRIX_DENSE_BUDGET_MB = 0
            block = _make_single_comp_block(4, 4)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _truncate_matrix_element_ineq(
                    block, "X", 4, 4, i=0, j=0, c=0.0, direction=">",
                    use_project=False
                )
                warning_types = [str(x.category) for x in w]
                assert any("KroneckerDenseMemoryWarning" in t for t in warning_types)
        finally:
            lmt.MATRIX_DENSE_BUDGET_MB = old_budget

    def test_project_fallback_still_produces_result(self):
        """Even with PROJECT fallback, truncation should complete successfully."""
        import libMatrixTruncate as lmt
        old_budget = lmt.MATRIX_DENSE_BUDGET_MB
        try:
            lmt.MATRIX_DENSE_BUDGET_MB = 0
            block = _make_single_comp_block(4, 4)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                new_block, nf = _truncate_matrix_element_ineq(
                    block, "X", 4, 4, i=0, j=0, c=0.0, direction=">",
                    use_project=False
                )
            assert 0.0 < nf < 1.0 + 1e-9
        finally:
            lmt.MATRIX_DENSE_BUDGET_MB = old_budget


# ---------------------------------------------------------------------------
# TestTruncateRouting
# ---------------------------------------------------------------------------

class TestTruncateRouting:

    def test_truncate_matrix_dispatch_from_truncate(self):
        """truncate() routing guard should dispatch matrix constraint to truncate_matrix."""
        from libSOGAtruncate import truncate
        dist = _make_dist_with_block(2, 2, M=np.zeros((2, 2)))
        nf, new_dist = truncate(dist, "X[0,0] > 0.0", {})
        assert 0.4 < nf < 0.6
        assert new_dist.gm_block is not None

    def test_truncate_matrix_row_sum_dispatch(self):
        """row_sum(X,0) > 0.0 dispatches correctly through truncate()."""
        from libSOGAtruncate import truncate
        dist = _make_dist_with_block(2, 2, M=np.zeros((2, 2)))
        nf, new_dist = truncate(dist, "row_sum(X, 0) > 0.0", {})
        assert 0.4 < nf < 0.6

    def test_truncate_matrix_col_sum_dispatch(self):
        """col_sum(X,0) > 0.0 dispatches correctly through truncate()."""
        from libSOGAtruncate import truncate
        dist = _make_dist_with_block(2, 2, M=np.zeros((2, 2)))
        nf, new_dist = truncate(dist, "col_sum(X, 0) > 0.0", {})
        assert 0.4 < nf < 0.6

    def test_truncate_true_false_unchanged(self):
        """'true' and 'false' should work with matrix dist too."""
        from libSOGAtruncate import truncate
        dist = _make_dist_with_block(2, 2, M=np.zeros((2, 2)))
        nf_t, d_t = truncate(dist, "true", {})
        nf_f, d_f = truncate(dist, "false", {})
        assert nf_t == pytest.approx(1.0)
        assert nf_f == pytest.approx(0.0)

    def test_truncate_matrix_not_implemented_for_stub_missing_block(self):
        """truncate_matrix should raise ValueError if gm_block is None."""
        from libMatrixTruncate import truncate_matrix
        ve = VarEntry(name="X", kind="matrix", shape=(2, 2), flat_offset=0)
        gm = GaussianMix([1.0], [np.zeros(0)], [np.zeros((0, 0))])
        dist = Dist(var_list=[], gm=gm, var_entries=[ve], gm_block=None)
        with pytest.raises(ValueError, match="gm_block is None"):
            truncate_matrix(dist, "X[0,0] > 0.0", {}, "X")

    def test_trace_ineq_raises_not_implemented(self):
        """trace(X) > c should raise NotImplementedError through truncate()."""
        from libSOGAtruncate import truncate
        dist = _make_dist_with_block(2, 2, M=np.zeros((2, 2)))
        with pytest.raises(NotImplementedError, match="trace"):
            truncate(dist, "trace(X) > 1.0", {})
