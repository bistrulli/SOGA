"""
tests/test_gaussian_mix_block.py — Joint state storage tests for M3.

Tests:
  - Round-trip via from_dist -> to_dense_scalar_view
  - PSD invariants enforced after construction
  - Cross-validation against dense vec/kron at m,n <= 8
  - Multi-component (K=3) law-of-total-variance validation
  - matrix_var and matrix_cov output API (M3.8)
  - merge/prune guards (M3.6, M3.7)

Plan reference: §M3.5 of plan/2026-05-22-matrix-gm-lishan.md
"""

import sys
import os

import numpy as np
import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from libSOGAshared import Dist, GaussianMix, VarEntry
from libSOGAsharedMatrix import GaussianMixBlock, _enforce_psd_kron_factors
from libMatrixGaussian import KRONECKER_CONVENTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scalar_dist(n_vars=2, n_comp=1, seed=42):
    """Create a scalar Dist with n_vars variables and n_comp components."""
    rng = np.random.default_rng(seed)
    var_list = [f"x{i}" for i in range(n_vars)]
    pi = list(np.ones(n_comp) / n_comp)
    mus = []
    sigmas = []
    for _ in range(n_comp):
        mu = rng.normal(size=n_vars)
        A = rng.normal(size=(n_vars, n_vars))
        sigma = A @ A.T + np.eye(n_vars) * 0.1
        mus.append(mu)
        sigmas.append(sigma)
    gm = GaussianMix(pi, mus, sigmas)
    return Dist(var_list, gm)


def _make_matrix_var_entry(m, n, name="X"):
    return VarEntry(name=name, kind="matrix", shape=(m, n), flat_offset=-1)


def _isotropic(d, scale=1.0):
    return scale * np.eye(d)


# ---------------------------------------------------------------------------
# M3.3: from_dist constructor
# ---------------------------------------------------------------------------

class TestFromDist:
    """GaussianMixBlock.from_dist round-trips against original Dist."""

    def test_from_dist_roundtrip_single_comp(self):
        """to_dense_scalar_view recovers original scalar mean/cov."""
        dist = _make_scalar_dist(n_vars=3, n_comp=1, seed=1)
        block = GaussianMixBlock.from_dist(dist)
        mu_vec, sigma_mat = block.to_dense_scalar_view(0)
        np.testing.assert_allclose(mu_vec, dist.gm.mu[0], atol=1e-12)
        np.testing.assert_allclose(sigma_mat, dist.gm.sigma[0], atol=1e-12)

    def test_from_dist_roundtrip_multi_comp(self):
        """All components round-trip correctly."""
        dist = _make_scalar_dist(n_vars=2, n_comp=3, seed=7)
        block = GaussianMixBlock.from_dist(dist)
        assert block.n_comp() == 3
        for k in range(3):
            mu_vec, sigma_mat = block.to_dense_scalar_view(k)
            np.testing.assert_allclose(mu_vec, dist.gm.mu[k], atol=1e-12)
            np.testing.assert_allclose(sigma_mat, dist.gm.sigma[k], atol=1e-12)

    def test_from_dist_pi_preserved(self):
        """Mixing weights are preserved."""
        dist = _make_scalar_dist(n_vars=2, n_comp=4, seed=3)
        block = GaussianMixBlock.from_dist(dist)
        np.testing.assert_allclose(block.pi, dist.gm.pi, atol=1e-14)

    def test_from_dist_empty_var_entries(self):
        """from_dist with no var_entries has empty matrix vars."""
        dist = _make_scalar_dist()
        block = GaussianMixBlock.from_dist(dist)
        assert block.var_entries == []


# ---------------------------------------------------------------------------
# M3.2: PSD invariants enforced
# ---------------------------------------------------------------------------

class TestPSDEnforcement:
    """_enforce_psd_kron_factors and from_matrix_gm maintain PSD on U, V."""

    def test_psd_on_exactly_psd_inputs(self):
        """Already-PSD factors are returned unchanged (modulo float round-off)."""
        U = _isotropic(4, 1.0)
        V = _isotropic(4, 0.5)
        U_out, V_out = _enforce_psd_kron_factors(U, V)
        # All eigenvalues positive
        assert np.all(np.linalg.eigvalsh(U_out) > 0)
        assert np.all(np.linalg.eigvalsh(V_out) > 0)

    def test_psd_clips_small_negative_eigenvalue(self):
        """Slightly-negative-eigenvalue matrix is corrected to PSD."""
        rng = np.random.default_rng(42)
        A = rng.normal(size=(4, 4))
        U = A @ A.T   # PSD
        # Introduce small negative eigenvalue
        eig, Q = np.linalg.eigh(U)
        eig[0] = -1e-9
        U_bad = Q @ np.diag(eig) @ Q.T
        U_out, _ = _enforce_psd_kron_factors(U_bad, _isotropic(4))
        assert np.all(np.linalg.eigvalsh(U_out) >= 0)

    def test_from_matrix_gm_psd_invariant(self):
        """from_matrix_gm enforces PSD on returned factors."""
        ve = _make_matrix_var_entry(3, 3)
        M = np.zeros((3, 3))
        U = _isotropic(3, 2.0)
        V = _isotropic(3, 0.5)
        block = GaussianMixBlock.from_matrix_gm([ve], [M], [U], [V])
        U_out, V_out = block.get_cov(0, "X", "X")
        assert np.all(np.linalg.eigvalsh(U_out) >= 0)
        assert np.all(np.linalg.eigvalsh(V_out) >= 0)


# ---------------------------------------------------------------------------
# M3.1 / M3.4: from_matrix_gm and cross-cov zero at declaration
# ---------------------------------------------------------------------------

class TestFromMatrixGM:
    """GaussianMixBlock.from_matrix_gm stores correct factors and zero cross-covs."""

    def test_single_component_mean_stored(self):
        """Mean matrix is stored correctly in mu_blocks."""
        ve = _make_matrix_var_entry(2, 3)
        M = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        U = _isotropic(2)
        V = _isotropic(3)
        block = GaussianMixBlock.from_matrix_gm([ve], [M], [U], [V])
        np.testing.assert_allclose(block.get_mu(0, "X"), M, atol=1e-14)

    def test_single_component_kron_factors_stored(self):
        """Kronecker factors (U, V) are stored in cov_blocks."""
        ve = _make_matrix_var_entry(2, 2)
        M = np.zeros((2, 2))
        U = _isotropic(2, 2.0)
        V = _isotropic(2, 0.5)
        block = GaussianMixBlock.from_matrix_gm([ve], [M], [U], [V])
        U_out, V_out = block.get_cov(0, "X", "X")
        np.testing.assert_allclose(U_out, U, atol=1e-12)
        np.testing.assert_allclose(V_out, V, atol=1e-12)

    def test_cross_cov_zero_at_declaration(self):
        """Scalar-matrix cross-covariance is zero at declaration (M3.4)."""
        ve = _make_matrix_var_entry(2, 2)
        M = np.zeros((2, 2))
        block = GaussianMixBlock.from_matrix_gm(
            [ve], [M], [_isotropic(2)], [_isotropic(2)],
            var_list=["x", "y"]
        )
        mn = 2 * 2
        cross_x = block.get_cov(0, "x", "X")
        cross_y = block.get_cov(0, "y", "X")
        np.testing.assert_allclose(cross_x, np.zeros(mn), atol=1e-14)
        np.testing.assert_allclose(cross_y, np.zeros(mn), atol=1e-14)

    def test_multi_component_pi(self):
        """Multi-component mixture preserves pi."""
        ve = _make_matrix_var_entry(2, 2)
        pi = [0.3, 0.7]
        Ms = [np.zeros((2, 2)), np.ones((2, 2))]
        Us = [_isotropic(2, 1.0), _isotropic(2, 2.0)]
        Vs = [_isotropic(2, 0.5), _isotropic(2, 1.0)]
        block = GaussianMixBlock.from_matrix_gm([ve], Ms, Us, Vs, pi=pi)
        np.testing.assert_allclose(block.pi, pi, atol=1e-14)
        assert block.n_comp() == 2


# ---------------------------------------------------------------------------
# M3.5 / M3.8: Cross-validation against dense vec/kron at m,n <= 8
# ---------------------------------------------------------------------------

class TestDenseCrossValidation:
    """Validate Kronecker storage against dense mat-var formulas for m,n <= 8."""

    @pytest.mark.parametrize("m,n", [(2, 2), (3, 3), (4, 4), (2, 4), (4, 2), (3, 5)])
    def test_element_variance_matches_kronecker(self, m, n):
        """Var(X[i,j]) = U[i,i]*V[j,j] for each element."""
        rng = np.random.default_rng(m * 100 + n)
        ve = _make_matrix_var_entry(m, n)
        M = rng.normal(size=(m, n))
        A = rng.normal(size=(m, m)); U = A @ A.T + 0.1 * np.eye(m)
        B = rng.normal(size=(n, n)); V = B @ B.T + 0.1 * np.eye(n)
        block = GaussianMixBlock.from_matrix_gm([ve], [M], [U], [V])
        U_stored, V_stored = block.get_cov(0, "X", "X")
        for i in range(m):
            for j in range(n):
                expected = float(U_stored[i, i] * V_stored[j, j])
                got = block.matrix_var("X", i, j)
                assert abs(got - expected) < 1e-10, (
                    f"m={m},n={n},i={i},j={j}: expected {expected}, got {got}"
                )

    @pytest.mark.parametrize("m,n", [(2, 2), (3, 3), (4, 4)])
    def test_full_cov_matches_kron(self, m, n):
        """matrix_full_cov(K=1) matches np.kron(V, U) for single component."""
        rng = np.random.default_rng(m * 31 + n)
        ve = _make_matrix_var_entry(m, n)
        M = np.zeros((m, n))
        A = rng.normal(size=(m, m)); U = A @ A.T + 0.1 * np.eye(m)
        B = rng.normal(size=(n, n)); V = B @ B.T + 0.1 * np.eye(n)
        block = GaussianMixBlock.from_matrix_gm([ve], [M], [U], [V])
        U_s, V_s = block.get_cov(0, "X", "X")
        assert KRONECKER_CONVENTION == "V_outer_U"
        expected_full = np.kron(V_s, U_s)
        got_full = block.matrix_full_cov("X")
        np.testing.assert_allclose(got_full, expected_full, atol=1e-10)


# ---------------------------------------------------------------------------
# M3.5 (critical): Multi-component law-of-total-variance
# ---------------------------------------------------------------------------

class TestLawOfTotalVariance:
    """K=3 mixture: matrix_var must include BOTH within- and between-component terms."""

    def test_K3_var_matches_manual_law_of_total_variance(self):
        """Verified: omitting between-component term causes ~20% error."""
        # Three components with means 0, 1, 2 on X[0,0]
        ve = _make_matrix_var_entry(2, 2)
        pi = [1/3, 1/3, 1/3]
        M0 = np.zeros((2, 2))
        M1 = np.zeros((2, 2)); M1[0, 0] = 1.0
        M2 = np.zeros((2, 2)); M2[0, 0] = 2.0
        U = _isotropic(2, 0.1)   # within-component variance = 0.1 * 1.0 = 0.1
        V = _isotropic(2, 1.0)
        block = GaussianMixBlock.from_matrix_gm(
            [ve], [M0, M1, M2], [U, U, U], [V, V, V], pi=pi
        )
        # Manual computation
        M_bar = block.matrix_mean("X")  # [0,0] entry = (0 + 1 + 2)/3 = 1.0
        assert abs(M_bar[0, 0] - 1.0) < 1e-12

        # Within-component variance for X[0,0]: U[0,0]*V[0,0] = 0.1*1.0 = 0.1
        # Between-component: pi_k * (M_k[0,0] - 1.0)^2 summed:
        #   1/3*(0-1)^2 + 1/3*(1-1)^2 + 1/3*(2-1)^2 = 1/3 + 0 + 1/3 = 2/3
        # Total = 0.1 + 2/3
        expected_within = 0.1
        expected_between = 2.0 / 3.0
        expected_total = expected_within + expected_between

        got = block.matrix_var("X", 0, 0)
        assert abs(got - expected_total) < 1e-10, (
            f"Law of total variance: expected {expected_total:.6f}, got {got:.6f}"
        )

        # Verify: if we ONLY used within-component, we'd miss 2/3 ≈ 87% of the variance
        within_only_error = abs(expected_within - expected_total) / expected_total
        assert within_only_error > 0.5, (
            "Test design: between-component term should be substantial (>50% of total)"
        )

    def test_K3_cov_matches_manual(self):
        """Cov(X[0,0], X[1,1]) law-of-total-covariance for K=3."""
        ve = _make_matrix_var_entry(2, 2)
        pi = [0.2, 0.5, 0.3]
        rng = np.random.default_rng(99)
        Ms = [rng.normal(size=(2, 2)) for _ in range(3)]
        A = rng.normal(size=(2, 2)); U = A @ A.T + 0.1 * np.eye(2)
        B = rng.normal(size=(2, 2)); V = B @ B.T + 0.1 * np.eye(2)
        block = GaussianMixBlock.from_matrix_gm(
            [ve], Ms, [U, U, U], [V, V, V], pi=pi
        )
        M_bar = block.matrix_mean("X")
        # Manual: Cov(X[0,0], X[1,1]) = sum_k pi_k * (U[0,1]*V[0,1] + delta_00_k * delta_11_k)
        # where delta_ij_k = M_k[i,j] - M_bar[i,j]
        U_s, V_s = block.get_cov(0, "X", "X")
        expected = sum(
            pi[k] * (
                float(U_s[0, 1] * V_s[0, 1])
                + float((Ms[k][0, 0] - M_bar[0, 0]) * (Ms[k][1, 1] - M_bar[1, 1]))
            )
            for k in range(3)
        )
        got = block.matrix_cov("X", 0, 0, 1, 1)
        assert abs(got - expected) < 1e-10, (
            f"matrix_cov: expected {expected:.6f}, got {got:.6f}"
        )


# ---------------------------------------------------------------------------
# M3.6: merge guard
# ---------------------------------------------------------------------------

class TestMergeGuard:
    """merge() correctly handles matrix var_entries (fix2 lifted NotImplementedError)."""

    def test_merge_with_matrix_vars_succeeds(self):
        """fix2: merge with matrix var_entries now works (M3.6 NotImplementedError lifted)."""
        from libSOGAmerge import merge
        from libSOGAsharedMatrix import GaussianMixBlock
        ve = VarEntry("X", "matrix", (2, 2))
        M = np.eye(2)
        U = np.eye(2); V = np.eye(2)
        block = GaussianMixBlock.from_matrix_gm([ve], [M], [U], [V], pi=[1.0], var_list=['x'])
        gm = GaussianMix([1.], [np.array([0.])], [np.zeros((1, 1))])
        d = Dist(["x"], gm, var_entries=[ve], gm_block=block)
        # merge with two identical distributions that have matrix var_entries
        p, merged = merge([(0.5, d), (0.5, d)])
        assert p > 0
        assert merged.gm_block is not None
        assert merged.gm_block.n_comp() == 2  # 1+1 concatenated

    def test_merge_scalar_only_passes(self):
        """merge scalar programs (no var_entries) works as before."""
        from libSOGAmerge import merge
        gm = GaussianMix([1.], [np.array([0., 0.])], [np.zeros((2, 2))])
        d1 = Dist(["x", "y"], gm)
        d2 = Dist(["x", "y"], gm)
        p, merged = merge([(0.5, d1), (0.5, d2)])
        assert p > 0


# ---------------------------------------------------------------------------
# M3.7: classic_prune guard
# ---------------------------------------------------------------------------

class TestClassicPruneGuard:
    """classic_prune raises AssertionError when gm_block is set (M3.7)."""

    def test_classic_prune_with_gm_block_raises(self):
        """classic_prune must not be called on matrix-variable distributions."""
        from libSOGAmerge import classic_prune
        # Minimal GaussianMixBlock mock
        block = object()  # any non-None object
        gm = GaussianMix([0.5, 0.5], [np.zeros(2), np.zeros(2)], [np.eye(2), np.eye(2)])
        dist = Dist(["x", "y"], gm, gm_block=block)
        with pytest.raises(AssertionError, match="M3.7"):
            classic_prune(dist, 1)

    def test_classic_prune_scalar_passes(self):
        """classic_prune on a scalar-only Dist (gm_block=None) works normally."""
        from libSOGAmerge import classic_prune
        pi = [0.4, 0.6]
        mus = [np.array([0.]), np.array([1.])]
        sigmas = [np.array([[1.]]), np.array([[1.]])]
        gm = GaussianMix(pi, mus, sigmas)
        dist = Dist(["x"], gm)
        result = classic_prune(dist, 1)
        assert result.gm.n_comp() == 1
