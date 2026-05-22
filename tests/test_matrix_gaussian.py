"""
Tests for src/libMatrixGaussian.py — M0.7

Covers:
- Closed-form operations: scale, transpose, add_constant, affine_left,
  affine_right, affine (all exact, validated against dense cross-check)
- Identity / no-op tests
- Isotropic-noise fast paths A and B in MatrixGaussian.add()
- Sign disambiguation in _nearest_kronecker (indefinite-U case)
- PSD invariants on U and V after every operation
- Sampling via Cholesky vs eigh square-root (distribution-level)
- _is_isotropic helper
- KRONECKER_CONVENTION constant
- matmul_independent (nearest-Kronecker approximation, MC-validated)
- add() general path (nearest-Kronecker projection with warning)
- from_scalar_dense / to_scalar_dense round-trip
"""

from __future__ import annotations

import sys
import os
import warnings

import numpy as np
import pytest

# Allow running from repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.libMatrixGaussian import (
    MatrixGaussian,
    KRONECKER_CONVENTION,
    _nearest_kronecker,
    _is_isotropic,
    to_scalar_dense,
    from_scalar_dense,
    IsotropyNearThresholdWarning,
    KroneckerApproxWarning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_psd_matrix(rng: np.random.Generator, d: int, offset: float = 0.1) -> np.ndarray:
    """Return a random (d×d) PSD matrix."""
    A = rng.standard_normal((d, d))
    return A @ A.T + offset * np.eye(d)


def _sample_matgaussian(mg: MatrixGaussian, n_samples: int, seed: int) -> np.ndarray:
    """Draw n_samples from MN(M, U, V) via Cholesky of V ⊗ U."""
    rng = np.random.default_rng(seed)
    m, n = mg.shape
    Cov = np.kron(mg.V, mg.U) + 1e-10 * np.eye(m * n)
    L = np.linalg.cholesky(Cov)
    samples = []
    for _ in range(n_samples):
        z = rng.standard_normal(m * n)
        vec_x = mg.M.flatten("F") + L @ z
        samples.append(vec_x.reshape((m, n), order="F"))
    return np.array(samples)


def _is_psd(M: np.ndarray, tol: float = -1e-9) -> bool:
    """Return True if M is symmetric and all eigenvalues >= tol."""
    if not np.allclose(M, M.T, atol=1e-10):
        return False
    eigs = np.linalg.eigvalsh(M)
    return bool(np.all(eigs >= tol))


# ---------------------------------------------------------------------------
# 1. Module-level assertions
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_convention_string(self):
        assert KRONECKER_CONVENTION == "V_outer_U"

    def test_convention_type(self):
        assert isinstance(KRONECKER_CONVENTION, str)


# ---------------------------------------------------------------------------
# 2. _is_isotropic helper
# ---------------------------------------------------------------------------

class TestIsIsotropic:
    def test_identity_is_isotropic(self):
        assert _is_isotropic(np.eye(4))

    def test_scaled_identity_is_isotropic(self):
        assert _is_isotropic(3.7 * np.eye(5))

    def test_diagonal_not_isotropic(self):
        D = np.diag([1.0, 2.0, 3.0])
        assert not _is_isotropic(D)

    def test_dense_psd_not_isotropic(self):
        rng = np.random.default_rng(42)
        M = _make_psd_matrix(rng, 4)
        assert not _is_isotropic(M)

    def test_zero_matrix_is_isotropic(self):
        # 0 * I is isotropic
        assert _is_isotropic(np.zeros((3, 3)))

    def test_near_identity_within_tolerance(self):
        M = np.eye(3) + 5e-11 * np.ones((3, 3))
        # small perturbation: should still pass atol=1e-10 test
        assert _is_isotropic(M)

    def test_non_square_returns_false(self):
        assert not _is_isotropic(np.eye(3)[:2, :])

    def test_1x1_always_isotropic(self):
        assert _is_isotropic(np.array([[7.0]]))

    def test_1d_array_returns_false(self):
        assert not _is_isotropic(np.ones(4))


# ---------------------------------------------------------------------------
# 3. MatrixGaussian construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic_construction(self):
        M = np.eye(3)
        U = np.eye(3)
        V = 2.0 * np.eye(3)
        mg = MatrixGaussian(M, U, V)
        assert mg.shape == (3, 3)
        assert mg.m == 3
        assert mg.n == 3

    def test_rectangular_construction(self):
        M = np.zeros((2, 5))
        U = np.eye(2)
        V = np.eye(5)
        mg = MatrixGaussian(M, U, V)
        assert mg.shape == (2, 5)
        assert mg.m == 2
        assert mg.n == 5

    def test_wrong_U_shape_raises(self):
        with pytest.raises(AssertionError):
            MatrixGaussian(np.eye(3), np.eye(2), np.eye(3))

    def test_wrong_V_shape_raises(self):
        with pytest.raises(AssertionError):
            MatrixGaussian(np.eye(3), np.eye(3), np.eye(2))

    def test_mean_returns_copy(self):
        mg = MatrixGaussian(np.eye(3), np.eye(3), np.eye(3))
        m = mg.mean()
        m[0, 0] = 999.0
        assert mg.M[0, 0] != 999.0


# ---------------------------------------------------------------------------
# 4. variance_element and variance_matrix
# ---------------------------------------------------------------------------

class TestVariance:
    def test_variance_element_analytical(self):
        rng = np.random.default_rng(0)
        U = _make_psd_matrix(rng, 3)
        V = _make_psd_matrix(rng, 4)
        mg = MatrixGaussian(np.zeros((3, 4)), U, V)
        for i in range(3):
            for j in range(4):
                expected = U[i, i] * V[j, j]
                assert abs(mg.variance_element(i, j) - expected) < 1e-14

    def test_variance_matrix_shape(self):
        mg = MatrixGaussian(np.zeros((3, 4)), np.eye(3), 2.0 * np.eye(4))
        assert mg.variance_matrix().shape == (3, 4)

    def test_variance_matrix_isotropic(self):
        mg = MatrixGaussian(np.zeros((3, 3)), 2.0 * np.eye(3), 3.0 * np.eye(3))
        # Var(X[i,j]) = 2 * 3 = 6 for all i,j
        assert np.allclose(mg.variance_matrix(), 6.0 * np.ones((3, 3)))


# ---------------------------------------------------------------------------
# 5. covariance_vec: dense validation
# ---------------------------------------------------------------------------

class TestCovarianceVec:
    def test_kron_convention(self):
        rng = np.random.default_rng(1)
        U = _make_psd_matrix(rng, 2)
        V = _make_psd_matrix(rng, 3)
        mg = MatrixGaussian(np.zeros((2, 3)), U, V)
        dense = mg.covariance_vec()
        expected = np.kron(V, U)
        assert np.allclose(dense, expected, atol=1e-14)

    def test_cov_symmetric(self):
        rng = np.random.default_rng(2)
        U = _make_psd_matrix(rng, 3)
        V = _make_psd_matrix(rng, 3)
        mg = MatrixGaussian(np.zeros((3, 3)), U, V)
        C = mg.covariance_vec()
        assert np.allclose(C, C.T, atol=1e-14)

    def test_cov_psd(self):
        rng = np.random.default_rng(3)
        U = _make_psd_matrix(rng, 3)
        V = _make_psd_matrix(rng, 3)
        mg = MatrixGaussian(np.zeros((3, 3)), U, V)
        assert _is_psd(mg.covariance_vec())


# ---------------------------------------------------------------------------
# 6. scale
# ---------------------------------------------------------------------------

class TestScale:
    def test_scale_mean(self):
        rng = np.random.default_rng(4)
        M = rng.standard_normal((3, 4))
        mg = MatrixGaussian(M, np.eye(3), np.eye(4))
        scaled = mg.scale(2.5)
        assert np.allclose(scaled.M, 2.5 * M)

    def test_scale_covariance(self):
        rng = np.random.default_rng(5)
        U = _make_psd_matrix(rng, 3)
        V = _make_psd_matrix(rng, 3)
        mg = MatrixGaussian(np.zeros((3, 3)), U, V)
        scaled = mg.scale(3.0)
        # U_new = c² * U, V_new = V
        assert np.allclose(scaled.U, 9.0 * U, atol=1e-14)
        assert np.allclose(scaled.V, V, atol=1e-14)

    def test_scale_one_is_noop(self):
        rng = np.random.default_rng(6)
        M = rng.standard_normal((3, 3))
        U = _make_psd_matrix(rng, 3)
        V = _make_psd_matrix(rng, 3)
        mg = MatrixGaussian(M, U, V)
        scaled = mg.scale(1.0)
        assert np.allclose(scaled.M, M, atol=1e-14)
        assert np.allclose(scaled.U, U, atol=1e-14)

    def test_scale_psd_preserved(self):
        rng = np.random.default_rng(7)
        mg = MatrixGaussian(np.zeros((3, 3)), _make_psd_matrix(rng, 3), _make_psd_matrix(rng, 3))
        assert _is_psd(mg.scale(2.0).U)
        assert _is_psd(mg.scale(2.0).V)


# ---------------------------------------------------------------------------
# 7. transpose
# ---------------------------------------------------------------------------

class TestTranspose:
    def test_transpose_mean(self):
        M = np.arange(6, dtype=float).reshape(2, 3)
        mg = MatrixGaussian(M, np.eye(2), np.eye(3))
        assert np.allclose(mg.transpose().M, M.T)

    def test_transpose_swaps_UV(self):
        rng = np.random.default_rng(8)
        U = _make_psd_matrix(rng, 2)
        V = _make_psd_matrix(rng, 3)
        mg = MatrixGaussian(np.zeros((2, 3)), U, V)
        t = mg.transpose()
        assert np.allclose(t.U, V, atol=1e-14)
        assert np.allclose(t.V, U, atol=1e-14)

    def test_transpose_involution(self):
        """transp(transp(X)) == X (no-op)."""
        rng = np.random.default_rng(9)
        M = rng.standard_normal((3, 4))
        U = _make_psd_matrix(rng, 3)
        V = _make_psd_matrix(rng, 4)
        mg = MatrixGaussian(M, U, V)
        tt = mg.transpose().transpose()
        assert np.allclose(tt.M, M, atol=1e-14)
        assert np.allclose(tt.U, U, atol=1e-14)
        assert np.allclose(tt.V, V, atol=1e-14)


# ---------------------------------------------------------------------------
# 8. add_constant
# ---------------------------------------------------------------------------

class TestAddConstant:
    def test_add_constant_shifts_mean(self):
        rng = np.random.default_rng(10)
        M = rng.standard_normal((3, 3))
        B = rng.standard_normal((3, 3))
        mg = MatrixGaussian(M, np.eye(3), np.eye(3))
        result = mg.add_constant(B)
        assert np.allclose(result.M, M + B, atol=1e-14)

    def test_add_constant_preserves_cov(self):
        rng = np.random.default_rng(11)
        U = _make_psd_matrix(rng, 3)
        V = _make_psd_matrix(rng, 3)
        mg = MatrixGaussian(np.zeros((3, 3)), U, V)
        B = rng.standard_normal((3, 3))
        result = mg.add_constant(B)
        assert np.allclose(result.U, U, atol=1e-14)
        assert np.allclose(result.V, V, atol=1e-14)

    def test_add_zero_is_noop(self):
        rng = np.random.default_rng(12)
        M = rng.standard_normal((3, 3))
        U = _make_psd_matrix(rng, 3)
        V = _make_psd_matrix(rng, 3)
        mg = MatrixGaussian(M, U, V)
        result = mg.add_constant(np.zeros((3, 3)))
        assert np.allclose(result.M, M, atol=1e-14)
        assert np.allclose(result.U, U, atol=1e-14)


# ---------------------------------------------------------------------------
# 9. affine_left and affine_right
# ---------------------------------------------------------------------------

class TestAffine:
    def test_affine_left_mean(self):
        rng = np.random.default_rng(13)
        A = rng.standard_normal((3, 3))
        M = rng.standard_normal((3, 4))
        mg = MatrixGaussian(M, np.eye(3), np.eye(4))
        result = mg.affine_left(A)
        assert np.allclose(result.M, A @ M, atol=1e-14)

    def test_affine_left_U_transform(self):
        rng = np.random.default_rng(14)
        A = rng.standard_normal((3, 3))
        U = _make_psd_matrix(rng, 3)
        V = _make_psd_matrix(rng, 4)
        mg = MatrixGaussian(np.zeros((3, 4)), U, V)
        result = mg.affine_left(A)
        expected_U = A @ U @ A.T
        assert np.allclose(result.U, expected_U, atol=1e-12)
        assert np.allclose(result.V, V, atol=1e-14)

    def test_affine_right_V_transform(self):
        rng = np.random.default_rng(15)
        B = rng.standard_normal((4, 3))
        U = _make_psd_matrix(rng, 2)
        V = _make_psd_matrix(rng, 4)
        mg = MatrixGaussian(np.zeros((2, 4)), U, V)
        result = mg.affine_right(B)
        expected_V = B.T @ V @ B
        assert np.allclose(result.V, expected_V, atol=1e-12)
        assert np.allclose(result.U, U, atol=1e-14)

    def test_affine_left_identity_is_noop(self):
        """affine_left(I_m) is a no-op."""
        rng = np.random.default_rng(16)
        m, n = 3, 4
        M = rng.standard_normal((m, n))
        U = _make_psd_matrix(rng, m)
        V = _make_psd_matrix(rng, n)
        mg = MatrixGaussian(M, U, V)
        result = mg.affine_left(np.eye(m))
        assert np.allclose(result.M, M, atol=1e-14)
        assert np.allclose(result.U, U, atol=1e-14)
        assert np.allclose(result.V, V, atol=1e-14)

    def test_affine_right_identity_is_noop(self):
        """affine_right(I_n) is a no-op."""
        rng = np.random.default_rng(17)
        m, n = 3, 4
        M = rng.standard_normal((m, n))
        U = _make_psd_matrix(rng, m)
        V = _make_psd_matrix(rng, n)
        mg = MatrixGaussian(M, U, V)
        result = mg.affine_right(np.eye(n))
        assert np.allclose(result.M, M, atol=1e-14)
        assert np.allclose(result.V, V, atol=1e-14)
        assert np.allclose(result.U, U, atol=1e-14)

    def test_affine_combined(self):
        rng = np.random.default_rng(18)
        A = rng.standard_normal((3, 3))
        B = rng.standard_normal((4, 2))
        C_bias = rng.standard_normal((3, 2))
        U = _make_psd_matrix(rng, 3)
        V = _make_psd_matrix(rng, 4)
        M = rng.standard_normal((3, 4))
        mg = MatrixGaussian(M, U, V)
        result = mg.affine(A, B, C_bias)
        expected_M = A @ M @ B + C_bias
        expected_U = A @ U @ A.T
        expected_V = B.T @ V @ B
        assert np.allclose(result.M, expected_M, atol=1e-12)
        assert np.allclose(result.U, expected_U, atol=1e-12)
        assert np.allclose(result.V, expected_V, atol=1e-12)

    @pytest.mark.parametrize("m,n", [(2, 3), (4, 4), (3, 5)])
    def test_affine_left_psd_preserved(self, m, n):
        rng = np.random.default_rng(m * 100 + n)
        A = rng.standard_normal((m, m))
        mg = MatrixGaussian(
            np.zeros((m, n)),
            _make_psd_matrix(rng, m),
            _make_psd_matrix(rng, n),
        )
        result = mg.affine_left(A)
        # U = A @ U @ A.T must be PSD (symmetry + eigenvalue check)
        assert _is_psd(result.U)
        assert _is_psd(result.V)


# ---------------------------------------------------------------------------
# 10. add — isotropic fast paths
# ---------------------------------------------------------------------------

class TestAddIsotropic:
    """Case A and B fast paths must be exact (no SVD involved)."""

    def test_add_isotropic_path_A_mean(self):
        """Case A: U_X = c*I, other fully isotropic."""
        c, a, b = 2.0, 0.5, 3.0
        mx = MatrixGaussian(np.ones((3, 4)), c * np.eye(3), _make_psd_matrix(np.random.default_rng(0), 4))
        other = MatrixGaussian(np.zeros((3, 4)), a * np.eye(3), b * np.eye(4))
        result = mx.add(other)
        expected_mean = mx.M + other.M
        assert np.allclose(result.M, expected_mean, atol=1e-14)

    def test_add_isotropic_path_A_U_unchanged(self):
        c, a, b = 2.0, 0.5, 3.0
        V_x = _make_psd_matrix(np.random.default_rng(1), 4)
        mx = MatrixGaussian(np.zeros((3, 4)), c * np.eye(3), V_x)
        other = MatrixGaussian(np.zeros((3, 4)), a * np.eye(3), b * np.eye(4))
        result = mx.add(other)
        assert np.allclose(result.U, c * np.eye(3), atol=1e-14)

    def test_add_isotropic_path_A_V_update(self):
        c, a, b = 2.0, 0.5, 3.0
        V_x = _make_psd_matrix(np.random.default_rng(2), 4)
        mx = MatrixGaussian(np.zeros((3, 4)), c * np.eye(3), V_x)
        other = MatrixGaussian(np.zeros((3, 4)), a * np.eye(3), b * np.eye(4))
        result = mx.add(other)
        expected_V = V_x + (a * b / c) * np.eye(4)
        assert np.allclose(result.V, expected_V, atol=1e-12)

    def test_add_isotropic_path_A_exact_vs_dense(self):
        """Fast path A must match the dense V⊗U + V_other⊗U_other sum."""
        c, a, b = 1.5, 2.0, 0.8
        rng = np.random.default_rng(3)
        V_x = _make_psd_matrix(rng, 3)
        mx = MatrixGaussian(np.zeros((2, 3)), c * np.eye(2), V_x)
        other = MatrixGaussian(np.zeros((2, 3)), a * np.eye(2), b * np.eye(3))
        result = mx.add(other)
        # Dense reference
        Cov_ref = np.kron(V_x, c * np.eye(2)) + np.kron(b * np.eye(3), a * np.eye(2))
        Cov_fast = np.kron(result.V, result.U)
        assert np.allclose(Cov_fast, Cov_ref, atol=1e-10)

    def test_add_isotropic_path_B(self):
        """Case B: V_X = d*I, other fully isotropic."""
        d, a, b = 3.0, 0.5, 2.0
        rng = np.random.default_rng(4)
        U_x = _make_psd_matrix(rng, 3)
        mx = MatrixGaussian(np.zeros((3, 4)), U_x, d * np.eye(4))
        other = MatrixGaussian(np.zeros((3, 4)), a * np.eye(3), b * np.eye(4))
        result = mx.add(other)
        expected_U = U_x + (a * b / d) * np.eye(3)
        assert np.allclose(result.U, expected_U, atol=1e-12)
        assert np.allclose(result.V, d * np.eye(4), atol=1e-14)

    def test_add_isotropic_path_B_exact_vs_dense(self):
        d, a, b = 3.0, 0.5, 2.0
        rng = np.random.default_rng(5)
        U_x = _make_psd_matrix(rng, 3)
        mx = MatrixGaussian(np.zeros((3, 4)), U_x, d * np.eye(4))
        other = MatrixGaussian(np.zeros((3, 4)), a * np.eye(3), b * np.eye(4))
        result = mx.add(other)
        Cov_ref = np.kron(d * np.eye(4), U_x) + np.kron(b * np.eye(4), a * np.eye(3))
        Cov_fast = np.kron(result.V, result.U)
        assert np.allclose(Cov_fast, Cov_ref, atol=1e-10)

    def test_add_both_fully_isotropic(self):
        """Both operands fully isotropic — path A (U_X check runs first)."""
        c = 2.0; V_val = 3.0; a = 1.0; b = 1.0
        mx = MatrixGaussian(np.zeros((3, 3)), c * np.eye(3), V_val * np.eye(3))
        other = MatrixGaussian(np.zeros((3, 3)), a * np.eye(3), b * np.eye(3))
        result = mx.add(other)
        # V_new = V_x + (a*b/c) * I, U_new = c * I
        expected_V = V_val * np.eye(3) + (a * b / c) * np.eye(3)
        assert np.allclose(result.V, expected_V, atol=1e-12)
        assert np.allclose(result.U, c * np.eye(3), atol=1e-14)


# ---------------------------------------------------------------------------
# 11. add — general path (nearest-Kronecker)
# ---------------------------------------------------------------------------

class TestAddGeneral:
    def test_add_general_mean(self):
        rng = np.random.default_rng(20)
        mg1 = MatrixGaussian(rng.standard_normal((3, 3)),
                             _make_psd_matrix(rng, 3), _make_psd_matrix(rng, 3))
        mg2 = MatrixGaussian(rng.standard_normal((3, 3)),
                             _make_psd_matrix(rng, 3), _make_psd_matrix(rng, 3))
        result = mg1.add(mg2)
        assert np.allclose(result.M, mg1.M + mg2.M, atol=1e-14)

    def test_add_general_psd(self):
        rng = np.random.default_rng(21)
        mg1 = MatrixGaussian(np.zeros((3, 3)),
                             _make_psd_matrix(rng, 3), _make_psd_matrix(rng, 3))
        mg2 = MatrixGaussian(np.zeros((3, 3)),
                             _make_psd_matrix(rng, 3), _make_psd_matrix(rng, 3))
        result = mg1.add(mg2)
        assert _is_psd(result.U)
        assert _is_psd(result.V)

    def test_add_general_shape_preserved(self):
        rng = np.random.default_rng(22)
        mg1 = MatrixGaussian(np.zeros((2, 5)),
                             _make_psd_matrix(rng, 2), _make_psd_matrix(rng, 5))
        mg2 = MatrixGaussian(np.zeros((2, 5)),
                             _make_psd_matrix(rng, 2), _make_psd_matrix(rng, 5))
        result = mg1.add(mg2)
        assert result.shape == (2, 5)


# ---------------------------------------------------------------------------
# 12. _nearest_kronecker — sign disambiguation
# ---------------------------------------------------------------------------

class TestNearestKronecker:
    def test_symmetric_output(self):
        rng = np.random.default_rng(30)
        U0 = _make_psd_matrix(rng, 3)
        V0 = _make_psd_matrix(rng, 4)
        Cov = np.kron(V0, U0)
        U_est, V_est = _nearest_kronecker(Cov, 3, 4)
        assert np.allclose(U_est, U_est.T, atol=1e-12)
        assert np.allclose(V_est, V_est.T, atol=1e-12)

    def test_psd_output(self):
        rng = np.random.default_rng(31)
        U0 = _make_psd_matrix(rng, 3)
        V0 = _make_psd_matrix(rng, 4)
        Cov = np.kron(V0, U0)
        U_est, V_est = _nearest_kronecker(Cov, 3, 4)
        assert _is_psd(U_est)
        assert _is_psd(V_est)

    def test_recovers_exact_kron(self):
        """When Cov is exactly V⊗U, nearest-Kronecker recovers it up to scaling."""
        rng = np.random.default_rng(32)
        U0 = _make_psd_matrix(rng, 3)
        V0 = _make_psd_matrix(rng, 4)
        Cov = np.kron(V0, U0)
        U_est, V_est = _nearest_kronecker(Cov, 3, 4)
        # Reconstruction must match Cov
        Cov_est = np.kron(V_est, U_est)
        rel_err = np.linalg.norm(Cov - Cov_est, 'fro') / np.linalg.norm(Cov, 'fro')
        assert rel_err < 1e-8

    def test_sign_disambiguation_indefinite_U(self):
        """Indefinite U but positive trace — old heuristic fails; new first-nonzero fix works.

        Construct a matrix with positive trace(U) but negative minimum eigenvalue.
        After SVD the u_vec might come out with wrong sign. Verify the reconstruction
        is positive (Cov_est[0,0] > 0) and that the returned U, V are PSD.
        """
        # Indefinite U: eigenvalues [-1, 3] → trace = 2 > 0 but min_eig < 0
        Q = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
        U_indef = Q @ np.diag([-1.0, 3.0]) @ Q.T
        V0 = 2.0 * np.eye(2)
        # Build Cov from U_indef (NOT PSD, just for the test of sign disambiguation)
        # We want the Kronecker projection of a matrix that "looks" like V⊗U_indef
        Cov = np.kron(V0, U_indef)
        U_est, V_est = _nearest_kronecker(Cov, 2, 2)
        # After clipping, U_est should be PSD (negative eigenvalue zeroed)
        assert _is_psd(U_est), f"U_est not PSD: eigvals={np.linalg.eigvalsh(U_est)}"
        assert _is_psd(V_est), f"V_est not PSD: eigvals={np.linalg.eigvalsh(V_est)}"
        # Reconstruction should be positive semi-definite
        Cov_est = np.kron(V_est, U_est)
        assert _is_psd(Cov_est)

    def test_first_nonzero_positive_convention(self):
        """u_vec first nonzero element must be positive after disambiguation."""
        rng = np.random.default_rng(33)
        U0 = _make_psd_matrix(rng, 3)
        V0 = _make_psd_matrix(rng, 3)
        # Negate the Cov to force sign flip
        Cov_pos = np.kron(V0, U0)
        U_est, V_est = _nearest_kronecker(Cov_pos, 3, 3)
        u_vec = U_est.flatten('F')
        # First non-trivially nonzero element must be positive
        for val in u_vec:
            if abs(val) > 1e-14:
                assert val > 0, f"First nonzero u_vec element {val} is negative"
                break


# ---------------------------------------------------------------------------
# 13. Sampling via Cholesky vs eigh (distribution-level)
# ---------------------------------------------------------------------------

class TestSamplingConsistency:
    """Verify sampled statistics match analytical moments within MC tolerance."""

    @pytest.mark.parametrize("m,n,seed", [(2, 3, 0), (3, 3, 1), (4, 2, 2)])
    def test_mean_from_samples(self, m, n, seed):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((m, n))
        U = _make_psd_matrix(rng, m)
        V = _make_psd_matrix(rng, n)
        mg = MatrixGaussian(M, U, V)
        samples = _sample_matgaussian(mg, n_samples=20000, seed=seed + 100)
        sample_mean = samples.mean(0)
        max_err = np.max(np.abs(sample_mean - M))
        # With 20k samples and typical variance O(1), max error << 0.1
        assert max_err < 0.05, f"Max |Δ mean| {max_err:.4e} > tolerance"

    @pytest.mark.parametrize("m,n,seed", [(2, 3, 10), (3, 3, 11)])
    def test_variance_from_samples(self, m, n, seed):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((m, n))
        U = _make_psd_matrix(rng, m, offset=0.5)
        V = _make_psd_matrix(rng, n, offset=0.5)
        mg = MatrixGaussian(M, U, V)
        samples = _sample_matgaussian(mg, n_samples=30000, seed=seed + 200)
        analytical_var = mg.variance_matrix()
        sample_var = samples.var(0)
        # 5% relative tolerance
        rel_err = np.max(np.abs(analytical_var - sample_var) / (np.abs(analytical_var) + 1e-10))
        assert rel_err < 0.10, f"Max rel |Δ var| {rel_err:.4e} > 10%"


# ---------------------------------------------------------------------------
# 14. matmul_independent — MC validation
# ---------------------------------------------------------------------------

class TestMatmulIndependent:
    def test_mean_analytical(self):
        rng = np.random.default_rng(40)
        M_X = rng.standard_normal((3, 4))
        M_Y = rng.standard_normal((4, 2))
        X = MatrixGaussian(M_X, np.eye(3), np.eye(4))
        Y = MatrixGaussian(M_Y, np.eye(4), np.eye(2))
        Z = X.matmul_independent(Y)
        assert np.allclose(Z.M, M_X @ M_Y, atol=1e-12)

    def test_shape(self):
        rng = np.random.default_rng(41)
        X = MatrixGaussian(np.zeros((3, 4)), np.eye(3), np.eye(4))
        Y = MatrixGaussian(np.zeros((4, 2)), np.eye(4), np.eye(2))
        Z = X.matmul_independent(Y)
        assert Z.shape == (3, 2)

    def test_variance_mc_small(self):
        """MC validation at m=n=4: max relative |Δ var| < 10%."""
        rng = np.random.default_rng(42)
        m = 4
        M_X = rng.standard_normal((m, m)) * 0.1
        M_Y = rng.standard_normal((m, m)) * 0.1
        U_X = 0.1 * np.eye(m)
        V_X = 0.1 * np.eye(m)
        U_Y = 0.1 * np.eye(m)
        V_Y = 0.1 * np.eye(m)
        X = MatrixGaussian(M_X, U_X, V_X)
        Y = MatrixGaussian(M_Y, U_Y, V_Y)
        Z_mg = X.matmul_independent(Y)
        # MC
        n_mc = 30000
        L_X = np.linalg.cholesky(np.kron(V_X, U_X) + 1e-10 * np.eye(m * m))
        L_Y = np.linalg.cholesky(np.kron(V_Y, U_Y) + 1e-10 * np.eye(m * m))
        rng2 = np.random.default_rng(43)
        mc_Z = []
        for _ in range(n_mc):
            Xv = M_X.flatten('F') + L_X @ rng2.standard_normal(m * m)
            Xs = Xv.reshape((m, m), order='F')
            Yv = M_Y.flatten('F') + L_Y @ rng2.standard_normal(m * m)
            Ys = Yv.reshape((m, m), order='F')
            mc_Z.append(Xs @ Ys)
        mc_Z = np.array(mc_Z)
        mc_var = mc_Z.var(0)
        analytical_var = Z_mg.variance_matrix()
        rel_err = np.max(np.abs(analytical_var - mc_var) / (np.abs(mc_var) + 1e-10))
        assert rel_err < 0.15, f"matmul MC var rel_err {rel_err:.4e} > 15%"


# ---------------------------------------------------------------------------
# 15. to_scalar_dense / from_scalar_dense round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("m,n", [(2, 3), (3, 3), (4, 2)])
    def test_to_scalar_dense_shape(self, m, n):
        rng = np.random.default_rng(50 + m + n)
        mg = MatrixGaussian(
            rng.standard_normal((m, n)),
            _make_psd_matrix(rng, m),
            _make_psd_matrix(rng, n),
        )
        vec, cov = to_scalar_dense(mg)
        assert vec.shape == (m * n,)
        assert cov.shape == (m * n, m * n)

    @pytest.mark.parametrize("m,n", [(2, 3), (3, 3)])
    def test_round_trip_mean(self, m, n):
        rng = np.random.default_rng(60 + m + n)
        M = rng.standard_normal((m, n))
        U = _make_psd_matrix(rng, m)
        V = _make_psd_matrix(rng, n)
        mg = MatrixGaussian(M, U, V)
        vec, cov = to_scalar_dense(mg)
        mg2 = from_scalar_dense(vec, cov, m, n)
        assert np.allclose(mg2.M, M, atol=1e-12)

    @pytest.mark.parametrize("m,n", [(2, 3), (3, 3)])
    def test_round_trip_cov(self, m, n):
        """Round-trip cov reconstruction matches to relative tolerance."""
        rng = np.random.default_rng(70 + m + n)
        U = _make_psd_matrix(rng, m)
        V = _make_psd_matrix(rng, n)
        mg = MatrixGaussian(np.zeros((m, n)), U, V)
        vec, cov = to_scalar_dense(mg)
        mg2 = from_scalar_dense(vec, cov, m, n)
        cov2 = to_scalar_dense(mg2)[1]
        rel_err = np.linalg.norm(cov - cov2, 'fro') / np.linalg.norm(cov, 'fro')
        assert rel_err < 1e-8, f"Round-trip cov rel_err {rel_err:.3e}"
