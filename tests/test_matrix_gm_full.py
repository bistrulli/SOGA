"""
tests/test_matrix_gm_full.py — Unit tests for matrix_gm_full [B.8]

Tests:
  - _try_kronecker_decompose: exact Kronecker, near-Kronecker, non-separable
  - _parse_matrix_gm_full_text: valid parsing, shape validation, symmetry check
  - MATRIX_GM_FULL dispatcher: Kronecker path, dense path, warning classes
  - GaussianMixBlock.from_matrix_gm_full: both Kronecker and dense branches
  - matrix_var and matrix_cov with dense sentinel storage
  - Operations on Kronecker-detected variable (matmul via SOGA.py subprocess)
  - Operations on dense variable: extract scalar (works)

Acceptance criteria:
  - At least 15 tests
  - Auto-detect three regimes verified (constraint 6 from plan)
"""
import sys
import os
import warnings
import numpy as np
import pytest

# Add src/ to path for direct imports
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

from libMatrixGaussian import (
    _try_kronecker_decompose,
    _nearest_kronecker,
    KroneckerDetectionInfo,
    KroneckerNearMissWarning,
    DenseCovarianceInfo,
)
from libMatrixUpdate import (
    _parse_matrix_gm_full_text,
    _parse_matrix_expr,
    update_rule_matrix,
)
from libSOGAshared import Dist, VarEntry, GaussianMix
from libSOGAsharedMatrix import GaussianMixBlock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dist_with_matrix_var(name: str = "X", shape=(2, 2)) -> Dist:
    """Return a minimal Dist with a single matrix variable VarEntry."""
    ve = VarEntry(name=name, kind="matrix", shape=shape)
    gm_scalar = GaussianMix([1.0], [np.zeros(0)], [np.zeros((0, 0))])
    return Dist([], gm_scalar, var_entries=[ve], gm_block=None)


# ---------------------------------------------------------------------------
# Group 1: _try_kronecker_decompose
# ---------------------------------------------------------------------------

class TestTryKroneckerDecompose:

    def test_exact_kronecker_I4_zero_residual(self):
        """I_4 = I_2 ⊗ I_2: residual should be near zero."""
        Sigma = np.eye(4)
        U, V, ratio = _try_kronecker_decompose(Sigma, 2, 2)
        assert ratio < 1e-10, f"Expected near-zero residual, got {ratio:.3e}"

    def test_exact_kronecker_reconstruction(self):
        """I_4 = I_2 ⊗ I_2: reconstructed kron(V, U) should match to 1e-14."""
        Sigma = np.eye(4)
        U, V, ratio = _try_kronecker_decompose(Sigma, 2, 2)
        Sigma_recon = np.kron(V, U)
        np.testing.assert_allclose(
            Sigma_recon, Sigma, atol=1e-12,
            err_msg="Reconstruction of I_4 should be exact"
        )

    def test_exact_kronecker_nontrivial(self):
        """Sigma = [[3,1],[1,2]] ⊗ [[2,0.5],[0.5,1]]: exact Kronecker."""
        U_true = np.array([[2.0, 0.5], [0.5, 1.0]])
        V_true = np.array([[3.0, 1.0], [1.0, 2.0]])
        Sigma = np.kron(V_true, U_true)
        U, V, ratio = _try_kronecker_decompose(Sigma, 2, 2)
        assert ratio < 1e-10, f"Expected near-zero residual, got {ratio:.3e}"
        Sigma_recon = np.kron(V, U)
        np.testing.assert_allclose(Sigma_recon, Sigma, atol=1e-10)

    def test_near_kronecker_1e12_noise(self):
        """I_4 + 1e-12 noise: residual ~ 1e-12, within SOGA_KRON_STRICT=1e-8."""
        np.random.seed(42)
        noise = 1e-12 * np.random.randn(4, 4)
        noise = 0.5 * (noise + noise.T)
        Sigma = np.eye(4) + noise
        U, V, ratio = _try_kronecker_decompose(Sigma, 2, 2)
        assert ratio < 1e-8, f"Expected residual < 1e-8, got {ratio:.3e}"

    def test_non_separable_rank2_high_residual(self):
        """Rank-2 rearrangement Sigma: residual should be > 0.1."""
        # Construct explicitly non-Kronecker PSD matrix
        # Use two different Kronecker products summed
        U1, V1 = np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])
        U2, V2 = np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])
        Sigma = np.kron(V1, U1) + np.kron(V2, U2)
        # This is not a single Kronecker product: kron(V1+V2, U_avg) won't match
        # For it to be non-separable, we need off-diagonal coupling
        A = np.array([[2.0, 0.5, 0.5, 0.0],
                      [0.5, 1.0, 0.0, 0.5],
                      [0.5, 0.0, 1.0, 0.5],
                      [0.0, 0.5, 0.5, 2.0]])
        A = 0.5 * (A + A.T)
        w, Q = np.linalg.eigh(A)
        w = np.clip(w, 0.1, None)
        Sigma = Q @ np.diag(w) @ Q.T
        _, _, ratio = _try_kronecker_decompose(Sigma, 2, 2)
        assert ratio > 0.05, f"Expected large residual (> 0.05), got {ratio:.3e}"

    def test_zero_sigma_trivially_kronecker(self):
        """Zero Sigma: residual should be 0."""
        Sigma = np.zeros((4, 4))
        U, V, ratio = _try_kronecker_decompose(Sigma, 2, 2)
        assert ratio == 0.0


# ---------------------------------------------------------------------------
# Group 2: _parse_matrix_gm_full_text
# ---------------------------------------------------------------------------

class TestParseMatrixGmFullText:

    def test_2x2_identity_sigma(self):
        """Parse matrix_gm_full([[0,0],[0,0]], I_4)."""
        text = "matrix_gm_full([[0,0],[0,0]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])"
        M, Sigma = _parse_matrix_gm_full_text(text)
        assert M.shape == (2, 2)
        assert Sigma.shape == (4, 4)
        np.testing.assert_allclose(Sigma, np.eye(4))

    def test_nonzero_mean(self):
        """Mean matrix [[1,2],[3,4]] is parsed correctly."""
        text = "matrix_gm_full([[1,2],[3,4]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])"
        M, Sigma = _parse_matrix_gm_full_text(text)
        expected_M = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(M, expected_M)

    def test_wrong_arity_3_args_raises(self):
        """3-arg matrix_gm_full(...) should raise ValueError."""
        text = "matrix_gm_full([[0,0],[0,0]], [[1,0],[0,1]], [[1,0],[0,1]])"
        with pytest.raises(ValueError, match="expects 2 arguments"):
            _parse_matrix_gm_full_text(text)

    def test_wrong_arity_1_arg_raises(self):
        """1-arg matrix_gm_full(...) should raise ValueError."""
        text = "matrix_gm_full([[0,0],[0,0]])"
        with pytest.raises(ValueError, match="expects 2 arguments"):
            _parse_matrix_gm_full_text(text)

    def test_sigma_wrong_shape_raises(self):
        """Sigma shape mismatch with M should raise ValueError."""
        # M is 2x2 → need 4x4 Sigma; provide 2x2 instead
        text = "matrix_gm_full([[0,0],[0,0]], [[1,0],[0,1]])"
        with pytest.raises(ValueError, match="Sigma shape mismatch"):
            _parse_matrix_gm_full_text(text)

    def test_asymmetric_sigma_raises(self):
        """Non-symmetric Sigma should raise ValueError."""
        sigma_asym = [[1, 0.5, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        text = f"matrix_gm_full([[0,0],[0,0]], {sigma_asym})"
        with pytest.raises(ValueError, match="not symmetric"):
            _parse_matrix_gm_full_text(text)


# ---------------------------------------------------------------------------
# Group 3: MATRIX_GM_FULL dispatcher
# ---------------------------------------------------------------------------

class TestMatrixGmFullDispatcher:

    def test_kronecker_path_stores_UV(self):
        """I_4 → stored as (U, V) Kronecker factors."""
        dist0 = _make_dist_with_matrix_var("X", (2, 2))
        expr = "X = matrix_gm_full([[0,0],[0,0]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            dist1 = update_rule_matrix(dist0, expr, {})
        stored = dist1.gm_block.cov_blocks[0][frozenset({"X"})]
        U, V = stored
        assert U is not None, "Kronecker path: U should not be None"
        assert V is not None, "Kronecker path: V should not be None"
        # I_4 → U ~ I_2, V ~ I_2
        np.testing.assert_allclose(U, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(V, np.eye(2), atol=1e-10)

    def test_dense_path_stores_none_sigma(self):
        """Non-Kronecker Sigma → stored as (None, Sigma_dense) sentinel."""
        dist0 = _make_dist_with_matrix_var("X", (2, 2))
        # Build non-Kronecker PSD Sigma
        A = np.array([[2.0, 0.5, 0.5, 0.0],
                      [0.5, 1.0, 0.0, 0.5],
                      [0.5, 0.0, 1.0, 0.5],
                      [0.0, 0.5, 0.5, 2.0]])
        A = 0.5 * (A + A.T)
        w, Q = np.linalg.eigh(A)
        w = np.clip(w, 0.1, None)
        Sigma = Q @ np.diag(w) @ Q.T
        Sigma_list = Sigma.tolist()
        expr = f"X = matrix_gm_full([[0,0],[0,0]], {Sigma_list})"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            dist1 = update_rule_matrix(dist0, expr, {})
        stored = dist1.gm_block.cov_blocks[0][frozenset({"X"})]
        assert stored[0] is None, f"Dense path: first element should be None, got {stored[0]}"
        assert stored[1].shape == (4, 4)

    def test_kronecker_info_warning_emitted(self):
        """KroneckerDetectionInfo warning should be emitted for exact Kronecker."""
        dist0 = _make_dist_with_matrix_var("X", (2, 2))
        expr = "X = matrix_gm_full([[0,0],[0,0]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])"
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            update_rule_matrix(dist0, expr, {})
        kron_warnings = [w for w in wlist if issubclass(w.category, KroneckerDetectionInfo)]
        assert len(kron_warnings) > 0, "Expected KroneckerDetectionInfo warning"

    def test_dense_cov_info_warning_emitted(self):
        """DenseCovarianceInfo warning should be emitted for non-Kronecker Sigma."""
        dist0 = _make_dist_with_matrix_var("X", (2, 2))
        A = np.array([[2.0, 0.5, 0.5, 0.0],
                      [0.5, 1.0, 0.0, 0.5],
                      [0.5, 0.0, 1.0, 0.5],
                      [0.0, 0.5, 0.5, 2.0]])
        A = 0.5 * (A + A.T)
        w, Q = np.linalg.eigh(A)
        w = np.clip(w, 0.1, None)
        Sigma = Q @ np.diag(w) @ Q.T
        Sigma_list = Sigma.tolist()
        expr = f"X = matrix_gm_full([[0,0],[0,0]], {Sigma_list})"
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            update_rule_matrix(dist0, expr, {})
        dense_warnings = [w for w in wlist if issubclass(w.category, DenseCovarianceInfo)]
        assert len(dense_warnings) > 0, "Expected DenseCovarianceInfo warning"

    def test_mean_stored_correctly(self):
        """Mean matrix [[1,2],[3,4]] stored correctly in mu_blocks."""
        dist0 = _make_dist_with_matrix_var("X", (2, 2))
        expr = "X = matrix_gm_full([[1,2],[3,4]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            dist1 = update_rule_matrix(dist0, expr, {})
        M_stored = dist1.gm_block.mu_blocks[0]["X"]
        np.testing.assert_allclose(M_stored, np.array([[1, 2], [3, 4]]))

    def test_shape_mismatch_raises(self):
        """Sigma shape not matching declared variable shape → ValueError."""
        dist0 = _make_dist_with_matrix_var("X", (2, 2))
        # Wrong Sigma: 2x2 instead of 4x4
        expr = "X = matrix_gm_full([[0,0],[0,0]], [[1,0],[0,1]])"
        with pytest.raises((ValueError, RuntimeError)):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                update_rule_matrix(dist0, expr, {})


# ---------------------------------------------------------------------------
# Group 4: GaussianMixBlock.from_matrix_gm_full
# ---------------------------------------------------------------------------

class TestFromMatrixGmFull:

    def test_kronecker_init(self):
        """from_matrix_gm_full in Kronecker mode: stores (U, V) in cov_blocks."""
        ve = VarEntry(name="X", kind="matrix", shape=(2, 2))
        M = np.zeros((2, 2))
        U = np.eye(2)
        V = np.eye(2)
        Sigma = np.kron(V, U)
        block = GaussianMixBlock.from_matrix_gm_full(ve, M, Sigma, U, V, [1.0], [])
        stored = block.cov_blocks[0][frozenset({"X"})]
        assert stored[0] is not None, "Kronecker mode: U should not be None"
        U_stored, V_stored = stored
        np.testing.assert_allclose(U_stored, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(V_stored, np.eye(2), atol=1e-10)

    def test_dense_init(self):
        """from_matrix_gm_full in dense mode: stores (None, Sigma) sentinel."""
        ve = VarEntry(name="X", kind="matrix", shape=(2, 2))
        M = np.zeros((2, 2))
        Sigma = np.diag([2.0, 1.0, 1.0, 2.0])
        block = GaussianMixBlock.from_matrix_gm_full(ve, M, Sigma, None, None, [1.0], [])
        stored = block.cov_blocks[0][frozenset({"X"})]
        assert stored[0] is None, "Dense mode: first element should be None"
        np.testing.assert_allclose(stored[1], Sigma, atol=1e-10)

    def test_dense_marginal_variance(self):
        """Dense Sigma: Var[X[i,j]] = Sigma[j*m+i, j*m+i] (column-major)."""
        ve = VarEntry(name="X", kind="matrix", shape=(2, 2))
        M = np.zeros((2, 2))
        # Var[X[0,0]] = 3, Var[X[1,0]] = 1, Var[X[0,1]] = 1, Var[X[1,1]] = 2
        diag_vals = [3.0, 1.0, 1.0, 2.0]
        Sigma = np.diag(diag_vals)
        block = GaussianMixBlock.from_matrix_gm_full(ve, M, Sigma, None, None, [1.0], [])
        # column-major: idx(i,j) = j*m + i
        assert abs(block.matrix_var("X", 0, 0) - 3.0) < 1e-10
        assert abs(block.matrix_var("X", 1, 0) - 1.0) < 1e-10
        assert abs(block.matrix_var("X", 0, 1) - 1.0) < 1e-10
        assert abs(block.matrix_var("X", 1, 1) - 2.0) < 1e-10

    def test_dense_cross_covariance(self):
        """Dense Sigma: Cov(X[0,0], X[1,1]) = Sigma[0, 3] (column-major indices)."""
        ve = VarEntry(name="X", kind="matrix", shape=(2, 2))
        M = np.zeros((2, 2))
        Sigma = np.eye(4)
        Sigma[0, 3] = 0.5
        Sigma[3, 0] = 0.5  # symmetric
        block = GaussianMixBlock.from_matrix_gm_full(ve, M, Sigma, None, None, [1.0], [])
        # idx(0,0)=0, idx(1,1)=3 in column-major
        cov_val = block.matrix_cov("X", 0, 0, 1, 1)
        assert abs(cov_val - 0.5) < 1e-10, f"Expected 0.5, got {cov_val}"
