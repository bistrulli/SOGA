"""
tests/test_update_matrix.py — Matrix assignment operation tests for M4.

Tests:
  - matrix_gm initialisation (M4.1)
  - affine_left A@X (M4.2) — dense cross-validation at m,n=4,8
  - affine_right X@B (M4.3)
  - add_const X+C (M4.4)
  - add_random X+N (M4.5): iso path A, iso path B, general path
  - scale c*X (M4.6)
  - transpose transp(X) (M4.7)
  - identity ops: affine(I), transp(transp(X)) = X, scale(1) no-op
  - PSD invariants after each op (M4.9)

Per-op acceptance: max |Delta| < 1e-10 vs dense ground truth for exact ops,
< 1e-3 for nearest-Kronecker path.

Plan reference: §M4.10 of plan/2026-05-22-matrix-gm-lishan.md
"""

import sys
import os
from copy import deepcopy

import numpy as np
import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from libSOGAshared import Dist, GaussianMix, VarEntry
from libSOGAsharedMatrix import GaussianMixBlock
from libSOGAupdate import update_rule_matrix
from libMatrixUpdate import (
    _matrix_affine_left,
    _matrix_affine_right,
    _matrix_add_const,
    _matrix_add_random,
    _matrix_scale,
    _matrix_transpose,
    _parse_matrix_gm_text,
    _parse_nested_list,
)
from libMatrixGaussian import KRONECKER_CONVENTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_dist(m: int, n: int, name: str = "X", seed: int = 1,
               pi_scale: float = 1.0, u_scale: float = 1.0, v_scale: float = 0.5,
               mean_val: float = 0.0) -> tuple:
    """Create a Dist + GaussianMixBlock for a single matrix variable."""
    rng = np.random.default_rng(seed)
    M = mean_val * np.ones((m, n))
    A = rng.normal(size=(m, m)); U = A @ A.T + u_scale * np.eye(m)
    B = rng.normal(size=(n, n)); V = B @ B.T + v_scale * np.eye(n)
    ve = VarEntry(name, "matrix", (m, n), -1)
    block = GaussianMixBlock.from_matrix_gm([ve], [M], [U], [V])
    gm = GaussianMix([1.], [np.array([])], [np.zeros((0, 0))])
    dist = Dist([], gm, var_entries=[ve], gm_block=block)
    return dist, M, U, V


def _var_element(block: GaussianMixBlock, k: int, name: str, i: int, j: int) -> float:
    """Var(X[i,j]) = U[i,i]*V[j,j] for single component."""
    U, V = block.get_cov(k, name, name)
    return float(U[i, i] * V[j, j])


def _check_psd(block: GaussianMixBlock, name: str) -> None:
    """Assert that U and V factors are PSD for all components."""
    for k in range(block.n_comp()):
        U, V = block.get_cov(k, name, name)
        assert np.all(np.linalg.eigvalsh(U) >= -1e-10), f"U not PSD in component {k}"
        assert np.all(np.linalg.eigvalsh(V) >= -1e-10), f"V not PSD in component {k}"


def _dense_kron(block: GaussianMixBlock, k: int, name: str) -> np.ndarray:
    """Return dense V⊗U for component k."""
    U, V = block.get_cov(k, name, name)
    assert KRONECKER_CONVENTION == "V_outer_U"
    return np.kron(V, U)


# ---------------------------------------------------------------------------
# M4.1 / M4: matrix_gm initialisation
# ---------------------------------------------------------------------------

class TestMatrixGMInit:
    """update_rule_matrix initialises gm_block from matrix_gm(M, U, V)."""

    def test_matrix_gm_init_2x2(self):
        """2x2 matrix_gm sets correct mean and K factors."""
        ve = VarEntry("X", "matrix", (2, 2), -1)
        gm = GaussianMix([1.], [np.array([])], [np.zeros((0, 0))])
        dist = Dist([], gm, var_entries=[ve])
        expr = "X=matrix_gm([[1,2],[3,4]],[[2,0],[0,2]],[[1,0],[0,1]])"
        result = update_rule_matrix(dist, expr, {})
        M = result.gm_block.matrix_mean("X")
        np.testing.assert_allclose(M, [[1, 2], [3, 4]], atol=1e-12)
        U, V = result.gm_block.get_cov(0, "X", "X")
        np.testing.assert_allclose(U, 2 * np.eye(2), atol=1e-12)
        np.testing.assert_allclose(V, np.eye(2), atol=1e-12)

    def test_matrix_gm_parser_4x4(self):
        """_parse_matrix_gm_text handles 4x4 matrices correctly."""
        zeros = "[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]"
        I4 = "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"
        half = "[[0.5,0,0,0],[0,0.5,0,0],[0,0,0.5,0],[0,0,0,0.5]]"
        M, U, V = _parse_matrix_gm_text(f"matrix_gm({zeros},{I4},{half})")
        np.testing.assert_allclose(M, np.zeros((4, 4)))
        np.testing.assert_allclose(U, np.eye(4))
        np.testing.assert_allclose(V, 0.5 * np.eye(4))

    def test_nested_list_parser_basic(self):
        """_parse_nested_list handles nested numeric lists without code execution."""
        result = _parse_nested_list("[[1.0, 0.5], [0.5, 2.0]]")
        expected = [[1.0, 0.5], [0.5, 2.0]]
        assert result == expected

    def test_nested_list_parser_floats(self):
        """_parse_nested_list handles scientific notation."""
        result = _parse_nested_list("[1e-3, 2.5e2]")
        assert abs(result[0] - 0.001) < 1e-15
        assert abs(result[1] - 250.0) < 1e-12


# ---------------------------------------------------------------------------
# M4.2: affine_left A @ X
# ---------------------------------------------------------------------------

class TestAffineLeft:
    """Y = A @ X: M_new = A@M, U_new = A@U@A^T, V unchanged."""

    @pytest.mark.parametrize("m,n", [(2, 2), (4, 4), (3, 5), (8, 4)])
    def test_mean_affine_left(self, m, n):
        """E[A@X] = A @ E[X]."""
        rng = np.random.default_rng(m * 7 + n)
        dist, M, U, V = _init_dist(m, n, seed=m + n)
        p = m  # A is square p×m for simplicity
        A = rng.normal(size=(p, m))
        block = deepcopy(dist.gm_block)
        _matrix_affine_left(block, 0, "X", A)
        M_new = block.get_mu(0, "X")
        np.testing.assert_allclose(M_new, A @ M, atol=1e-12)

    @pytest.mark.parametrize("m,n", [(2, 2), (4, 4), (4, 3)])
    def test_cov_affine_left(self, m, n):
        """V⊗U after A@X equals V⊗(A@U@A^T): cross-validation against dense."""
        rng = np.random.default_rng(m * 13 + n)
        dist, M, U, V = _init_dist(m, n, seed=m * 5 + n)
        A = rng.normal(size=(m, m))  # square for simplicity
        block = deepcopy(dist.gm_block)
        _matrix_affine_left(block, 0, "X", A)
        U_new, V_new = block.get_cov(0, "X", "X")
        # Dense ground truth
        Cov_gt = np.kron(V, A @ U @ A.T)
        Cov_got = _dense_kron(block, 0, "X")
        np.testing.assert_allclose(Cov_got, Cov_gt, atol=1e-10)

    def test_identity_affine_left_is_noop(self):
        """A = I_m: affine_left is a no-op on covariance."""
        m, n = 4, 4
        dist, M, U, V = _init_dist(m, n, seed=42)
        A = np.eye(m)
        block = deepcopy(dist.gm_block)
        _matrix_affine_left(block, 0, "X", A)
        U_new, V_new = block.get_cov(0, "X", "X")
        np.testing.assert_allclose(U_new, U, atol=1e-12)
        np.testing.assert_allclose(V_new, V, atol=1e-12)

    @pytest.mark.parametrize("m,n", [(2, 2), (4, 4)])
    def test_psd_after_affine_left(self, m, n):
        """PSD invariant holds after affine_left (M4.9)."""
        rng = np.random.default_rng(m + n + 1)
        dist, _, _, _ = _init_dist(m, n, seed=m + n + 1)
        A = rng.normal(size=(m, m))
        block = deepcopy(dist.gm_block)
        _matrix_affine_left(block, 0, "X", A)
        from libSOGAsharedMatrix import _enforce_psd_kron_factors
        _enforce_psd_kron_factors(*block.get_cov(0, "X", "X"))
        _check_psd(block, "X")


# ---------------------------------------------------------------------------
# M4.3: affine_right X @ B
# ---------------------------------------------------------------------------

class TestAffineRight:
    """Y = X @ B: M_new = M@B, V_new = B^T@V@B, U unchanged."""

    @pytest.mark.parametrize("m,n", [(2, 2), (4, 4), (3, 5)])
    def test_mean_affine_right(self, m, n):
        """E[X@B] = E[X] @ B."""
        rng = np.random.default_rng(m * 3 + n * 7)
        dist, M, U, V = _init_dist(m, n, seed=m * 3 + n)
        B = rng.normal(size=(n, n))
        block = deepcopy(dist.gm_block)
        _matrix_affine_right(block, 0, "X", B)
        np.testing.assert_allclose(block.get_mu(0, "X"), M @ B, atol=1e-12)

    @pytest.mark.parametrize("m,n", [(2, 2), (4, 4)])
    def test_cov_affine_right(self, m, n):
        """V⊗U after X@B equals (B^T@V@B)⊗U."""
        rng = np.random.default_rng(m * 11 + n)
        dist, M, U, V = _init_dist(m, n, seed=m + n * 5)
        B = rng.normal(size=(n, n))
        block = deepcopy(dist.gm_block)
        _matrix_affine_right(block, 0, "X", B)
        U_new, V_new = block.get_cov(0, "X", "X")
        Cov_gt = np.kron(B.T @ V @ B, U)
        Cov_got = _dense_kron(block, 0, "X")
        np.testing.assert_allclose(Cov_got, Cov_gt, atol=1e-10)

    def test_identity_affine_right_is_noop(self):
        """B = I_n: affine_right is a no-op."""
        m, n = 4, 4
        dist, M, U, V = _init_dist(m, n, seed=99)
        B = np.eye(n)
        block = deepcopy(dist.gm_block)
        _matrix_affine_right(block, 0, "X", B)
        U_new, V_new = block.get_cov(0, "X", "X")
        np.testing.assert_allclose(V_new, V, atol=1e-12)
        np.testing.assert_allclose(U_new, U, atol=1e-12)


# ---------------------------------------------------------------------------
# M4.4: add_const X + C
# ---------------------------------------------------------------------------

class TestAddConst:
    """Y = X + C: mean shifts, covariance unchanged."""

    def test_mean_add_const(self):
        """E[X + C] = E[X] + C."""
        dist, M, U, V = _init_dist(3, 3, seed=5)
        C = np.ones((3, 3)) * 2.5
        block = deepcopy(dist.gm_block)
        _matrix_add_const(block, 0, "X", C)
        np.testing.assert_allclose(block.get_mu(0, "X"), M + C, atol=1e-12)

    def test_cov_add_const_unchanged(self):
        """Covariance is unchanged by adding a constant."""
        dist, M, U, V = _init_dist(3, 3, seed=5)
        C = np.ones((3, 3))
        block = deepcopy(dist.gm_block)
        _matrix_add_const(block, 0, "X", C)
        U_new, V_new = block.get_cov(0, "X", "X")
        np.testing.assert_allclose(U_new, U, atol=1e-12)
        np.testing.assert_allclose(V_new, V, atol=1e-12)

    def test_zero_const_is_noop(self):
        """Adding C=0 is a no-op on mean."""
        dist, M, U, V = _init_dist(4, 4, seed=10)
        block = deepcopy(dist.gm_block)
        _matrix_add_const(block, 0, "X", np.zeros((4, 4)))
        np.testing.assert_allclose(block.get_mu(0, "X"), M, atol=1e-12)


# ---------------------------------------------------------------------------
# M4.5: add_random X + N
# ---------------------------------------------------------------------------

class TestAddRandom:
    """Y = X + N: isotropic fast paths and general nearest-Kronecker."""

    def _make_isotropic_pair(self, m, n):
        """Create a GaussianMixBlock with X and N variables, both isotropic.

        Uses update_rule_matrix to init both variables via matrix_gm(...)
        to exercise the real initialisation path.
        """
        ve_X = VarEntry("X", "matrix", (m, n), -1)
        ve_N = VarEntry("N", "matrix", (m, n), -1)
        gm = GaussianMix([1.], [np.array([])], [np.zeros((0, 0))])
        dist = Dist([], gm, var_entries=[ve_X, ve_N])

        zeros = "[[" + "],[".join(",".join(["0"] * n) for _ in range(m)) + "]]"
        Im = "[[" + "],[".join(",".join(["1" if i==j else "0" for j in range(m)]) for i in range(m)) + "]]"
        half_In = "[[" + "],[".join(",".join(["0.5" if i==j else "0" for j in range(n)]) for i in range(n)) + "]]"
        In = "[[" + "],[".join(",".join(["1" if i==j else "0" for j in range(n)]) for i in range(n)) + "]]"
        qIm = "[[" + "],[".join(",".join(["0.25" if i==j else "0" for j in range(m)]) for i in range(m)) + "]]"

        dist = update_rule_matrix(dist, f"X=matrix_gm({zeros},{Im},{half_In})", {})
        dist = update_rule_matrix(dist, f"N=matrix_gm({zeros},{qIm},{In})", {})

        U_X = np.eye(m)
        V_X = 0.5 * np.eye(n)
        U_N = 0.25 * np.eye(m)
        V_N = np.eye(n)
        return dist.gm_block, U_X, V_X, U_N, V_N

    def test_iso_path_A_mean(self):
        """Iso path A: E[X+N] = E[X] + E[N]."""
        m, n = 4, 4
        block, U_X, V_X, U_N, V_N = self._make_isotropic_pair(m, n)
        _matrix_add_random(block, 0, "X", "N")
        np.testing.assert_allclose(block.get_mu(0, "X"), np.zeros((m, n)), atol=1e-12)

    def test_iso_path_A_variance(self):
        """Iso path A: Var(X+N)[0,0] = U_X[0,0]*V_X[0,0] + U_N[0,0]*V_N[0,0]."""
        m, n = 4, 4
        block, U_X, V_X, U_N, V_N = self._make_isotropic_pair(m, n)
        _matrix_add_random(block, 0, "X", "N")
        U_new, V_new = block.get_cov(0, "X", "X")
        # Expected: V_new = V_X + (U_N[0,0]*V_N[0,0]/U_X[0,0])*I = 0.5*I + 0.25*I = 0.75*I
        # U_new = U_X = I
        expected_var = float(U_new[0, 0] * V_new[0, 0])
        expected_true = 1.0 * 0.5 + 0.25 * 1.0  # = 0.75
        assert abs(expected_var - expected_true) < 1e-10

    def test_iso_path_A_cov_structure(self):
        """Iso path A: V_new is still isotropic after iso add."""
        m, n = 4, 4
        block, *_ = self._make_isotropic_pair(m, n)
        _matrix_add_random(block, 0, "X", "N")
        U_new, V_new = block.get_cov(0, "X", "X")
        # V_new should be isotropic: V_X + (a*b/c)*I = 0.75*I
        np.testing.assert_allclose(V_new, 0.75 * np.eye(n), atol=1e-10)

    def test_iso_path_B_variance(self):
        """Iso path B: V_X isotropic, N isotropic → U_new updated."""
        m, n = 3, 5
        ve_X = VarEntry("X", "matrix", (m, n), -1)
        ve_N = VarEntry("N", "matrix", (m, n), -1)
        gm = GaussianMix([1.], [np.array([])], [np.zeros((0, 0))])
        dist = Dist([], gm, var_entries=[ve_X, ve_N])

        # Build matrix string helpers for 3x5
        rng = np.random.default_rng(77)
        A = rng.normal(size=(m, m)); U_X = A @ A.T + np.eye(m)
        V_X = 2.0 * np.eye(n)  # isotropic
        U_N = 0.5 * np.eye(m)
        V_N = 1.0 * np.eye(n)

        def mat2str(M):
            return "[" + ",".join("[" + ",".join(str(v) for v in row) + "]" for row in M) + "]"

        zeros = mat2str(np.zeros((m, n)))
        dist = update_rule_matrix(dist, f"X=matrix_gm({zeros},{mat2str(U_X)},{mat2str(V_X)})", {})
        dist = update_rule_matrix(dist, f"N=matrix_gm({zeros},{mat2str(U_N)},{mat2str(V_N)})", {})

        block = dist.gm_block
        _matrix_add_random(block, 0, "X", "N")
        U_new, V_new = block.get_cov(0, "X", "X")
        # Expected: U_new = U_X + (U_N[0,0]*V_N[0,0]/V_X[0,0])*I = U_X + 0.25*I
        U_expected = U_X + 0.25 * np.eye(m)
        np.testing.assert_allclose(U_new, U_expected, atol=1e-10)

    def test_psd_after_add_random_general(self):
        """PSD invariant holds after general (nearest-Kronecker) add."""
        m, n = 4, 4
        rng = np.random.default_rng(123)
        ve_X = VarEntry("X", "matrix", (m, n), -1)
        ve_N = VarEntry("N", "matrix", (m, n), -1)
        gm = GaussianMix([1.], [np.array([])], [np.zeros((0, 0))])
        dist = Dist([], gm, var_entries=[ve_X, ve_N])

        A = rng.normal(size=(m, m)); U_X = A @ A.T + np.eye(m)
        B = rng.normal(size=(n, n)); V_X = B @ B.T + np.eye(n)
        C = rng.normal(size=(m, m)); U_N = C @ C.T + np.eye(m)
        D = rng.normal(size=(n, n)); V_N = D @ D.T + np.eye(n)

        def mat2str(M):
            return "[" + ",".join("[" + ",".join(str(v) for v in row) + "]" for row in M) + "]"

        zeros = mat2str(np.zeros((m, n)))
        dist = update_rule_matrix(dist, f"X=matrix_gm({zeros},{mat2str(U_X)},{mat2str(V_X)})", {})
        dist = update_rule_matrix(dist, f"N=matrix_gm({zeros},{mat2str(U_N)},{mat2str(V_N)})", {})

        block = dist.gm_block
        _matrix_add_random(block, 0, "X", "N")
        _check_psd(block, "X")


# ---------------------------------------------------------------------------
# M4.6: scale c * X
# ---------------------------------------------------------------------------

class TestScale:
    """Y = c * X: M_new = c*M, U_new = c^2*U, V unchanged."""

    @pytest.mark.parametrize("c", [2.0, 0.5, -1.0, 0.0])
    def test_mean_scale(self, c):
        """E[c*X] = c * E[X]."""
        dist, M, U, V = _init_dist(3, 3, seed=5, mean_val=1.0)
        block = deepcopy(dist.gm_block)
        _matrix_scale(block, 0, "X", c)
        np.testing.assert_allclose(block.get_mu(0, "X"), c * M, atol=1e-12)

    def test_cov_scale(self):
        """Cov scales by c^2 (absorbed into U)."""
        dist, M, U, V = _init_dist(4, 4, seed=7)
        c = 3.0
        block = deepcopy(dist.gm_block)
        _matrix_scale(block, 0, "X", c)
        U_new, V_new = block.get_cov(0, "X", "X")
        np.testing.assert_allclose(U_new, (c ** 2) * U, atol=1e-10)
        np.testing.assert_allclose(V_new, V, atol=1e-12)

    def test_scale_1_is_noop(self):
        """c=1: scale is a no-op."""
        dist, M, U, V = _init_dist(3, 3, seed=5)
        block = deepcopy(dist.gm_block)
        _matrix_scale(block, 0, "X", 1.0)
        np.testing.assert_allclose(block.get_mu(0, "X"), M, atol=1e-12)
        U_new, V_new = block.get_cov(0, "X", "X")
        np.testing.assert_allclose(U_new, U, atol=1e-10)


# ---------------------------------------------------------------------------
# M4.7: transpose
# ---------------------------------------------------------------------------

class TestTranspose:
    """Y = transp(X): M_new = M^T, U_new = V, V_new = U."""

    @pytest.mark.parametrize("m,n", [(2, 3), (4, 4), (3, 5)])
    def test_mean_transpose(self, m, n):
        """E[X^T] = E[X]^T."""
        dist, M, U, V = _init_dist(m, n, seed=m + n * 3, mean_val=1.0)
        block = deepcopy(dist.gm_block)
        _matrix_transpose(block, 0, "X")
        np.testing.assert_allclose(block.get_mu(0, "X"), M.T, atol=1e-12)

    @pytest.mark.parametrize("m,n", [(2, 3), (4, 4)])
    def test_factors_swap_on_transpose(self, m, n):
        """U_new = V_old, V_new = U_old after transpose."""
        dist, M, U, V = _init_dist(m, n, seed=42)
        block = deepcopy(dist.gm_block)
        _matrix_transpose(block, 0, "X")
        U_new, V_new = block.get_cov(0, "X", "X")
        np.testing.assert_allclose(U_new, V, atol=1e-12)
        np.testing.assert_allclose(V_new, U, atol=1e-12)

    @pytest.mark.parametrize("m,n", [(2, 3), (4, 4)])
    def test_double_transpose_is_identity(self, m, n):
        """transp(transp(X)) = X on both mean and covariance."""
        dist, M, U, V = _init_dist(m, n, seed=33, mean_val=1.5)
        block = deepcopy(dist.gm_block)
        _matrix_transpose(block, 0, "X")
        _matrix_transpose(block, 0, "X")
        np.testing.assert_allclose(block.get_mu(0, "X"), M, atol=1e-12)
        U_new, V_new = block.get_cov(0, "X", "X")
        np.testing.assert_allclose(U_new, U, atol=1e-12)
        np.testing.assert_allclose(V_new, V, atol=1e-12)


# ---------------------------------------------------------------------------
# M4 integration: update_rule_matrix end-to-end
# ---------------------------------------------------------------------------

class TestUpdateRuleMatrixIntegration:
    """End-to-end tests via update_rule_matrix dispatcher."""

    def _make_blank_dist(self, *var_names, shape=(4, 4)):
        """Create a Dist with multiple matrix VarEntries."""
        ves = [VarEntry(n, "matrix", shape, -1) for n in var_names]
        gm = GaussianMix([1.], [np.array([])], [np.zeros((0, 0))])
        return Dist([], gm, var_entries=ves)

    def test_matrix_gm_init_4x4(self):
        """matrix_gm(M, U, V) initialises gm_block correctly."""
        dist = self._make_blank_dist("X")
        z = "[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]"
        I = "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"
        h = "[[0.5,0,0,0],[0,0.5,0,0],[0,0,0.5,0],[0,0,0,0.5]]"
        result = update_rule_matrix(dist, f"X=matrix_gm({z},{I},{h})", {})
        assert result.gm_block is not None
        np.testing.assert_allclose(result.gm_block.matrix_mean("X"), np.zeros((4, 4)))
        assert abs(result.gm_block.matrix_var("X", 0, 0) - 0.5) < 1e-12

    def test_lishan_forward_pass_4x4(self):
        """C = A@X + N matches expected moments for 4x4 Lishan case."""
        dist = self._make_blank_dist("X", "N", "C")
        z = "[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]"
        I = "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"
        h = "[[0.5,0,0,0],[0,0.5,0,0],[0,0,0.5,0],[0,0,0,0.5]]"
        qI = "[[0.25,0,0,0],[0,0.25,0,0],[0,0,0.25,0],[0,0,0,0.25]]"

        dist = update_rule_matrix(dist, f"X=matrix_gm({z},{I},{h})", {})
        dist = update_rule_matrix(dist, f"N=matrix_gm({z},{qI},{I})", {})
        dist = update_rule_matrix(dist, f"C=matrix_gm({z},{I},{I})", {})

        A = np.eye(4)
        data = {"A_kernel": A.tolist()}
        dist = update_rule_matrix(dist, "C=A_kernel@X", data)
        dist = update_rule_matrix(dist, "C=C+N", {})

        M_C = dist.gm_block.matrix_mean("C")
        var_C00 = dist.gm_block.matrix_var("C", 0, 0)

        np.testing.assert_allclose(M_C, np.zeros((4, 4)), atol=1e-12)
        # Var(C[0,0]) = 1*0.5 + 0.25*1 = 0.75
        assert abs(var_C00 - 0.75) < 1e-10

    def test_affine_right_via_dispatcher(self):
        """C = X @ B dispatches correctly."""
        dist = self._make_blank_dist("X", "C")
        I = "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"
        z = "[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]"
        h = "[[0.5,0,0,0],[0,0.5,0,0],[0,0,0.5,0],[0,0,0,0.5]]"
        dist = update_rule_matrix(dist, f"X=matrix_gm({z},{I},{h})", {})
        dist = update_rule_matrix(dist, f"C=matrix_gm({z},{I},{I})", {})
        B = 2.0 * np.eye(4)
        data = {"B_mat": B.tolist()}
        dist = update_rule_matrix(dist, "C=X@B_mat", data)
        U_C, V_C = dist.gm_block.get_cov(0, "C", "C")
        # V_new = B^T @ V_X @ B = 2I @ 0.5I @ 2I = 2.0 * I
        np.testing.assert_allclose(V_C, 2.0 * np.eye(4), atol=1e-10)

    def test_scale_via_dispatcher(self):
        """C = 3 * X scales correctly."""
        dist = self._make_blank_dist("X")
        I = "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"
        h = "[[0.5,0,0,0],[0,0.5,0,0],[0,0,0.5,0],[0,0,0,0.5]]"
        z = "[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]"
        dist = update_rule_matrix(dist, f"X=matrix_gm({z},{I},{h})", {})
        dist = update_rule_matrix(dist, "X=3*X", {})
        var_X00 = dist.gm_block.matrix_var("X", 0, 0)
        # Var(3X[0,0]) = 9 * 1.0 * 0.5 = 4.5
        assert abs(var_X00 - 4.5) < 1e-10

    def test_psd_invariant_after_affine_left(self):
        """PSD holds on U, V after A@X for non-trivial A."""
        dist = self._make_blank_dist("X", "C")
        I = "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"
        h = "[[0.5,0,0,0],[0,0.5,0,0],[0,0,0.5,0],[0,0,0,0.5]]"
        z = "[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]"
        dist = update_rule_matrix(dist, f"X=matrix_gm({z},{I},{h})", {})
        dist = update_rule_matrix(dist, f"C=matrix_gm({z},{I},{I})", {})
        rng = np.random.default_rng(5)
        A = rng.normal(size=(4, 4))
        data = {"A_k": A.tolist()}
        dist = update_rule_matrix(dist, "C=A_k@X", data)
        _check_psd(dist.gm_block, "C")


# ---------------------------------------------------------------------------
# A1/A2 — scalar = matrix[i,j] cross-cov + observe back-prop
# ---------------------------------------------------------------------------

class TestExtractScalarCrossCov:
    """A1 (research-note 04): extract_scalar_from_matrix stores
    `Cov(y, vec(X)) = V[:,j] ⊗ U[:,i]` per Gupta & Nagar 1999 Thm 2.3.1.
    """

    def _make_X_2x2_iso(self):
        """X ~ MN(0, I, I) — independent 2x2 standard normal entries."""
        from libMatrixUpdate import update_rule_matrix
        ve_X = VarEntry(name="X", kind="matrix", shape=(2, 2), flat_offset=-1)
        dist = Dist(
            var_list=[],
            gm=GaussianMix([1.0], [np.zeros(0)], [np.zeros((0, 0))]),
            var_entries=[ve_X],
            gm_block=GaussianMixBlock(
                var_list=[],
                var_entries=[],
                pi=[1.0],
                mu_blocks=[{}],
                cov_blocks=[{}],
            ),
        )
        I = "[[1,0],[0,1]]"
        z = "[[0,0],[0,0]]"
        return update_rule_matrix(dist, f"X=matrix_gm({z},{I},{I})", {})

    def test_extract_cross_cov_value(self):
        """Cov(y, vec(X)) stored as np.kron(V[:,j], U[:,i]) (column-major)."""
        from libMatrixUpdate import extract_scalar_from_matrix
        dist0 = self._make_X_2x2_iso()
        dist1 = extract_scalar_from_matrix(dist0, "y", "X", 0, 0)
        # Expected: V[:,0] ⊗ U[:,0] = [1,0] ⊗ [1,0] = [1,0,0,0]
        stored = dist1.gm_block.cov_blocks[0][frozenset({"y", "X"})]
        np.testing.assert_allclose(stored, np.array([1, 0, 0, 0]), atol=1e-12)

    def test_extract_marginal_moments(self):
        """E[y] = M[i,j], Var[y] = U[i,i]·V[j,j]."""
        from libMatrixUpdate import extract_scalar_from_matrix
        dist0 = self._make_X_2x2_iso()
        dist1 = extract_scalar_from_matrix(dist0, "y", "X", 0, 1)
        assert dist1.var_list == ["y"]
        np.testing.assert_allclose(dist1.gm.mu[0][0], 0.0, atol=1e-12)
        # U[0,0]=1, V[1,1]=1 → Var(y)=1
        np.testing.assert_allclose(dist1.gm.sigma[0][0, 0], 1.0, atol=1e-12)


class TestObserveExtractedScalarBackprop:
    """A2 (research-note 04): observe(y > c) on an extracted scalar updates
    the source matrix's mean via the stored cross-cov.

    Analytic ground truth: for X ~ MN(0, I, I), y = X[i,j] ~ N(0,1),
    E[y | y > 0] = sqrt(2/pi) ≈ 0.79788.
    """

    def _run_program(self, src_text):
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".soga", delete=False) as f:
            f.write(src_text)
            f.flush()
            r = subprocess.run(
                [sys.executable, "SOGA.py", "-f", f.name],
                cwd=SRC, capture_output=True, text=True, timeout=30,
            )
        return r.stdout + r.stderr

    def test_observe_y_propagates_to_matrix_mean(self):
        """E[X[0,0]] after observe(y > 0) ≈ sqrt(2/pi) for X ~ MN(0, I, I)."""
        src = (
            "matrix[2][2] X;\n"
            "X = matrix_gm([[0,0],[0,0]], [[1,0],[0,1]], [[1,0],[0,1]]);\n"
            "y = X[0,0];\n"
            "observe(y > 0);\n"
        )
        out = self._run_program(src)
        assert "E[y]" in out
        assert "E[X]" in out
        # Parse E[y] line
        for line in out.splitlines():
            if "E[y]:" in line:
                e_y = float(line.split(":")[1].strip())
                expected = np.sqrt(2.0 / np.pi)
                assert abs(e_y - expected) < 1e-4, f"E[y]={e_y}, expected={expected}"
        # Parse E[X] block — find the X[0,0] value (first number after E[X]:)
        x_block = out.split("E[X]:")[1].splitlines()
        first_row = None
        for ln in x_block[1:]:
            ln = ln.strip().lstrip("[")
            if ln and ln[0].isdigit() or (ln and ln[0] == "-"):
                first_row = ln
                break
        assert first_row is not None
        x00 = float(first_row.split()[0].rstrip("]"))
        expected = np.sqrt(2.0 / np.pi)
        assert abs(x00 - expected) < 1e-4, f"E[X[0,0]]={x00}, expected={expected}"

    def test_correlated_backprop(self):
        """U with off-diagonal coupling: E[X[1,0]] ≈ 0.5·sqrt(2/pi) after observe(y > 0)
        where y = X[0,0] and U=[[1,0.5],[0.5,1]] (row correlation 0.5)."""
        src = (
            "matrix[2][2] X;\n"
            "X = matrix_gm([[0,0],[0,0]], [[1.0,0.5],[0.5,1.0]], [[1.0,0.0],[0.0,1.0]]);\n"
            "y = X[0,0];\n"
            "observe(y > 0);\n"
        )
        out = self._run_program(src)
        x_block = out.split("E[X]:")[1].splitlines()
        rows = []
        for ln in x_block[1:]:
            stripped = ln.strip().lstrip("[").rstrip("]")
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    rows.append([float(p.rstrip("]")) for p in parts])
                    if len(rows) == 2:
                        break
                except ValueError:
                    pass
        assert len(rows) == 2, f"expected 2 rows, got {rows}"
        c = np.sqrt(2.0 / np.pi)
        # X[0,0] post = c, X[1,0] post = 0.5 * c (Kalman gain via U[0,1]·V[0,0] = 0.5)
        assert abs(rows[0][0] - c) < 1e-3
        assert abs(rows[1][0] - 0.5 * c) < 1e-3
        # X[0,1] and X[1,1] independent → stay 0
        assert abs(rows[0][1]) < 1e-3
        assert abs(rows[1][1]) < 1e-3
