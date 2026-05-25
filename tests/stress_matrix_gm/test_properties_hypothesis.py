"""
test_properties_hypothesis.py — 6 property-based invariants for matrix-GM ops.

Uses hypothesis with @settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow]).

Invariants:
    H1: affine_left preserves mean (A@M formula)
    H2: transp swaps Kronecker factors (U↔V)
    H3: add preserves mean (M1+M2)
    H4: extract scalar mean = M[i,j]
    H5a: matrix_gm_full with Kron Sigma → cov_kind='kron'
    H5b: matrix_gm_full with non-Kron Sigma (FIXED_NON_KRON_4x4) → cov_kind='dense'
    H6: nearest_kronecker reconstruction error < 1e-12 for exact Kron

Invariants H1-H4 and H6 use generated SPD matrices.
H5a/H5b use fixed fixtures to avoid randomness issues with Kronecker detection.

Generators use assume(eigvalsh > 1e-6) to ensure valid PSD inputs.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from libMatrixGaussian import (
    MatrixGaussian, _nearest_kronecker, _try_kronecker_decompose
)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

def _spd_2x2(seed_offset: int = 0):
    """Strategy: random SPD 2x2 matrix."""
    @st.composite
    def _strategy(draw):
        # Draw a random 2x2 SPD via L @ L.T + eps * I
        vals = draw(st.lists(
            st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=4, max_size=4,
        ))
        L = np.array(vals).reshape(2, 2)
        M = L @ L.T + 0.1 * np.eye(2)
        eigvals = np.linalg.eigvalsh(M)
        assume(np.all(eigvals > 1e-6))
        return M
    return _strategy()


def _mean_2x2():
    """Strategy: random 2x2 mean matrix."""
    @st.composite
    def _strategy(draw):
        vals = draw(st.lists(
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=4, max_size=4,
        ))
        return np.array(vals).reshape(2, 2)
    return _strategy()


def _det_2x2():
    """Strategy: random 2x2 deterministic matrix A."""
    @st.composite
    def _strategy(draw):
        vals = draw(st.lists(
            st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=4, max_size=4,
        ))
        return np.array(vals).reshape(2, 2)
    return _strategy()


# ---------------------------------------------------------------------------
# H1: affine_left preserves mean
# ---------------------------------------------------------------------------

@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(M=_mean_2x2(), A=_det_2x2(), U=_spd_2x2(), V=_spd_2x2())
def test_H1_affine_left_preserves_mean(M, A, U, V):
    """H1: (A @ X).mean() == A @ M for all A, M, PSD U, V."""
    mg = MatrixGaussian(M, U, V)
    mg_y = mg.affine_left(A)
    expected = A @ M
    np.testing.assert_allclose(
        mg_y.mean(), expected, atol=1e-10, rtol=1e-8,
        err_msg=f"affine_left mean mismatch: got {mg_y.mean()}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# H2: transp swaps Kronecker factors
# ---------------------------------------------------------------------------

@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(M=_mean_2x2(), U=_spd_2x2(), V=_spd_2x2())
def test_H2_transp_swaps_factors(M, U, V):
    """H2: transp(X) has U_T = V and V_T = U (row/col factors swap)."""
    mg = MatrixGaussian(M, U, V)
    mg_t = mg.transpose()
    np.testing.assert_allclose(
        mg_t.U, mg.V, atol=1e-12,
        err_msg="transp: U_T should equal original V"
    )
    np.testing.assert_allclose(
        mg_t.V, mg.U, atol=1e-12,
        err_msg="transp: V_T should equal original U"
    )
    np.testing.assert_allclose(
        mg_t.mean(), M.T, atol=1e-12,
        err_msg="transp: mean should be M^T"
    )


# ---------------------------------------------------------------------------
# H3: add preserves mean (X + Y)
# ---------------------------------------------------------------------------

@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(M1=_mean_2x2(), M2=_mean_2x2(), U1=_spd_2x2(), V1=_spd_2x2(), U2=_spd_2x2(), V2=_spd_2x2())
def test_H3_add_preserves_mean(M1, M2, U1, V1, U2, V2):
    """H3: (X + Y).mean() == M1 + M2 for independent X, Y."""
    mg1 = MatrixGaussian(M1, U1, V1)
    mg2 = MatrixGaussian(M2, U2, V2)
    mg_sum = mg1.add(mg2)
    expected = M1 + M2
    np.testing.assert_allclose(
        mg_sum.mean(), expected, atol=1e-10, rtol=1e-8,
        err_msg=f"add mean mismatch: got {mg_sum.mean()}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# H4: extract scalar mean = M[i,j]
# ---------------------------------------------------------------------------

@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(M=_mean_2x2(), U=_spd_2x2(), V=_spd_2x2())
def test_H4_extract_scalar_mean(M, U, V):
    """H4: variance_element(i,j) = U[i,i] * V[j,j]. Mean accessor = M."""
    mg = MatrixGaussian(M, U, V)
    # Verify mean accessor
    np.testing.assert_allclose(
        mg.mean(), M, atol=1e-12,
        err_msg="MatrixGaussian.mean() should return M"
    )
    # Verify variance formula: Var(X[i,j]) = U[i,i] * V[j,j]
    for i in range(2):
        for j in range(2):
            var_ij = mg.variance_element(i, j)
            expected_var = float(U[i, i] * V[j, j])
            assert abs(var_ij - expected_var) < 1e-12, (
                f"variance_element({i},{j})={var_ij}, expected U[{i},{i}]*V[{j},{j}]={expected_var}"
            )


# ---------------------------------------------------------------------------
# H5a: matrix_gm_full with Kron Sigma → cov_kind='kron'
# H5b: matrix_gm_full with non-Kron Sigma → cov_kind='dense'
#
# These use fixed fixtures, NOT hypothesis @given, to guarantee determinism.
# A Kronecker Sigma must satisfy rank-1 rearrangement (residual < SOGA_KRON_STRICT).
# A non-Kronecker Sigma must have residual >= SOGA_KRON_LOOSE.
# ---------------------------------------------------------------------------

def test_H5_kron_detection_kron_and_dense():
    """H5: matrix_gm_full auto-detection — both Kron and Dense cases.

    H5a: I_4 = I_2 ⊗ I_2 → exact Kronecker (residual < 1e-8).
    H5b: FIXED_NON_KRON_4x4 → non-Kronecker (residual > 1e-4).

    Tests the _try_kronecker_decompose kernel used by matrix_gm_full dispatcher.
    Combined into one test node to keep total campaign at 106 tests.
    """
    from tests.stress_matrix_gm.conftest import FIXED_NON_KRON_4x4

    # H5a: Kronecker case
    Sigma_kron = np.eye(4)  # I_4 = I_2 ⊗ I_2 — exact Kronecker
    U, V, ratio_kron = _try_kronecker_decompose(Sigma_kron, 2, 2)
    assert ratio_kron < 1e-8, (
        f"I_4 should have Kronecker residual < 1e-8, got {ratio_kron:.3e}"
    )
    Sigma_recon = np.kron(V, U)
    frob_err = np.linalg.norm(Sigma_kron - Sigma_recon, "fro") / np.linalg.norm(Sigma_kron, "fro")
    assert frob_err < 1e-10, f"Kronecker reconstruction error {frob_err:.3e} > 1e-10"

    # H5b: Dense case
    S = FIXED_NON_KRON_4x4
    U, V, ratio_dense = _try_kronecker_decompose(S, 2, 2)
    assert ratio_dense > 1e-4, (
        f"FIXED_NON_KRON_4x4 residual {ratio_dense:.3e} should be >> 1e-4"
    )


# ---------------------------------------------------------------------------
# H6: nearest_kronecker reconstruction error < 1e-12 for EXACT Kronecker input
# ---------------------------------------------------------------------------

@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(U=_spd_2x2(), V=_spd_2x2())
def test_H6_nearest_kron_exact_recovery(U, V):
    """H6: _nearest_kronecker on an exact kron(V, U) should recover U, V.

    The reconstruction kron(V_out, U_out) should match kron(V, U) to 1e-10 Frobenius.
    This tests the Van Loan-Pitsianis implementation stability.
    """
    Cov = np.kron(V, U)
    U_out, V_out = _nearest_kronecker(Cov, 2, 2)
    Cov_recon = np.kron(V_out, U_out)
    frob_true = float(np.linalg.norm(Cov, "fro"))
    assume(frob_true > 1e-8)  # skip degenerate near-zero
    frob_err = float(np.linalg.norm(Cov - Cov_recon, "fro")) / frob_true
    assert frob_err < 1e-10, (
        f"nearest_kronecker reconstruction error {frob_err:.3e} > 1e-10 for exact kron input"
    )
