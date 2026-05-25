"""
Matrix-Variate Gaussian math kernel for SOGA — production module.

X is matrix-Gaussian with M (m×n) mean and Kronecker-product covariance:

    vec(X) ~ N(vec(M), V ⊗ U)

where U is m×m (row covariance) and V is n×n (column covariance).
Convention: KRONECKER_CONVENTION = "V_outer_U" — V is the *left* factor,
U is the *right* factor in every kron(V, U) call in this module.

Storage cost: m² + n² entries vs (mn)² for the dense form.
At m=n=32: 2048 vs 1048576 → ~512× saving.

Closed-form exact operations implemented:
- scale(c)                    → c·X ~ MN(c·M, c²·U, V)
- transpose()                 → X^T ~ MN(M^T, V, U)
- add_constant(B)             → X + B ~ MN(M + B, U, V)
- affine_left(A)              → A·X ~ MN(A·M, A·U·A^T, V)
- affine_right(B)             → X·B ~ MN(M·B, U, B^T·V·B)
- affine(A, B, C=None)        → A·X·B + C

Approximate operations (nearest-Kronecker projection):
- add(other)                  → X + Y with general covariance projected to Kronecker
- matmul_independent(other)   → X·Y (two random matrices, moment-matched)

Isotropic fast paths in add() avoid densifying kron(V, U) when one operand
has isotropic row or column factor — critical to avoid 800 MB blowup at K=50
components, m=n=32.

References:
- Gupta & Nagar (2000), Matrix Variate Distributions, Chapman & Hall.
- Petersen & Pedersen (2012), The Matrix Cookbook §10.
- Van Loan & Pitsianis (1993), Approximation with Kronecker products.

Moved from experiments/matrix_gm_poc_2026-05-21/matrix_gm.py into production
src/ as part of M0.3. PoC original preserved in-place for offline validation.
"""

from __future__ import annotations

import logging
import warnings
import numpy as np

logger = logging.getLogger(__name__)

# Convention assertion: every np.kron(V, U) site in this module uses V as the
# LEFT factor and U as the RIGHT factor.  Assert this invariant with a module
# constant so downstream code can validate against it.
KRONECKER_CONVENTION = "V_outer_U"


# ---------------------------------------------------------------------------
# Warning classes
# ---------------------------------------------------------------------------

class IsotropyNearThresholdWarning(UserWarning):
    """Emitted when a matrix is near-isotropic within 10× the tolerance."""


class KroneckerApproxWarning(UserWarning):
    """Emitted when a nearest-Kronecker projection exceeds an error threshold."""


class KroneckerDetectionInfo(UserWarning):
    """Informational: matrix_gm_full detected exact Kronecker structure.

    Emitted when residual_ratio < SOGA_KRON_STRICT and the covariance is
    stored as efficient (U, V) Kronecker factors.
    """


class KroneckerNearMissWarning(UserWarning):
    """Emitted when residual is between SOGA_KRON_STRICT and SOGA_KRON_LOOSE.

    The covariance is stored as dense for safety, but it is close to Kronecker.
    Check whether the Sigma was intended to be exactly Kronecker.
    """


class DenseCovarianceInfo(UserWarning):
    """Informational: matrix_gm_full detected non-Kronecker structure.

    Emitted when residual_ratio >= SOGA_KRON_LOOSE and the covariance is
    stored as a dense sentinel (None, Sigma).
    """


class StaleCrossCovWarning(UserWarning):
    """Scalar variable extracted before an observe/write on a correlated
    matrix variable; its moments are now stale (Gap3/F4).

    Emitted when a scalar variable has a non-zero cross-covariance with a
    matrix variable that is about to be conditioned (via observe or element
    write).  The scalar's moments will NOT be updated by the operation —
    they remain at the pre-condition value.

    To avoid stale moments: extract scalar variables AFTER any observe or
    element-write operations on correlated matrix variables.
    See docs/LIMITATIONS.md §Gap3 and §F4.
    """


# ---------------------------------------------------------------------------
# Isotropy helper
# ---------------------------------------------------------------------------

def _is_isotropic(M: np.ndarray, atol: float = 1e-10, rtol: float = 1e-8) -> bool:
    """Return True if M is a positive scalar multiple of the identity.

    Tolerance chosen at atol=1e-10, rtol=1e-8 to avoid misrouting after chains
    of affine ops with condition number ~1e4 (diagonal drift ~2e-11 can fail
    stricter 1e-12).  If the deviation is within 10× of the tolerance boundary,
    emit IsotropyNearThresholdWarning at DEBUG level for diagnostics.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    c = M[0, 0]
    target = c * np.eye(M.shape[0])
    deviation = float(np.max(np.abs(M - target)))
    scale = float(np.max(np.abs(M))) if np.max(np.abs(M)) > 0 else 1.0
    threshold = atol + rtol * scale
    if deviation <= threshold:
        return True
    # Near-threshold diagnostic
    if deviation <= 10.0 * threshold:
        warnings.warn(
            IsotropyNearThresholdWarning(
                f"_is_isotropic near threshold: deviation={deviation:.3e}, "
                f"threshold={threshold:.3e}, variable shape={M.shape}"
            ),
            stacklevel=3,
        )
    return False


# ---------------------------------------------------------------------------
# Nearest-Kronecker decomposition
# ---------------------------------------------------------------------------

def _nearest_kronecker(Cov: np.ndarray, m: int, n: int):
    """Find U (m×m), V (n×n) such that Cov ≈ V ⊗ U, minimising Frobenius
    error. Algorithm: Van Loan & Pitsianis (1993).

    Sign disambiguation: first-nonzero-positive convention on u_vec BEFORE
    eigenvalue clipping.  This replaces the PoC 'trace(U) < 0' heuristic which
    fails when U is indefinite but has positive trace.
    """
    # Rearrange Cov (mn × mn) into R (m², n²) via the V ⊗ U convention.
    # Block (j, j') of Cov is Cov[j*m:(j+1)*m, j'*m:(j'+1)*m] = V[j,j'] * U.
    R = np.zeros((m * m, n * n))
    for j in range(n):
        for jj in range(n):
            block = Cov[j * m:(j + 1) * m, jj * m:(jj + 1) * m]  # m × m
            R[:, j * n + jj] = block.flatten('F')

    Us, sigma_sv, Vt = np.linalg.svd(R, full_matrices=False)
    s = sigma_sv[0]
    u_vec = Us[:, 0] * np.sqrt(s)
    v_vec = Vt[0, :] * np.sqrt(s)

    # Sign disambiguation: first-nonzero-positive convention on u_vec.
    # Find the first element of u_vec with |value| > 1e-14; if it is negative,
    # flip both u_vec and v_vec so the sign is absorbed consistently.
    for val in u_vec:
        if abs(val) > 1e-14:
            if val < 0:
                u_vec = -u_vec
                v_vec = -v_vec
            break

    U = u_vec.reshape((m, m), order='F')
    V = v_vec.reshape((n, n), order='F')

    # Force symmetry
    U = 0.5 * (U + U.T)
    V = 0.5 * (V + V.T)

    # Clip negative eigenvalues to zero (PSD enforcement on Kronecker factors)
    for K in (U, V):
        w, Q = np.linalg.eigh(K)
        w = np.clip(w, 0.0, None)
        K[:] = Q @ np.diag(w) @ Q.T

    return U, V


# ---------------------------------------------------------------------------
# Kronecker decompose helper (B.4) — auto-detect for matrix_gm_full
# ---------------------------------------------------------------------------

def _try_kronecker_decompose(
    Sigma: np.ndarray,
    m: int,
    n: int,
    tol: float = 1e-8,
) -> tuple:
    """Test whether Sigma (mn × mn) is Kronecker-separable and decompose.

    Uses the Van Loan-Pitsianis rank-1 SVD on the rearrangement R[Sigma].
    The residual ratio is s_2 / s_1 (second / first singular value of R).
    If R has rank exactly 1 the ratio is 0 (within machine precision).

    Parameters
    ----------
    Sigma : np.ndarray shape (mn, mn)
        Full covariance to test.
    m : int
        Row dimension of the matrix variable.
    n : int
        Column dimension of the matrix variable.
    tol : float
        Residual-ratio threshold.  If residual < tol the decomposition is
        considered exact and the Kronecker factors (U, V) are returned.
        Configurable via env var SOGA_KRON_STRICT (default 1e-8).

    Returns
    -------
    U : np.ndarray (m, m)   — row Kronecker factor (best-fit)
    V : np.ndarray (n, n)   — column Kronecker factor (best-fit)
    residual_ratio : float  — s_2 / s_1 (0 ≤ ratio ≤ 1)
        ratio ≈ 0 → exact Kronecker (within numerical precision)
        0 < ratio < tol → within threshold (still stored as Kronecker)
        ratio >= tol → not separable (stored as dense)

    Notes
    -----
    The (U, V) factors are always returned regardless of the residual; the
    caller decides whether to use them based on the returned ratio.
    Sign disambiguation and PSD clipping are applied by _nearest_kronecker.
    """
    # Build rearrangement matrix R (m² × n²)
    R = np.zeros((m * m, n * n))
    for j in range(n):
        for jj in range(n):
            block = Sigma[j * m:(j + 1) * m, jj * m:(jj + 1) * m]
            R[:, j * n + jj] = block.flatten('F')

    _, sigma_sv, _ = np.linalg.svd(R, full_matrices=False)

    # Residual ratio: s_2 / s_1
    s1 = float(sigma_sv[0]) if len(sigma_sv) > 0 else 0.0
    s2 = float(sigma_sv[1]) if len(sigma_sv) > 1 else 0.0
    if s1 < 1e-14:
        # Near-zero Sigma: trivially Kronecker (zero matrix)
        residual_ratio = 0.0
    else:
        residual_ratio = s2 / s1

    # Always extract best (U, V) via _nearest_kronecker
    U, V = _nearest_kronecker(Sigma, m, n)
    return U, V, residual_ratio


# ---------------------------------------------------------------------------
# MatrixGaussian class
# ---------------------------------------------------------------------------

class MatrixGaussian:
    """X ~ MN(M, U, V): X in R^{m x n} with vec(X) ~ N(vec(M), V ⊗ U).

    Convention (KRONECKER_CONVENTION = "V_outer_U"):
    - Covariance of vec(X) is V ⊗ U.
    - Per-element variance: Var(X[i,j]) = U[i,i] * V[j,j].
    - Covariance between X[i,j] and X[i',j']: Cov = U[i,i'] * V[j,j'].

    Storage: M (m × n), U (m × m), V (n × n).
    """

    def __init__(self, M: np.ndarray, U: np.ndarray, V: np.ndarray):
        M = np.asarray(M, dtype=float)
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        m, n = M.shape
        assert U.shape == (m, m), f"U should be {m}x{m}, got {U.shape}"
        assert V.shape == (n, n), f"V should be {n}x{n}, got {V.shape}"
        self.M = M
        self.U = U
        self.V = V

    @property
    def shape(self):
        return self.M.shape

    @property
    def m(self):
        return self.M.shape[0]

    @property
    def n(self):
        return self.M.shape[1]

    def mean(self) -> np.ndarray:
        return self.M.copy()

    def variance_element(self, i: int, j: int) -> float:
        return float(self.U[i, i] * self.V[j, j])

    def variance_matrix(self) -> np.ndarray:
        """Element-wise variance: Var(X[i,j]) for every (i,j)."""
        return np.outer(np.diag(self.U), np.diag(self.V))

    def covariance_vec(self) -> np.ndarray:
        """Full vectorised covariance: V ⊗ U in R^{(m·n) x (m·n)}.

        Used for validation only; do not call on large m·n.
        Assertion: convention is KRONECKER_CONVENTION = "V_outer_U".
        """
        assert KRONECKER_CONVENTION == "V_outer_U"
        return np.kron(self.V, self.U)

    def __repr__(self):
        return f"MN(M={self.shape}, |U|={self.U.shape}, |V|={self.V.shape})"

    # ----- closed-form exact operations -----

    def scale(self, c: float) -> "MatrixGaussian":
        """c · X ~ MN(c·M, c²·U, V).  Scale absorbed into U per convention."""
        return MatrixGaussian(c * self.M, (c * c) * self.U, self.V.copy())

    def transpose(self) -> "MatrixGaussian":
        """X^T ~ MN(M^T, V, U).  Row/column factors swap."""
        return MatrixGaussian(self.M.T, self.V.copy(), self.U.copy())

    def add_constant(self, B: np.ndarray) -> "MatrixGaussian":
        """X + B (deterministic B) ~ MN(M + B, U, V).  Covariance unchanged."""
        B = np.asarray(B, dtype=float)
        assert B.shape == self.shape
        return MatrixGaussian(self.M + B, self.U.copy(), self.V.copy())

    def affine_left(self, A: np.ndarray) -> "MatrixGaussian":
        """A · X ~ MN(A·M, A·U·A^T, V).  A deterministic, p×m."""
        A = np.asarray(A, dtype=float)
        return MatrixGaussian(A @ self.M, A @ self.U @ A.T, self.V.copy())

    def affine_right(self, B: np.ndarray) -> "MatrixGaussian":
        """X · B ~ MN(M·B, U, B^T·V·B).  B deterministic, n×q."""
        B = np.asarray(B, dtype=float)
        return MatrixGaussian(self.M @ B, self.U.copy(), B.T @ self.V @ B)

    def affine(self, A: np.ndarray, B: np.ndarray, C: np.ndarray = None) -> "MatrixGaussian":
        """A · X · B + C ~ MN(A·M·B + C, A·U·A^T, B^T·V·B)."""
        out = self.affine_left(A).affine_right(B)
        if C is not None:
            out = out.add_constant(C)
        return out

    def add(self, other: "MatrixGaussian") -> "MatrixGaussian":
        """X + Y (both matrix-Gaussian, INDEPENDENT).

        Isotropic fast paths avoid materializing kron(V, U) when possible:

        Case A (exact): U_X = c·I_m AND other.U = a·I_m, other.V = b·I_n.
            Cov_sum = V_X ⊗ (c·I) + (b·I) ⊗ (a·I)
                    = V_X ⊗ (c·I) + (a·b)·I_{mn}
                    = (V_X + (a·b/c)·I_n) ⊗ (c·I_m)
            → V_new = V_X + (a·b/c)·I_n, U_new = U_X. No SVD.

        Case B (exact): V_X = d·I_n AND other is fully isotropic.
            → U_new = U_X + (a·b/d)·I_m, V_new = V_X. No SVD.

        General (approximate): densify kron(V, U) and use nearest-Kronecker.
        Projection error is logged; KroneckerApproxWarning emitted if > 5%.
        """
        assert self.shape == other.shape
        assert KRONECKER_CONVENTION == "V_outer_U"
        M_out = self.M + other.M

        # Case A: U_X is isotropic AND other is fully isotropic
        if _is_isotropic(self.U) and _is_isotropic(other.U) and _is_isotropic(other.V):
            c = float(self.U[0, 0])
            a = float(other.U[0, 0])
            b = float(other.V[0, 0])
            if c > 0:
                V_new = self.V + (a * b / c) * np.eye(self.n)
                U_new = self.U.copy()
                logger.debug("MatrixGaussian.add: isotropic fast path A")
                return MatrixGaussian(M_out, U_new, V_new)

        # Case B: V_X is isotropic AND other is fully isotropic
        if _is_isotropic(self.V) and _is_isotropic(other.U) and _is_isotropic(other.V):
            d = float(self.V[0, 0])
            a = float(other.U[0, 0])
            b = float(other.V[0, 0])
            if d > 0:
                U_new = self.U + (a * b / d) * np.eye(self.m)
                V_new = self.V.copy()
                logger.debug("MatrixGaussian.add: isotropic fast path B")
                return MatrixGaussian(M_out, U_new, V_new)

        # General: full densification + nearest-Kronecker projection
        logger.debug("MatrixGaussian.add: general path (nearest-Kronecker projection)")
        assert KRONECKER_CONVENTION == "V_outer_U"
        Cov_full = np.kron(self.V, self.U) + np.kron(other.V, other.U)
        U_out, V_out = _nearest_kronecker(Cov_full, self.m, self.n)

        # Log projection error
        Cov_approx = np.kron(V_out, U_out)
        norm_full = float(np.linalg.norm(Cov_full, 'fro'))
        if norm_full > 0:
            eps_proj = float(np.linalg.norm(Cov_full - Cov_approx, 'fro')) / norm_full
            if eps_proj > 0.05:
                warnings.warn(
                    KroneckerApproxWarning(
                        f"MatrixGaussian.add projection error {eps_proj:.3%} > 5%"
                    ),
                    stacklevel=2,
                )
            logger.debug(f"MatrixGaussian.add projection error: {eps_proj:.3e}")

        return MatrixGaussian(M_out, U_out, V_out)

    def matmul_independent(self, other: "MatrixGaussian") -> "MatrixGaussian":
        """X · Y (both random, INDEPENDENT) — moment-matched nearest matrix-Gaussian.

        X is m×k, Y is k×n; result is m×n.
        E[X·Y] = E[X] · E[Y] = M_X · M_Y.
        Exact covariance computed, then projected to nearest Kronecker form.
        """
        assert self.shape[1] == other.shape[0]
        assert KRONECKER_CONVENTION == "V_outer_U"
        m, k = self.shape
        _, n = other.shape
        M_X, U_X, V_X = self.M, self.U, self.V
        M_Y, U_Y, V_Y = other.M, other.U, other.V
        M_out = M_X @ M_Y

        # Term A: U_X[i,i'] * trace(V_X · U_Y) * V_Y[j,j']
        sA = np.trace(V_X @ U_Y)
        CovA = sA * np.kron(V_Y, U_X)

        # Term B: U_X[i,i'] * (M_Y^T · V_X · M_Y)[j,j']
        MtVM = M_Y.T @ V_X @ M_Y
        CovB = np.kron(MtVM, U_X)

        # Term C: V_Y[j,j'] * (M_X · U_Y · M_X^T)[i,i']
        MxUYMxT = M_X @ U_Y @ M_X.T
        CovC = np.kron(V_Y, MxUYMxT)

        Cov = CovA + CovB + CovC
        U_out, V_out = _nearest_kronecker(Cov, m, n)
        return MatrixGaussian(M_out, U_out, V_out)


# ---------------------------------------------------------------------------
# Helpers for testing and cross-validation
# ---------------------------------------------------------------------------

def to_scalar_dense(mg: MatrixGaussian) -> tuple[np.ndarray, np.ndarray]:
    """Convert to (vec_mean, full_dense_cov) for cross-validation.

    Returns vec(M) in column-major ('F') order and V ⊗ U dense matrix.
    """
    return mg.M.flatten('F'), mg.covariance_vec()


def from_scalar_dense(mean_vec: np.ndarray, cov_dense: np.ndarray, m: int, n: int) -> MatrixGaussian:
    """Reconstruct best matrix-Gaussian approximation from dense covariance."""
    M = mean_vec.reshape((m, n), order='F')
    U, V = _nearest_kronecker(cov_dense, m, n)
    return MatrixGaussian(M, U, V)
