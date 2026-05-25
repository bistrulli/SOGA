"""
analytical_ground_truth.py — Closed-form analytical formulas for matrix-GM ops.

Each function implements the exact formula for an operation on a matrix-variate
Gaussian MN(M, U, V) where vec(X) ~ N(vec(M), V ⊗ U).

Convention: KRONECKER_CONVENTION = "V_outer_U" (matches libMatrixGaussian.py).
    Var(X[i,j])      = U[i,i] * V[j,j]
    Cov(X[i,j], X[i',j']) = U[i,i'] * V[j,j']

All functions return (mean, cov_structure) where cov_structure depends on the op.
For Kronecker-preserving ops: cov_structure = (U_out, V_out).
For extract: cov_structure = (variance_scalar, cross_cov_vec).
For observe/truncate (Tallis): returns (mu_post, sigma_post) scalars.

References:
    Gupta & Nagar (2000), Matrix Variate Distributions, Chapman & Hall.
    Petersen & Pedersen (2012), The Matrix Cookbook §10.
    Tallis (1961), Ann. Math. Stat. 32(1), 223-229 (truncated normal).
    Isserlis (1918), Biometrika 12(1/2), 134-139 (moment formula for products).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# agt_extract — scalar extraction X[i,j]
# ---------------------------------------------------------------------------

def agt_extract(
    M: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    i: int,
    j: int,
) -> tuple:
    """Extract scalar y = X[i,j] from X ~ MN(M, U, V).

    LaTeX formula:
        E[y] = M_{ij}
        Var[y] = U_{ii} V_{jj}
        Cov(y, vec(X)) = (V[:,j] ⊗ U[:,i])  [in V⊗U column-major order]

    Returns
    -------
    mean_y : float
    var_y  : float
    cross_cov : np.ndarray shape (mn,) in column-major V⊗U ordering
    """
    M = np.asarray(M, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    m, n = M.shape

    mean_y = float(M[i, j])
    var_y = float(U[i, i] * V[j, j])

    # Cov(X[i,j], X[i',j']) = U[i,i'] * V[j,j']
    # Cross-cov vec in column-major order: entry (i', j') = U[i,i'] * V[j,j']
    cross_cov = np.zeros(m * n)
    for ip in range(m):
        for jp in range(n):
            # column-major index: ip + jp * m
            cross_cov[ip + jp * m] = U[i, ip] * V[j, jp]

    return mean_y, var_y, cross_cov


# ---------------------------------------------------------------------------
# agt_left_affine — Y = A @ X
# ---------------------------------------------------------------------------

def agt_left_affine(
    A: np.ndarray,
    M: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
) -> tuple:
    """Left affine Y = A @ X for X ~ MN(M, U, V), A deterministic p×m.

    LaTeX formula:
        E[Y] = A M
        U_Y  = A U A^T
        V_Y  = V   (column factor unchanged)

    Returns (M_Y, U_Y, V_Y).
    """
    A = np.asarray(A, dtype=float)
    M = np.asarray(M, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    M_Y = A @ M
    U_Y = A @ U @ A.T
    V_Y = V.copy()
    return M_Y, U_Y, V_Y


# ---------------------------------------------------------------------------
# agt_right_affine — Y = X @ B
# ---------------------------------------------------------------------------

def agt_right_affine(
    M: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    B: np.ndarray,
) -> tuple:
    """Right affine Y = X @ B for X ~ MN(M, U, V), B deterministic n×q.

    LaTeX formula:
        E[Y] = M B
        U_Y  = U   (row factor unchanged)
        V_Y  = B^T V B

    Returns (M_Y, U_Y, V_Y).
    """
    M = np.asarray(M, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    B = np.asarray(B, dtype=float)

    M_Y = M @ B
    U_Y = U.copy()
    V_Y = B.T @ V @ B
    return M_Y, U_Y, V_Y


# ---------------------------------------------------------------------------
# agt_sum — Y = X1 + X2 (independent)
# ---------------------------------------------------------------------------

def agt_sum(
    M1: np.ndarray,
    U1: np.ndarray,
    V1: np.ndarray,
    M2: np.ndarray,
    U2: np.ndarray,
    V2: np.ndarray,
) -> tuple:
    """Sum of two independent X1 ~ MN(M1,U1,V1) and X2 ~ MN(M2,U2,V2).

    LaTeX formula (full-covariance form):
        E[Y]   = M1 + M2
        Cov[Y] = V1 ⊗ U1 + V2 ⊗ U2  (exact dense covariance)

    For Kronecker-preserving cases (isotropic fast paths), the result can be
    expressed as (U_out, V_out).  This function returns the full dense covariance
    to serve as the ground truth for all cases.

    Returns (M_Y, Cov_dense) where Cov_dense is (mn × mn).
    """
    M1 = np.asarray(M1, dtype=float)
    M2 = np.asarray(M2, dtype=float)
    m, n = M1.shape

    M_Y = M1 + M2
    Cov_Y = np.kron(V1, U1) + np.kron(V2, U2)
    return M_Y, Cov_Y


# ---------------------------------------------------------------------------
# agt_transp — Y = X^T
# ---------------------------------------------------------------------------

def agt_transp(
    M: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
) -> tuple:
    """Transpose Y = X^T for X ~ MN(M, U, V).

    LaTeX formula:
        E[Y] = M^T
        U_Y  = V   (row ↔ col swap)
        V_Y  = U

    Returns (M_Y, U_Y, V_Y) where M_Y is n×m, U_Y is n×n, V_Y is m×m.
    """
    M = np.asarray(M, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    M_Y = M.T
    U_Y = V.copy()
    V_Y = U.copy()
    return M_Y, U_Y, V_Y


# ---------------------------------------------------------------------------
# agt_isserlis — Z = X @ Y (two independent random matrices, Isserlis 3-term)
# ---------------------------------------------------------------------------

def agt_isserlis(
    M1: np.ndarray,
    U1: np.ndarray,
    V1: np.ndarray,
    M2: np.ndarray,
    U2: np.ndarray,
    V2: np.ndarray,
) -> tuple:
    """Random×random matmul Z = X @ Y (X m×k, Y k×n), Isserlis 3-term formula.

    X ~ MN(M1, U1, V1) and Y ~ MN(M2, U2, V2) are INDEPENDENT.
    X is m×k, Y is k×n.

    LaTeX (Isserlis / 2nd-moment identity):
        E[Z] = M1 @ M2
        Cov[Z] = A + B + C  where:
            A = tr(V1 @ U2) * (V2 ⊗ U1)
            B = (M2^T V1 M2) ⊗ U1
            C = V2 ⊗ (M1 U2 M1^T)

    This is the exact formula implemented in libMatrixGaussian.matmul_independent.
    Returns (M_Z, Cov_dense) where Cov_dense is (mn × mn).

    Note: a nearest-Kronecker projection is applied in production code to
    get (U_Z, V_Z); here we return the exact dense covariance for ground truth.
    """
    M1 = np.asarray(M1, dtype=float)
    U1 = np.asarray(U1, dtype=float)
    V1 = np.asarray(V1, dtype=float)
    M2 = np.asarray(M2, dtype=float)
    U2 = np.asarray(U2, dtype=float)
    V2 = np.asarray(V2, dtype=float)

    M_Z = M1 @ M2

    # Term A: tr(V1 @ U2) * (V2 ⊗ U1)
    sA = np.trace(V1 @ U2)
    CovA = sA * np.kron(V2, U1)

    # Term B: (M2^T @ V1 @ M2) ⊗ U1
    MtVM = M2.T @ V1 @ M2
    CovB = np.kron(MtVM, U1)

    # Term C: V2 ⊗ (M1 @ U2 @ M1^T)
    MxUYMxT = M1 @ U2 @ M1.T
    CovC = np.kron(V2, MxUYMxT)

    Cov_Z = CovA + CovB + CovC
    return M_Z, Cov_Z


# ---------------------------------------------------------------------------
# agt_schur_write — X[i,j] = c (element write via Schur complement)
# ---------------------------------------------------------------------------

def agt_schur_write(
    M: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    i: int,
    j: int,
    c: float,
) -> tuple:
    """Assign X[i,j] = c (deterministic constant).

    The element write is modelled as an observe-and-condition on the scalar
    y = X[i,j] == c.  This is a degenerate conditioning (delta observation).

    LaTeX (Schur complement / linear Gaussian conditioning):
        Prior:  [y, X_rest] joint Gaussian
        Condition on y = c.

        E[X[i',j'] | y=c] = M[i',j'] + Cov(X[i',j'], y) / Var(y) * (c - M[i,j])
                          = M[i',j'] + (U[i',i] * V[j',j]) / (U[i,i] * V[j,j]) * (c - M[i,j])

        Var[X[i',j'] | y=c] = Var[X[i',j']] - Cov(X[i',j'], y)^2 / Var(y)
                             = U[i',i'] V[j',j'] - (U[i',i] V[j',j])^2 / (U[i,i] V[j,j])

    In Kronecker form the updated covariance is no longer exactly Kronecker after
    the Schur complement densification.  Production code densifies and re-projects.
    This function returns (M_out, Cov_dense_out) for exact ground truth.

    Returns (M_out, Cov_dense_out) where M_out is (m, n) and Cov_dense_out is (mn, mn).
    """
    M = np.asarray(M, dtype=float).copy()
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    m, n = M.shape

    # Dense covariance before write
    Cov = np.kron(V, U)

    # Index of (i, j) in column-major vectorization
    idx = i + j * m  # column-major: vec(X)[i + j*m] = X[i,j]

    var_y = float(Cov[idx, idx])
    assert var_y > 1e-14, f"Var(X[{i},{j}]) is near-zero: {var_y}"

    # Update vec(M): E[X | y=c] = vec(M) + Cov[:,idx] / var_y * (c - M[i,j])
    vec_M = M.flatten(order='F')
    delta = c - vec_M[idx]
    vec_M_out = vec_M + (Cov[:, idx] / var_y) * delta

    # Update Cov: Cov_out = Cov - Cov[:,idx:idx+1] @ Cov[idx:idx+1,:] / var_y
    Cov_out = Cov - np.outer(Cov[:, idx], Cov[idx, :]) / var_y

    M_out = vec_M_out.reshape((m, n), order='F')
    return M_out, Cov_out


# ---------------------------------------------------------------------------
# agt_tallis_truncate — truncated normal (scalar observe y > c)
# ---------------------------------------------------------------------------

def agt_tallis_truncate(
    mu: float,
    sigma: float,
    a: float,
    b: float = np.inf,
) -> tuple:
    """Moments of truncated normal Y ~ N(mu, sigma^2) conditioned on a < Y <= b.

    LaTeX (Tallis 1961 / Cohen 1991):
        Let alpha = (a - mu) / sigma,  beta = (b - mu) / sigma.
        Let phi = standard normal PDF,  Phi = standard normal CDF.
        Z = Phi(beta) - Phi(alpha)  (truncation mass).

        E[Y | a < Y <= b] = mu + sigma * (phi(alpha) - phi(beta)) / Z
        Var[Y | a < Y <= b] = sigma^2 * (
            1 + (alpha*phi(alpha) - beta*phi(beta)) / Z
              - ((phi(alpha) - phi(beta)) / Z)^2
        )

    Parameters
    ----------
    mu : float
        Prior mean.
    sigma : float
        Prior standard deviation (sigma > 0).
    a : float
        Lower bound (can be -inf).
    b : float
        Upper bound (default +inf).

    Returns
    -------
    mu_post : float
    var_post : float
    weight : float  — renormalization mass Z (probability of the constraint)
    """
    assert sigma > 0, f"sigma must be > 0, got {sigma}"

    alpha = (a - mu) / sigma if not np.isinf(a) else -np.inf
    beta = (b - mu) / sigma if not np.isinf(b) else np.inf

    phi_a = float(stats.norm.pdf(alpha)) if not np.isinf(alpha) else 0.0
    phi_b = float(stats.norm.pdf(beta)) if not np.isinf(beta) else 0.0
    Phi_a = float(stats.norm.cdf(alpha)) if not np.isinf(alpha) else 0.0
    Phi_b = float(stats.norm.cdf(beta)) if not np.isinf(beta) else 1.0

    Z = Phi_b - Phi_a
    if Z < 1e-300:
        raise ValueError(f"Truncation mass Z={Z:.3e} is effectively zero")

    mu_post = mu + sigma * (phi_a - phi_b) / Z
    var_post = sigma ** 2 * (
        1.0
        + (alpha * phi_a - beta * phi_b) / Z
        - ((phi_a - phi_b) / Z) ** 2
    )
    return float(mu_post), float(var_post), float(Z)
