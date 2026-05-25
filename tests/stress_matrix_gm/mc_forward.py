"""
mc_forward.py — Forward Monte Carlo sampler for matrix-variate Gaussian.

Implements vectorized sampling of X ~ MN(M, U, V) via the Kronecker
covariance Sigma = kron(V, U) + jitter.

Public API:
    sample_MN(M, U, V, n_samples, rng) -> np.ndarray shape (n_samples, m, n)

Sampling strategy:
    1. Build Sigma = kron(V, U) + 1e-12 * I_{mn}  (jitter for PSD stability)
    2. Cholesky decompose Sigma = L @ L.T
    3. Draw z ~ N(0, I_{mn}) per sample
    4. Compute vec(X) = vec(M) + L @ z
    5. Reshape from vec (column-major F order) to (m, n)

The 1e-12 jitter absorbs floating-point imprecision in kron(V, U) without
meaningfully perturbing the distribution (it is < 1e-10 of any test covariance).

Unit tests (inline, run with pytest):
    test_sample_MN_mean_convergence  — sample mean converges to M
    test_sample_MN_cov_reconstruction — sample covariance reconstructs kron(V, U)
    test_sample_MN_1x1_degenerate    — 1x1 case: scalar Gaussian
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def sample_MN(
    M: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample X ~ MN(M, U, V) via Cholesky decomposition of kron(V, U).

    Parameters
    ----------
    M : np.ndarray shape (m, n)
        Mean matrix.
    U : np.ndarray shape (m, m)
        Row covariance factor (Kronecker right factor).
    V : np.ndarray shape (n, n)
        Column covariance factor (Kronecker left factor).
    n_samples : int
        Number of iid samples to draw.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    samples : np.ndarray shape (n_samples, m, n)
        Array of iid matrix-variate Gaussian samples.

    Notes
    -----
    Convention: vec(X) ~ N(vec(M), V ⊗ U).
    This matches KRONECKER_CONVENTION = "V_outer_U" in libMatrixGaussian.py.
    The 1e-12 jitter is applied to the full covariance for numerical stability;
    it is absorbed by Cholesky without affecting the distribution meaningfully.
    """
    M = np.asarray(M, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    m, n = M.shape
    mn = m * n

    # Build full Kronecker covariance + jitter
    Sigma = np.kron(V, U) + 1e-12 * np.eye(mn)

    # Cholesky
    L = np.linalg.cholesky(Sigma)

    # Draw standard normal noise: shape (mn, n_samples)
    Z = rng.standard_normal((mn, n_samples))

    # vec(X) samples: shape (mn, n_samples)
    vec_M = M.flatten(order='F')  # column-major vec convention
    vec_X = vec_M[:, None] + L @ Z  # (mn, n_samples)

    # Reshape each column to (m, n) using column-major order
    # Result shape: (n_samples, m, n)
    samples = vec_X.T.reshape(n_samples, n, m).transpose(0, 2, 1)
    # Note: vec_X is in column-major order, so vec_X[k] has shape (mn,)
    # Reshape to (m, n) via Fortran (column-major) order:
    samples = np.array([
        col.reshape((m, n), order='F') for col in vec_X.T
    ])
    return samples  # shape (n_samples, m, n)


# ---------------------------------------------------------------------------
# Inline unit tests
# ---------------------------------------------------------------------------

def test_sample_MN_mean_convergence():
    """Sample mean should converge to M within 3-sigma bound (N=1e5)."""
    rng = np.random.default_rng(42)
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    n_samples = 100_000

    samples = sample_MN(M, U, V, n_samples, rng)
    assert samples.shape == (n_samples, 2, 2), f"Shape mismatch: {samples.shape}"

    sample_mean = samples.mean(axis=0)
    # Each entry X[i,j] ~ N(M[i,j], U[i,i]*V[j,j]) = N(M[i,j], 1)
    # 3-sigma bound on mean error: 3/sqrt(N) = 3/sqrt(1e5) ≈ 0.0095
    tol = 4.0 / np.sqrt(n_samples)  # 4-sigma conservative
    np.testing.assert_allclose(
        sample_mean, M, atol=tol,
        err_msg=f"Sample mean did not converge to M within {tol:.4f}"
    )


def test_sample_MN_cov_reconstruction():
    """Sample covariance should reconstruct kron(V, U) within tolerance."""
    rng = np.random.default_rng(42)
    M = np.array([[0.0, 0.0], [0.0, 0.0]])
    U = np.array([[2.0, 0.5], [0.5, 1.0]])
    V = np.array([[3.0, 1.0], [1.0, 2.0]])
    n_samples = 100_000

    samples = sample_MN(M, U, V, n_samples, rng)
    # Flatten each sample in F-order to get vec(X)
    vecs = np.array([s.flatten(order='F') for s in samples])  # (N, 4)
    Cov_sample = np.cov(vecs.T)  # (4, 4)
    Cov_true = np.kron(V, U)

    frob_true = np.linalg.norm(Cov_true, 'fro')
    frob_err = np.linalg.norm(Cov_sample - Cov_true, 'fro') / frob_true
    # 3-sigma bound: 3*sqrt(2/N) ≈ 1.34% for N=1e5
    tol_frob = 0.05  # 5% conservative
    assert frob_err < tol_frob, (
        f"Frobenius relative error {frob_err:.3%} exceeds tolerance {tol_frob:.3%}"
    )


def test_sample_MN_1x1_degenerate():
    """1x1 case: X ~ MN(mu, sigma2, 1) == N(mu, sigma2). Check moments."""
    rng = np.random.default_rng(42)
    M = np.array([[5.0]])
    U = np.array([[4.0]])  # variance = 4
    V = np.array([[1.0]])
    n_samples = 200_000

    samples = sample_MN(M, U, V, n_samples, rng)
    assert samples.shape == (n_samples, 1, 1)

    vals = samples[:, 0, 0]
    mean_sample = float(np.mean(vals))
    var_sample = float(np.var(vals))

    # True: mean=5, var=4
    # 3-sigma on mean: 3*sqrt(4/N) = 3*2/sqrt(N) ≈ 0.013 for N=2e5
    # 3-sigma on var: 3*sqrt(2/N)*4 ≈ 0.076 for N=2e5
    assert abs(mean_sample - 5.0) < 0.05, f"Mean mismatch: {mean_sample}"
    assert abs(var_sample - 4.0) < 0.20, f"Variance mismatch: {var_sample}"
