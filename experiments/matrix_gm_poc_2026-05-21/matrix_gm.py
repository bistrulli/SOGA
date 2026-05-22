"""
Matrix-Variate Gaussian standalone implementation — proof of concept.

X is matrix-Gaussian with M m×n mean and Kronecker-product covariance:

    vec(X) ~ N(vec(M), V ⊗ U)

where U is m×m (row covariance) and V is n×n (column covariance). Storage is
m² + n² instead of (mn)². For m=n=32: 2048 entries vs 1048576 → 500× saving.

This file implements the closed-form operations sufficient to model:
- Bayesian linear regression and similar linear-Gaussian computations
- Affine transformations (deterministic_left · X · deterministic_right + bias)
- Additive matrix noise (fault injection as the most natural model)

For matmul of two random matrix-Gaussians (the hard case), we provide a
moment-matched approximation that returns the nearest matrix-Gaussian.

References:
- Gupta & Nagar (2000), "Matrix Variate Distributions", Chapman & Hall.
- Petersen & Pedersen (2012), "The Matrix Cookbook" §10.
- Van Loan & Pitsianis (1993), "Approximation with Kronecker products".
"""

from __future__ import annotations
import numpy as np


class MatrixGaussian:
    """X ~ MN(M, U, V): X ∈ R^{m × n} with vec(X) ~ N(vec(M), V ⊗ U).

    Storage: M (m × n), U (m × m), V (n × n). Per-element variance is
    Var(X[i,j]) = U[i,i] * V[j,j]. Covariance between X[i,j] and X[i',j']
    is Cov = U[i,i'] * V[j,j'].
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
        """Full vectorised covariance: V ⊗ U  ∈ R^{(m·n) × (m·n)}.

        Used for validation only; do not call on large m·n.
        """
        return np.kron(self.V, self.U)

    def __repr__(self):
        return f"MN(M={self.shape}, |U|={self.U.shape}, |V|={self.V.shape})"

    # ----- closed-form exact operations -----

    def scale(self, c: float) -> "MatrixGaussian":
        """c · X ~ MN(c·M, c²·U, V).  (Or equivalently MN(c·M, U, c²·V).)"""
        return MatrixGaussian(c * self.M, (c * c) * self.U, self.V)

    def transpose(self) -> "MatrixGaussian":
        """X^T ~ MN(M^T, V, U)."""
        return MatrixGaussian(self.M.T, self.V.copy(), self.U.copy())

    def add_constant(self, B: np.ndarray) -> "MatrixGaussian":
        """X + B (deterministic B) ~ MN(M + B, U, V)."""
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
        """X + Y (both matrix-Gaussian, INDEPENDENT) — approximate as
        matrix-Gaussian by moment matching. In general X + Y is matrix-Gaussian
        only when U_X = U_Y or V_X = V_Y; otherwise the resulting covariance
        is U_X⊗V_X + U_Y⊗V_Y which is NOT Kronecker. We return the best-fit
        Kronecker approximation via nearest-Kronecker decomposition.
        """
        assert self.shape == other.shape
        M_out = self.M + other.M
        # Full covariance:  V_X ⊗ U_X + V_Y ⊗ U_Y. Approximate.
        Cov_full = np.kron(self.V, self.U) + np.kron(other.V, other.U)
        U_out, V_out = _nearest_kronecker(Cov_full, self.m, self.n)
        return MatrixGaussian(M_out, U_out, V_out)

    def matmul_independent(self, other: "MatrixGaussian") -> "MatrixGaussian":
        """X · Y (both random, INDEPENDENT) — exact moment matching to
        nearest matrix-Gaussian. X is m×k, Y is k×n; result is m×n.

        E[X·Y] = E[X] · E[Y] = M_X · M_Y.
        Cov of vec(X·Y) is dense in general; we approximate the result's
        covariance by its nearest-Kronecker form.
        """
        assert self.shape[1] == other.shape[0]
        M_out = self.M @ other.M

        # Compute true Cov(vec(XY))
        # Cov(XY_ij, XY_i'j') = sum_{l,l'} ( U_X[i,i'] V_X[l,l'] + M_X[i,l] M_X[i',l'] )
        #                                  * ( U_Y[l,l'] V_Y[j,j'] + M_Y[l,j] M_Y[l',j'] )
        #                       - M_X·M_Y at (i,j) and (i',j')
        # Exact but expensive O(m² k² n²). For PoC keep it simple.
        m, k = self.shape
        _, n = other.shape
        M_X, U_X, V_X = self.M, self.U, self.V
        M_Y, U_Y, V_Y = other.M, other.U, other.V

        # Build the full cov as a (mn × mn) matrix
        Cov = np.zeros((m * n, m * n))
        for i in range(m):
            for j in range(n):
                for ii in range(m):
                    for jj in range(n):
                        # sum_{l, l'}
                        # term1: U_X[i,ii] V_X[l,l'] * U_Y[l,l'] V_Y[j,jj]
                        # term2: U_X[i,ii] V_X[l,l'] * M_Y[l,j] M_Y[l',jj]
                        # term3: M_X[i,l] M_X[ii,l'] * U_Y[l,l'] V_Y[j,jj]
                        term1 = U_X[i, ii] * V_Y[j, jj] * np.trace(V_X @ U_Y)
                        term2 = U_X[i, ii] * (V_X @ M_Y[:, [j]] @ M_Y[:, [jj]].T).trace()
                        # Wait, simpler: use vectorisation
                        # Actually let me just compute it cleanly with einsum:
                        pass
                        # Replaced below
                        Cov[i * n + j, ii * n + jj] = 0.0  # placeholder

        # Cleaner computation via einsum:
        # Cov[i,j; i',j'] = sum_{l,l'} [
        #   U_X[i,i'] V_X[l,l'] U_Y[l,l'] V_Y[j,j']             (random*random term)
        # + U_X[i,i'] V_X[l,l'] M_Y[l,j] M_Y[l',j']             (random_X * mean_Y)
        # + M_X[i,l] M_X[i',l'] U_Y[l,l'] V_Y[j,j']             (mean_X * random_Y)
        # ]
        # Term A: U_X[i,i'] * (sum_{l,l'} V_X[l,l'] U_Y[l,l']) * V_Y[j,j']
        #       = U_X[i,i'] * trace(V_X · U_Y^T) * V_Y[j,j']
        #       Note V_X and U_Y both symmetric → trace(V_X · U_Y)
        sA = np.trace(V_X @ U_Y)
        CovA = sA * np.kron(V_Y, U_X)  # (mn × mn)

        # Term B: U_X[i,i'] * sum_{l,l'} V_X[l,l'] M_Y[l,j] M_Y[l',j']
        #       = U_X[i,i'] * (M_Y^T · V_X · M_Y)[j,j']
        MtVM = M_Y.T @ V_X @ M_Y  # (n × n)
        CovB = np.kron(MtVM, U_X)  # (mn × mn)

        # Term C: V_Y[j,j'] * sum_{l,l'} M_X[i,l] M_X[i',l'] U_Y[l,l']
        #       = V_Y[j,j'] * (M_X · U_Y · M_X^T)[i,i']
        MxUYMxT = M_X @ U_Y @ M_X.T  # (m × m)
        CovC = np.kron(V_Y, MxUYMxT)  # (mn × mn)

        Cov = CovA + CovB + CovC  # exact, mn × mn

        # Project to nearest Kronecker form
        U_out, V_out = _nearest_kronecker(Cov, m, n)
        return MatrixGaussian(M_out, U_out, V_out)


def _nearest_kronecker(Cov: np.ndarray, m: int, n: int):
    """Find U (m×m), V (n×n) such that Cov ≈ V ⊗ U, minimising
    Frobenius error. Algorithm: Van Loan & Pitsianis (1993).
    """
    # Reshape Cov (mn × mn) into a rearranged tensor R(i,j,i',j') and
    # then to a matrix of shape (m², n²); top singular value gives the
    # rank-1 approximation = nearest Kronecker.
    # vec(U) outer vec(V).
    R = np.zeros((m * m, n * n))
    for j in range(n):
        for jj in range(n):
            block = Cov[j * m:(j + 1) * m, jj * m:(jj + 1) * m]  # m × m
            R[:, j * n + jj] = block.flatten('F')  # vec(block) into row j*n+jj
    # Note: the rearrangement convention depends on Kronecker order; we use
    # V ⊗ U  which means Cov[i + m·j, i' + m·j'] = V[j,j'] · U[i,i'].
    Us, sigma, Vt = np.linalg.svd(R, full_matrices=False)
    # Rank-1: largest singular value
    s = sigma[0]
    u_vec = Us[:, 0] * np.sqrt(s)
    v_vec = Vt[0, :] * np.sqrt(s)
    U = u_vec.reshape((m, m), order='F')
    V = v_vec.reshape((n, n), order='F')
    # Force symmetry and PSD enforcement
    U = 0.5 * (U + U.T); V = 0.5 * (V + V.T)
    # Sign convention: U and V can each carry a sign; ensure positive trace
    if np.trace(U) < 0:
        U = -U
        V = -V
    # Clip negative eigenvalues to zero (numerical PSD)
    for K in (U, V):
        w, Q = np.linalg.eigh(K)
        w = np.clip(w, 0, None)
        K[:] = Q @ np.diag(w) @ Q.T
    return U, V


# ----- helpers for testing -----

def to_scalar_dense(mg: MatrixGaussian) -> tuple[np.ndarray, np.ndarray]:
    """Convert matrix-Gaussian to (vec_mean, full_dense_cov). Used for
    cross-validation with naive scalar implementations."""
    return mg.M.flatten('F'), mg.covariance_vec()


def from_scalar_dense(mean_vec: np.ndarray, cov_dense: np.ndarray, m: int, n: int):
    """Reverse: rebuild best matrix-Gaussian approximation from dense cov."""
    M = mean_vec.reshape((m, n), order='F')
    U, V = _nearest_kronecker(cov_dense, m, n)
    return MatrixGaussian(M, U, V)
