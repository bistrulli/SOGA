"""
libSOGAsharedMatrix.py — Kronecker-structured joint state for matrix-variate GM.

Implements GaussianMixBlock: the joint state representation for programs that
mix scalar and matrix-Gaussian variables.  Each mixture component stores:
  - per-variable mean: scalar → float, matrix → np.ndarray (m, n)
  - per-variable-pair covariance:
      (scalar, scalar) pair  → float
      (matrix, matrix) same  → (U: (m,m), V: (n,n)) Kronecker factors
      (scalar, matrix) cross → np.ndarray (mn,) flat cross-covariance (V⊗U order)
      (matrix, matrix) diff  → NotImplementedError in v1

M3 plan references:
  §M3.1 — GaussianMixBlock class definition
  §M3.2 — PSD enforcement protocol on Kronecker factors
  §M3.3 — from_dist constructor (pure-scalar Dist → GaussianMixBlock)
  §M3.4 — matrix_gm initialisation (independent of scalar vars at declaration)
  §M3.5 — tests/test_gaussian_mix_block.py

Kronecker convention (matches libMatrixGaussian.KRONECKER_CONVENTION):
  vec(X) ~ N(vec(M), V ⊗ U)
  Var(X[i,j]) = U[i,i] * V[j,j]
  Cov(X[i,j], X[i',j']) = U[i,i'] * V[j,j']
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple, Any, Union

import numpy as np

from libSOGAshared import (
    Dist, GaussianMix, VarEntry, NumericalError,
    make_psd, make_sym,
)
from libMatrixGaussian import KRONECKER_CONVENTION, _nearest_kronecker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases for cov_blocks values
# ---------------------------------------------------------------------------

# scalar-scalar: float
# matrix-matrix (same var): (U_array, V_array)
# scalar-matrix cross: flat np.ndarray shape (mn,)
CovValue = Union[float, Tuple[np.ndarray, np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# PSD enforcement on Kronecker factors (M3.2)
# ---------------------------------------------------------------------------

def _enforce_psd_kron_factors(
    U: np.ndarray, V: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Enforce PSD on Kronecker factors U and V individually.

    Protocol (per plan M3.2 / numerical memo §1):
      1. make_sym(U), make_sym(V)
      2. sign disambiguation — first-nonzero-positive convention on the leading
         eigenvector is already handled inside libMatrixGaussian._nearest_kronecker;
         here we only enforce PSD on already-extracted factors.
      3. make_psd(U), make_psd(V)
      4. optional debug assertion: min_eig(kron(V, U)) >= -1e-12 under SOGA_DEBUG=1

    Raises NumericalError (from make_psd) if PSD cannot be achieved after retries.
    """
    import os
    U = make_sym(U.copy())
    V = make_sym(V.copy())
    U = make_psd(U)
    V = make_psd(V)

    if os.environ.get("SOGA_DEBUG"):
        # Dense check — only at debug time, never in production loops
        kron_full = np.kron(V, U)
        min_eig = float(np.min(np.linalg.eigvalsh(kron_full)))
        if min_eig < -1e-12:
            logger.warning(
                "SOGA_DEBUG: min_eig(V⊗U) = %.4e < -1e-12 after PSD enforcement",
                min_eig,
            )
    return U, V


# ---------------------------------------------------------------------------
# GaussianMixBlock
# ---------------------------------------------------------------------------

class GaussianMixBlock:
    """Joint state for a Gaussian mixture with scalar + matrix variables.

    Stores per-component means and covariances in a block structure that avoids
    materialising the full (d_scalar + mn) × (d_scalar + mn) joint covariance.

    Parameters
    ----------
    var_list : list[str]
        Scalar variable names (same order as Dist.var_list / GaussianMix.mu).
    var_entries : list[VarEntry]
        Matrix variable entries (kind='matrix').
    pi : list[float]
        Mixture weights (sum to 1).
    mu_blocks : list[dict[str, Any]]
        Per-component per-variable mean.  Key = var name.
        Scalar var → np.ndarray shape (1,)  [to keep uniform dict access]
        Matrix var → np.ndarray shape (m, n)
    cov_blocks : list[dict[frozenset, Any]]
        Per-component per-pair covariance.
        Key = frozenset({name_a, name_b}) — but for self-covariance of a matrix
        variable the key is frozenset({name}) (a frozenset of one element).
        Value types:
          (scalar, scalar): float (the [i,j] covariance entry)
          (matrix, matrix) same var: (U, V) tuple of ndarrays
          (scalar, matrix) cross: np.ndarray shape (mn,) in V⊗U column-major order
          (matrix, matrix) different vars: NotImplementedError in v1
    log_pi : list[float] | None
        Log-weights for numerical stability (M5.5).  None until M5 initialises
        the log-weight infrastructure.  After any truncation step log_pi is set
        and kept in sync with pi.
    """

    def __init__(
        self,
        var_list: List[str],
        var_entries: List[VarEntry],
        pi: List[float],
        mu_blocks: List[Dict[str, np.ndarray]],
        cov_blocks: List[Dict[FrozenSet, CovValue]],
        log_pi: Optional[List[float]] = None,
    ):
        self.var_list = list(var_list)
        self.var_entries = list(var_entries)
        self.pi = list(pi)
        self.mu_blocks = mu_blocks
        self.cov_blocks = cov_blocks
        self.log_pi = log_pi

    # ------------------------------------------------------------------
    # Basic accessors
    # ------------------------------------------------------------------

    def n_comp(self) -> int:
        return len(self.pi)

    def _matrix_var_shape(self, var_name: str) -> Tuple[int, int]:
        for ve in self.var_entries:
            if ve.name == var_name:
                return ve.shape
        raise KeyError(f"No matrix variable '{var_name}' in var_entries")

    def get_mu(self, k: int, var_name: str) -> np.ndarray:
        """Mean of variable `var_name` in component k.

        Returns shape (m, n) for matrix vars, shape (1,) for scalar vars.
        """
        return self.mu_blocks[k][var_name]

    def get_cov(self, k: int, var_a: str, var_b: str) -> CovValue:
        """Covariance block for (var_a, var_b) in component k.

        For matrix self-covariance: returns (U, V) tuple.
        For scalar self-covariance: returns float.
        For cross (scalar, matrix): returns np.ndarray shape (mn,).
        For two different matrix variables: raises NotImplementedError (v1).
        """
        if var_a == var_b:
            key = frozenset({var_a})
        else:
            # Check for two-different-matrix-var case
            a_is_mat = any(ve.name == var_a for ve in self.var_entries)
            b_is_mat = any(ve.name == var_b for ve in self.var_entries)
            if a_is_mat and b_is_mat:
                raise NotImplementedError(
                    f"[v1] Cross-covariance between two different matrix variables "
                    f"('{var_a}', '{var_b}') is not supported in v1.  "
                    "See plan/2026-05-22-matrix-gm-lishan.md §M3.1."
                )
            key = frozenset({var_a, var_b})
        return self.cov_blocks[k][key]

    def to_dense_scalar_view(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the scalar part of component k as (mu_vec, sigma_matrix).

        Used to feed the scalar truncate path when a test/observe condition
        involves only scalar variables.

        Returns
        -------
        mu_vec : np.ndarray shape (d_scalar,)
        sigma_matrix : np.ndarray shape (d_scalar, d_scalar)
        """
        d = len(self.var_list)
        mu_vec = np.zeros(d)
        sigma_matrix = np.zeros((d, d))
        for i, vi in enumerate(self.var_list):
            mu_vec[i] = float(self.mu_blocks[k][vi][0])
            for j, vj in enumerate(self.var_list):
                if i == j:
                    sigma_matrix[i, j] = float(self.get_cov(k, vi, vi))
                else:
                    sigma_matrix[i, j] = float(self.get_cov(k, vi, vj))
        return mu_vec, sigma_matrix

    # ------------------------------------------------------------------
    # Mixture moments for matrix variables (M3.8)
    # ------------------------------------------------------------------

    def matrix_mean(self, var_name: str) -> np.ndarray:
        """Mixture mean for a matrix variable: M_bar = sum_k pi_k * M_k."""
        m, n = self._matrix_var_shape(var_name)
        M_bar = np.zeros((m, n))
        for k in range(self.n_comp()):
            M_bar += self.pi[k] * self.mu_blocks[k][var_name]
        return M_bar

    def matrix_var(self, var_name: str, i: int, j: int) -> float:
        """Per-element variance via law of total variance (mixture).

        Var(X[i,j]) = sum_k pi_k * (U_k[i,i]*V_k[j,j] + (M_k[i,j] - M_bar[i,j])^2)

        Both within-component (U_k[i,i]*V_k[j,j]) and between-component
        (M_k[i,j] - M_bar[i,j])^2 terms are included.  Omitting the
        between-component term would give wrong variance for multi-component
        mixtures (~20% error per plan cross-review).

        fix4: handles dense sentinel (None, Sigma) storage — reads variance
        from the dense (mn × mn) covariance via column-major index j*m+i.
        """
        M_bar = self.matrix_mean(var_name)
        ve = next((v for v in self.var_entries if v.name == var_name), None)
        total = 0.0
        for k in range(self.n_comp()):
            stored = self.cov_blocks[k].get(frozenset({var_name}))
            if stored is not None and stored[0] is None:
                # Dense sentinel: read variance from Sigma[idx, idx]
                Sigma = stored[1]
                m_v = ve.shape[0] if ve else Sigma.shape[0] // Sigma.shape[0]
                idx = j * (Sigma.shape[0] // (ve.shape[1] if ve else 1)) + i if ve else j * int(Sigma.shape[0] ** 0.5) + i
                # More robust: infer m from var_entries
                if ve is not None:
                    m_v, n_v = ve.shape
                    idx = j * m_v + i
                else:
                    m_v = int(Sigma.shape[0] ** 0.5)
                    idx = j * m_v + i
                within = float(Sigma[idx, idx])
            else:
                U_k, V_k = self.get_cov(k, var_name, var_name)
                within = float(U_k[i, i] * V_k[j, j])
            between = float((self.mu_blocks[k][var_name][i, j] - M_bar[i, j]) ** 2)
            total += self.pi[k] * (within + between)
        return total

    def matrix_cov(
        self, var_name: str, i1: int, j1: int, i2: int, j2: int
    ) -> float:
        """Mixture covariance between X[i1,j1] and X[i2,j2].

        Cov(X[i1,j1], X[i2,j2]) = sum_k pi_k * (
            U_k[i1,i2] * V_k[j1,j2]
          + (M_k[i1,j1] - M_bar[i1,j1]) * (M_k[i2,j2] - M_bar[i2,j2])
        )
        """
        M_bar = self.matrix_mean(var_name)
        total = 0.0
        for k in range(self.n_comp()):
            U_k, V_k = self.get_cov(k, var_name, var_name)
            within = float(U_k[i1, i2] * V_k[j1, j2])
            between = float(
                (self.mu_blocks[k][var_name][i1, j1] - M_bar[i1, j1])
                * (self.mu_blocks[k][var_name][i2, j2] - M_bar[i2, j2])
            )
            total += self.pi[k] * (within + between)
        return total

    def matrix_full_cov(self, var_name: str) -> np.ndarray:
        """Full (mn × mn) vectorised covariance matrix for var_name (dense).

        Expensive — for validation and small m*n only.  Uses the law of total
        covariance across all element pairs (i1,j1), (i2,j2).
        """
        m, n = self._matrix_var_shape(var_name)
        mn = m * n
        M_bar = self.matrix_mean(var_name)
        M_bar_vec = M_bar.flatten("F")  # column-major / V⊗U convention
        Cov = np.zeros((mn, mn))
        for k in range(self.n_comp()):
            U_k, V_k = self.get_cov(k, var_name, var_name)
            kron_k = np.kron(V_k, U_k)  # V⊗U convention
            M_k_vec = self.mu_blocks[k][var_name].flatten("F")
            delta_k = M_k_vec - M_bar_vec
            Cov += self.pi[k] * (kron_k + np.outer(delta_k, delta_k))
        return Cov

    def matrix_kron_factors(
        self, var_name: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (U, V) Kronecker factors if stored as Kronecker.

        In v1, always returns the first-component factors (K=1) or None if
        the variable has been densified post-observe (M5 DENSE strategy
        stores a dense covariance — not yet implemented in M3).  For K>1
        this returns the mixture-averaged factors (informational only).
        """
        if self.n_comp() == 0:
            return None
        U0, V0 = self.get_cov(0, var_name, var_name)
        if U0 is None:
            return None
        return U0, V0

    # ------------------------------------------------------------------
    # Constructor: from_dist (M3.3)
    # ------------------------------------------------------------------

    @classmethod
    def from_dist(
        cls,
        dist: Dist,
        var_entries: Optional[List[VarEntry]] = None,
    ) -> "GaussianMixBlock":
        """Build a GaussianMixBlock from a scalar Dist (zero matrix variables).

        The Dist's scalar GaussianMix becomes the scalar part of the block.
        No matrix variables are added.  Used for testing and cross-validation.

        Parameters
        ----------
        dist : Dist
            Scalar-only distribution.
        var_entries : list[VarEntry] | None
            Matrix variable entries to include (empty or None means none).
        """
        var_list = dist.var_list
        ves = var_entries if var_entries else []
        pi = list(dist.gm.pi)
        mu_blocks = []
        cov_blocks = []
        for k in range(dist.gm.n_comp()):
            mu_k = {}
            cov_k = {}
            for i, vname in enumerate(var_list):
                mu_k[vname] = np.array([dist.gm.mu[k][i]])
            for i, vi in enumerate(var_list):
                for j, vj in enumerate(var_list):
                    if i <= j:
                        key = frozenset({vi, vj}) if vi != vj else frozenset({vi})
                        val = float(dist.gm.sigma[k][i, j])
                        cov_k[key] = val
            mu_blocks.append(mu_k)
            cov_blocks.append(cov_k)
        return cls(var_list, ves, pi, mu_blocks, cov_blocks)

    # ------------------------------------------------------------------
    # Constructor: from_matrix_gm (M3.4)
    # ------------------------------------------------------------------

    @classmethod
    def from_matrix_gm(
        cls,
        var_entries: List[VarEntry],
        means: List[np.ndarray],         # per-component, per-matrix-var means
        us: List[np.ndarray],            # per-component U factors
        vs: List[np.ndarray],            # per-component V factors
        pi: Optional[List[float]] = None,
        var_list: Optional[List[str]] = None,
    ) -> "GaussianMixBlock":
        """Initialise a GaussianMixBlock from matrix_gm(...) constructor data.

        At declaration, the matrix variable is assumed INDEPENDENT of all
        prior scalar variables → cross-covariances are initialised to zero
        (M3.4 spec: 'scalar-matrix cross-covariance is zero at declaration').

        Parameters
        ----------
        var_entries : list[VarEntry]
            Matrix variable entries (one or more).
        means : list[np.ndarray]
            Per-component mean matrices (m, n) for each VarEntry.
            Length = n_comp * len(var_entries) — outer list is per-component.
        us : list[np.ndarray]
            Per-component U factors (m, m).
        vs : list[np.ndarray]
            Per-component V factors (n, n).
        pi : list[float] | None
            Mixing weights.  Defaults to [1.0] (single component).
        var_list : list[str] | None
            Scalar variable names (empty if no scalars yet).
        """
        if pi is None:
            pi = [1.0]
        if var_list is None:
            var_list = []
        n_comp = len(pi)
        assert len(means) == n_comp
        assert len(us) == n_comp
        assert len(vs) == n_comp

        mu_blocks = []
        cov_blocks = []
        for k in range(n_comp):
            mu_k = {}
            cov_k = {}
            for ve in var_entries:
                mu_k[ve.name] = np.asarray(means[k], dtype=float)
                U_k, V_k = _enforce_psd_kron_factors(
                    np.asarray(us[k], dtype=float),
                    np.asarray(vs[k], dtype=float),
                )
                cov_k[frozenset({ve.name})] = (U_k, V_k)
            # Scalar means start at 0, scalar-scalar cov = 0
            for sv in var_list:
                mu_k[sv] = np.array([0.0])
            for sv in var_list:
                cov_k[frozenset({sv})] = 0.0
                for ve in var_entries:
                    # scalar-matrix cross-cov = 0 at declaration
                    m_var, n_var = ve.shape
                    cov_k[frozenset({sv, ve.name})] = np.zeros(m_var * n_var)
            mu_blocks.append(mu_k)
            cov_blocks.append(cov_k)
        return cls(var_list, var_entries, pi, mu_blocks, cov_blocks)

    # ------------------------------------------------------------------
    # Renormalise (used after truncation)
    # ------------------------------------------------------------------

    def renormalise(self) -> None:
        """Renormalise pi in-place using log_pi if available, else linear sum."""
        if self.log_pi is not None:
            max_lp = max(self.log_pi)
            shifted = [lp - max_lp for lp in self.log_pi]
            exp_vals = [np.exp(lp) for lp in shifted]
            s = sum(exp_vals)
            self.pi = [e / s for e in exp_vals]
        else:
            s = sum(self.pi)
            if s > 0:
                self.pi = [p / s for p in self.pi]

    def __repr__(self) -> str:
        return (
            f"GaussianMixBlock("
            f"n_comp={self.n_comp()}, "
            f"scalars={self.var_list}, "
            f"matrices={[ve.name for ve in self.var_entries]})"
        )
