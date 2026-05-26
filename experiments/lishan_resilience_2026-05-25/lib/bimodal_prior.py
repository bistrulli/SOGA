"""
M5.1 — Bimodal Bernoulli prior utilities for Step 3 analysis.

Represents a bimodal input distribution:
    B[i,j] ~ p * delta(V_high) + (1-p) * delta(V_low)

where p is the mixing weight and V_high, V_high are the two support points.

Two approximation modes:
    1. 2-component GM (primary, for sparse A where m_dense < threshold):
       Keeps exact bimodal structure — two separate flat-input runs.
       Most accurate for A=I_32 since D[r,s] = B[r,s] directly.

    2. Single-Gaussian moment-match (secondary, for dense A):
       Replaces bimodal with Gaussian N(mu_p, sigma2_p).
       mu_p = p * V_high + (1-p) * V_low
       sigma2_p = p * (1-p) * (V_high - V_low)^2
       Valid only when CLT applies (m_dense >= 16 nonzero entries per row).

HONESTY DISCLAIMER: Input-side fault model. NOT register-level injection.
Strada Q discipline.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np


class BimodalApproximationWarning(UserWarning):
    """Issued when single-Gaussian moment-match is used for intermediate p values."""
    pass


@dataclass(frozen=True)
class BimodalPrior:
    """Immutable description of a bimodal Bernoulli input prior.

    Attributes:
        p: Mixing weight in [0, 1]. P(B[i,j] = V_high) = p.
        V_low: Lower support point (e.g., 0.0 for inactive neuron).
        V_high: Upper support point (e.g., 1.0 for active neuron).
        m: Number of rows of the input matrix B.
        n: Number of columns of the input matrix B.
    """
    p: float
    V_low: float
    V_high: float
    m: int
    n: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"Mixing weight p={self.p} must be in [0, 1]")
        if self.V_high < self.V_low:
            raise ValueError(f"V_high={self.V_high} must be >= V_low={self.V_low}")
        if self.m <= 0 or self.n <= 0:
            raise ValueError(f"Matrix dimensions m={self.m}, n={self.n} must be positive")

    @property
    def mu(self) -> float:
        """First moment: E[B[i,j]] = p * V_high + (1-p) * V_low."""
        return self.p * self.V_high + (1.0 - self.p) * self.V_low

    @property
    def sigma2(self) -> float:
        """Second central moment: Var[B[i,j]] = p*(1-p)*(V_high - V_low)^2."""
        return self.p * (1.0 - self.p) * (self.V_high - self.V_low) ** 2

    @property
    def sigma(self) -> float:
        """Standard deviation sqrt(sigma2)."""
        return float(np.sqrt(max(self.sigma2, 0.0)))

    @property
    def is_degenerate(self) -> bool:
        """True if prior collapses to a point mass (p=0 or p=1 or V_high=V_low)."""
        return self.sigma2 < 1e-30

    def to_matrix_gaussian(
        self,
        m_dense_threshold: int = 16,
        warn_range: Tuple[float, float] = (0.2, 0.8),
        U_B_scale: float = 1e-10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Convert bimodal prior to matrix-Gaussian parameters (M_B, U_B, V_B).

        Returns the parameters for a single-Gaussian approximation.
        For Step 3 with sparse A, prefer 2-component GM via component_baselines().

        Args:
            m_dense_threshold: If m_dense < threshold, this function is less
                accurate (2-component GM preferred). Not enforced here but
                BimodalApproximationWarning is issued.
            warn_range: (p_lo, p_hi) — warn if p is in this range (bimodal regime).
            U_B_scale: Minimum diagonal scale for U_B when sigma2 ~ 0.

        Returns:
            (M_B, U_B, V_B, mode_str)
            - M_B: (m, n) mean matrix
            - U_B: (m, m) row covariance
            - V_B: (n, n) column covariance (identity)
            - mode_str: 'moment-match' or 'degenerate'
        """
        if warn_range[0] < self.p < warn_range[1]:
            warnings.warn(
                f"BimodalPrior.to_matrix_gaussian: p={self.p:.2f} is in the bimodal "
                f"range {warn_range}. Single-Gaussian moment-match may underestimate "
                f"SDC. Consider 2-component GM path (use predict_bimodal_sweep with "
                f"sparse A). See LIMITATIONS.md.",
                BimodalApproximationWarning,
                stacklevel=2,
            )

        if self.is_degenerate:
            # Point mass at mu (degenerate: V=0)
            M_B = np.full((self.m, self.n), self.mu, dtype=np.float64)
            U_B = U_B_scale * np.eye(self.m)
            V_B = np.eye(self.n)
            return M_B, U_B, V_B, "degenerate"

        M_B = np.full((self.m, self.n), self.mu, dtype=np.float64)
        U_B = max(self.sigma2, U_B_scale) * np.eye(self.m)
        V_B = np.eye(self.n)
        return M_B, U_B, V_B, "moment-match"

    def component_baselines(
        self,
        U_B_scale: float = 1e-10,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return two matrix-Gaussian parameter sets for 2-component GM.

        Component 0 (weight p): flat input B = V_high * ones(m, n)
        Component 1 (weight 1-p): flat input B = V_low * ones(m, n)

        Returns:
            ((M_high, U_high, V_high), (M_low, U_low, V_low))
        """
        M_high = np.full((self.m, self.n), self.V_high, dtype=np.float64)
        U_high = U_B_scale * np.eye(self.m)
        V_high_mat = np.eye(self.n)

        M_low = np.full((self.m, self.n), self.V_low, dtype=np.float64)
        U_low = U_B_scale * np.eye(self.m)
        V_low_mat = np.eye(self.n)

        return (M_high, U_high, V_high_mat), (M_low, U_low, V_low_mat)


def bimodal_to_matrix_gaussian(
    p: float,
    V_low: float,
    V_high: float,
    m: int,
    n: int,
    m_dense_threshold: int = 16,
    warn_range: Tuple[float, float] = (0.2, 0.8),
    U_B_scale: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Convenience wrapper: create BimodalPrior and return matrix-Gaussian parameters.

    For sparse A (m_dense < m_dense_threshold), the 2-component GM path is
    preferred. See predict_bimodal_sweep() in predict_resilience_soga.py.

    Args:
        p: Mixing weight in [0, 1].
        V_low, V_high: Support points.
        m, n: Input matrix dimensions.
        m_dense_threshold: Density threshold for approximation warning.
        warn_range: p range where BimodalApproximationWarning is issued.
        U_B_scale: Near-zero variance floor.

    Returns:
        (M_B, U_B, V_B, mode_str)
    """
    prior = BimodalPrior(p=p, V_low=V_low, V_high=V_high, m=m, n=n)
    return prior.to_matrix_gaussian(
        m_dense_threshold=m_dense_threshold,
        warn_range=warn_range,
        U_B_scale=U_B_scale,
    )


def p_critical_sdc(
    V_low: float,
    V_high: float,
    eps: float,
) -> float:
    """Estimate the critical p where SDC risk transitions from low to high.

    For the bimodal Bernoulli prior with moment-matched Gaussian:
    sigma_p = sqrt(p*(1-p)) * |V_high - V_low|

    The SDC transition occurs when sigma_p ~ eps * |mu_p|:
        sqrt(p*(1-p)) * |Delta_V| ~ eps * |p*V_high + (1-p)*V_low|

    Returns the p in [0,1] where this approximately holds (root-finding).
    Returns np.nan if no such p exists or if V_high == V_low.
    """
    if abs(V_high - V_low) < 1e-15:
        return float("nan")

    delta_v = abs(V_high - V_low)

    def residual(p_val: float) -> float:
        mu = p_val * V_high + (1.0 - p_val) * V_low
        sigma = np.sqrt(max(p_val * (1.0 - p_val), 0.0)) * delta_v
        return sigma - eps * abs(mu)

    # Try a simple scan to find the crossing
    p_arr = np.linspace(0.01, 0.99, 200)
    res = np.array([residual(p) for p in p_arr])
    sign_changes = np.where(np.diff(np.sign(res)))[0]
    if len(sign_changes) == 0:
        return float("nan")

    # Linear interpolation to approximate crossing
    idx = sign_changes[0]
    p0, p1 = p_arr[idx], p_arr[idx + 1]
    r0, r1 = res[idx], res[idx + 1]
    if abs(r1 - r0) < 1e-15:
        return float(p0)
    p_crit = p0 - r0 * (p1 - p0) / (r1 - r0)
    return float(np.clip(p_crit, 0.0, 1.0))
