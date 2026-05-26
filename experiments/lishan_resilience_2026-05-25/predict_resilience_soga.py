"""
M3 — SOGA analytical prediction of resilience curves.

HONESTY DISCLAIMER: Input-side fault model. NOT register-level injection.
Strada Q discipline.

Architecture: Option 2 (unanimous expert recommendation) — analytical
marginalization at output, NOT explicit fault-mixture propagation through SOGA.

1. SOGA computes D_baseline = A @ B as a single matrix-Gaussian.
2. For each (output cell (r,s), fault class c, input row i):
   - Compute per-component tail-Gaussian Pr(|D[r,s] - D_golden[r,s]| > threshold)
   - Use H4 symmetry reduction: per output cell (r,s), only faults in input
     column j=s shift the mean; faults in j≠s leave D[r,s] unchanged (MSK).
   - For A=I_32: only i=r gives nonzero A[r,i] → K_per_cell = 6 (1+5 classes).
   - For general A: K_per_cell = 1 + 5 * m_distinct where m_distinct is the
     number of distinct nonzero values among {A[r,i]: i in [m]}.
3. Aggregate weighted by P(class) and P(fault_cell in j=s).

References:
    - plan M3.2, M3.3, M3.4, M3.5
    - libMatrixGaussian.MatrixGaussian + affine_left
    - lib/fault_model.py for shift computations
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, EXP_DIR)

from libMatrixGaussian import MatrixGaussian
from scipy.special import logsumexp as sp_logsumexp
from lib.fault_model import (
    FaultClass,
    FAULT_CLASS_PROB,
    check_overflow,
    shift_moments,
    tail_gauss,
    MantissaApproximationWarning,
    HIGH_EXP_K_VALUES,
    LOW_EXP_K_VALUES,
    HIGH_MANTISSA_E2P2,
    LOW_MANTISSA_E2P2,
    _FLOAT32_MAX_LOG,
)


# ---------------------------------------------------------------------------
# M3.1: compute_baseline — matrix-Gaussian via libMatrixGaussian
# ---------------------------------------------------------------------------

def compute_baseline(
    A: np.ndarray,
    M_B: np.ndarray,
    U_B: np.ndarray,
    V_B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute D_baseline = A @ B using libMatrixGaussian.

    D = A @ B ~ MN(A @ M_B, A @ U_B @ A^T, V_B)

    Args:
        A: Kernel matrix (m x m), float64.
        M_B: B mean matrix (m x n).
        U_B: B row covariance (m x m).
        V_B: B column covariance (n x n).

    Returns:
        (M_D, U_D, V_D): Parameters of D distribution.
    """
    B_mg = MatrixGaussian(M=M_B, U=U_B, V=V_B)
    D_mg = B_mg.affine_left(A)
    return D_mg.M, D_mg.U, D_mg.V


# ---------------------------------------------------------------------------
# M3.2 + M3.3 + M3.4: compute_sdc_vectorized — fully numpy-vectorized (H3)
# ---------------------------------------------------------------------------

def compute_sdc_vectorized(
    M_D: np.ndarray,
    U_D: np.ndarray,
    V_D: np.ndarray,
    A: np.ndarray,
    v: float,
    eps: float,
    p_fault: float,
    m_input: Optional[int] = None,
    n_input: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Fully vectorized analytical prediction of (Pr_MSK, Pr_SDC, Pr_OTR).

    Vectorizes over all output cells (r,s) and input rows i simultaneously
    using numpy broadcasting. Target: <1s per v for 32×32 (H3).

    H4 symmetry: for D[r,s] = sum_i A[r,i]*B[i,s], only faults in column j=s
    of B shift D[r,s]. All other fault positions (j≠s) → MSK.

    Returns aggregate (Pr_MSK, Pr_SDC, Pr_OTR) averaged over output cells.
    """
    m = A.shape[0]
    n = M_D.shape[1]
    m_in = m_input or m
    n_in = n_input or n

    # Shapes used in broadcasting:
    # A: (m, m_in) → A[r, i]
    # M_D diagonal: (m,) → mu_golden[r, s] = M_D[r, s] (same for all s for flat v)
    # For flat v input: M_D = v * A @ ones = v * A.sum(axis=1)[:, None] * ones_n
    # After compute_baseline with flat v: M_D[r,s] = v * sum_i A[r,i]

    # mu_golden: shape (m, n) — same for all s with flat B
    mu_golden = M_D  # (m, n)

    # threshold per (r,s): eps * |mu_golden[r,s]|, with fallback to eps
    threshold = np.where(np.abs(mu_golden) > 1e-10, eps * np.abs(mu_golden), eps)  # (m, n)

    # sigma_base per (r,s) from Kronecker structure: sigma^2 = U_D[r,r] * V_D[s,s]
    U_diag = np.diag(U_D)  # (m,)
    V_diag = np.diag(V_D)  # (n,)
    sigma_base_sq = np.outer(U_diag, V_diag)  # (m, n)
    sigma_base = np.sqrt(np.maximum(sigma_base_sq, 0.0))  # (m, n)

    # P(fault at (i, j=s)) = 1/(m_in * n_in) per cell
    # After H4: we sum over i; the n_in factor cancels with the 1/n_in denominator
    # (see below)
    p_fault_cell = 1.0 / (m_in * n_in)

    # Accumulate SDC and OTR contributions: shape (m, n)
    p_sdc_total = np.zeros((m, n), dtype=np.float64)
    p_otr_total = np.zeros((m, n), dtype=np.float64)

    # A matrix for broadcasting: A[r, i] with r in [m], i in [m_in]
    # We need delta_mean[r, i] = E[delta|v] * A[r, i] for each class

    v_scalar = float(v)
    log_abs_v = np.log(abs(v_scalar)) if abs(v_scalar) > 0 else -np.inf

    # Pre-compute A[r, i]: shape (m, m_in)
    A_mat = A[:m, :m_in]  # (m, m_in)
    log_abs_A = np.where(np.abs(A_mat) > 0, np.log(np.abs(A_mat)), -np.inf)  # (m, m_in)

    for fc in FaultClass:
        p_c = FAULT_CLASS_PROB[fc]

        if fc == FaultClass.SIGN:
            # delta = -2*v (deterministic, Var=0)
            scale = -2.0
            E_delta = v_scalar * scale
            Var_delta = 0.0
            is_ovf_scalar = False  # SIGN never overflows

            # delta_mean on output D[r,s] = E_delta * A[r,i], shape (m, m_in)
            delta_mean_mat = E_delta * A_mat  # (m, m_in)
            sigma_shift_mat = np.zeros_like(delta_mean_mat)

            # mu_post[r, s, i] = mu_golden[r, s] + delta_mean_mat[r, i]
            # But we sum over i weighted by p_fault_cell
            # shape: broadcast (m,n,1) + (m,1,m_in) → (m,n,m_in) then sum over i
            mu_post = mu_golden[:, :, None] + delta_mean_mat[:, None, :]  # (m, n, m_in)
            sigma_post = sigma_base[:, :, None]  # (m, n, 1) (Var_delta=0)

            p_sdc_mat = _tail_gauss_broadcast(mu_post, sigma_post, threshold[:, :, None])
            # p_sdc_mat: (m, n, m_in); sum over i weighted by p_fault_cell
            p_sdc_contribution = p_c * p_fault_cell * p_sdc_mat.sum(axis=2)  # (m, n)
            p_sdc_total += p_sdc_contribution

        elif fc in (FaultClass.HIGH_EXP, FaultClass.LOW_EXP):
            k_values = HIGH_EXP_K_VALUES if fc == FaultClass.HIGH_EXP else LOW_EXP_K_VALUES
            n_k = len(k_values)
            p_sub = 1.0 / n_k  # uniform over k-values within class

            for k in k_values:
                scale = float(2**k - 1)
                E_delta_k = v_scalar * scale
                log_scale = float(np.log(scale))

                # H1: overflow check for each (r, i)
                log_mag = log_abs_v + log_scale + log_abs_A  # (m, m_in)
                is_ovf = log_mag > _FLOAT32_MAX_LOG  # (m, m_in) bool

                # OTR contribution: per (r,s), sum over i where is_ovf
                # D[r,s] OTR if the fault at (i,s) for this k causes overflow
                # = p_fault_cell * p_c * p_sub * sum_i(is_ovf[r,i])
                # This is the same for all s (no s-dependence in overflow check)
                otr_per_r = (is_ovf * p_fault_cell * p_c * p_sub).sum(axis=1)  # (m,)
                p_otr_total += otr_per_r[:, None]  # broadcast to (m, n)

                # SDC for non-overflow i
                non_ovf = ~is_ovf  # (m, m_in)
                if not non_ovf.any():
                    continue

                delta_mean_mat = E_delta_k * A_mat  # (m, m_in)
                # Zero out overflow cases
                delta_mean_mat_safe = np.where(non_ovf, delta_mean_mat, 0.0)

                mu_post = mu_golden[:, :, None] + delta_mean_mat_safe[:, None, :]  # (m,n,m_in)
                sigma_post = sigma_base[:, :, None]  # Var_delta=0 for exact EXP

                p_sdc_mat = _tail_gauss_broadcast(mu_post, sigma_post, threshold[:, :, None])
                # Zero out overflow positions
                p_sdc_mat = np.where(non_ovf[None, :, :].transpose(1, 0, 2), p_sdc_mat, 0.0)
                # Note: need shape alignment (m, n, m_in)
                p_sdc_mat = np.where(
                    non_ovf[:, None, :],  # (m, 1, m_in)
                    p_sdc_mat,
                    0.0,
                )

                p_sdc_contribution = p_c * p_sub * p_fault_cell * p_sdc_mat.sum(axis=2)
                p_sdc_total += p_sdc_contribution

        else:  # MANTISSA classes (moment-matched)
            # E_delta = 0 (sign symmetry)
            # Var_delta = v^2 * E_2p2
            E_2p2 = HIGH_MANTISSA_E2P2 if fc == FaultClass.HIGH_MANTISSA else LOW_MANTISSA_E2P2
            Var_delta = v_scalar**2 * E_2p2
            sigma_delta_scalar = float(np.sqrt(Var_delta))

            # sigma_shift on D[r,s] = sigma_delta * |A[r,i]|
            sigma_shift_mat = sigma_delta_scalar * np.abs(A_mat)  # (m, m_in)
            # mu_post = mu_golden (no mean shift; E[delta]=0)
            mu_post = mu_golden[:, :, None]  # (m, n, 1) broadcast over i

            sigma_post = np.sqrt(
                sigma_base[:, :, None]**2 + sigma_shift_mat[:, None, :]**2
            )  # (m, n, m_in)

            p_sdc_mat = _tail_gauss_broadcast(mu_post, sigma_post, threshold[:, :, None])
            p_sdc_contribution = p_c * p_fault_cell * p_sdc_mat.sum(axis=2)  # (m, n)
            p_sdc_total += p_sdc_contribution

    # Apply p_fault scaling
    # p_sdc_total currently = E[SDC | fault occurs] summed over all (i,class) weighted by p_fault_cell
    # Multiplied by n_in to cancel: per (r,s), sum over i*class = sum over column j=s faults
    # The H4 factor: we only counted j=s faults, which has prob 1/n_in of all faults
    # But we need to sum over all faults → multiply back by n_in
    p_sdc_exec = p_fault * n_in * p_sdc_total  # (m, n)
    p_otr_exec = p_fault * n_in * p_otr_total  # (m, n)
    p_msk_exec = 1.0 - p_sdc_exec - p_otr_exec  # (m, n)

    # Clip and average over output cells
    p_sdc_exec = np.clip(p_sdc_exec, 0, 1)
    p_otr_exec = np.clip(p_otr_exec, 0, 1)
    p_msk_exec = np.clip(p_msk_exec, 0, 1)

    return (
        float(np.mean(p_msk_exec)),
        float(np.mean(p_sdc_exec)),
        float(np.mean(p_otr_exec)),
    )


def _tail_gauss_broadcast(
    mu_post: np.ndarray,
    sigma_post: np.ndarray,
    threshold: np.ndarray,
) -> np.ndarray:
    """Vectorized tail_gauss for 3D arrays. Returns P(|X| > threshold) shape-compatible."""
    from scipy.stats import norm as scipy_norm

    degenerate = sigma_post < 1e-300
    sigma_safe = np.where(degenerate, 1.0, sigma_post)

    z_upper = (threshold - mu_post) / sigma_safe
    z_lower = (-threshold - mu_post) / sigma_safe

    log_p_upper = scipy_norm.logsf(z_upper)
    log_p_lower = scipy_norm.logcdf(z_lower)

    log_p_sdc = sp_logsumexp(
        np.stack([log_p_upper, log_p_lower], axis=-1), axis=-1
    )
    p_sdc = np.exp(np.clip(log_p_sdc, -750.0, 0.0))

    # Override degenerate: deterministic check
    det_sdc = (np.abs(mu_post) > threshold).astype(np.float64)
    result = np.where(degenerate, det_sdc, p_sdc)
    zero_thresh = threshold <= 0.0
    result = np.where(zero_thresh & ~degenerate, 1.0, result)
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# M3.2 + M3.3: compute_per_cell_SDC — loop-based reference (for validation)
# ---------------------------------------------------------------------------

def compute_per_cell_SDC(
    M_D: np.ndarray,
    U_D: np.ndarray,
    V_D: np.ndarray,
    A: np.ndarray,
    v: float,
    eps: float,
    p_fault: float,
    m_input: Optional[int] = None,
    n_input: Optional[int] = None,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Compute (P_MSK, P_SDC, P_OTR) for each output cell (r,s).

    Uses H4 symmetry reduction: for output cell (r,s), only faults in
    input column j=s shift the mean. Faults in j≠s leave D[r,s] unchanged
    (i.e., contribute to MSK). This is mathematically exact for 2MM.

    Per-output-cell K reduction:
    - Only i such that A[r,i] != 0 gives non-trivial shift (for j=s fixed).
    - For A=I_32: only i=r gives A[r,i]=1; K=6 (1 baseline + 5 classes).
    - For general A: K <= 1 + 5 * m (all rows) in worst case.

    Args:
        M_D, U_D, V_D: Baseline D distribution from compute_baseline.
        A: Kernel matrix (m x m).
        v: Flat input value (all B cells = v for Step 1).
        eps: SDC relative threshold.
        p_fault: Per-execution fault probability.
        m_input, n_input: Input dimensions (default: A.shape).

    Returns:
        dict {(r,s): {'MSK': float, 'SDC': float, 'OTR': float}}
    """
    m = A.shape[0]
    n = M_D.shape[1]
    m_in = m_input or m
    n_in = n_input or n

    results = {}

    # Per output cell (r,s): only faults in column j=s matter
    # P(fault in column j=s) = m_in / (m_in * n_in) = 1/n_in
    # (uniform over all m_in * n_in input cells; fault is in row i, col j=s)
    p_fault_in_col_s = m_in / (m_in * n_in)  # = 1/n_in

    # Suppress mantissa warnings for performance (already documented)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MantissaApproximationWarning)

        for r in range(m):
            for s in range(n):
                mu_golden = float(M_D[r, s])

                # Per-cell variance from the Gaussian distribution
                # D[r,s] ~ N(mu_golden, sigma_D[r,s]^2)
                # sigma_D[r,s]^2 = U_D[r,r] * V_D[s,s] (Kronecker structure)
                sigma_base_sq = float(U_D[r, r]) * float(V_D[s, s])
                sigma_base = float(np.sqrt(max(sigma_base_sq, 0.0)))

                threshold = eps * abs(mu_golden) if abs(mu_golden) > 1e-10 else eps

                # Initialize: probability mass NOT in column j=s is always MSK
                # P(fault NOT in col s) = (n_in - 1) / n_in → contributes to MSK
                p_not_col_s = (n_in - 1) / n_in
                # P(fault in col s) = 1/n_in → needs further analysis

                p_sdc_from_col_s = 0.0
                p_otr_from_col_s = 0.0

                # Iterate over input rows i (H4: only j=s matters)
                # P(fault at (i, j=s)) = 1 / (m_in * n_in) each
                p_fault_cell = 1.0 / (m_in * n_in)

                for i in range(m_in):
                    A_ri = float(A[r, i])

                    for fc in FaultClass:
                        p_c = FAULT_CLASS_PROB[fc]

                        # H1: overflow pre-check (float64 internal, float32 threshold)
                        is_ovf, _ = check_overflow(float(v), fc, A_ri)
                        if is_ovf:
                            # OTR: classify entire output as OTR
                            p_otr_from_col_s += p_fault_cell * p_c
                            continue

                        # Shift on input B[i,s] → shift on output D[r,s] = delta * A[r,i]
                        E_delta, Var_delta = shift_moments(float(v), fc)
                        delta_mean_on_output = float(E_delta) * A_ri
                        sigma_shift_on_output = float(np.sqrt(max(Var_delta, 0.0))) * abs(A_ri)

                        mu_post = mu_golden + delta_mean_on_output
                        sigma_post = float(np.sqrt(sigma_base**2 + sigma_shift_on_output**2))

                        # H2: tail probability in log-space
                        p_sdc_c = float(tail_gauss(mu_post, sigma_post, threshold))
                        p_sdc_from_col_s += p_fault_cell * p_c * p_sdc_c

                # Total per-execution probabilities
                # Given a fault occurs:
                #   - P(fault NOT in j=s) * P(fault given exec) * 1 = (n-1)/n * p_fault → MSK
                #   - P(fault in j=s) * p_fault * P(SDC from col s) → SDC
                # Note: p_sdc_from_col_s already accounts for cell probability
                p_sdc_total = p_fault * n_in * p_sdc_from_col_s
                p_otr_total = p_fault * n_in * p_otr_from_col_s
                p_msk_total = 1.0 - p_sdc_total - p_otr_total

                results[(r, s)] = {
                    "MSK": float(np.clip(p_msk_total, 0, 1)),
                    "SDC": float(np.clip(p_sdc_total, 0, 1)),
                    "OTR": float(np.clip(p_otr_total, 0, 1)),
                }

    return results


# ---------------------------------------------------------------------------
# Aggregate results to kernel-level (r,s) -> average over all cells
# ---------------------------------------------------------------------------

def aggregate_results(
    per_cell_results: Dict[Tuple[int, int], Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate per-cell SDC/MSK/OTR to kernel-level averages."""
    if not per_cell_results:
        return {"MSK": 0.0, "SDC": 0.0, "OTR": 0.0}

    msk_vals = [r["MSK"] for r in per_cell_results.values()]
    sdc_vals = [r["SDC"] for r in per_cell_results.values()]
    otr_vals = [r["OTR"] for r in per_cell_results.values()]

    return {
        "MSK": float(np.mean(msk_vals)),
        "SDC": float(np.mean(sdc_vals)),
        "OTR": float(np.mean(otr_vals)),
    }


# ---------------------------------------------------------------------------
# M3.6 helper: v_sweep prediction
# ---------------------------------------------------------------------------

def predict_v_sweep(
    A: np.ndarray,
    M_B_flat: float,
    v_list: List[float],
    eps: float = 1e-3,
    p_fault: float = 1.0,
    U_B_scale: float = 1e-6,  # near-zero baseline uncertainty for flat B
    V_B_scale: float = 1e-6,
) -> Dict[float, Dict[str, float]]:
    """Sweep v_list and compute SOGA resilience predictions.

    For Step 1 (flat input): B has mean = v * ones(m,n) and near-zero variance.
    Uses compute_sdc_vectorized (H3) for <1s per v-point at 32x32.

    Args:
        A: Kernel matrix.
        M_B_flat: Not used (overridden by v per-iteration).
        v_list: Input values to sweep.
        eps: SDC threshold.
        p_fault: Per-execution fault probability.
        U_B_scale: Small value for near-zero row variance of flat B.
        V_B_scale: Small value for near-zero col variance of flat B.

    Returns:
        dict {v: {'MSK': float, 'SDC': float, 'OTR': float}}
    """
    m, n = A.shape
    results = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MantissaApproximationWarning)

        for v in v_list:
            M_B = np.full((m, n), v, dtype=np.float64)
            U_B = U_B_scale * np.eye(m)
            V_B = V_B_scale * np.eye(n)

            M_D, U_D, V_D = compute_baseline(A, M_B, U_B, V_B)

            pr_msk, pr_sdc, pr_otr = compute_sdc_vectorized(
                M_D, U_D, V_D, A, v, eps, p_fault, m_input=m, n_input=n
            )
            results[v] = {
                "MSK": float(np.clip(pr_msk, 0, 1)),
                "SDC": float(np.clip(pr_sdc, 0, 1)),
                "OTR": float(np.clip(pr_otr, 0, 1)),
            }

    return results


# ---------------------------------------------------------------------------
# M5.2 extension: predict with non-trivial U_B, V_B (Step 3 bimodal)
# ---------------------------------------------------------------------------

def predict_bimodal_sweep(
    A: np.ndarray,
    p_list: List[float],
    V_low: float,
    V_high: float,
    eps: float = 1e-3,
    p_fault: float = 1.0,
    m_dense_threshold: int = 16,
) -> Dict[float, Dict[str, float]]:
    """Sweep bimodal mixing weight p and compute SOGA resilience predictions.

    For Step 3 (bimodal Bernoulli prior):
    - If A is sparse (m_dense < threshold): use 2-component GM primary path
      (compute two separate baselines for v=V_high and v=V_low, aggregate).
    - If A is dense (m_dense >= threshold): use single-Gaussian moment-match
      via compute_per_cell_SDC with non-trivial U_B, V_B.

    Args:
        A: Kernel matrix.
        p_list: Mixing weights to sweep.
        V_low, V_high: The two Bernoulli levels.
        eps: SDC threshold.
        p_fault: Per-execution fault probability.
        m_dense_threshold: Minimum nonzero entries per row for dense A.

    Returns:
        dict {p: {'MSK': float, 'SDC': float, 'OTR': float, 'mode': str}}
    """
    m, n = A.shape

    # Determine A density
    m_dense = int(np.max([np.sum(np.abs(A[r]) > 1e-10) for r in range(m)]))
    use_two_component = m_dense < m_dense_threshold

    results = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MantissaApproximationWarning)

        for p in p_list:
            if use_two_component:
                # 2-component GM: component 1 is v=V_high (weight p),
                #                 component 2 is v=V_low (weight 1-p)
                # Aggregate Pr(SDC) = p * Pr_SDC_high + (1-p) * Pr_SDC_low
                # (exact for Bernoulli at Bernoulli level)
                agg_high = _predict_for_v(A, V_high, eps, p_fault, m, n)
                agg_low = _predict_for_v(A, V_low, eps, p_fault, m, n)

                # Weighted average (degenerate: p=0 or p=1 handled gracefully)
                msk = p * agg_high["MSK"] + (1 - p) * agg_low["MSK"]
                sdc = p * agg_high["SDC"] + (1 - p) * agg_low["SDC"]
                otr = p * agg_high["OTR"] + (1 - p) * agg_low["OTR"]
                mode = "2-component-GM"
            else:
                # Single-Gaussian moment-match (CLT valid for dense A)
                # Bernoulli per-cell: mu = p*V_high + (1-p)*V_low
                #                   sigma^2 = p*(1-p)*(V_high - V_low)^2
                mu_v = p * V_high + (1 - p) * V_low
                sigma2_v = p * (1 - p) * (V_high - V_low) ** 2

                M_B = np.full((m, n), mu_v, dtype=np.float64)
                U_B = max(sigma2_v, 1e-30) * np.eye(m)
                V_B = np.eye(n)

                M_D, U_D, V_D = compute_baseline(A, M_B, U_B, V_B)
                per_cell = compute_per_cell_SDC(
                    M_D, U_D, V_D, A, mu_v, eps, p_fault, m_input=m, n_input=n
                )
                agg = aggregate_results(per_cell)
                msk, sdc, otr = agg["MSK"], agg["SDC"], agg["OTR"]
                mode = "single-Gaussian-CLT"

            results[p] = {
                "MSK": float(np.clip(msk, 0, 1)),
                "SDC": float(np.clip(sdc, 0, 1)),
                "OTR": float(np.clip(otr, 0, 1)),
                "mode": mode,
            }

    return results


def _predict_for_v(
    A: np.ndarray,
    v: float,
    eps: float,
    p_fault: float,
    m: int,
    n: int,
    U_B_scale: float = 1e-10,
) -> Dict[str, float]:
    """Internal helper: predict for a single flat input value v (vectorized)."""
    M_B = np.full((m, n), v, dtype=np.float64)
    U_B = U_B_scale * np.eye(m)
    V_B = np.eye(n)
    M_D, U_D, V_D = compute_baseline(A, M_B, U_B, V_B)
    pr_msk, pr_sdc, pr_otr = compute_sdc_vectorized(
        M_D, U_D, V_D, A, v, eps, p_fault, m_input=m, n_input=n
    )
    return {"MSK": float(np.clip(pr_msk, 0, 1)),
            "SDC": float(np.clip(pr_sdc, 0, 1)),
            "OTR": float(np.clip(pr_otr, 0, 1))}
