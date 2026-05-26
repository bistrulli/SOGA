"""
Bit-exact analytical prediction of resilience curves for 2MM 32x32 float32.

HONESTY DISCLAIMER: Input-side fault model. Faults are injected into B
BEFORE D = A @ B is computed. NOT register-level injection. Strada Q discipline.

Architecture: "MC analytical" — same per-bit IEEE 754 struct.pack/unpack routine as
MC reference (simulate_fi_mc.flip_bit), but with closed-form analytical aggregation
instead of sampling.

This is the PRIMARY predictor (bit-exact). The 5-class moment-matched model is
preserved as predict_resilience_soga_5class.py (historical reference).

Key design decisions:
  1. Per-bit XOR shift from lib/bit_fault_table.py (bit-perfect vs MC reference)
  2. OTR aggregation uses PER-EXECUTION semantics (aligned with MC):
     MC classifies the ENTIRE EXECUTION as OTR if ANY output cell is non-finite.
     For A=I_32 (identity): only D[i,j] is affected by B[i,j] fault -> per-execution
     OTR = P(bit b is_special for the faulted cell).
  3. SDC classification uses same epsilon threshold as MC reference.
  4. 2-component exact aggregation for Bernoulli prior with A=I_32 (no CLT needed).

References:
  - plan/2026-05-26-bit-exact-fault-model.md (R2)
  - lib/DESIGN_BIT_EXACT.md §OTR_SEMANTICS
  - simulate_fi_mc.classify_outcome (MC reference)
"""

from __future__ import annotations

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
from lib.bit_fault_table import bit_fault_table_array


# ---------------------------------------------------------------------------
# Reuse compute_baseline from 5-class model (no changes needed)
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
# R2.2 + R2.2b: compute_per_cell_SDC — bit-exact with aligned OTR semantics
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
    """Compute (P_MSK, P_SDC, P_OTR) for each output cell (r,s) using bit-exact XOR shifts.

    OTR AGGREGATION (R2.2b, aligned with MC):
    MC classify_outcome returns (0,0,1) if np.any(~np.isfinite(D_perturbed)) — PER-EXECUTION.
    For A=I_32: a fault at B[i,j] only affects D[i,j], so OTR = P(v_post is_special).
    For general A: OTR = P(any output cell non-finite) = P(fault bit is_special AND A[r,i]!=0).

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

    # Pre-compute bit-fault table for v: shape (1, 32, 2) -> (32, 2)
    v_arr = np.array([float(v)], dtype=np.float64)
    fault_table = bit_fault_table_array(v_arr)[0]  # (32, 2): (delta, is_special)
    deltas = fault_table[:, 0]       # (32,) float64 shift per bit
    is_special = fault_table[:, 1].astype(bool)  # (32,) bool

    # Count special bits (bits that produce non-finite output)
    n_special = int(np.sum(is_special))
    n_finite = 32 - n_special

    results = {}

    # For each output cell (r, s):
    for r in range(m):
        for s in range(n):
            mu_golden = float(M_D[r, s])
            sigma_base_sq = float(U_D[r, r]) * float(V_D[s, s])
            sigma_base = float(np.sqrt(max(sigma_base_sq, 0.0)))
            threshold = eps * abs(mu_golden) if abs(mu_golden) > 1e-10 else eps

            # Accumulate per-bit contributions
            # Uniform fault model: fault at (i, j, bit) with prob p_fault/(m_in*n_in*32)
            # H4: for output cell (r,s), only faults in column j=s affect D[r,s]
            # For column j=s: sum over i in [m_in] and all 32 bits

            p_sdc_cell = 0.0
            p_otr_cell = 0.0  # per-cell OTR (not used for aggregation below)

            p_fault_per_scenario = 1.0 / (m_in * n_in * 32)

            for i in range(m_in):
                A_ri = float(A[r, i])

                for b in range(32):
                    # OTR: bit b produces non-finite v_post
                    if is_special[b]:
                        # If A_ri != 0: the output D[r,s] would be corrupted by
                        # Inf/NaN propagation -> OTR for this cell
                        if abs(A_ri) > 0.0:
                            p_otr_cell += p_fault_per_scenario
                        continue

                    # Non-special: compute delta on output D[r,s]
                    delta_b = deltas[b] * A_ri  # shift propagated through A

                    # SDC: |delta_b| > threshold (for near-zero sigma_base)
                    # For non-zero sigma_base: use deterministic check since
                    # sigma_base is near-zero for flat input (U_B_scale = 1e-6)
                    if sigma_base < 1e-10:
                        # Deterministic classification
                        if abs(delta_b) > threshold:
                            p_sdc_cell += p_fault_per_scenario
                    else:
                        # Probabilistic: Gaussian uncertainty around delta_b
                        from scipy.stats import norm as scipy_norm
                        from scipy.special import logsumexp as sp_logsumexp
                        # P(|delta_b + noise| > threshold) where noise ~ N(0, sigma_base^2)
                        z_upper = (threshold - delta_b) / sigma_base
                        z_lower = (-threshold - delta_b) / sigma_base
                        log_p_sdc = sp_logsumexp([
                            scipy_norm.logsf(z_upper),
                            scipy_norm.logcdf(z_lower)
                        ])
                        p_sdc_cell += p_fault_per_scenario * float(np.exp(
                            np.clip(log_p_sdc, -750.0, 0.0)
                        ))

            # Scale by p_fault — NO n_in factor.
            # p_fault_per_scenario = 1/(m_in*n_in*32) already represents uniform sampling
            # of (fault_cell=(i,j), bit=b). Summing over (i in [m_in], b in [32]) for fixed j=s
            # gives the per-(r,s) SDC probability directly without renormalization.
            p_sdc_exec = float(np.clip(p_fault * p_sdc_cell, 0, 1))
            p_otr_exec = float(np.clip(p_fault * p_otr_cell, 0, 1))
            p_msk_exec = float(np.clip(1.0 - p_sdc_exec - p_otr_exec, 0, 1))

            results[(r, s)] = {
                "MSK": p_msk_exec,
                "SDC": p_sdc_exec,
                "OTR": p_otr_exec,
            }

    return results


# ---------------------------------------------------------------------------
# R2.3: compute_sdc_vectorized — vectorized bit-exact aggregation
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
    """Vectorized bit-exact analytical prediction of (Pr_MSK, Pr_SDC, Pr_OTR).

    Fully numpy-vectorized over all (r, i, b) dimensions simultaneously.
    Performance target: <200ms per v-point for 32x32.

    OTR semantics (R2.2b): aligned with MC per-execution OTR.
    For A=I_32: P_OTR_exec = n_special / 32 (uniform over 32 bits, only 1 matters
    per execution = bit 30 on v~1.0 giving +Inf).

    Args:
        M_D, U_D, V_D: Baseline distribution.
        A: Kernel matrix (m x m).
        v: Flat input value.
        eps: SDC threshold.
        p_fault: Per-execution fault probability.
        m_input, n_input: Input dimensions.

    Returns:
        (Pr_MSK, Pr_SDC, Pr_OTR) averaged over output cells.
    """
    m = A.shape[0]
    n = M_D.shape[1]
    m_in = m_input or m
    n_in = n_input or n

    # Bit-fault table for v: shape (32, 2)
    v_arr = np.array([float(v)], dtype=np.float64)
    fault_table = bit_fault_table_array(v_arr)[0]  # (32, 2)
    deltas = fault_table[:, 0]       # (32,) delta per bit
    is_special = fault_table[:, 1].astype(bool)  # (32,)

    # OTR (per-execution, aligned with MC): P(any output cell non-finite | fault)
    # For A=I_32: only one cell affected per fault; OTR iff bit is special
    # Aggregated over all (i,j,b) uniformly: p_otr = p_fault * n_special/32
    # For general A: at least one A[r,i]!=0 => non-finite propagates
    # Check: for each (i,b) with is_special, does any A[r,i]!=0? (for identity: yes iff i<m)
    n_special_effective = int(np.sum(is_special))  # bits that cause non-finite for nonzero A_ri
    p_otr_exec = p_fault * n_special_effective / 32.0

    # SDC: vectorized over all (r, i, b) simultaneously
    # delta_out[r, i, b] = deltas[b] * A[r, i] — shift on D[r,s] from fault at (i, j=s, b)
    # Shape: A (m, m_in), deltas (32,) -> delta_out (m, m_in, 32)
    A_mat = A[:m, :m_in]  # (m, m_in)
    delta_out = A_mat[:, :, None] * deltas[None, None, :]  # (m, m_in, 32)

    # Mask special bits (they go to OTR, not SDC)
    not_special = ~is_special  # (32,)
    delta_finite = delta_out * not_special[None, None, :]  # (m, m_in, 32); 0 for special bits

    # threshold[r, s]: (m, n) — for flat input, threshold is same for all s
    # threshold[r, :] = eps * |M_D[r, 0]| (same for all s since B is flat)
    mu_golden = M_D  # (m, n)
    threshold = np.where(np.abs(mu_golden) > 1e-10, eps * np.abs(mu_golden), eps)  # (m, n)

    # sigma_base: (m, n) — near-zero for flat input (U_B_scale << 1)
    U_diag = np.diag(U_D)  # (m,)
    V_diag = np.diag(V_D)  # (n,)
    sigma_base = np.sqrt(np.maximum(np.outer(U_diag, V_diag), 0.0))  # (m, n)

    # p_per_scen: probability of each (fault_cell, bit) combination
    p_per_scen = 1.0 / (m_in * n_in * 32)

    if np.all(sigma_base < 1e-10):
        # Near-deterministic (flat input with tiny variance): use deterministic SDC check
        # For flat input: threshold[r, s] is same for all s (same M_D[r,s] for all s)
        # Use threshold[:, 0]: (m,)
        thresh_1d = threshold[:, 0]  # (m,) — representative for all s

        # SDC flag: |delta_finite[r, i, b]| > threshold[r]
        # Shape: (m, m_in, 32), threshold broadcast: (m, 1, 1)
        sdc_flag = (np.abs(delta_finite) > thresh_1d[:, None, None]) & not_special[None, None, :]

        # P_SDC[r,s] = p_fault * p_per_scen * sum_{i,b} sdc_flag[r,i,b]
        # NOTE: No n_in multiplication here. The p_per_scen = 1/(m_in*n_in*32) already
        # represents the probability of each (fault_cell=(i,j), bit=b) uniformly sampled.
        # For H4: output cell (r,s) is only affected by faults in column j=s. Faults in
        # j≠s give zero delta (for flat input with H4). So sdc_flag already sums over
        # only the relevant (i=r, b=SDC bits) — no extra factor needed.
        # The MC classify_outcome reports fraction of SDC cells per execution, which equals
        # p_per_scen * sum_{(i,j,b)} I(fault (i,j,b) makes (r,s) SDC) (averaged over (r,s)).
        p_sdc_per_r = p_fault * p_per_scen * sdc_flag.sum(axis=(1, 2))  # (m,) — NO n_in factor

        # Broadcast to (m, n) — for flat input, SDC is same for all s
        p_sdc_exec_arr = np.clip(np.tile(p_sdc_per_r[:, None], (1, n)), 0, 1)  # (m, n)
    else:
        # Non-degenerate sigma_base: vectorized Gaussian tail probability
        # P_SDC[r,s] = sum_{i,b} p_per_scen * P(|delta_finite[r,i,b] + noise| > threshold[r,s])
        # where noise ~ N(0, sigma_base[r,s]^2)
        from scipy.stats import norm as scipy_norm
        from scipy.special import logsumexp as sp_logsumexp

        # Expand dimensions for broadcasting:
        # delta_finite: (m, m_in, 32) -> (m, 1, m_in, 32) for (r, s, i, b)
        # threshold: (m, n) -> (m, n, 1, 1)
        # sigma_base: (m, n) -> (m, n, 1, 1)
        d4 = delta_finite[:, None, :, :]  # (m, 1, m_in, 32)
        th4 = threshold[:, :, None, None]  # (m, n, 1, 1)
        sg4 = np.where(sigma_base[:, :, None, None] > 1e-300, sigma_base[:, :, None, None], 1.0)

        z_upper = (th4 - d4) / sg4
        z_lower = (-th4 - d4) / sg4

        log_p_sdc_4d = sp_logsumexp(
            np.stack([scipy_norm.logsf(z_upper), scipy_norm.logcdf(z_lower)], axis=-1),
            axis=-1,
        )  # (m, n, m_in, 32)
        p_sdc_4d = np.exp(np.clip(log_p_sdc_4d, -750.0, 0.0))

        # Mask special bits
        p_sdc_4d *= not_special[None, None, None, :]

        # Sum over (i, b) and scale — NO n_in factor (see deterministic path above)
        p_sdc_exec_arr = np.clip(
            p_fault * p_per_scen * p_sdc_4d.sum(axis=(2, 3)),
            0, 1,
        )  # (m, n)

    p_msk_exec_arr = np.clip(1.0 - p_sdc_exec_arr - p_otr_exec, 0, 1)

    return (
        float(np.mean(p_msk_exec_arr)),
        float(np.mean(p_sdc_exec_arr)),
        float(np.clip(p_otr_exec, 0, 1)),
    )


# ---------------------------------------------------------------------------
# Aggregate per-cell results
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
# R2.2: predict_v_sweep — bit-exact Step 1 sweep
# ---------------------------------------------------------------------------

def predict_v_sweep(
    A: np.ndarray,
    M_B_flat: float,
    v_list: List[float],
    eps: float = 1e-3,
    p_fault: float = 1.0,
    U_B_scale: float = 1e-6,
    V_B_scale: float = 1e-6,
) -> Dict[float, Dict[str, float]]:
    """Sweep v_list and compute bit-exact SOGA resilience predictions.

    For Step 1 (flat input): B has mean = v * ones(m,n) and near-zero variance.
    Uses compute_sdc_vectorized for performance.

    Args:
        A: Kernel matrix.
        M_B_flat: Not used directly (overridden by v per-iteration).
        v_list: Input values to sweep.
        eps: SDC threshold.
        p_fault: Per-execution fault probability.
        U_B_scale: Small row variance for near-deterministic flat B.
        V_B_scale: Small col variance.

    Returns:
        dict {v: {'MSK': float, 'SDC': float, 'OTR': float}}
    """
    m, n = A.shape
    results = {}

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
# R4.1: predict_bimodal_sweep — bit-exact Step 3 bimodal Bernoulli
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
    """Sweep bimodal mixing weight p with bit-exact kernel.

    For Step 3 (bimodal Bernoulli prior):
    - Sparse A (m_dense < threshold): 2-component exact dispatch
      Pr(SDC|p) = p * Pr_SDC(V_high) + (1-p) * Pr_SDC(V_low)
      (exact for Bernoulli with independent per-cell inputs)
    - Dense A: single Gaussian moment-match path (CLT valid, kept for completeness)

    The bit-exact kernel is plumbed into _predict_for_v (replaces 5-class model).

    Args:
        A: Kernel matrix.
        p_list: Mixing weights to sweep.
        V_low, V_high: The two Bernoulli levels.
        eps: SDC threshold.
        p_fault: Per-execution fault probability.
        m_dense_threshold: Minimum nonzero entries per row for dense A path.

    Returns:
        dict {p: {'MSK': float, 'SDC': float, 'OTR': float, 'mode': str}}
    """
    m, n = A.shape

    # Determine A density
    m_dense = int(np.max([np.sum(np.abs(A[r]) > 1e-10) for r in range(m)]))
    use_two_component = m_dense < m_dense_threshold

    results = {}

    for p in p_list:
        if use_two_component:
            # 2-component exact: Pr(SDC|p) = p * Pr_SDC_high + (1-p) * Pr_SDC_low
            agg_high = _predict_for_v(A, V_high, eps, p_fault, m, n)
            agg_low = _predict_for_v(A, V_low, eps, p_fault, m, n)

            msk = p * agg_high["MSK"] + (1 - p) * agg_low["MSK"]
            sdc = p * agg_high["SDC"] + (1 - p) * agg_low["SDC"]
            otr = p * agg_high["OTR"] + (1 - p) * agg_low["OTR"]
            mode = "2-component-GM"
        else:
            # Single-Gaussian moment-match (CLT valid for dense A)
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
    """Internal helper: predict for a single flat input value v (bit-exact)."""
    M_B = np.full((m, n), v, dtype=np.float64)
    U_B = U_B_scale * np.eye(m)
    V_B = np.eye(n)
    M_D, U_D, V_D = compute_baseline(A, M_B, U_B, V_B)
    pr_msk, pr_sdc, pr_otr = compute_sdc_vectorized(
        M_D, U_D, V_D, A, v, eps, p_fault, m_input=m, n_input=n
    )
    return {
        "MSK": float(np.clip(pr_msk, 0, 1)),
        "SDC": float(np.clip(pr_sdc, 0, 1)),
        "OTR": float(np.clip(pr_otr, 0, 1)),
    }
