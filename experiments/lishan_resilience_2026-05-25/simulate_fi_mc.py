"""
M2 — Monte Carlo fault injection reference simulator.

HONESTY DISCLAIMER: Input-side fault model. Faults are injected into B
BEFORE D = A @ B is computed. NOT register-level injection. Strada Q discipline.

Usage:
    python3 experiments/lishan_resilience_2026-05-25/simulate_fi_mc.py
    (or import as library for run_mc_step1.py)

Implements:
    - flip_bit(value, bit_index) -> float32
    - simulate_one_fi(A, B, fault_cell, bit) -> D_perturbed
    - simulate_v_sweep(A, v_list, n_samples, seed) -> dict
    - simulate_p_sweep(A, p_list, V_low, V_high, n_samples, seed) -> dict
      (Step 3: bimodal Bernoulli prior)
"""

from __future__ import annotations

import os
import struct
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)


# ---------------------------------------------------------------------------
# M2.1: flip_bit — IEEE 754 float32 bit-flip via struct
# ---------------------------------------------------------------------------

def flip_bit(value: float, bit_index: int) -> float:
    """Flip bit `bit_index` (0=LSB, 31=MSB) of IEEE 754 float32.

    Args:
        value: Input float (will be cast to float32 before bit manipulation).
        bit_index: Which bit to flip (0-31).

    Returns:
        float value after XOR of the specified bit (as float64).

    Verification cases:
        flip_bit(1.0, 31) = -1.0   (sign flip on 1.0)
        flip_bit(-1.0, 31) = 1.0   (sign flip on -1.0)
        flip_bit(0.0, 31) = -0.0   (sign flip on 0 → -0.0; == 0 in Python)
        flip_bit(1.0, 23) ≠ 1.0    (flip lowest exponent bit → different value)
    """
    if not (0 <= bit_index <= 31):
        raise ValueError(f"bit_index must be in [0,31], got {bit_index}")

    # Cast to float32 first to ensure proper IEEE 754 representation
    val32 = np.float32(value)
    # Pack float32 as big-endian unsigned 32-bit int
    packed = struct.pack('>f', val32)
    bits = struct.unpack('>I', packed)[0]
    # XOR the specified bit
    bits ^= (1 << bit_index)
    # Unpack back to float32
    result_packed = struct.pack('>I', bits)
    result = struct.unpack('>f', result_packed)[0]
    return float(result)


def flip_bit_array(values: np.ndarray, bit_index: int) -> np.ndarray:
    """Vectorized flip_bit for an array of float32 values."""
    vals32 = values.astype(np.float32)
    # View as uint32 for bitwise operations
    bits = vals32.view(np.uint32).copy()
    bits ^= np.uint32(1 << bit_index)
    return bits.view(np.float32).astype(np.float64)


# ---------------------------------------------------------------------------
# M2.2: simulate_one_fi — single fault injection
# ---------------------------------------------------------------------------

def simulate_one_fi(
    A: np.ndarray,
    B: np.ndarray,
    fault_cell: Tuple[int, int],
    bit: int,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply single bit-flip fault to B, recompute D = A @ B.

    Args:
        A: Kernel matrix (m x m), float64.
        B: Input matrix (m x n), float32 or float64.
        fault_cell: (row, col) index of faulted cell in B.
        bit: Bit index to flip (0-31).
        eps: SDC threshold fraction (not used here, returned in classification).

    Returns:
        (D_baseline, D_perturbed, mask) where:
            D_baseline: A @ B (golden output, float64)
            D_perturbed: A @ B_perturbed (fault output, float64)
            mask: boolean matrix, True where |D_perturbed[r,s] - D_baseline[r,s]|
                  > eps * |D_baseline[r,s]|
    """
    B_f32 = B.astype(np.float32)
    D_baseline = (A @ B_f32.astype(np.float64))

    B_perturbed = B_f32.copy()
    r, c = fault_cell
    B_perturbed[r, c] = np.float32(flip_bit(float(B_perturbed[r, c]), bit))

    D_perturbed = (A @ B_perturbed.astype(np.float64))

    # Classify: SDC, OTR, MSK
    # OTR: any output cell is inf or nan
    is_otr = np.any(~np.isfinite(D_perturbed))

    return D_baseline, D_perturbed, is_otr


# ---------------------------------------------------------------------------
# M2.3: simulate_v_sweep — full Step 1 MC sweep
# ---------------------------------------------------------------------------

def classify_outcome(
    D_baseline: np.ndarray,
    D_perturbed: np.ndarray,
    eps: float,
) -> Tuple[float, float, float]:
    """Classify per-output-cell outcomes and return aggregate fractions.

    Returns:
        (Pr_MSK, Pr_SDC, Pr_OTR): fractions summing to 1.0
    """
    # OTR: any output cell non-finite
    if np.any(~np.isfinite(D_perturbed)):
        return 0.0, 0.0, 1.0

    diff = np.abs(D_perturbed - D_baseline)
    golden_abs = np.abs(D_baseline)
    # Threshold per cell: eps * |D_baseline[r,s]|
    # For cells where D_baseline ~ 0, use absolute eps
    threshold = np.where(golden_abs > 1e-10, eps * golden_abs, eps)
    sdc_mask = diff > threshold

    n_total = D_baseline.size
    n_sdc = int(np.sum(sdc_mask))
    n_msk = n_total - n_sdc

    return n_msk / n_total, n_sdc / n_total, 0.0


def simulate_v_sweep(
    A: np.ndarray,
    v_list: List[float],
    n_samples: int = 1000,
    seed: int = 42,
    eps: float = 1e-3,
    p_fault: float = 1.0,
) -> Dict[float, Dict[str, float]]:
    """MC fault injection sweep over input values v.

    For each v:
    - Generate flat B (all entries = v) as float32
    - Sample n_samples (fault_cell, bit) pairs uniformly
    - For each sample: apply flip_bit, compute D_perturbed, classify MSK/SDC/OTR
    - Aggregate per-output-cell classification

    Args:
        A: Kernel matrix (m x m).
        v_list: List of input values to sweep.
        n_samples: Number of MC samples per v-point.
        seed: Random seed for reproducibility.
        eps: SDC relative threshold.
        p_fault: Probability of fault occurrence per execution (scales results).

    Returns:
        dict {v: {'MSK': float, 'SDC': float, 'OTR': float, 'time_s': float}}
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape[0], A.shape[1]
    results = {}

    for v in v_list:
        t0 = time.time()
        B_flat = np.full((m, n), v, dtype=np.float32)
        D_baseline = A @ B_flat.astype(np.float64)

        msk_count = 0
        sdc_count = 0
        otr_count = 0

        for _ in range(n_samples):
            # Sample fault cell uniformly from (m*n) cells
            flat_idx = int(rng.integers(0, m * n))
            fault_row = flat_idx // n
            fault_col = flat_idx % n
            # Sample bit uniformly from 32 bits
            bit = int(rng.integers(0, 32))

            _, D_perturbed, is_otr_single = simulate_one_fi(
                A, B_flat, (fault_row, fault_col), bit, eps
            )
            pr_msk, pr_sdc, pr_otr = classify_outcome(D_baseline, D_perturbed, eps)

            msk_count += pr_msk
            sdc_count += pr_sdc
            otr_count += pr_otr

        # Average over samples, then scale by p_fault
        pr_msk_avg = msk_count / n_samples
        pr_sdc_avg = sdc_count / n_samples
        pr_otr_avg = otr_count / n_samples

        # Apply p_fault: with probability (1-p_fault) no fault → MSK
        pr_sdc_exec = p_fault * pr_sdc_avg
        pr_otr_exec = p_fault * pr_otr_avg
        pr_msk_exec = 1.0 - pr_sdc_exec - pr_otr_exec

        elapsed = time.time() - t0
        results[v] = {
            "MSK": float(np.clip(pr_msk_exec, 0, 1)),
            "SDC": float(np.clip(pr_sdc_exec, 0, 1)),
            "OTR": float(np.clip(pr_otr_exec, 0, 1)),
            "time_s": elapsed,
            "n_samples": n_samples,
        }

    return results


# ---------------------------------------------------------------------------
# M2.3 extension: simulate_p_sweep — Step 3 bimodal Bernoulli
# ---------------------------------------------------------------------------

def simulate_p_sweep(
    A: np.ndarray,
    p_list: List[float],
    V_low: float,
    V_high: float,
    n_samples: int = 1000,
    seed: int = 42,
    eps: float = 1e-3,
    p_fault: float = 1.0,
) -> Dict[float, Dict[str, float]]:
    """MC fault injection sweep over bimodal Bernoulli mixing weight p.

    For each p:
    - Generate B where each cell is V_high with probability p, else V_low
    - Apply fault injection as in simulate_v_sweep
    - Classify outcomes

    Args:
        A: Kernel matrix (m x m).
        p_list: List of mixing weights p in [0, 1].
        V_low, V_high: The two Bernoulli levels.
        n_samples: MC samples per p-point.
        seed: RNG seed.
        eps: SDC threshold.
        p_fault: Per-execution fault probability.

    Returns:
        dict {p: {'MSK': float, 'SDC': float, 'OTR': float, ...}}
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape[0], A.shape[1]
    results = {}

    for p in p_list:
        t0 = time.time()
        msk_count = 0
        sdc_count = 0
        otr_count = 0

        for _ in range(n_samples):
            # Sample a B matrix from Bernoulli(p)
            bern_mask = rng.random((m, n)) < p
            B_sample = np.where(bern_mask, V_high, V_low).astype(np.float32)
            D_baseline = A @ B_sample.astype(np.float64)

            # Sample fault cell and bit
            flat_idx = int(rng.integers(0, m * n))
            fault_row = flat_idx // n
            fault_col = flat_idx % n
            bit = int(rng.integers(0, 32))

            _, D_perturbed, _ = simulate_one_fi(
                A, B_sample, (fault_row, fault_col), bit, eps
            )
            pr_msk, pr_sdc, pr_otr = classify_outcome(D_baseline, D_perturbed, eps)

            msk_count += pr_msk
            sdc_count += pr_sdc
            otr_count += pr_otr

        pr_sdc_exec = p_fault * sdc_count / n_samples
        pr_otr_exec = p_fault * otr_count / n_samples
        pr_msk_exec = 1.0 - pr_sdc_exec - pr_otr_exec

        elapsed = time.time() - t0
        results[p] = {
            "MSK": float(np.clip(pr_msk_exec, 0, 1)),
            "SDC": float(np.clip(pr_sdc_exec, 0, 1)),
            "OTR": float(np.clip(pr_otr_exec, 0, 1)),
            "time_s": elapsed,
            "n_samples": n_samples,
        }

    return results
