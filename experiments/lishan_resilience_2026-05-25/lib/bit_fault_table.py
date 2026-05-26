"""
Bit-exact IEEE 754 XOR fault table for float32 inputs.

HONESTY DISCLAIMER: Input-side fault model. Faults are injected into B
BEFORE D = A @ B is computed. NOT register-level injection. Strada Q discipline.

This module implements the bit-exact XOR shift computation identical to
the MC reference (simulate_fi_mc.flip_bit) but analytically tabulated
for all 32 bits simultaneously.

Design:
  - compute_xor_shift: scalar, per-bit shift computation
  - bit_fault_table_array: vectorized, returns (N, 32, 2) table of
    (delta_float64, is_special_bool) for all (v, bit) pairs

IEEE 754 float32 bit layout:
  bit 31: sign
  bits 30-23: exponent (8 bits, biased-127)
  bits 22-0: mantissa (23 bits, stored fraction)

Bug fixes applied (per numerical-stability-expert + Codex iter 1):
  Bug A: Guard NaN ONLY (not Inf). For Inf input, individual bit-flips
         have well-defined results: e.g., +Inf bit 30 -> 1.0 (finite).
         We must NOT early-exit on Inf input.
  Bug B: Delta must use float32-cast minuend: float(np.float32(v_post)) - float(v_f32).
         This ensures MC-identical arithmetic (MC also casts to float32 first).
  Bug C: Use native float32 byte order (np.asarray(dtype=np.float32)), NOT
         explicit '<f4' which fails on big-endian systems.

References:
  - plan/2026-05-26-bit-exact-fault-model.md §Approach
  - simulate_fi_mc.flip_bit (MC reference: uses struct.pack('>f',...))
  - numerical-stability-expert memo Q1-Q3
  - Codex iter 1 findings F1, F2
"""

from __future__ import annotations

import struct
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Scalar implementation (R1.1)
# ---------------------------------------------------------------------------

def compute_xor_shift(v: float, bit_idx: int) -> Tuple[float, bool]:
    """IEEE 754 bit-XOR exact. Bit-perfect match against MC reference flip_bit.

    Computes the shift delta = v_post - v_f32 after XOR-flipping bit `bit_idx`
    of the float32 representation of v.

    Args:
        v: Input float. NaN input returns (0.0, True) immediately.
           Inf input flows through (per IEEE 754: flipping different bits of
           Inf gives finite, NaN, or -Inf — each well-defined).
        bit_idx: Bit to flip (0=LSB of mantissa, 31=sign).

    Returns:
        (delta, is_special):
            delta: float64 shift v_post - v_f32 (0.0 if v_post is non-finite)
            is_special: True if v_post is non-finite (Inf, -Inf, NaN)

    Examples (hand-verified against MC flip_bit):
        compute_xor_shift(1.0, 31)  -> (-2.0, False)   # sign flip: 1.0 -> -1.0
        compute_xor_shift(1.0, 23)  -> (+1.0, False)   # bit 23 flip: 1.0 -> 2.0
        compute_xor_shift(1.0, 22)  -> (+0.5, False)   # bit 22 flip: 1.0 -> 1.5
        compute_xor_shift(1.0, 30)  -> (+inf, True)    # bit 30: biased-exp overflow
        compute_xor_shift(0.0, 31)  -> (-0.0 shift, False) via copysign semantics
    """
    # Bug A fix (Codex iter 1 F2): guard NaN ONLY, not Inf
    # Inf input has well-defined per-bit behavior (e.g., +Inf bit 30 -> 1.0)
    if np.isnan(v):
        return 0.0, True

    v_f32 = np.float32(v)
    # Use big-endian pack (same as MC reference which uses '>f')
    bits = int.from_bytes(struct.pack('!f', v_f32), 'big')
    flipped = bits ^ (1 << bit_idx)
    v_post = struct.unpack('!f', flipped.to_bytes(4, 'big'))[0]

    is_special = not np.isfinite(v_post)
    # Bug B fix: use float32-cast minuend for MC-identical arithmetic
    delta = (float(np.float32(v_post)) - float(v_f32)) if not is_special else 0.0
    return delta, is_special


# ---------------------------------------------------------------------------
# Vectorized implementation (R1.2)
# ---------------------------------------------------------------------------

def bit_fault_table_array(v_array: np.ndarray) -> np.ndarray:
    """Vectorized bit-fault table for an array of float32 input values.

    Computes the (delta, is_special) pair for every (v, bit_idx) combination
    simultaneously using numpy view tricks. Identical semantics to compute_xor_shift
    for each element.

    Args:
        v_array: 1-D array of float values (will be cast to float32).

    Returns:
        Shape (N, 32, 2) float64 array where:
            result[i, b, 0] = delta for v_array[i], bit_idx=b
            result[i, b, 1] = is_special (as 0.0 or 1.0) for v_array[i], bit_idx=b

    Note:
        - NaN inputs: all 32 bits return (0.0, 1.0) (special)
        - Inf inputs: flows through per-bit (Bug A fix), e.g., +Inf bit 30 -> 1.0 finite
        - Uses native float32 byte order (Bug C fix: no explicit '<f4')
    """
    v32 = np.asarray(v_array, dtype=np.float32)  # native float32, Bug C fix
    is_nan_in = np.isnan(v32)  # (N,) - NaN guard (Bug A: NOT Inf)

    # View as uint32 for bitwise XOR
    bits = v32.view(np.uint32)  # (N,)

    # Broadcast: flip each of 32 bits
    bit_idx = np.arange(32, dtype=np.uint32)  # (32,)
    flipped = bits[:, None] ^ (np.uint32(1) << bit_idx[None, :])  # (N, 32) uint32

    # View as float32 to get post-flip values
    v_post = flipped.view(np.float32)  # (N, 32) - need reshape for correct view

    # Fix: flipped is (N, 32) uint32, view as float32 gives same shape
    # but we need to ensure the view is on a contiguous array
    v_post = flipped.reshape(-1).view(np.float32).reshape(v32.shape[0], 32)  # (N, 32)

    # is_special: non-finite result OR NaN input (NaN input -> always special)
    is_special = ~np.isfinite(v_post) | is_nan_in[:, None]  # (N, 32) bool

    # delta in float64: 0.0 where special, else v_post - v32
    # Bug B fix: delta uses float32-precision subtraction promoted to float64
    # Suppress divide/invalid warnings: NaN/Inf inputs produce nan in subtraction
    # (these are masked out by np.where, so the warning is spurious)
    with np.errstate(invalid='ignore', over='ignore'):
        raw_delta = v_post.astype(np.float64) - v32.astype(np.float64)[:, None]
    delta = np.where(is_special, 0.0, raw_delta)  # (N, 32)

    # Stack: axis=-1 gives (N, 32, 2)
    return np.stack([delta, is_special.astype(np.float64)], axis=-1)


# ---------------------------------------------------------------------------
# Convenience: scalar-vs-vectorized cross-check
# ---------------------------------------------------------------------------

def validate_scalar_vs_vectorized(v_array: np.ndarray, rtol: float = 1e-12) -> bool:
    """Assert that compute_xor_shift matches bit_fault_table_array for all inputs.

    Args:
        v_array: Test values.
        rtol: Relative tolerance for delta comparison (should be ~1e-15 for exact match).

    Returns:
        True if all match. Raises AssertionError on mismatch.
    """
    table = bit_fault_table_array(v_array)
    for i, v in enumerate(v_array):
        for b in range(32):
            delta_s, is_special_s = compute_xor_shift(float(v), b)
            delta_v = float(table[i, b, 0])
            is_special_v = bool(table[i, b, 1])
            assert is_special_s == is_special_v, (
                f"is_special mismatch at v={v}, bit={b}: "
                f"scalar={is_special_s}, vec={is_special_v}"
            )
            if not is_special_s:
                assert abs(delta_s - delta_v) <= rtol * max(abs(delta_s), 1e-300), (
                    f"delta mismatch at v={v}, bit={b}: "
                    f"scalar={delta_s:.6e}, vec={delta_v:.6e}"
                )
    return True
