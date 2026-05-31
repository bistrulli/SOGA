"""Shared int32 PolyBench-2MM kernel + bit-flip primitives + Wilson CI.

Used by mc_register.py (register-level FI) and mc_input_side.py (input-side FI).
Single canonical implementation of the kernel so the two fault models differ ONLY
in WHERE the fault is injected, never in the arithmetic.

Conventions (see POLYBENCH_REFERENCE.md):
  - square N x N, alpha=1, beta=0  ->  tmp = A@B, D = tmp@C
  - A, C: deterministic int32 modular init;  B: FLAT input, all entries = v
  - int32 two's-complement wraparound at every multiply and every add
  - 2*N^3 accumulator FMA sites, enumerated i-outer, j-middle, k-inner,
    phase 1 then phase 2
  - SDC = (output != baseline) exactly (eps = 0 for int32); OTR = never (no Inf/NaN)
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

INT32_MIN = -0x80000000
INT32_MAX = 0x7FFFFFFF
UINT32_MASK = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# int32 two's-complement primitives
# ---------------------------------------------------------------------------

def wrap32(x: int) -> int:
    """Wrap a Python int into signed int32 range (two's complement)."""
    x &= UINT32_MASK
    return x - 0x100000000 if x >= 0x80000000 else x


def flip_bit_int32(value: int, bit: int) -> int:
    """XOR bit `bit` (0=LSB, 31=sign) of the int32 two's-complement pattern of value.

    Verification:
        flip_bit_int32(0, 0)  =  1
        flip_bit_int32(1, 0)  =  0          (bit was 1 -> delta = -1)
        flip_bit_int32(1, 31) = -2147483647 (sign bit set -> +2^31 then wrap)
        flip_bit_int32(-1, 31)=  2147483647 (sign bit cleared)
    """
    if not (0 <= bit <= 31):
        raise ValueError(f"bit must be in [0,31], got {bit}")
    u = value & UINT32_MASK          # unsigned two's-complement pattern
    u ^= (1 << bit)
    u &= UINT32_MASK
    return u - 0x100000000 if u >= 0x80000000 else u


# ---------------------------------------------------------------------------
# Deterministic PolyBench init (int32, square N), flat input B
# ---------------------------------------------------------------------------

def polybench_AC(n: int) -> Tuple[List[List[int]], List[List[int]]]:
    """A[i][j]=(i*j+1)%n,  C[i][j]=(i*(j+3)+1)%n  (deterministic int32 kernels)."""
    A = [[(i * j + 1) % n for j in range(n)] for i in range(n)]
    C = [[(i * (j + 3) + 1) % n for j in range(n)] for i in range(n)]
    return A, C


# ---------------------------------------------------------------------------
# Step-by-step kernel with optional register-level fault
# ---------------------------------------------------------------------------

def kernel_2mm(
    A: List[List[int]],
    C: List[List[int]],
    v: int,
    n: int,
    fault_site: int = -1,
    fault_bit: int = -1,
) -> List[List[int]]:
    """Execute the int32 2MM step-by-step; D = (A@B)@C with B flat = v.

    If fault_site >= 0, XOR fault_bit into the accumulator immediately after the
    add at global FMA site `fault_site`. Sites are enumerated 0..2N^3-1
    (phase 1 then phase 2, i-outer/j-middle/k-inner).

    Returns D as an N x N list of signed int32 Python ints.
    """
    site = 0
    # Phase 1: tmp[i][j] = sum_k A[i][k] * B[k][j], B[k][j] = v
    tmp = [[0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for j in range(n):
            acc = 0
            for k in range(n):
                acc = wrap32(acc + wrap32(Ai[k] * v))
                if site == fault_site:
                    acc = flip_bit_int32(acc, fault_bit)
                site += 1
            tmp[i][j] = acc
    # Phase 2: D[i][j] = sum_k tmp[i][k] * C[k][j]
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        ti = tmp[i]
        for j in range(n):
            acc = 0
            for k in range(n):
                acc = wrap32(acc + wrap32(ti[k] * C[k][j]))
                if site == fault_site:
                    acc = flip_bit_int32(acc, fault_bit)
                site += 1
            D[i][j] = acc
    return D


def n_fault_sites(n: int) -> int:
    """Total register FMA sites = 2 * N^3."""
    return 2 * n * n * n


# ---------------------------------------------------------------------------
# Input-side kernel: fault injected into a B cell BEFORE execution
# ---------------------------------------------------------------------------

def kernel_2mm_input_fault(
    A: List[List[int]],
    C: List[List[int]],
    v: int,
    n: int,
    fault_cell: Optional[Tuple[int, int]] = None,
    fault_bit: int = -1,
) -> List[List[int]]:
    """Same kernel, but the fault flips a bit of one B cell before any compute.

    B is flat = v except B[fault_cell] = flip_bit_int32(v, fault_bit).
    """
    # Build B as a flat-v matrix with one corrupted cell.
    if fault_cell is not None and fault_bit >= 0:
        fr, fc = fault_cell
        corrupted = flip_bit_int32(v, fault_bit)
    else:
        fr = fc = -1
        corrupted = v

    site_ignore = -1  # no register fault here
    # Re-implement phase 1 with per-(k,j) B value (column j of B).
    tmp = [[0] * n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for j in range(n):
            acc = 0
            for k in range(n):
                bkj = corrupted if (k == fr and j == fc) else v
                acc = wrap32(acc + wrap32(Ai[k] * bkj))
            tmp[i][j] = acc
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        ti = tmp[i]
        for j in range(n):
            acc = 0
            for k in range(n):
                acc = wrap32(acc + wrap32(ti[k] * C[k][j]))
            D[i][j] = acc
    return D


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% CI for k successes out of n binomial trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - rad), min(1.0, centre + rad))


def matrices_equal(d1: List[List[int]], d2: List[List[int]]) -> bool:
    """Exact equality test (eps = 0 SDC for int32)."""
    return d1 == d2
