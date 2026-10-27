"""Shared model of PolyBench-GPU GEMM under an NVBitFI transient campaign.

Kernel (sgrauerg/polybenchGpu CUDA/GEMM/gemm.cu, verified 2026-10-26):
    c[i*NJ+j] *= BETA;
    for (k = 0; k < NK; k++) c[i*NJ+j] += ALPHA * a[i*NK+k] * b[k*NJ+j];
SDC check (verified): stdout prints the COUNT of elements whose
percentDiff(cpu, gpu) > 0.05 %; percentDiff = 0 when both |v| < 0.01.
NVBitFI default campaign (verified, scripts/params.py): G_GP + FLIP_SINGLE_BIT,
uniform over dynamic instructions that write a general-purpose register,
bit = uniform 0..31 of the 32-bit destination register.

ASSUMED (not verified, no cuobjdump yet): the per-thread dynamic G_GP
instruction mix in `thread_sites`. It is derived from the CUDA source with the
accumulator kept in a register, the loop unrolled by `unroll`, and 64-bit
pointer increments split in lo/hi halves. Replace the class counts with the
NVBitFI profiler output (step 1 of every campaign) when available: per-class
outcome rates do not depend on the mix, only the aggregate does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ALPHA = np.float32(32412.0)
BETA = np.float32(2123.0)
EPS_POLYBENCH = 0.05 / 100.0          # PERCENT_DIFF_ERROR_THRESHOLD, as a fraction
SMALL_FLOAT_VAL = 1e-8
ABS_FLOOR = 0.01

# Fault-site classes. ARITH = value-carrying registers SOGA propagates;
# STRUCT = index/address/loop-counter registers (structural, not modeled by SOGA).
ARITH_CLASSES = ("LDG.c", "FMUL.beta", "LDG.a", "LDG.b", "FMUL.alpha", "FFMA")
STRUCT_CLASSES = ("S2R/IMAD.idx", "IMAD.addr_c", "IMAD.addr_a", "IMAD.addr_b",
                  "IADD.ptr_a.lo", "IADD.ptr_a.hi", "IADD.ptr_b.lo",
                  "IADD.ptr_b.hi", "IADD.k")
ALL_CLASSES = ARITH_CLASSES + STRUCT_CLASSES
CLASS_ID = {c: i for i, c in enumerate(ALL_CLASSES)}


@dataclass(frozen=True)
class Site:
    cls: str
    k: int  # loop iteration for per-iteration sites; counter value after a block for IADD; -1 prologue


def thread_sites(n: int, unroll: int) -> list[Site]:
    """ASSUMED dynamic G_GP sequence of one thread (identical for all threads: no divergence)."""
    if n % unroll != 0:
        raise ValueError(f"n={n} must be a multiple of unroll={unroll}")
    sites = [Site("S2R/IMAD.idx", -1)] * 6
    sites += [Site(c, -1) for c in ("IMAD.addr_c", "IMAD.addr_a", "IMAD.addr_b",
                                    "LDG.c", "FMUL.beta")]
    for k in range(n):
        sites += [Site(c, k) for c in ("LDG.a", "LDG.b", "FMUL.alpha", "FFMA")]
        if (k + 1) % unroll == 0:
            sites += [Site(c, k + 1) for c in ("IADD.ptr_a.lo", "IADD.ptr_a.hi",
                                               "IADD.ptr_b.lo", "IADD.ptr_b.hi",
                                               "IADD.k")]
    return sites


def class_counts(n: int, unroll: int) -> dict[str, int]:
    counts = {c: 0 for c in ALL_CLASSES}
    for s in thread_sites(n, unroll):
        counts[s.cls] += 1
    return counts


# --------------------------------------------------------------------------- #
# float32 register semantics
# --------------------------------------------------------------------------- #
def flip_f32(v: np.ndarray, bit: np.ndarray | int) -> np.ndarray:
    """XOR one bit of the IEEE-754 binary32 encoding (NVBitFI FLIP_SINGLE_BIT)."""
    u = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
    mask = np.left_shift(np.uint32(1), np.asarray(bit, dtype=np.uint32))
    return np.bitwise_xor(u, mask).view(np.float32)


def mul32(x: np.ndarray | np.floating, y: np.ndarray | np.floating) -> np.ndarray:
    """FMUL: exact product in float64, one rounding to float32."""
    with np.errstate(over="ignore", invalid="ignore"):
        return (np.asarray(x, np.float64) * np.asarray(y, np.float64)).astype(np.float32)


def fma32(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """FFMA x*y+z with a single float32 rounding (float64 intermediate; double rounding is rare)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return (np.asarray(x, np.float64) * np.asarray(y, np.float64)
                + np.asarray(z, np.float64)).astype(np.float32)


def percent_diff(ref: np.ndarray, out: np.ndarray) -> np.ndarray:
    """polybenchUtilFuncts.h percentDiff, vectorized; NaN propagates (and then passes the check)."""
    ref = np.asarray(ref, np.float64)
    out = np.asarray(out, np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pd = 100.0 * np.abs(np.abs(ref - out) / np.abs(ref + SMALL_FLOAT_VAL))
    both_small = (np.abs(ref) < ABS_FLOOR) & (np.abs(out) < ABS_FLOOR)
    return np.where(both_small, 0.0, pd)


def is_sdc(pd: np.ndarray, eps: float) -> np.ndarray:
    """gemm.cu counts a mismatch iff percentDiff > threshold; NaN > x is False -> NaN is masked."""
    with np.errstate(invalid="ignore"):
        return np.asarray(pd) > 100.0 * eps


# --------------------------------------------------------------------------- #
# Input distributions for A (B and C0 are fixed U[0,1] instances)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InputDist:
    name: str
    sample: Callable[[np.random.Generator, tuple[int, ...]], np.ndarray]
    # SOGA side: list of (weight, kind, params); kind in {"uniform", "gauss", "dirac"}
    soga_parts: tuple[tuple[float, str, tuple[float, ...]], ...]
    e_sq: float  # E[a^2], so that E||A||_F^2 = n^2 * e_sq


_SPARSE_P = 0.1
_SPARSE_HI = float(np.sqrt(10.0))  # E[a^2] = 0.1 * 10/3 = 1/3, same as U[0,1]
_N_SD = float(np.sqrt(1.0 / 3.0))  # E[a^2] = 1/3, same as U[0,1]


def _sparse(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    keep = rng.random(shape) < _SPARSE_P
    return np.where(keep, rng.uniform(0.0, _SPARSE_HI, shape), 0.0)


DISTS: tuple[InputDist, ...] = (
    InputDist("U[0,1]", lambda r, s: r.uniform(0.0, 1.0, s),
              ((1.0, "uniform", (0.0, 1.0)),), 1.0 / 3.0),
    InputDist("N(0,1/3)", lambda r, s: r.normal(0.0, _N_SD, s),
              ((1.0, "gauss", (0.0, _N_SD)),), 1.0 / 3.0),
    InputDist("Sparse10%", _sparse,
              ((1.0 - _SPARSE_P, "dirac", (0.0,)),
               (_SPARSE_P, "uniform", (0.0, _SPARSE_HI))), 1.0 / 3.0),
    InputDist("U[0,10]", lambda r, s: r.uniform(0.0, 10.0, s),
              ((1.0, "uniform", (0.0, 10.0)),), 100.0 / 3.0),
)
DIST_BY_NAME = {d.name: d for d in DISTS}


def make_inputs(dist: InputDist, n: int, seed_a: int, seed_fixed: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A ~ dist (seed_a); B, C0 ~ U[0,1] fixed across all campaigns (seed_fixed)."""
    A = dist.sample(np.random.default_rng(seed_a), (n, n)).astype(np.float32)
    rf = np.random.default_rng(seed_fixed)
    B = rf.uniform(0.0, 1.0, (n, n)).astype(np.float32)
    C0 = rf.uniform(0.0, 1.0, (n, n)).astype(np.float32)
    return A, B, C0
