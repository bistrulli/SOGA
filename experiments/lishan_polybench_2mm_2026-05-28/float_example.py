"""Minimal float32 register-FI on a single inner product (a GEMM/2mm output cell):
out = sum_k x_k. Demonstrates that arithmetic masking depends on the input's
MAGNITUDE DISTRIBUTION, not its Frobenius norm.

EXACT enumeration over every (term, bit) single-bit-flip (no sampling). For each
fault we recompute the float32 accumulation and classify the output vs golden:
  - OTR  : output is Inf/NaN
  - MSK  : output within tolerance of golden  (eps=0 => bit-exact, i.e. rounding-masked)
  - SDC  : otherwise

We compare two inputs with IDENTICAL Frobenius norm:
  - balanced   : all terms equal           (no term is small relative to the sum)
  - concentrated: one large term + small rest (small terms' low bits fall below ULP)
"""
from __future__ import annotations

import math
import numpy as np


def flip_bit_f32(x: float, b: int) -> float:
    arr = np.array([x], dtype=np.float32)
    u = arr.view(np.uint32)
    u[0] ^= np.uint32(1) << np.uint32(b)
    return float(arr[0])


def dot_f32(x: np.ndarray) -> np.float32:
    acc = np.float32(0.0)
    for xi in x:
        acc = np.float32(acc) + np.float32(xi)
    return np.float32(acc)


def classify(out: float, golden: float, eps: float) -> str:
    if not np.isfinite(out):
        return "OTR"
    if eps == 0.0:
        return "MSK" if np.float32(out) == np.float32(golden) else "SDC"
    thr = eps * abs(float(golden)) if abs(float(golden)) > 0 else eps
    return "MSK" if abs(float(out) - float(golden)) <= thr else "SDC"


def msk_rate(x: np.ndarray, eps: float) -> float:
    x = x.astype(np.float32)
    golden = dot_f32(x)
    n_terms = len(x)
    masked = 0
    total = 0
    for k in range(n_terms):
        for b in range(32):
            xf = x.copy()
            xf[k] = np.float32(flip_bit_f32(float(x[k]), b))
            out = dot_f32(xf)
            total += 1
            if classify(out, golden, eps) == "MSK":
                masked += 1
    return masked / total


def frob(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.asarray(x, dtype=np.float64) ** 2)))


def make_pair(n: int, norm: float, big_frac: float):
    """balanced and concentrated inputs with the SAME Frobenius norm.
    concentrated: one term carries big_frac of the norm^2, the rest share the rest."""
    balanced = np.full(n, norm / math.sqrt(n), dtype=np.float32)
    big = math.sqrt(big_frac) * norm
    rest = math.sqrt((1 - big_frac) * norm ** 2 / (n - 1))
    concentrated = np.array([big] + [rest] * (n - 1), dtype=np.float32)
    return balanced, concentrated


if __name__ == "__main__":
    N = 16
    NORM = 4.0
    bal, con = make_pair(N, NORM, big_frac=0.95)

    print(f"inner product length N={N}, Frobenius norm = {NORM}")
    print(f"balanced:     all terms = {bal[0]:.4f}   ||x||={frob(bal):.4f}  out={float(dot_f32(bal)):.4f}")
    print(f"concentrated: big={con[0]:.4f}, rest={con[1]:.4f}  ||x||={frob(con):.4f}  out={float(dot_f32(con)):.4f}")
    print()
    print(f"{'criterion':>22} | {'balanced MSK':>13} | {'concentrated MSK':>17}")
    print("-" * 60)
    for label, eps in [("bit-exact (eps=ULP)", 0.0), ("tolerance eps=1e-3", 1e-3),
                       ("tolerance eps=1e-2", 1e-2), ("tolerance eps=5e-2", 5e-2)]:
        rb = msk_rate(bal, eps)
        rc = msk_rate(con, eps)
        print(f"{label:>22} | {rb*100:>11.1f} % | {rc*100:>15.1f} %")
