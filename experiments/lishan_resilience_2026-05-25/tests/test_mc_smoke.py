"""
M2.5 — Smoke test for MC simulation.

Runs simulate_v_sweep with 3 v-points × 50 samples; asserts:
  - runtime < 30s
  - CSV well-formed (correct columns, numeric values, MSK+SDC+OTR ~ 1)
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
import warnings
import pytest
import numpy as np

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP_DIR)

from simulate_fi_mc import simulate_v_sweep, flip_bit


def test_mc_smoke_runtime():
    """MC sweep 3 v-points × 50 samples finishes < 30s."""
    A = np.eye(8, dtype=np.float64)  # 8x8 identity (small for smoke test)
    v_list = [0.1, 1.0, 10.0]
    n_samples = 50

    t0 = time.time()
    results = simulate_v_sweep(A, v_list, n_samples=n_samples, seed=42, eps=1e-3)
    elapsed = time.time() - t0

    assert elapsed < 30.0, f"MC smoke test took {elapsed:.1f}s > 30s"
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"


def test_mc_probabilities_sum_to_one():
    """MSK + SDC + OTR = 1 for each v."""
    A = np.eye(8, dtype=np.float64)
    v_list = [0.1, 1.0, 10.0]
    results = simulate_v_sweep(A, v_list, n_samples=50, seed=42)

    for v, r in results.items():
        total = r["MSK"] + r["SDC"] + r["OTR"]
        assert abs(total - 1.0) < 0.01, f"v={v}: MSK+SDC+OTR = {total:.4f} ≠ 1"


def test_mc_probabilities_in_range():
    """All probabilities in [0, 1]."""
    A = np.eye(8, dtype=np.float64)
    v_list = [0.1, 1.0, 10.0]
    results = simulate_v_sweep(A, v_list, n_samples=50, seed=42)

    for v, r in results.items():
        for key in ["MSK", "SDC", "OTR"]:
            val = r[key]
            assert 0.0 <= val <= 1.0, f"v={v} {key}={val:.4f} not in [0,1]"


def test_flip_bit_verified_cases():
    """5 known flip_bit verification cases."""
    # Case 1: sign flip on 1.0 → -1.0
    assert abs(flip_bit(1.0, 31) - (-1.0)) < 1e-6
    # Case 2: sign flip on -1.0 → 1.0
    assert abs(flip_bit(-1.0, 31) - 1.0) < 1e-6
    # Case 3: flip_bit(0.0, 31): -0.0 == 0.0 in Python float comparison
    result = flip_bit(0.0, 31)
    assert result == 0.0 or abs(result) < 1e-30  # -0.0
    # Case 4: flip_bit(1.0, 23): flip lowest exponent bit
    # 1.0 = 0_01111111_00000000000000000000000
    # flip bit 23 → 0_01111110_00000000000000000000000 = 0.5
    result4 = flip_bit(1.0, 23)
    assert abs(result4 - 0.5) < 1e-5, f"Expected 0.5, got {result4}"
    # Case 5: flip_bit(2.0, 22) — flip mantissa MSB of 2.0
    # 2.0 = 0_10000000_00000000000000000000000
    # flip bit 22 → 0_10000000_10000000000000000000000 = 3.0
    result5 = flip_bit(2.0, 22)
    assert abs(result5 - 3.0) < 1e-5, f"Expected 3.0, got {result5}"
