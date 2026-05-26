"""
M0.4 — Smoke tests: import + baseline loadability.

Tests:
  1. lib.fault_model is importable
  2. results/A_kernel.npz exists and is a valid 32x32 matrix
  3. results/A_identity.npz is I_32
  4. results/baseline_DUV.npz has correct keys and shapes
  5. config.json is loadable and has required keys
"""

from __future__ import annotations

import json
import os
import sys
import pytest
import numpy as np

# Ensure lib is importable
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(EXP_DIR))  # experiments/ parent
sys.path.insert(0, EXP_DIR)                   # lishan_resilience_2026-05-25/

RESULTS_DIR = os.path.join(EXP_DIR, "results")


def test_import_fault_model():
    """lib.fault_model imports without error."""
    from lib.fault_model import FaultClass, tail_gauss, check_overflow, shift_moments
    assert FaultClass.SIGN is not None
    assert callable(tail_gauss)


def test_import_lib_matrix_gaussian():
    """libMatrixGaussian imports from src/."""
    REPO_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from libMatrixGaussian import MatrixGaussian
    assert MatrixGaussian is not None


def test_a_kernel_npz_exists():
    """A_kernel.npz exists in results/."""
    path = os.path.join(RESULTS_DIR, "A_kernel.npz")
    assert os.path.exists(path), f"Missing: {path}. Run setup_matrices.py first."


def test_a_kernel_shape_and_dtype():
    """A_kernel is 32x32 float64."""
    data = np.load(os.path.join(RESULTS_DIR, "A_kernel.npz"))
    A = data["A"]
    assert A.shape == (32, 32), f"Expected (32,32), got {A.shape}"
    assert A.dtype == np.float64, f"Expected float64, got {A.dtype}"


def test_a_identity_is_identity():
    """A_identity.npz is I_32."""
    data = np.load(os.path.join(RESULTS_DIR, "A_identity.npz"))
    A = data["A"]
    assert np.allclose(A, np.eye(32)), "A_identity is not I_32"


def test_baseline_duv_loadable():
    """baseline_DUV.npz has M_D, U_D, V_D with correct shapes."""
    path = os.path.join(RESULTS_DIR, "baseline_DUV.npz")
    assert os.path.exists(path), f"Missing: {path}. Run setup_matrices.py first."
    data = np.load(path)
    for key in ("M_D", "U_D", "V_D"):
        assert key in data, f"Missing key '{key}' in baseline_DUV.npz"
    assert data["M_D"].shape == (32, 32)
    assert data["U_D"].shape == (32, 32)
    assert data["V_D"].shape == (32, 32)


def test_baseline_values_correct():
    """For A=I_32 and B~MN(0, I, I): D=B, so M_D=0, U_D=I, V_D=I."""
    data = np.load(os.path.join(RESULTS_DIR, "baseline_DUV.npz"))
    assert np.allclose(data["M_D"], 0.0), "M_D should be zero for zero-mean B"
    assert np.allclose(data["U_D"], np.eye(32)), "U_D should be I_32 for A=I,B has U=I"
    assert np.allclose(data["V_D"], np.eye(32)), "V_D should be I_32"


def test_config_json_loadable():
    """config.json loads and has required keys."""
    config_path = os.path.join(EXP_DIR, "config.json")
    assert os.path.exists(config_path), f"Missing: {config_path}"
    with open(config_path) as f:
        cfg = json.load(f)
    for key in ("seed", "eps", "kernel", "fault_model", "step1", "step3", "honesty_disclaimer"):
        assert key in cfg, f"Missing key '{key}' in config.json"
    assert cfg["seed"] == 42
    assert cfg["eps"] == 1e-3
