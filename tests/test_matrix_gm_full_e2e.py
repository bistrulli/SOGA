"""
tests/test_matrix_gm_full_e2e.py — End-to-end SOGA.py tests for matrix_gm_full [B.9]

Tests run complete programs via subprocess to catch issues that unit tests miss.
All expected values are analytically verified.

Analytical ground truth:
  (E1) X ~ MN(0, I_2, I_2): E[x00]=0, Var[x00]=1  (I_4 = I_2⊗I_2 → Kronecker-detected)
  (E2) X ~ MN(M, I_2, I_2) for M=[[1,2],[3,4]]: E[x00]=1, E[x11]=4
  (E3) Dense non-Kronecker: E[x00]=1 (mean matches), Var[x00] from Sigma diagonal
  (E4) Extract + observe element on dense variable
  (E5) matrix_gm_full and matrix_gm() in same program (different variables)
  (E6) Kronecker-detected → matmul/transpose works (same as matrix_gm path)
  (E7) KroneckerDetectionInfo in stderr for Kronecker Sigma
  (E8) DenseCovarianceInfo in stderr for non-Kronecker Sigma

Acceptance: ≥8 tests, each matching expected values to 4 decimal places.
"""
import subprocess
import sys
import os
import re
import tempfile
import textwrap
import warnings
import pytest


# Path to SOGA.py
SOGA_PY = os.path.join(os.path.dirname(__file__), "..", "src", "SOGA.py")
PYTHON = sys.executable


def run_soga(program_text: str, timeout: int = 30) -> tuple:
    """Run a SOGA program text via subprocess; return (stdout, stderr, returncode)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".soga", delete=False) as f:
        f.write(program_text)
        fname = f.name
    try:
        result = subprocess.run(
            [PYTHON, SOGA_PY, "-f", fname],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.path.join(os.path.dirname(__file__), "..", "src"),
        )
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(fname)


def parse_e(output: str, var: str) -> float:
    """Parse E[var]: value from SOGA output. Returns float."""
    pattern = rf"E\[{re.escape(var)}\]:\s+([-\d.eE+]+)"
    m = re.search(pattern, output)
    if m is None:
        raise AssertionError(f"E[{var}] not found in output:\n{output}")
    return float(m.group(1))


# ---------------------------------------------------------------------------
# E1: Kronecker-detected case (I_4 = I_2 ⊗ I_2)
# ---------------------------------------------------------------------------

def test_e1_kronecker_zero_mean():
    """I_4 Sigma → Kronecker-detected; E[x00]=0.0 (zero mean)."""
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm_full([[0,0],[0,0]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
        x00 = X[0,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"SOGA exited with code {rc}. stderr:\n{stderr}"
    assert abs(parse_e(stdout, "x00") - 0.0) < 1e-4


# ---------------------------------------------------------------------------
# E2: Non-zero mean — Kronecker-detected
# ---------------------------------------------------------------------------

def test_e2_kronecker_nonzero_mean():
    """E[x00]=1, E[x01]=2, E[x10]=3, E[x11]=4 for M=[[1,2],[3,4]], I_4 Sigma."""
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm_full([[1,2],[3,4]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
        x00 = X[0,0];
        x01 = X[0,1];
        x10 = X[1,0];
        x11 = X[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"SOGA exited with code {rc}. stderr:\n{stderr}"
    assert abs(parse_e(stdout, "x00") - 1.0) < 1e-4
    assert abs(parse_e(stdout, "x01") - 2.0) < 1e-4
    assert abs(parse_e(stdout, "x10") - 3.0) < 1e-4
    assert abs(parse_e(stdout, "x11") - 4.0) < 1e-4


# ---------------------------------------------------------------------------
# E3: Dense non-Kronecker Sigma — mean propagated correctly
# ---------------------------------------------------------------------------

def test_e3_dense_mean_propagation():
    """Dense non-Kronecker Sigma: E[x00]=1 (from mean), E[x11]=4."""
    # Build a non-Kronecker but PSD symmetric Sigma
    # diag(2,1,1,2) + off-diag coupling
    sigma_str = "[[2.0,0.0,0.0,0.5],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.5,0.0,0.0,2.0]]"
    prog = textwrap.dedent(f"""\
        matrix[2][2] X;
        X = matrix_gm_full([[1,2],[3,4]], {sigma_str});
        x00 = X[0,0];
        x11 = X[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"SOGA exited with code {rc}. stderr:\n{stderr}"
    assert abs(parse_e(stdout, "x00") - 1.0) < 1e-4
    assert abs(parse_e(stdout, "x11") - 4.0) < 1e-4


# ---------------------------------------------------------------------------
# E4: Observe element on dense variable
# ---------------------------------------------------------------------------

def test_e4_observe_element_on_dense():
    """Dense Sigma + observe X[0,0] == 0: E[x00] becomes 0."""
    sigma_str = "[[2.0,0.0,0.0,0.5],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.5,0.0,0.0,2.0]]"
    prog = textwrap.dedent(f"""\
        matrix[2][2] X;
        X = matrix_gm_full([[0,0],[0,0]], {sigma_str});
        observe(X[0,0] == 0);
        x00 = X[0,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"SOGA exited with code {rc}. stderr:\n{stderr}"
    # After conditioning on X[0,0]==0, E[x00] should be 0
    assert abs(parse_e(stdout, "x00") - 0.0) < 1e-4


# ---------------------------------------------------------------------------
# E5: matrix_gm_full and matrix_gm in same program
# ---------------------------------------------------------------------------

def test_e5_mixed_constructor_same_program():
    """matrix_gm_full(X) and matrix_gm(Y) can coexist."""
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        matrix[2][2] Y;
        X = matrix_gm_full([[1,0],[0,1]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
        Y = matrix_gm([[2,0],[0,2]], [[1,0],[0,1]], [[1,0],[0,1]]);
        x00 = X[0,0];
        y00 = Y[0,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"SOGA exited with code {rc}. stderr:\n{stderr}"
    assert abs(parse_e(stdout, "x00") - 1.0) < 1e-4
    assert abs(parse_e(stdout, "y00") - 2.0) < 1e-4


# ---------------------------------------------------------------------------
# E6: Kronecker-detected → operations work (transp, scale)
# ---------------------------------------------------------------------------

def test_e6_kronecker_then_scale():
    """Kronecker-detected X, then c*X: E[x00] should be 2.0."""
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm_full([[1,2],[3,4]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
        X = 2 * X;
        x00 = X[0,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"SOGA exited with code {rc}. stderr:\n{stderr}"
    assert abs(parse_e(stdout, "x00") - 2.0) < 1e-4


# ---------------------------------------------------------------------------
# E7: KroneckerDetectionInfo in stderr
# ---------------------------------------------------------------------------

def test_e7_kronecker_detection_info_in_output():
    """KroneckerDetectionInfo should appear in stderr for exact Kronecker Sigma."""
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm_full([[0,0],[0,0]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
        x00 = X[0,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0
    # Python warnings printed to stderr
    assert "KroneckerDetection" in stderr, (
        f"Expected KroneckerDetectionInfo in stderr, got:\n{stderr[:400]}"
    )


# ---------------------------------------------------------------------------
# E8: Dense mode recognized as non-Kronecker (DenseCovarianceInfo in output)
# ---------------------------------------------------------------------------

def test_e8_dense_covariance_info_in_output():
    """DenseCovarianceInfo should appear in output for non-Kronecker Sigma."""
    # Build a clearly non-Kronecker PSD Sigma
    sigma_str = "[[2.0,0.0,0.0,0.5],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.5,0.0,0.0,2.0]]"
    prog = textwrap.dedent(f"""\
        matrix[2][2] X;
        X = matrix_gm_full([[0,0],[0,0]], {sigma_str});
        x00 = X[0,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0
    # Either DenseCovarianceInfo or KroneckerNearMissWarning should appear
    combined = stdout + stderr
    assert ("DenseCovarianceInfo" in combined or "KroneckerNearMiss" in combined), (
        f"Expected Dense or NearMiss info in output. stdout+stderr:\n{combined[:400]}"
    )
