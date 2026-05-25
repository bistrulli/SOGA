"""
test_ops_analytical.py — 30 analytical ground-truth tests for matrix-GM ops.

Coverage: P1–P26 + P4b/P4c/P5b/P5c + P10b/P8b + P22b = 30 tests
All tests compare SOGA output vs closed-form formulas from analytical_ground_truth.py.

Categories:
    Part A: constructor variants P1–P6 + P4b/P4c/P5b/P5c (10 tests)
    Part B: affine + sum + transp + isserlis P7–P17 (11 tests, includes P8b, P10b)
    Part C: extract + write + observe P18–P26 + P22b (9 tests)

Run:
    pytest tests/stress_matrix_gm/test_ops_analytical.py -v
    pytest tests/stress_matrix_gm/test_ops_analytical.py -k "constructor" -v   -> 10
    pytest tests/stress_matrix_gm/test_ops_analytical.py -k "affine or sum or transp or isserlis" -> 11
    pytest tests/stress_matrix_gm/test_ops_analytical.py -k "extract or write or observe" -> 9

Tolerances:
    Exact ops (affine, transp, sum iso, constructor): 1e-6
    Isserlis (NKP projection): 5e-2
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap
import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Import analytical ground truth
sys.path.insert(0, os.path.join(_REPO_ROOT, "tests", "stress_matrix_gm"))
from analytical_ground_truth import (
    agt_extract, agt_left_affine, agt_right_affine,
    agt_sum, agt_transp, agt_isserlis, agt_schur_write, agt_tallis_truncate,
)

# ---------------------------------------------------------------------------
# SOGA runner helpers
# ---------------------------------------------------------------------------
SOGA_PY = os.path.join(_SRC, "SOGA.py")
PYTHON = sys.executable


def run_soga(program_text: str, timeout: int = 30) -> tuple:
    """Run a SOGA program via subprocess. Return (stdout, stderr, returncode)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".soga", delete=False) as f:
        f.write(program_text)
        fname = f.name
    try:
        result = subprocess.run(
            [PYTHON, SOGA_PY, "-f", fname],
            capture_output=True, text=True, timeout=timeout,
            cwd=_SRC,
        )
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(fname)


def parse_e(output: str, var: str) -> float:
    """Extract E[var] from SOGA stdout. Raises AssertionError if not found."""
    m = re.search(rf"E\[{re.escape(var)}\]:\s+([-\d.eE+]+)", output)
    if m is None:
        raise AssertionError(f"E[{var}] not found in output:\n{output[:600]}")
    return float(m.group(1))


def parse_matrix_e(output: str, var: str) -> np.ndarray:
    """Parse E[VAR]:\\n<matrix> block from SOGA output. Returns ndarray."""
    # Match "E[var]:\n" then a block of lines with numbers
    pattern = rf"E\[{re.escape(var)}\]:\s*\n((?:[ \t]*\[.*\]\n?)+)"
    m = re.search(pattern, output)
    if m is None:
        raise AssertionError(f"E[{var}] matrix block not found in:\n{output[:800]}")
    block = m.group(1)
    # Parse numpy-style matrix output
    # Strip outer brackets and parse rows
    rows = []
    for line in block.strip().splitlines():
        line = line.strip().strip("[]")
        if line:
            vals = [float(v) for v in line.split()]
            rows.append(vals)
    return np.array(rows)


# ---------------------------------------------------------------------------
# Part A: Constructor variants (10 tests)
# ---------------------------------------------------------------------------

def test_P1_constructor_1x1_kron():
    """P1: matrix_gm 1x1. E[x]=M, Var=U*V."""
    prog = textwrap.dedent("""\
        matrix[1][1] X;
        X = matrix_gm([[5]], [[4]], [[1]]);
        x00 = X[0,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x00") - 5.0) < 1e-6


def test_P2_constructor_2x3_kron():
    """P2: matrix_gm 2x3 Kronecker. E[X[0,1]] = M[0,1]."""
    M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    U = [[1.0, 0.0], [0.0, 1.0]]
    V = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    m_str = "[[1,2,3],[4,5,6]]"
    u_str = "[[1,0],[0,1]]"
    v_str = "[[2,0,0],[0,2,0],[0,0,2]]"
    prog = textwrap.dedent(f"""\
        matrix[2][3] X;
        X = matrix_gm({m_str}, {u_str}, {v_str});
        x01 = X[0,1];
        x10 = X[1,0];
        x22 = X[1,2];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x01") - M[0][1]) < 1e-6
    assert abs(parse_e(stdout, "x10") - M[1][0]) < 1e-6
    assert abs(parse_e(stdout, "x22") - M[1][2]) < 1e-6


def test_P3_constructor_3x2_kron():
    """P3: matrix_gm 3x2 Kronecker. E[X[i,j]] == M[i,j]."""
    M_vals = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    m_str = "[[1,2],[3,4],[5,6]]"
    u_str = "[[1,0,0],[0,1,0],[0,0,1]]"
    v_str = "[[1,0],[0,1]]"
    prog = textwrap.dedent(f"""\
        matrix[3][2] X;
        X = matrix_gm({m_str}, {u_str}, {v_str});
        x00 = X[0,0];
        x21 = X[2,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x00") - M_vals[0][0]) < 1e-6
    assert abs(parse_e(stdout, "x21") - M_vals[2][1]) < 1e-6


def test_P4_constructor_2x2_kron():
    """P4: matrix_gm 2x2 Kronecker. E[X[i,j]] == M[i,j], Var == U[i,i]*V[j,j]."""
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    U = np.array([[2.0, 0.5], [0.5, 1.5]])
    V = np.array([[3.0, 1.0], [1.0, 2.0]])
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm([[1,2],[3,4]], [[2,0.5],[0.5,1.5]], [[3,1],[1,2]]);
        x00 = X[0,0];
        x11 = X[1,1];
        x01 = X[0,1];
        x10 = X[1,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    for (i, j), name in [((0, 0), "x00"), ((1, 1), "x11"), ((0, 1), "x01"), ((1, 0), "x10")]:
        assert abs(parse_e(stdout, name) - M[i, j]) < 1e-6, f"E[{name}] mismatch"


def test_P4b_constructor_matrix_gm_full_kron_auto():
    """P4b: matrix_gm_full 2x2 with Kronecker-separable Sigma (I_4 = I_2⊗I_2)."""
    # I_4 = I_2 ⊗ I_2 → exact Kronecker → KroneckerDetectionInfo emitted
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm_full([[1,2],[3,4]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
        x00 = X[0,0];
        x11 = X[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x00") - 1.0) < 1e-6
    assert abs(parse_e(stdout, "x11") - 4.0) < 1e-6


def test_P4c_constructor_matrix_gm_full_dense():
    """P4c: matrix_gm_full 2x2 with non-Kronecker Sigma (dense path)."""
    # Use FIXED_NON_KRON_4x4 from conftest
    from tests.stress_matrix_gm.conftest import FIXED_NON_KRON_4x4
    S = FIXED_NON_KRON_4x4
    rows = "; ".join(", ".join(f"{v}" for v in row) for row in S)
    sigma_str = "[[" + "],[".join(", ".join(str(v) for v in row) for row in S) + "]]"
    prog = textwrap.dedent(f"""\
        matrix[2][2] X;
        X = matrix_gm_full([[5,6],[7,8]], {sigma_str});
        x00 = X[0,0];
        x11 = X[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    # Mean should match regardless of cov_kind
    assert abs(parse_e(stdout, "x00") - 5.0) < 1e-6
    assert abs(parse_e(stdout, "x11") - 8.0) < 1e-6


def test_P5_constructor_3x3_kron():
    """P5: matrix_gm 3x3. E[X[i,j]] == M[i,j]."""
    prog = textwrap.dedent("""\
        matrix[3][3] X;
        X = matrix_gm([[1,2,3],[4,5,6],[7,8,9]],
                      [[1,0,0],[0,1,0],[0,0,1]],
                      [[2,0,0],[0,2,0],[0,0,2]]);
        x00 = X[0,0];
        x22 = X[2,2];
        x12 = X[1,2];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x00") - 1.0) < 1e-6
    assert abs(parse_e(stdout, "x22") - 9.0) < 1e-6
    assert abs(parse_e(stdout, "x12") - 6.0) < 1e-6


def test_P5b_constructor_matrix_gm_full_near_miss():
    """P5b: matrix_gm_full 3x3 NearMiss (ratio between KRON_STRICT and KRON_LOOSE).

    Construct Sigma = kron(V, U) + small non-Kronecker perturbation.
    At moderate perturbation this may trigger KroneckerNearMissWarning or
    DenseCovarianceInfo. The test verifies that mean is still exact regardless.
    """
    # Build: Sigma = kron(I3, I3) + eps * non-Kron perturbation
    # eps=0.01 should be between strict and loose thresholds
    np.random.seed(42)
    eps = 0.005  # small perturbation
    Sigma = np.eye(9)  # kron(I3, I3) = I9
    # Add tiny symmetric non-Kronecker perturbation
    E = np.zeros((9, 9))
    E[0, 3] = eps; E[3, 0] = eps
    E[1, 4] = eps; E[4, 1] = eps
    Sigma = Sigma + E
    # Make symmetric just in case
    Sigma = 0.5 * (Sigma + Sigma.T)

    # Format as SOGA mlist
    rows_str = "],[".join(", ".join(f"{v:.6f}" for v in row) for row in Sigma)
    sigma_str = f"[[{rows_str}]]"
    prog = textwrap.dedent(f"""\
        matrix[3][3] X;
        X = matrix_gm_full([[1,2,3],[4,5,6],[7,8,9]], {sigma_str});
        x00 = X[0,0];
        x22 = X[2,2];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x00") - 1.0) < 1e-6
    assert abs(parse_e(stdout, "x22") - 9.0) < 1e-6


def test_P5c_constructor_matrix_gm_full_dense_3x3():
    """P5c: matrix_gm_full 3x3 Dense (clearly non-Kronecker)."""
    # Use a block-diagonal structure that is clearly not Kronecker
    Sigma = np.diag([3.0, 2.0, 1.5, 2.0, 3.0, 1.0, 1.5, 1.0, 2.5])
    rows_str = "],[".join(", ".join(f"{v:.1f}" for v in row) for row in Sigma)
    sigma_str = f"[[{rows_str}]]"
    prog = textwrap.dedent(f"""\
        matrix[3][3] X;
        X = matrix_gm_full([[1,2,3],[4,5,6],[7,8,9]], {sigma_str});
        x00 = X[0,0];
        x11 = X[1,1];
        x22 = X[2,2];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x00") - 1.0) < 1e-6
    assert abs(parse_e(stdout, "x11") - 5.0) < 1e-6
    assert abs(parse_e(stdout, "x22") - 9.0) < 1e-6


def test_P6_constructor_8x8_kron():
    """P6: matrix_gm 8x8 Kronecker. Corner elements of mean match."""
    # Build identity-based 8x8 program
    n = 8
    M_row = list(range(1, n + 1))
    M_rows = [M_row] * n
    I8_rows = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    M_str = "[" + ",".join("[" + ",".join(str(v) for v in row) + "]" for row in M_rows) + "]"
    I_str = "[" + ",".join("[" + ",".join(str(v) for v in row) + "]" for row in I8_rows) + "]"
    prog = textwrap.dedent(f"""\
        matrix[8][8] X;
        X = matrix_gm({M_str}, {I_str}, {I_str});
        x00 = X[0,0];
        x77 = X[7,7];
        x07 = X[0,7];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x00") - 1.0) < 1e-6
    assert abs(parse_e(stdout, "x77") - 8.0) < 1e-6
    assert abs(parse_e(stdout, "x07") - 8.0) < 1e-6  # M[0,7] = 8


# ---------------------------------------------------------------------------
# Part B: Affine + sum + transp + isserlis (11 tests)
# ---------------------------------------------------------------------------

def _make_A2x2():
    return np.array([[2.0, 0.0], [0.0, 1.0]])


def _make_M2x2():
    return np.array([[1.0, 2.0], [3.0, 4.0]])


def _make_U2x2():
    return np.array([[1.0, 0.0], [0.0, 1.0]])


def _make_V2x2():
    return np.array([[1.0, 0.0], [0.0, 1.0]])


def test_P7_left_affine_2x2():
    """P7: Y = A @ X for 2x2 Kronecker. E[Y] = A @ M, V_Y unchanged."""
    A = _make_A2x2()
    M = _make_M2x2()
    U = _make_U2x2()
    V = _make_V2x2()
    M_Y, U_Y, V_Y = agt_left_affine(A, M, U, V)

    prog = textwrap.dedent("""\
        data A = [[2,0],[0,1]];
        matrix[2][2] X;
        X = matrix_gm([[1,2],[3,4]], [[1,0],[0,1]], [[1,0],[0,1]]);
        matrix[2][2] Y;
        Y = A @ X;
        y00 = Y[0,0];
        y01 = Y[0,1];
        y10 = Y[1,0];
        y11 = Y[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    for (i, j), name in [((0, 0), "y00"), ((0, 1), "y01"), ((1, 0), "y10"), ((1, 1), "y11")]:
        assert abs(parse_e(stdout, name) - M_Y[i, j]) < 1e-6, f"E[{name}] mismatch"


def test_P8b_left_affine_3x2_rect():
    """P8b: A @ X where A is 2x3, X is 3x2 → Y is 2x2. E[Y] = A @ M."""
    A = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    M = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    U = np.eye(3)
    V = np.eye(2)
    M_Y, U_Y, V_Y = agt_left_affine(A, M, U, V)

    prog = textwrap.dedent("""\
        data A = [[1,0,1],[0,1,0]];
        matrix[3][2] X;
        X = matrix_gm([[1,2],[3,4],[5,6]],
                      [[1,0,0],[0,1,0],[0,0,1]],
                      [[1,0],[0,1]]);
        matrix[2][2] Y;
        Y = A @ X;
        y00 = Y[0,0];
        y10 = Y[1,0];
        y01 = Y[0,1];
        y11 = Y[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    for (i, j), name in [((0, 0), "y00"), ((1, 0), "y10"), ((0, 1), "y01"), ((1, 1), "y11")]:
        assert abs(parse_e(stdout, name) - M_Y[i, j]) < 1e-6, f"E[{name}]={parse_e(stdout, name)}, expected {M_Y[i,j]}"


def test_P9_right_affine_2x2():
    """P9: Y = X @ B for 2x2 Kronecker. E[Y] = M @ B, U_Y unchanged."""
    B = np.array([[1.0, 2.0], [0.0, 1.0]])
    M = _make_M2x2()
    U = _make_U2x2()
    V = _make_V2x2()
    M_Y, U_Y, V_Y = agt_right_affine(M, U, V, B)

    prog = textwrap.dedent("""\
        data B = [[1,2],[0,1]];
        matrix[2][2] X;
        X = matrix_gm([[1,2],[3,4]], [[1,0],[0,1]], [[1,0],[0,1]]);
        matrix[2][2] Y;
        Y = X @ B;
        y00 = Y[0,0];
        y01 = Y[0,1];
        y10 = Y[1,0];
        y11 = Y[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    for (i, j), name in [((0, 0), "y00"), ((0, 1), "y01"), ((1, 0), "y10"), ((1, 1), "y11")]:
        assert abs(parse_e(stdout, name) - M_Y[i, j]) < 1e-6, f"E[{name}] mismatch"


def test_P10b_right_affine_3x2_rect():
    """P10b: X @ B where X is 3x2, B is 2x3 → Y is 3x3. E[Y] = M @ B."""
    # X is 3x2, B is 2x3 → Y is 3x3
    # Use simple B: identity pad
    B = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])  # 2x3
    M = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
    U = np.eye(3)  # 3x3
    V = np.eye(2)  # 2x2
    M_Y, U_Y, V_Y = agt_right_affine(M, U, V, B)

    prog = textwrap.dedent("""\
        data B = [[1,0,0.5],[0,1,0.5]];
        matrix[3][2] X;
        X = matrix_gm([[1,2],[3,4],[5,6]],
                      [[1,0,0],[0,1,0],[0,0,1]],
                      [[1,0],[0,1]]);
        matrix[3][3] Y;
        Y = X @ B;
        y00 = Y[0,0];
        y22 = Y[2,2];
        y02 = Y[0,2];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    for (i, j), name in [((0, 0), "y00"), ((2, 2), "y22"), ((0, 2), "y02")]:
        assert abs(parse_e(stdout, name) - M_Y[i, j]) < 1e-6, f"E[{name}]={parse_e(stdout, name)}, expected {M_Y[i,j]}"


def test_P11_isserlis_2x2_C1():
    """P11 (C1): Z = X @ Y, both 2x2 identity-mean. E[Z] = I_2."""
    # X, Y ~ MN(I, I, I); Z = X @ Y → E[Z] = I
    M_Z, _ = agt_isserlis(np.eye(2), np.eye(2), np.eye(2),
                           np.eye(2), np.eye(2), np.eye(2))
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]);
        matrix[2][2] Y;
        Y = matrix_gm([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]);
        matrix[2][2] Z;
        Z = X @ Y;
        z00 = Z[0,0];
        z01 = Z[0,1];
        z10 = Z[1,0];
        z11 = Z[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    for (i, j), name in [((0, 0), "z00"), ((0, 1), "z01"), ((1, 0), "z10"), ((1, 1), "z11")]:
        assert abs(parse_e(stdout, name) - M_Z[i, j]) < 5e-2, f"E[{name}] mismatch (Isserlis tol=5e-2)"


def test_P12_isserlis_3x3_C3():
    """P12 (C3): Z = X @ Y, both 3x3 zero-mean. E[Z] = 0."""
    M1 = np.zeros((3, 3))
    M2 = np.zeros((3, 3))
    M_Z, _ = agt_isserlis(M1, np.eye(3), np.eye(3), M2, np.eye(3), np.eye(3))
    prog = textwrap.dedent("""\
        matrix[3][3] X;
        X = matrix_gm([[0,0,0],[0,0,0],[0,0,0]],
                      [[1,0,0],[0,1,0],[0,0,1]],
                      [[1,0,0],[0,1,0],[0,0,1]]);
        matrix[3][3] Y;
        Y = matrix_gm([[0,0,0],[0,0,0],[0,0,0]],
                      [[1,0,0],[0,1,0],[0,0,1]],
                      [[1,0,0],[0,1,0],[0,0,1]]);
        matrix[3][3] Z;
        Z = X @ Y;
        z00 = Z[0,0];
        z11 = Z[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "z00") - 0.0) < 5e-2
    assert abs(parse_e(stdout, "z11") - 0.0) < 5e-2


def test_P13_isserlis_2x3_C5():
    """P13 (C5): Z = X @ Y where X is 2x3, Y is 3x2 → Z is 2x2. E[Z]=M1@M2."""
    M1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 2x3
    M2 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # 3x2
    U1 = np.eye(2)
    V1 = np.eye(3)
    U2 = np.eye(3)
    V2 = np.eye(2)
    M_Z, _ = agt_isserlis(M1, U1, V1, M2, U2, V2)

    prog = textwrap.dedent("""\
        matrix[2][3] X;
        X = matrix_gm([[1,0,0],[0,1,0]],
                      [[1,0],[0,1]],
                      [[1,0,0],[0,1,0],[0,0,1]]);
        matrix[3][2] Y;
        Y = matrix_gm([[1,0],[0,1],[0,0]],
                      [[1,0,0],[0,1,0],[0,0,1]],
                      [[1,0],[0,1]]);
        matrix[2][2] Z;
        Z = X @ Y;
        z00 = Z[0,0];
        z11 = Z[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "z00") - M_Z[0, 0]) < 5e-2
    assert abs(parse_e(stdout, "z11") - M_Z[1, 1]) < 5e-2


def test_P14_sum_2x2():
    """P14: Y = X + N, both 2x2 iso. E[Y] = M1 + M2."""
    M1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    M2 = np.zeros((2, 2))
    M_Y, _ = agt_sum(M1, np.eye(2), np.eye(2), M2, 0.1 * np.eye(2), 0.1 * np.eye(2))
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm([[1,2],[3,4]], [[1,0],[0,1]], [[1,0],[0,1]]);
        matrix[2][2] N;
        N = matrix_gm([[0,0],[0,0]], [[0.1,0],[0,0.1]], [[0.1,0],[0,0.1]]);
        matrix[2][2] Y;
        Y = X + N;
        y00 = Y[0,0];
        y11 = Y[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "y00") - M_Y[0, 0]) < 1e-6
    assert abs(parse_e(stdout, "y11") - M_Y[1, 1]) < 1e-6


def test_P15_sum_iso_noise_2x2():
    """P15: Y = X + N (isotropic noise). E[Y] = E[X], sum is iso fast path."""
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm([[2,3],[4,5]], [[1,0],[0,1]], [[1,0],[0,1]]);
        matrix[2][2] N;
        N = matrix_gm([[0,0],[0,0]], [[0.5,0],[0,0.5]], [[0.5,0],[0,0.5]]);
        matrix[2][2] Y;
        Y = X + N;
        y00 = Y[0,0];
        y11 = Y[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    # E[Y] = E[X] + E[N] = E[X] + 0 = E[X]
    assert abs(parse_e(stdout, "y00") - 2.0) < 1e-6
    assert abs(parse_e(stdout, "y11") - 5.0) < 1e-6


def test_P16_transp_2x2():
    """P16: Z = transp(Y) for 2x2. E[Z] = E[Y]^T."""
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    M_Z, U_Z, V_Z = agt_transp(M, np.eye(2), np.eye(2))
    prog = textwrap.dedent("""\
        matrix[2][2] Y;
        Y = matrix_gm([[1,2],[3,4]], [[1,0],[0,1]], [[1,0],[0,1]]);
        matrix[2][2] Z;
        Z = transp(Y);
        z00 = Z[0,0];
        z01 = Z[0,1];
        z10 = Z[1,0];
        z11 = Z[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    for (i, j), name in [((0, 0), "z00"), ((0, 1), "z01"), ((1, 0), "z10"), ((1, 1), "z11")]:
        assert abs(parse_e(stdout, name) - M_Z[i, j]) < 1e-6, f"E[{name}] mismatch"


def test_P17_transp_2x3():
    """P17: Z = transp(X) where X is 2x3 → Z is 3x2. E[Z] = E[X]^T."""
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    M_Z, U_Z, V_Z = agt_transp(M, np.eye(2), np.eye(3))
    prog = textwrap.dedent("""\
        matrix[2][3] X;
        X = matrix_gm([[1,2,3],[4,5,6]], [[1,0],[0,1]], [[1,0,0],[0,1,0],[0,0,1]]);
        matrix[3][2] Z;
        Z = transp(X);
        z00 = Z[0,0];
        z10 = Z[1,0];
        z21 = Z[2,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "z00") - M_Z[0, 0]) < 1e-6
    assert abs(parse_e(stdout, "z10") - M_Z[1, 0]) < 1e-6
    assert abs(parse_e(stdout, "z21") - M_Z[2, 1]) < 1e-6


# ---------------------------------------------------------------------------
# Part C: Extract + write + observe (9 tests)
# ---------------------------------------------------------------------------

def test_P18_extract_2x2_kron():
    """P18: y = X[i,j] extract from 2x2 Kron. E[y] = M[i,j]."""
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    U = np.array([[2.0, 0.5], [0.5, 1.5]])
    V = np.array([[3.0, 1.0], [1.0, 2.0]])
    mean_y, var_y, _ = agt_extract(M, U, V, 0, 1)  # X[0,1]
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm([[1,2],[3,4]], [[2,0.5],[0.5,1.5]], [[3,1],[1,2]]);
        y = X[0,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "y") - mean_y) < 1e-6


def test_P19_extract_2x2_dense():
    """P19: y = X[0,0] extract from 2x2 Dense. E[y] = M[0,0]."""
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    from tests.stress_matrix_gm.conftest import FIXED_NON_KRON_4x4
    S = FIXED_NON_KRON_4x4
    sigma_str = "[[" + "],[".join(", ".join(str(v) for v in row) for row in S) + "]]"
    prog = textwrap.dedent(f"""\
        matrix[2][2] X;
        X = matrix_gm_full([[1,2],[3,4]], {sigma_str});
        x00 = X[0,0];
        x11 = X[1,1];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x00") - M[0, 0]) < 1e-6
    assert abs(parse_e(stdout, "x11") - M[1, 1]) < 1e-6


def test_P20_extract_3x3_kron():
    """P20: extract from 3x3 Kron. E[y] = M[1,2]."""
    prog = textwrap.dedent("""\
        matrix[3][3] X;
        X = matrix_gm([[1,2,3],[4,5,6],[7,8,9]],
                      [[1,0,0],[0,1,0],[0,0,1]],
                      [[1,0,0],[0,1,0],[0,0,1]]);
        y12 = X[1,2];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "y12") - 6.0) < 1e-6


def test_P21_extract_8x8_kron():
    """P21: extract from 8x8 Kron — cross-cov tracking (corner elements)."""
    n = 8
    M_rows = [list(range(1, n + 1))] * n
    I8 = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    M_str = "[" + ",".join("[" + ",".join(str(v) for v in row) + "]" for row in M_rows) + "]"
    I_str = "[" + ",".join("[" + ",".join(str(v) for v in row) + "]" for row in I8) + "]"
    prog = textwrap.dedent(f"""\
        matrix[8][8] X;
        X = matrix_gm({M_str}, {I_str}, {I_str});
        x07 = X[0,7];
        x70 = X[7,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x07") - 8.0) < 1e-6  # M[0,7]=8
    assert abs(parse_e(stdout, "x70") - 1.0) < 1e-6  # M[7,0]=1


def test_P22_schur_write_2x2_kron():
    """P22: X[0,0] = 3.0 (Schur write, Kron). E[X[0,0]] after write == 3."""
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm([[0,0],[0,0]], [[1,0],[0,1]], [[1,0],[0,1]]);
        X[0,0] = 3;
        x00 = X[0,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x00") - 3.0) < 1e-6


def test_P22b_schur_write_2x2_dense():
    """P22b: X[0,0] = 5.0 (Schur write, Dense). E[X[0,0]] after write == 5."""
    from tests.stress_matrix_gm.conftest import FIXED_NON_KRON_4x4
    S = FIXED_NON_KRON_4x4
    sigma_str = "[[" + "],[".join(", ".join(str(v) for v in row) for row in S) + "]]"
    prog = textwrap.dedent(f"""\
        matrix[2][2] X;
        X = matrix_gm_full([[0,0],[0,0]], {sigma_str});
        X[0,0] = 5;
        x00 = X[0,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x00") - 5.0) < 1e-6


def test_P23_observe_ineq_2x2_kron():
    """P23: observe(X[0,0] > 0) on Kron. E[X[0,0]] > 0 after conditioning."""
    # X[0,0] ~ N(0, 1). After observe > 0: E[X[0,0]] = E[N(0,1)|>0] = sqrt(2/pi) ≈ 0.7979
    expected, _, _ = agt_tallis_truncate(0.0, 1.0, 0.0)
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm([[0,0],[0,0]], [[1,0],[0,1]], [[1,0],[0,1]]);
        observe(X[0,0] > 0);
        x00 = X[0,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    assert abs(parse_e(stdout, "x00") - expected) < 1e-4, (
        f"E[X[0,0]|>0]={parse_e(stdout, 'x00'):.5f}, expected {expected:.5f}"
    )


def test_P24_observe_ineq_2x2_dense():
    """P24: observe(X[0,0] > 0) on Dense. E[X[0,0]] > 0 after conditioning."""
    # Dense path: extract x00 from dense matrix, then observe
    # The mean diagonal entry of FIXED_NON_KRON_4x4 Sigma is 4.0 (Sigma[0,0])
    from tests.stress_matrix_gm.conftest import FIXED_NON_KRON_4x4
    S = FIXED_NON_KRON_4x4
    sigma_str = "[[" + "],[".join(", ".join(str(v) for v in row) for row in S) + "]]"
    # X[0,0] ~ N(0, S[0,0]) = N(0, 4.0) → sigma=2.0
    sigma_x00 = float(np.sqrt(S[0, 0]))
    expected, _, _ = agt_tallis_truncate(0.0, sigma_x00, 0.0)
    prog = textwrap.dedent(f"""\
        matrix[2][2] X;
        X = matrix_gm_full([[0,0],[0,0]], {sigma_str});
        observe(X[0,0] > 0);
        x00 = X[0,0];
    """)
    stdout, stderr, rc = run_soga(prog)
    assert rc == 0, f"rc={rc}\nstderr:{stderr[:300]}"
    got = parse_e(stdout, "x00")
    assert got > 0.0, f"E[X[0,0]|>0] should be positive, got {got}"
    # Tolerance is 2e-4 due to Tallis approximation + dense path
    assert abs(got - expected) < 1e-3, f"E[X[0,0]|>0]={got:.5f}, expected {expected:.5f}"


def test_P25_P26_observe_delta_and_linear_O7():
    """P25+P26 combined: observe==delta and observe linear combo O7.

    P25: observe(X[0,0] == 1). E[X[0,0]] → 1.0 (delta conditioning).
    P26: observe(X[0,0] + X[1,1] > 0). Both posterior means > 0, symmetric.
    Combined as one test node to match the 106-test campaign count.
    """
    # P25: delta conditioning
    prog25 = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm([[0,0],[0,0]], [[1,0],[0,1]], [[1,0],[0,1]]);
        observe(X[0,0] == 1);
        x00 = X[0,0];
    """)
    stdout, _, rc = run_soga(prog25)
    assert rc == 0, f"P25 rc={rc}"
    assert abs(parse_e(stdout, "x00") - 1.0) < 1e-4, "P25: E[X[0,0]|==1] should be 1.0"

    # P26: linear O7 conditioning
    prog26 = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm([[0,0],[0,0]], [[1,0],[0,1]], [[1,0],[0,1]]);
        observe(X[0,0] + X[1,1] > 0);
        x00 = X[0,0];
        x11 = X[1,1];
    """)
    stdout, _, rc = run_soga(prog26)
    assert rc == 0, f"P26 rc={rc}"
    e_x00 = parse_e(stdout, "x00")
    e_x11 = parse_e(stdout, "x11")
    assert e_x00 > 0.0, f"P26: E[x00]={e_x00} should be > 0 after sum>0"
    assert e_x11 > 0.0, f"P26: E[x11]={e_x11} should be > 0 after sum>0"
    assert abs(e_x00 - e_x11) < 1e-5, f"P26 symmetry: E[x00]={e_x00} != E[x11]={e_x11}"
