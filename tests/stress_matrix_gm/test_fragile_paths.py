"""
test_fragile_paths.py — 5 xfail probes for known fragile paths in matrix-GM.

All 5 tests are marked ``pytest.mark.xfail(strict=True)`` — they MUST fail.
An xpass (test unexpectedly passes) fails the whole run and signals a silent fix
that needs to be verified and the xfail marker removed.

Documented bugs / gaps (see docs/LIMITATIONS.md for full detail):

  F1  (C1 + Path 5): observe on a matrix variable densifies the covariance via
      the Schur-complement path (M5 dense strategy).  A subsequent A@X affine
      call attempts to unpack the dense sentinel (None, Sigma) as (U, V) in
      _matrix_affine_left, which calls A @ None.  The subprocess worker crashes
      with TypeError before writing its result to the multiprocessing Queue; the
      main process then hits the queue.get() timeout and prints
      "Warning: SOGA Timeout occurred".  Net effect: E[Y] never appears in
      stdout.  Assert "E[Y]" in stdout → AssertionError.

  F2a (C7 right-affine): X = matrix_gm_full(M, Sigma_dense) stores a dense
      sentinel (None, Sigma).  A subsequent Y = X @ B call reaches
      _matrix_affine_right, which calls ``block.get_cov(k, lhs, lhs)``
      returning (None, Sigma_dense) and then executes ``U_x.copy()``
      where U_x is None → AttributeError.

  F2b (C7 left-affine): Same sentinel, Y = A @ X path.  _matrix_affine_left
      calls ``A @ U_x`` where U_x is None → ValueError (matmul operand
      dimensions).

  F3  (Gap 3 — stale cross-cov after observe): After ``y0 = X[0,0]``, an
      observe(X[1,0] > 0) updates the matrix distribution of X via Tallis
      truncation and Kalman-style mean shift.  Because the cross-covariance
      Cov(y0, X) is stored in a separate flat scalar entry (not re-derived from
      X's updated Kronecker factors), the scalar y0 is NOT Kalman-updated.
      E[y0] remains at the prior mean while E[X[0,0]] shifts upward.

  F4  (extract + write cross-cov): After ``y = X[0,0]`` and ``X[0,0] = 5.0``
      (Schur element write that sets X[0,0] to a deterministic constant),
      the cross-covariance Cov(y, X[0,0]) should be 0 because the new X[0,0]
      is deterministic.  But the stale cross-cov block in cov_blocks still
      carries the pre-write value U[0,0]*V[0,0] != 0.

Plan reference: plan/2026-05-25-matrix-gm-stress-test.md §Iter 5 — Fragile paths
Fix target: dedicated plan (not this one — see LIMITATIONS.md for follow-up).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

SOGA_PY = os.path.join(_SRC, "SOGA.py")
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# SOGA subprocess runner (same as test_ops_analytical.py)
# ---------------------------------------------------------------------------

def _run_soga(program_text: str, timeout: int = 15) -> tuple:
    """Run a SOGA program via subprocess. Return (stdout, stderr, returncode)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".soga", delete=False) as f:
        f.write(program_text)
        fname = f.name
    try:
        result = subprocess.run(
            [PYTHON, SOGA_PY, "-f", fname, "-t", str(timeout - 2)],
            capture_output=True, text=True, timeout=timeout,
            cwd=_SRC,
        )
        return result.stdout, result.stderr, result.returncode
    finally:
        os.unlink(fname)


def _parse_matrix_e(output: str, var: str) -> np.ndarray:
    """Parse E[VAR]:\n<matrix> block from SOGA output. Returns ndarray."""
    pattern = rf"E\[{re.escape(var)}\]:\s*\n((?:[ \t]*\[.*\]\n?)+)"
    m = re.search(pattern, output)
    if m is None:
        raise AssertionError(
            f"E[{var}] matrix block not found in output:\n{output[:600]}"
        )
    rows = []
    for line in m.group(1).strip().splitlines():
        line = line.strip().strip("[]")
        if line:
            vals = [float(v) for v in line.split()]
            rows.append(vals)
    return np.array(rows)


# ---------------------------------------------------------------------------
# F1: observe(X[i,j] > c) followed by Y = A @ X crashes subprocess
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    match=r"Expected 'E\[Y\]' in SOGA stdout",
    reason=(
        "C1+Path5: observe on matrix variable densifies covariance sentinel. "
        "Subsequent A@X in _matrix_affine_left calls A @ None (TypeError). "
        "Subprocess crashes → SOGA prints 'Warning: SOGA Timeout occurred' "
        "with no E[Y] in stdout → assert below raises AssertionError."
    ),
)
def test_F1_observe_then_left_affine_hangs():
    """F1: observe(X[0,0]>0) then Y = A@X — subprocess worker crashes silently.

    Expected broken behaviour: SOGA stdout contains 'Warning: SOGA Timeout
    occurred' and does NOT contain 'E[Y]' because the multiprocessing worker
    crashes with TypeError (A @ None) before writing to the Queue.

    The test ASSERTS that E[Y] is present in stdout; this assertion fails
    (AssertionError) because it is not — confirming the bug is still present.
    """
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        matrix[2][2] A;
        X = matrix_gm([[1.0,0.0],[0.0,0.0]], [[1.0,0.0],[0.0,1.0]], [[1.0,0.0],[0.0,1.0]]);
        A = matrix_gm([[1.0,0.0],[0.0,1.0]], [[0.1,0.0],[0.0,0.1]], [[0.1,0.0],[0.0,0.1]]);
        observe(X[0,0] > 0);
        Y = A @ X;
    """)
    stdout, stderr, rc = _run_soga(prog, timeout=10)
    # This assertion MUST fail while the bug exists.
    # If it passes (xpass), the bug was silently fixed — update the xfail marker.
    assert "E[Y]" in stdout, (
        f"Expected 'E[Y]' in SOGA stdout but got:\n{stdout[:400]}\n"
        f"(stderr: {stderr[:200]})"
    )


# ---------------------------------------------------------------------------
# F2a: X = matrix_gm_full(M, Sigma_dense); Y = X @ B → AttributeError
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason=(
        "C7: _matrix_affine_right called on dense sentinel (None, Sigma). "
        "get_cov returns (None, Sigma_dense); code executes U_x.copy() where "
        "U_x is None → AttributeError: 'NoneType' object has no attribute 'copy'."
    ),
)
def test_F2a_dense_sentinel_right_affine_fails():
    """F2a: Y = X @ B when X is stored as dense sentinel → AttributeError.

    Reproduced directly against the internal API (no subprocess) to avoid
    the timeout path and get the raw exception from _matrix_affine_right.
    """
    from libSOGAshared import VarEntry
    from libSOGAsharedMatrix import GaussianMixBlock
    from libMatrixUpdate import _matrix_affine_right

    m, n = 2, 2
    M = np.array([[1.0, 0.5], [0.3, 0.8]])
    Sigma = np.array([
        [2.0, 0.4, 0.3, 0.1],
        [0.4, 1.5, 0.2, 0.3],
        [0.3, 0.2, 1.8, 0.2],
        [0.1, 0.3, 0.2, 1.2],
    ])
    ve = VarEntry("X", "matrix", (m, n))
    pi = [1.0]
    mu_blocks = [{"X": M.copy()}]
    cov_blocks = [{frozenset({"X"}): (None, Sigma.copy())}]
    block = GaussianMixBlock([], [ve], pi, mu_blocks, cov_blocks)

    B = np.array([[1.0, 0.3], [0.0, 1.0]])
    # This call MUST raise AttributeError while the bug exists.
    # If it succeeds (xpass), the bug was fixed — update the xfail marker.
    _matrix_affine_right(block, 0, "X", B)


# ---------------------------------------------------------------------------
# F2b: X = matrix_gm_full(M, Sigma_dense); Y = A @ X → ValueError
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        "C7: _matrix_affine_left called on dense sentinel (None, Sigma). "
        "get_cov returns (None, Sigma_dense); code executes A @ U_x where "
        "U_x is None (a 0-d object) → ValueError: matmul: Input operand 1 "
        "does not have enough dimensions."
    ),
)
def test_F2b_dense_sentinel_left_affine_fails():
    """F2b: Y = A @ X when X is stored as dense sentinel → ValueError.

    Reproduced directly against the internal API (no subprocess) to avoid
    the timeout path and get the raw exception from _matrix_affine_left.
    """
    from libSOGAshared import VarEntry
    from libSOGAsharedMatrix import GaussianMixBlock
    from libMatrixUpdate import _matrix_affine_left

    m, n = 2, 2
    M = np.array([[1.0, 0.5], [0.3, 0.8]])
    Sigma = np.array([
        [2.0, 0.4, 0.3, 0.1],
        [0.4, 1.5, 0.2, 0.3],
        [0.3, 0.2, 1.8, 0.2],
        [0.1, 0.3, 0.2, 1.2],
    ])
    ve = VarEntry("X", "matrix", (m, n))
    pi = [1.0]
    mu_blocks = [{"X": M.copy()}]
    cov_blocks = [{frozenset({"X"}): (None, Sigma.copy())}]
    block = GaussianMixBlock([], [ve], pi, mu_blocks, cov_blocks)

    A = np.array([[2.0, 0.5], [0.0, 1.0]])
    # This call MUST raise ValueError while the bug exists.
    # If it succeeds (xpass), the bug was fixed — update the xfail marker.
    _matrix_affine_left(block, 0, "X", A)


# ---------------------------------------------------------------------------
# F3: Gap 3 — extracted scalar not Kalman-updated after observe
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    match=r"Expected updated mean E\[y0\]=",
    reason=(
        "Gap3: after y0 = X[0,0] and observe(X[1,0] > 0), the scalar y0 "
        "is NOT Kalman-updated even though X[0,0] and X[1,0] are correlated "
        "(off-diagonal U factor). E[y0] stays at the prior mean while "
        "E[X[0,0]] shifts upward via the Tallis+Kalman path on the matrix "
        "distribution. The stale cross-cov in cov_blocks is never propagated "
        "back to the scalar y0 slot."
    ),
)
def test_F3_stale_cross_cov_after_observe():
    """F3: E[y0] is stale after observe(X[1,0]>0) with correlated rows.

    Setup: X ~ MN(M, U, V) with M[0,0]=1.0 and U off-diagonal 0.5 (rows
    correlated).  After observe(X[1,0]>0), the Tallis truncation shifts
    E[X[0,0]] from 1.0 to ~1.4 (Kalman-style update via row cross-cov).
    E[y0] must also shift because y0 = X[0,0] was extracted before observe.
    The test asserts |E[y0] - E_X_00_post| < 0.05 — this fails because
    E[y0] stays at 1.0 while E[X[0,0]] moves to ~1.4.
    """
    prog = textwrap.dedent("""\
        matrix[2][2] X;
        X = matrix_gm([[1.0,0.0],[0.0,0.0]], [[1.0,0.5],[0.5,1.0]], [[1.0,0.0],[0.0,1.0]]);
        y0 = X[0,0];
        observe(X[1,0] > 0);
    """)
    stdout, stderr, rc = _run_soga(prog, timeout=20)
    assert rc == 0, f"SOGA returned rc={rc}\nstderr:{stderr[:300]}"

    # Parse E[y0] (scalar) from stdout
    m = re.search(r"E\[y0\]:\s+([-\d.eE+]+)", stdout)
    assert m is not None, f"E[y0] not found in:\n{stdout[:400]}"
    e_y0 = float(m.group(1))

    # Parse E[X] matrix and extract X[0,0]
    E_X = _parse_matrix_e(stdout, "X")
    e_X00_post = float(E_X[0, 0])

    # The test ASSERTS that y0 was updated to match the posterior X[0,0].
    # This MUST fail while Gap3 exists: e_y0 == 1.0 (prior) but
    # e_X00_post == ~1.4 (Kalman-shifted via truncation + row cross-cov).
    assert abs(e_y0 - e_X00_post) < 0.05, (
        f"Expected updated mean E[y0]={e_y0:.5f} to match E[X[0,0]]_post="
        f"{e_X00_post:.5f} (Kalman update via row cross-cov), but got "
        f"abs difference {abs(e_y0 - e_X00_post):.5f} >= 0.05. "
        f"Gap3: stale cross-cov prevents back-propagation of observe update."
    )


# ---------------------------------------------------------------------------
# F4: extract + write — cross-cov not zeroed after Schur element write
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    match=r"Expected Cov\(y, X\[0,0\]\) to be zero after Schur write",
    reason=(
        "F4: after y = X[0,0] and X[0,0] = 5.0 (Schur write makes X[0,0] "
        "deterministic), Cov(y, X[0,0]) should be 0 because X[0,0] is now "
        "a known constant. But the stale cross-cov entry in cov_blocks still "
        "holds the pre-write value U[0,0]*V[0,0] != 0."
    ),
)
def test_F4_extract_write_stale_cross_cov():
    """F4: Cov(y, X[0,0]) != 0 after X[0,0] = 5.0 (should be 0).

    Setup: X ~ MN(M, U, V) with U[0,0]=1.0, V[0,0]=1.0 → Var(X[0,0])=1.0.
    y = X[0,0] extracted first.  Then X[0,0] = 5.0 is a Schur element write
    that conditions X on X[0,0]=5 and densifies its covariance.  After this,
    X[0,0] is deterministic (variance 0), so Cov(y, X[0,0]) must equal 0
    because y captures the OLD random X[0,0] and the NEW X[0,0]=5 is a
    constant independent of y.  The stale cov_blocks still holds the
    pre-write cross-cov, causing the test assertion to fail.
    """
    from libSOGAshared import VarEntry
    from libSOGAsharedMatrix import GaussianMixBlock

    # Build block: X ~ MN([[1,0],[0,0]], I_2, I_2)
    m, n = 2, 2
    M = np.array([[1.0, 0.0], [0.0, 0.0]])
    U = np.eye(2)
    V = np.eye(2)
    ve = VarEntry("X", "matrix", (m, n))
    pi = [1.0]
    mu_blocks = [{"X": M.copy()}]
    cov_blocks = [{frozenset({"X"}): (U.copy(), V.copy())}]
    # Scalar y: mean = M[0,0] = 1.0, variance = U[0,0]*V[0,0] = 1.0
    # Cross-cov Cov(y, vec(X)) encodes Cov(X[0,0], X[i,j]) = U[0,i]*V[0,j]
    # For now, y is stored as a separate scalar variable in var_list.
    # We simulate the post-extract state: var_list=['y'], cov_blocks has
    # 'y' → float(1.0) and cross-cov frozenset({'y','X'}) → flat array
    # representing Cov(y, vec(X)) = [U[0,0]*V[0,0], U[0,0]*V[0,1],
    #                                  U[1,0]*V[0,0], U[1,0]*V[0,1]] = [1,0,0,0]
    cross_cov_y_X = np.array([
        U[0, 0] * V[0, 0],  # Cov(y, X[0,0])
        U[0, 0] * V[0, 1],  # Cov(y, X[1,0]) — column-major index j*m+i
        U[0, 1] * V[0, 0],  # Cov(y, X[0,1])  wait: col-major: j=0,i=0 → 0; j=0,i=1 → 1; etc.
        U[0, 1] * V[0, 1],  # Cov(y, X[1,1])
    ], dtype=float)
    # Column-major (vec convention): idx = j*m + i
    # Cov(y, X[i,j]) = U[0,i] * V[0,j] for extract y=X[0,0]
    # Correct column-major order: (0,0)→0, (1,0)→1, (0,1)→2, (1,1)→3
    cross_cov_y_X = np.array([
        float(U[0, 0] * V[0, 0]),  # idx 0: (i=0,j=0)
        float(U[0, 1] * V[0, 0]),  # idx 1: (i=1,j=0)
        float(U[0, 0] * V[0, 1]),  # idx 2: (i=0,j=1)
        float(U[0, 1] * V[0, 1]),  # idx 3: (i=1,j=1)
    ])

    # Inject the cross-cov entry (simulating state after y = X[0,0])
    mu_blocks[0]["y"] = np.array([M[0, 0]])
    cov_blocks[0][frozenset({"y"})] = float(U[0, 0] * V[0, 0])  # Var(y)
    cov_blocks[0][frozenset({"y", "X"})] = cross_cov_y_X.copy()

    block = GaussianMixBlock(["y"], [ve], pi, mu_blocks, cov_blocks)

    # Simulate the Schur element write X[0,0] = 5.0
    # After the write, X[0,0] is deterministic.  The cross-cov Cov(y, X[0,0])
    # should become 0 because X[0,0] = 5 (constant) → Cov(y, 5) = 0.
    from libMatrixUpdate import _matrix_element_write_component
    _matrix_element_write_component(block, 0, "X", m, n, 0, 0, 5.0, 0.0)

    # The cross-cov vector for (y, X) post-write should have index 0 = 0
    # because Cov(y, X[0,0]) where X[0,0] is now deterministic must be 0.
    cross_cov_after = block.cov_blocks[0].get(frozenset({"y", "X"}))
    if cross_cov_after is None:
        cross_cov_y_X00_after = 0.0  # entry removed → Cov = 0 (correct)
    elif hasattr(cross_cov_after, "__len__"):
        cross_cov_y_X00_after = float(cross_cov_after[0])
    else:
        cross_cov_y_X00_after = float(cross_cov_after)

    # This assertion MUST fail while F4 exists: the stale cross-cov is
    # != 0 after the write (it retains the pre-write U[0,0]*V[0,0]=1.0).
    # The assertion MUST fail while F4 exists: cross_cov_y_X00_after == 1.0
    # (stale, non-zero) when the correct value is 0 (X[0,0] deterministic).
    assert abs(cross_cov_y_X00_after) < 1e-10, (
        f"Expected Cov(y, X[0,0]) to be zero after Schur write X[0,0]=5.0 "
        f"(X[0,0] is deterministic → Cov(y, const)=0) but got "
        f"{cross_cov_y_X00_after:.6e}. "
        f"F4: stale cross-cov not zeroed after element write."
    )
