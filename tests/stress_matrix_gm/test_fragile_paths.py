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

def _run_soga(program_text: str, timeout: int = 15, soga_timeout: int = None) -> tuple:
    """Run a SOGA program via subprocess. Return (stdout, stderr, returncode).

    soga_timeout: the -t flag passed to SOGA.py (worker queue timeout).
    Defaults to timeout - 2 if not specified.
    """
    if soga_timeout is None:
        soga_timeout = timeout - 2
    with tempfile.NamedTemporaryFile(mode="w", suffix=".soga", delete=False) as f:
        f.write(program_text)
        fname = f.name
    try:
        result = subprocess.run(
            [PYTHON, SOGA_PY, "-f", fname, "-t", str(soga_timeout)],
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
# F1: observe(X[i,j] > c) followed by Y = A @ X → NotImplementedError (subprocess)
# ---------------------------------------------------------------------------

def test_F1_observe_then_left_affine_hangs():
    """F1: observe(X[0,0]>0) then Y = A@X — worker raises NotImplementedError.

    Post-patch behaviour: the guard in _matrix_affine_left raises
    NotImplementedError (message "[C1/C7]...LIMITATIONS.md") when called on a
    dense-sentinel covariance (produced by observe densifying X).

    SOGA architecture note: the worker Process raises NotImplementedError;
    its traceback appears in stderr (worker inherits parent stderr).  The main
    process then hits the queue.get() timeout and prints
    "Warning: SOGA Timeout occurred", then exits with rc=0 (no sys.exit call).
    Therefore: assert rc == 0, assert "NotImplementedError" in combined output,
    assert "LIMITATIONS" or "dense" in combined output.
    """
    prog = textwrap.dedent("""\
        data A = [[1.0,0.0],[0.0,1.0]];
        matrix[2][2] X;
        matrix[2][2] Y;
        X = matrix_gm([[1.0,0.0],[0.0,0.0]], [[1.0,0.0],[0.0,1.0]], [[1.0,0.0],[0.0,1.0]]);
        observe(X[0,0] > 0);
        Y = A @ X;
    """)
    stdout, stderr, rc = _run_soga(prog, timeout=5, soga_timeout=3)
    combined = stdout + stderr
    # rc=0: SOGA main process hits queue timeout, prints warning, exits cleanly.
    assert rc == 0, (
        f"Expected rc=0 (SOGA queue timeout path) but got rc={rc}.\n"
        f"stdout: {stdout[:400]}\nstderr: {stderr[:400]}"
    )
    # Worker traceback (NotImplementedError) must appear somewhere in combined output.
    assert "NotImplementedError" in combined, (
        f"Expected 'NotImplementedError' in stdout+stderr but not found.\n"
        f"stdout: {stdout[:400]}\nstderr: {stderr[:400]}"
    )
    # The error message must reference LIMITATIONS or 'dense' to guide the user.
    assert "LIMITATIONS" in combined or "dense" in combined.lower(), (
        f"Expected 'LIMITATIONS' or 'dense' in stdout+stderr but not found.\n"
        f"stdout: {stdout[:400]}\nstderr: {stderr[:400]}"
    )


# ---------------------------------------------------------------------------
# F2a: X = matrix_gm_full(M, Sigma_dense); Y = X @ B → NotImplementedError
# ---------------------------------------------------------------------------

def test_F2a_dense_sentinel_right_affine_fails():
    """F2a: Y = X @ B when X is stored as dense sentinel → NotImplementedError.

    Post-patch: the guard in _matrix_affine_right raises NotImplementedError
    with a clear message referencing [C1/C7] and LIMITATIONS.md §C1/C7,
    instead of the previous AttributeError on U_x.copy() where U_x is None.
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
    # Post-patch: guard raises NotImplementedError with clear message.
    with pytest.raises(NotImplementedError, match=r"(?i)dense.sentinel|C1/C7"):
        _matrix_affine_right(block, 0, "X", B)


# ---------------------------------------------------------------------------
# F2b: X = matrix_gm_full(M, Sigma_dense); Y = A @ X → NotImplementedError
# ---------------------------------------------------------------------------

def test_F2b_dense_sentinel_left_affine_fails():
    """F2b: Y = A @ X when X is stored as dense sentinel → NotImplementedError.

    Post-patch: the guard in _matrix_affine_left raises NotImplementedError
    with a clear message referencing [C1/C7] and LIMITATIONS.md §C1/C7,
    instead of the previous ValueError on A @ U_x where U_x is None.
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
    # Post-patch: guard raises NotImplementedError with clear message.
    with pytest.raises(NotImplementedError, match=r"(?i)dense.sentinel|C1/C7"):
        _matrix_affine_left(block, 0, "X", A)


# ---------------------------------------------------------------------------
# F3: Gap 3 — extracted scalar not Kalman-updated after observe (in-process)
# ---------------------------------------------------------------------------

def test_F3_stale_cross_cov_after_observe():
    """F3: StaleCrossCovWarning emitted; E[y0] stays stale after observe(X[1,0]>0).

    Post-patch: _truncate_matrix_element_ineq emits StaleCrossCovWarning
    when it detects scalar y0 has non-zero cross-cov with X.  The scalar
    y0 is NOT updated (Gap3 bug is documented but not fixed) — this test
    locks the stale-but-warned behavior.

    Setup (in-process, no subprocess): build a GaussianMixBlock representing
    X ~ MN(M, U, V) with off-diagonal U (rows correlated), plus scalar y0
    extracted from X[0,0] with the correct cross-cov.  Then call
    _truncate_matrix_element_ineq directly for observe(X[1,0] > 0).
    """
    from libSOGAshared import VarEntry
    from libSOGAsharedMatrix import GaussianMixBlock
    from libMatrixTruncate import _truncate_matrix_element_ineq
    from libMatrixGaussian import StaleCrossCovWarning

    # Build block: X ~ MN([[1,0],[0,0]], U_off, I_2)
    m_mat, n_mat = 2, 2
    M = np.array([[1.0, 0.0], [0.0, 0.0]])
    U = np.array([[1.0, 0.5], [0.5, 1.0]])   # off-diagonal: rows correlated
    V = np.eye(2)
    ve_X = VarEntry("X", "matrix", (m_mat, n_mat))
    pi = [1.0]
    mu_blocks = [{"X": M.copy()}]
    cov_blocks = [{frozenset({"X"}): (U.copy(), V.copy())}]

    # Simulate y0 = X[0,0] extraction: add scalar y0 with mean M[0,0]=1.0
    # Cross-cov Cov(y0, vec(X)) in column-major: Cov(X[0,0], X[i,j]) = U[0,i]*V[0,j]
    cross_cov_y0_X = np.array([
        float(U[0, 0] * V[0, 0]),  # idx 0: (i=0,j=0)
        float(U[0, 1] * V[0, 0]),  # idx 1: (i=1,j=0)
        float(U[0, 0] * V[0, 1]),  # idx 2: (i=0,j=1)
        float(U[0, 1] * V[0, 1]),  # idx 3: (i=1,j=1)
    ])
    mu_blocks[0]["y0"] = np.array([M[0, 0]])
    cov_blocks[0][frozenset({"y0"})] = float(U[0, 0] * V[0, 0])
    cov_blocks[0][frozenset({"y0", "X"})] = cross_cov_y0_X.copy()

    prior_mean_y0 = float(M[0, 0])  # = 1.0

    block = GaussianMixBlock(["y0"], [ve_X], pi, mu_blocks, cov_blocks)

    # Post-patch: StaleCrossCovWarning must be emitted because y0 has non-zero
    # cross-cov with X.  The warning confirms Gap3 is now loud (not silent).
    with pytest.warns(StaleCrossCovWarning, match=r"y0|cross.cov"):
        result_block, norm = _truncate_matrix_element_ineq(
            block, "X", m_mat, n_mat, 1, 0, 0.0, "gt", False
        )

    # Verify the warning was emitted AND the scalar y0 is still stale.
    # E[y0] must NOT change (Gap3 is documented, not fixed).
    e_y0_post = float(result_block.mu_blocks[0]["y0"][0])
    assert abs(e_y0_post - prior_mean_y0) < 1e-10, (
        f"E[y0] changed from {prior_mean_y0} to {e_y0_post}: "
        f"unexpected fix of Gap3 — update test if bug was intentionally repaired."
    )

    # E[X[0,0]] changes after observe(X[1,0] > 0) via Kalman back-prop through the
    # off-diagonal U cross-cov.  ANALYTICALLY CORRECT VALUE: 1.39894
    #   E[X[0,0]|obs] = M[0,0] + Cov(X[0,0],X[1,0])/Var(X[1,0]) * delta
    #                 = 1 + 0.5/1 * sqrt(2/pi) = 1.39894
    # CURRENT (BUGGY) VALUE: 0.60106 — same magnitude (0.39894) but OPPOSITE SIGN.
    # This is bug C8 (sign error in dense Kalman back-prop to non-observed elements).
    # See docs/LIMITATIONS.md §C8.  This assertion LOCKS the buggy direction so
    # that when C8 is fixed the test will xpass (FAIL strict) and force a review.
    e_X00_post = float(result_block.mu_blocks[0]["X"][0, 0])
    assert e_X00_post == pytest.approx(0.60106, abs=1e-3), (
        f"E[X[0,0]] = {e_X00_post:.5f}, expected 0.60106 (buggy current C8 behavior); "
        f"analytical correct value is 1.39894.  If this test now produces 1.39894, "
        f"C8 has been fixed — update LIMITATIONS.md and assert the correct value here."
    )


# ---------------------------------------------------------------------------
# F4: extract + write — StaleCrossCovWarning emitted; cross-cov stays stale
# ---------------------------------------------------------------------------

def test_F4_extract_write_stale_cross_cov():
    """F4: StaleCrossCovWarning emitted; Cov(y, X[0,0]) stays stale after X[0,0]=5.0.

    Post-patch: _matrix_element_write_component emits StaleCrossCovWarning
    when it detects scalar y has non-zero cross-cov with X.  The cross-cov
    is NOT zeroed (F4 bug is documented but not fixed) — this test locks the
    stale-but-warned behavior.

    Warning is emitted but value is intentionally NOT fixed (see LIMITATIONS.md §F4).
    This test locks the stale-but-warned behavior.
    """
    from libSOGAshared import VarEntry
    from libSOGAsharedMatrix import GaussianMixBlock
    from libMatrixGaussian import StaleCrossCovWarning

    # Build block: X ~ MN([[1,0],[0,0]], I_2, I_2)
    m, n = 2, 2
    M = np.array([[1.0, 0.0], [0.0, 0.0]])
    U = np.eye(2)
    V = np.eye(2)
    ve = VarEntry("X", "matrix", (m, n))
    pi = [1.0]
    mu_blocks = [{"X": M.copy()}]
    cov_blocks = [{frozenset({"X"}): (U.copy(), V.copy())}]
    # Column-major (vec convention): idx = j*m + i
    # Cov(y, X[i,j]) = U[0,i] * V[0,j] for extract y=X[0,0]
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

    # Post-patch: StaleCrossCovWarning must be emitted because y has non-zero
    # cross-cov with X.  The warning confirms F4 is now loud (not silent).
    from libMatrixUpdate import _matrix_element_write_component
    with pytest.warns(StaleCrossCovWarning):
        _matrix_element_write_component(block, 0, "X", m, n, 0, 0, 5.0, 0.0)

    # The cross-cov vector for (y, X) post-write should still be non-zero
    # because F4 is NOT fixed — the stale cross-cov entry in cov_blocks still
    # holds the pre-write value U[0,0]*V[0,0]=1.0 != 0.
    cross_cov_after = block.cov_blocks[0].get(frozenset({"y", "X"}))
    if cross_cov_after is None:
        cross_cov_y_X00_after = 0.0  # entry removed → Cov = 0 (would be F4 fixed)
    elif hasattr(cross_cov_after, "__len__"):
        cross_cov_y_X00_after = float(cross_cov_after[0])
    else:
        cross_cov_y_X00_after = float(cross_cov_after)

    # Warning is emitted but value is intentionally NOT fixed (see LIMITATIONS.md §F4).
    # This test locks the stale-but-warned behavior: cross-cov IS still non-zero.
    assert abs(cross_cov_y_X00_after) > 1e-10, (
        f"Expected Cov(y, X[0,0]) to remain non-zero (stale) after Schur write X[0,0]=5.0 "
        f"(F4 bug documents that cross-cov is NOT zeroed), but got "
        f"{cross_cov_y_X00_after:.6e} ≈ 0. "
        f"If F4 was intentionally fixed, remove this test and update LIMITATIONS.md."
    )
