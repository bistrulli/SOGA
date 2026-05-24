"""
tests/test_loop_matrix.py — fix1: loop-variable resolution in matrix indexing.

Tests that matrix[i,j] extraction works when i and j are loop-counter
identifiers resolved via data[name][0], not just numeric literals.

Analytical ground truth: for X = diag(1,2,3,4) with U=I, V=I,
  E[d] after d = X[i,i] is exactly M[i,i] = (i+1).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from libSOGAshared import Dist, GaussianMix, VarEntry
from libSOGAsharedMatrix import GaussianMixBlock
from libSOGAupdate import update_rule
from libMatrixTruncate import _classify_constraint, _resolve_index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_diag_dist(n: int = 4) -> tuple:
    """Return (dist, diag_vals) for X = diag(1..n) with U=I, V=I."""
    ve = VarEntry('X', 'matrix', (n, n))
    M = np.diag([float(k + 1) for k in range(n)])
    U = np.eye(n)
    V = np.eye(n)
    block = GaussianMixBlock.from_matrix_gm(
        [ve], [M], [U], [V], pi=[1.0], var_list=['d']
    )
    gm = GaussianMix([1.0], [np.array([0.0])], [np.array([[0.0]])])
    dist = Dist(['d'], gm, var_entries=[ve], gm_block=block)
    return dist, [float(k + 1) for k in range(n)]


# ---------------------------------------------------------------------------
# fix1.1 — update_rule with IDV index
# ---------------------------------------------------------------------------

class TestLoopVarIndexExtraction:
    """fix1.1: update_rule accepts X[i,j] with identifier indices."""

    def test_diagonal_numeric_index(self):
        """Baseline: numeric literals still work after fix."""
        dist, diag = _make_diag_dist(4)
        data = {}
        result = update_rule(dist, 'd = X[0,0]', data)
        assert abs(float(result.gm.mu[0][0]) - 1.0) < 1e-10

    def test_diagonal_loop_var_row_col(self):
        """fix1: d = X[i,i] with i as loop counter — 4 iterations."""
        dist, diag = _make_diag_dist(4)
        for i in range(4):
            data = {'i': [i]}
            result = update_rule(dist, 'd = X[i, i]', data)
            expected = diag[i]  # exact closed-form: M[i,i] = i+1
            got = float(result.gm.mu[0][0])
            assert abs(got - expected) < 1e-10, (
                f"i={i}: E[d]={got} != expected={expected}"
            )

    def test_off_diagonal_loop_vars(self):
        """fix1: d = X[i,j] with i,j as separate loop vars."""
        n = 3
        ve = VarEntry('X', 'matrix', (n, n))
        M = np.arange(1., n * n + 1.).reshape(n, n, order='F')  # col-major
        U = np.eye(n); V = np.eye(n)
        block = GaussianMixBlock.from_matrix_gm(
            [ve], [M], [U], [V], pi=[1.0], var_list=['d']
        )
        gm = GaussianMix([1.0], [np.array([0.0])], [np.array([[0.0]])])
        dist = Dist(['d'], gm, var_entries=[ve], gm_block=block)
        for i in range(n):
            for j in range(n):
                data = {'i': [i], 'j': [j]}
                result = update_rule(dist, 'd = X[i, j]', data)
                expected = float(M[i, j])
                got = float(result.gm.mu[0][0])
                assert abs(got - expected) < 1e-10, (
                    f"X[{i},{j}]: E[d]={got} != {expected}"
                )

    def test_mixed_numeric_and_loop_var(self):
        """fix1: d = X[i,0] — one fixed, one loop variable."""
        dist, diag = _make_diag_dist(4)
        for i in range(4):
            data = {'i': [i]}
            result = update_rule(dist, 'd = X[i, 0]', data)
            # Column 0, row i: M[i,0] = 0 for i>0, M[0,0]=1 (diagonal matrix)
            ve = dist.var_entries[0]
            M = dist.gm_block.mu_blocks[0]['X']
            expected = float(M[i, 0])
            got = float(result.gm.mu[0][0])
            assert abs(got - expected) < 1e-10

    def test_index_var_not_in_data_raises(self):
        """fix1: missing loop var raises KeyError with informative message."""
        dist, _ = _make_diag_dist(4)
        data = {}  # 'i' missing
        with pytest.raises((KeyError, TypeError)):
            update_rule(dist, 'd = X[i, i]', data)

    def test_var_at_zero_index(self):
        """fix1: loop var = 0 should extract correctly (not None guard)."""
        dist, diag = _make_diag_dist(4)
        data = {'i': [0]}
        result = update_rule(dist, 'd = X[i, i]', data)
        assert abs(float(result.gm.mu[0][0]) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# fix1.2 — _classify_constraint with IDV indices
# ---------------------------------------------------------------------------

class TestClassifyConstraintIDV:
    """fix1.2: _classify_constraint resolves IDV in index positions."""

    def test_element_ineq_numeric(self):
        """Baseline: numeric index still classified correctly."""
        c = _classify_constraint('X[1,2] > 0.5', 'X')
        assert c['type'] == 'ELEMENT_INEQ'
        assert c['i'] == 1 and c['j'] == 2

    def test_element_ineq_loop_var(self):
        """fix1.2: X[i,j] > c with loop vars resolved via data."""
        data = {'i': [2], 'j': [3]}
        c = _classify_constraint('X[i,j] > 0.5', 'X', data)
        assert c['type'] == 'ELEMENT_INEQ'
        assert c['i'] == 2 and c['j'] == 3

    def test_row_sum_ineq_loop_var(self):
        """fix1.2: row_sum(X, i) > c with loop var."""
        data = {'i': [1]}
        c = _classify_constraint('row_sum(X, i) > 0.0', 'X', data)
        assert c['type'] == 'ROW_SUM_INEQ'
        assert c['i'] == 1

    def test_col_sum_ineq_loop_var(self):
        """fix1.2: col_sum(X, j) <= c with loop var."""
        data = {'j': [0]}
        c = _classify_constraint('col_sum(X, j) <= 1.0', 'X', data)
        assert c['type'] == 'COL_SUM_INEQ'
        assert c['j'] == 0


# ---------------------------------------------------------------------------
# _resolve_index helper
# ---------------------------------------------------------------------------

class TestResolveIndex:
    """Unit tests for fix1.2 _resolve_index helper."""

    def test_numeric_string(self):
        assert _resolve_index('0') == 0
        assert _resolve_index('3') == 3
        assert _resolve_index('10') == 10

    def test_identifier_in_data(self):
        data = {'i': [2], 'loop_k': [7]}
        assert _resolve_index('i', data) == 2
        assert _resolve_index('loop_k', data) == 7

    def test_identifier_missing_raises(self):
        with pytest.raises(KeyError):
            _resolve_index('missing_var', {})

    def test_no_data_with_identifier_raises(self):
        with pytest.raises(KeyError):
            _resolve_index('i', None)


# ---------------------------------------------------------------------------
# End-to-end: t9_loop_index via start_SOGA (sanity)
# ---------------------------------------------------------------------------

def test_t9_loop_index_end_to_end():
    """E2E: for i in range(4) { d = X[i,i]; } — final d = X[3,3] = 4."""
    import subprocess, tempfile, os
    soga_prog = """\
matrix[4][4] X;
X = matrix_gm([[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
for i in range(4) {
    d = X[i, i];
} end for;
"""
    with tempfile.NamedTemporaryFile(suffix='.soga', mode='w', delete=False) as f:
        f.write(soga_prog)
        fname = f.name
    try:
        root = os.path.join(os.path.dirname(__file__), '..')
        result = subprocess.run(
            [os.path.join(root, '.venv/bin/python3'), 'src/SOGA.py', '-f', fname],
            capture_output=True, text=True, cwd=root
        )
        assert result.returncode == 0, f"SOGA crashed:\n{result.stderr}"
        # After 4 iterations, d = X[3,3] = 4.0
        assert 'E[d]: 4.0' in result.stdout or 'E[d]: 4' in result.stdout, (
            f"Expected E[d]=4.0, got:\n{result.stdout}"
        )
    finally:
        os.unlink(fname)
