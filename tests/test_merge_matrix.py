"""
tests/test_merge_matrix.py — fix2: merge with matrix variables.

Tests component concatenation (research note 05 §Problem 1), weight rescaling,
variable-scope handling, and ranking_prune synchronisation with gm_block.

Analytical ground truth:
  2-branch merge with p_then=0.5, p_else=0.5, each branch has K=1 component
  → merged has K=2 components with pi = [0.5, 0.5].
  Mixture mean = 0.5 * M_then + 0.5 * M_else.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from copy import deepcopy

from libSOGAshared import Dist, GaussianMix, VarEntry
from libSOGAsharedMatrix import GaussianMixBlock
from libSOGAmerge import merge, ranking_prune, _merge_matrix_gm_blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_matrix_dist(
    mat_name: str, M: np.ndarray, U: np.ndarray = None, V: np.ndarray = None,
    pi: float = 1.0, scalar_mu: float = 0.0, scalar_var: float = 1.0,
    scalar_name: str = 'theta'
) -> Dist:
    """Build a Dist with one matrix var and one scalar."""
    m, n = M.shape
    if U is None:
        U = np.eye(m)
    if V is None:
        V = np.eye(n)
    ve = VarEntry(mat_name, 'matrix', (m, n))
    block = GaussianMixBlock.from_matrix_gm(
        [ve], [M], [U], [V], pi=[pi], var_list=[scalar_name]
    )
    gm = GaussianMix([pi], [np.array([scalar_mu])], [np.array([[scalar_var]])])
    return Dist([scalar_name], gm, var_entries=[ve], gm_block=block)


# ---------------------------------------------------------------------------
# fix2.1 — Component concatenation
# ---------------------------------------------------------------------------

class TestMergeMatrixConcatenation:
    """fix2.1: merge concatenates components and rescales weights."""

    def test_two_branch_merge_weights(self):
        """Merged pi = [p_then * pi_k, p_else * pi_k] renormalised."""
        dist_then = _make_matrix_dist('X', np.array([[1., 2.], [3., 4.]]))
        dist_else = _make_matrix_dist('X', np.array([[5., 6.], [7., 8.]]))
        p_then, p_else = 0.5, 0.5
        current_p, merged = merge([(p_then, dist_then), (p_else, dist_else)])
        assert abs(current_p - 1.0) < 1e-10
        assert merged.gm_block is not None
        assert merged.gm_block.n_comp() == 2
        # Equal weights
        assert abs(merged.gm_block.pi[0] - 0.5) < 1e-10
        assert abs(merged.gm_block.pi[1] - 0.5) < 1e-10

    def test_two_branch_merge_means_preserved(self):
        """Per-component means are preserved exactly after merge."""
        M_then = np.array([[1., 2.], [3., 4.]])
        M_else = np.array([[5., 6.], [7., 8.]])
        dist_then = _make_matrix_dist('X', M_then)
        dist_else = _make_matrix_dist('X', M_else)
        _, merged = merge([(0.5, dist_then), (0.5, dist_else)])
        blk = merged.gm_block
        assert np.allclose(blk.mu_blocks[0]['X'], M_then)
        assert np.allclose(blk.mu_blocks[1]['X'], M_else)

    def test_single_branch_passthrough(self):
        """Single-element list_dist passes through unchanged."""
        dist = _make_matrix_dist('X', np.eye(2))
        p, merged = merge([(0.7, dist)])
        assert p == 0.7
        assert merged is dist  # exact same object

    def test_asymmetric_weights(self):
        """Asymmetric branch weights rescale correctly."""
        dist_then = _make_matrix_dist('X', np.eye(2), scalar_mu=1.0)
        dist_else = _make_matrix_dist('X', 2 * np.eye(2), scalar_mu=-1.0)
        p_then, p_else = 0.7, 0.3
        current_p, merged = merge([(p_then, dist_then), (p_else, dist_else)])
        assert abs(current_p - 1.0) < 1e-10
        blk = merged.gm_block
        assert abs(blk.pi[0] - 0.7) < 1e-10
        assert abs(blk.pi[1] - 0.3) < 1e-10

    def test_mixture_mean_correct(self):
        """Mixture mean = sum_k pi_k * M_k matches weighted average."""
        M_then = np.array([[1., 0.], [0., 1.]])
        M_else = np.array([[3., 0.], [0., 3.]])
        dist_then = _make_matrix_dist('X', M_then)
        dist_else = _make_matrix_dist('X', M_else)
        _, merged = merge([(0.5, dist_then), (0.5, dist_else)])
        blk = merged.gm_block
        mean = blk.matrix_mean('X')
        expected = 0.5 * M_then + 0.5 * M_else  # = [[2,0],[0,2]]
        assert np.allclose(mean, expected)

    def test_scalar_gm_also_merged(self):
        """Scalar gm is also merged consistently with gm_block."""
        dist_then = _make_matrix_dist('X', np.eye(2), scalar_mu=1.0)
        dist_else = _make_matrix_dist('X', np.eye(2), scalar_mu=-1.0)
        _, merged = merge([(0.5, dist_then), (0.5, dist_else)])
        # scalar gm should have 2 components
        assert merged.gm.n_comp() == 2
        # aligned with gm_block
        assert merged.gm.n_comp() == merged.gm_block.n_comp()

    def test_kron_factors_preserved(self):
        """Kronecker factors (U, V) are preserved per-component after merge."""
        U1 = 2.0 * np.eye(2)
        V1 = 3.0 * np.eye(2)
        U2 = 4.0 * np.eye(2)
        V2 = 5.0 * np.eye(2)
        dist_then = _make_matrix_dist('X', np.eye(2), U1, V1)
        dist_else = _make_matrix_dist('X', np.eye(2), U2, V2)
        _, merged = merge([(0.5, dist_then), (0.5, dist_else)])
        blk = merged.gm_block
        U_out0, V_out0 = blk.get_cov(0, 'X', 'X')
        U_out1, V_out1 = blk.get_cov(1, 'X', 'X')
        assert np.allclose(U_out0, U1)
        assert np.allclose(V_out0, V1)
        assert np.allclose(U_out1, U2)
        assert np.allclose(V_out1, V2)


# ---------------------------------------------------------------------------
# fix2.3 — ranking_prune with gm_block
# ---------------------------------------------------------------------------

class TestRankingPruneWithGmBlock:
    """fix2.3: ranking_prune drops both scalar gm and gm_block components."""

    def _make_merged_dist(self, n_comp: int = 4) -> Dist:
        """Build a Dist with n_comp components for pruning tests."""
        ve = VarEntry('X', 'matrix', (2, 2))
        pi = [1.0 / n_comp] * n_comp
        mu_blocks = [{'X': float(k) * np.eye(2)} for k in range(n_comp)]
        cov_blocks = [{frozenset({'X'}): (np.eye(2), np.eye(2))} for _ in range(n_comp)]
        block = GaussianMixBlock(
            var_list=['s'], var_entries=[ve], pi=list(pi),
            mu_blocks=mu_blocks, cov_blocks=cov_blocks
        )
        gm_mu = [np.array([float(k)]) for k in range(n_comp)]
        gm_sigma = [np.array([[1.0]]) for _ in range(n_comp)]
        gm = GaussianMix(list(pi), gm_mu, gm_sigma)
        return Dist(['s'], gm, var_entries=[ve], gm_block=block)

    def test_ranking_prune_reduces_components(self):
        dist = self._make_merged_dist(4)
        dist_pruned = ranking_prune(dist, Kmax=2)
        assert dist_pruned.gm.n_comp() == 2
        assert dist_pruned.gm_block.n_comp() == 2

    def test_ranking_prune_sync_gm_and_block(self):
        """gm and gm_block have same number of components after prune."""
        dist = self._make_merged_dist(6)
        dist_pruned = ranking_prune(dist, Kmax=3)
        assert dist_pruned.gm.n_comp() == dist_pruned.gm_block.n_comp()

    def test_ranking_prune_weights_renormalised(self):
        """After prune, pi sums to 1."""
        dist = self._make_merged_dist(5)
        dist_pruned = ranking_prune(dist, Kmax=2)
        assert abs(sum(dist_pruned.gm.pi) - 1.0) < 1e-10
        assert abs(sum(dist_pruned.gm_block.pi) - 1.0) < 1e-10

    def test_ranking_prune_no_op_when_already_small(self):
        """No pruning done if n_comp <= Kmax."""
        dist = self._make_merged_dist(3)
        dist_pruned = ranking_prune(dist, Kmax=5)
        assert dist_pruned.gm.n_comp() == 3
        assert dist_pruned.gm_block.n_comp() == 3

    def test_ranking_prune_no_gm_block(self):
        """Scalar-only dist: gm_block=None, ranking_prune still works."""
        pi = [0.1, 0.5, 0.3, 0.1]
        mu = [np.array([float(k)]) for k in range(4)]
        sigma = [np.array([[1.0]]) for _ in range(4)]
        gm = GaussianMix(pi, mu, sigma)
        dist = Dist(['s'], gm)
        dist_pruned = ranking_prune(dist, Kmax=2)
        assert dist_pruned.gm.n_comp() == 2


# ---------------------------------------------------------------------------
# fix2.2 — Variable present in only one branch
# ---------------------------------------------------------------------------

class TestMergeVariableScope:
    """fix2.2: variable missing in one branch gets zero-placeholder."""

    def test_missing_var_gets_zero_placeholder(self):
        """Branch without matrix var gets injected with zero mean/cov."""
        ve = VarEntry('X', 'matrix', (2, 2))
        M = np.array([[1., 0.], [0., 1.]])
        U = np.eye(2); V = np.eye(2)
        # Branch with matrix var
        block_with = GaussianMixBlock.from_matrix_gm(
            [ve], [M], [U], [V], pi=[1.0], var_list=['s']
        )
        gm_with = GaussianMix([1.0], [np.array([1.0])], [np.array([[1.0]])])
        dist_with = Dist(['s'], gm_with, var_entries=[ve], gm_block=block_with)

        # Branch without matrix var (gm only)
        gm_without = GaussianMix([1.0], [np.array([-1.0])], [np.array([[1.0]])])
        # Build a block with a zero entry for X to satisfy merge
        block_without = GaussianMixBlock.from_matrix_gm(
            [ve], [np.zeros((2, 2))], [np.zeros((2, 2))], [np.zeros((2, 2))],
            pi=[1.0], var_list=['s']
        )
        dist_without = Dist(['s'], gm_without, var_entries=[ve], gm_block=block_without)

        _, merged = merge([(0.5, dist_with), (0.5, dist_without)])
        assert merged.gm_block is not None
        assert merged.gm_block.n_comp() == 2


# ---------------------------------------------------------------------------
# End-to-end: t10_if_else_matrix via subprocess
# ---------------------------------------------------------------------------

def test_t10_if_else_matrix_end_to_end():
    """E2E: if theta > 0.5 { y=X[0,0]; } else { y=X[1,1]; } — E[y] ≈ 2.5."""
    import subprocess, tempfile
    soga_prog = """\
matrix[2][2] X;
X = matrix_gm([[1,2],[3,4]], [[1,0],[0,1]], [[1,0],[0,1]]);
theta = uniform([0,1], 2);
if theta > 0.5 {
    y = X[0,0];
} else {
    y = X[1,1];
} end if;
"""
    with tempfile.NamedTemporaryFile(suffix='.soga', mode='w', delete=False) as f:
        f.write(soga_prog)
        fname = f.name
    try:
        root = os.path.join(os.path.dirname(__file__), '..')
        result = subprocess.run(
            [os.path.join(root, '.venv/bin/python3'), 'src/SOGA.py', '-f', fname],
            capture_output=True, text=True, cwd=root, timeout=30
        )
        assert result.returncode == 0, f"SOGA crashed:\n{result.stderr[-500:]}"
        # E[y] ≈ 0.5*M[0,0] + 0.5*M[1,1] = 0.5*1 + 0.5*4 = 2.5 (±0.1 for GM approx)
        import re
        m = re.search(r'E\[y\]:\s*([\d.]+)', result.stdout)
        assert m is not None, f"E[y] not found in output:\n{result.stdout}"
        ey = float(m.group(1))
        assert abs(ey - 2.5) < 0.1, f"E[y]={ey} not close to 2.5"
    finally:
        os.unlink(fname)
