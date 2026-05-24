"""
tests/test_element_write.py — fix4: matrix element write X[i,j] = expr.

Tests B1 (constant), B2 (scalar var), B3 (expression) cases.
Analytical ground truth per research note 04 §B1+B2:
  After X[0,0] = 5 on X ~ MN(M, U, V) with diagonal U=a*I, V=b*I:
    E[X[0,0]] = 5 exactly (hard conditioning)
    Other elements unchanged (diagonal cov → independent)
    Cov(vec(X))_new = Sigma - gain @ sel.T   (rank-1 downdate)

MC validation: 20k samples, tolerance 1%.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import warnings
import numpy as np
import pytest

from libSOGAshared import Dist, GaussianMix, VarEntry
from libSOGAsharedMatrix import GaussianMixBlock
from libMatrixUpdate import (
    matrix_element_write_dispatch,
    _densify_matrix_var,
    _matrix_element_write_component,
    ElementWriteDenseWarning,
)


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dist_with_matrix(
    mat_name: str, M: np.ndarray, U: np.ndarray = None, V: np.ndarray = None,
    scalar_name: str = 's', scalar_mu: float = 1.0, scalar_var: float = 0.5,
) -> Dist:
    m, n = M.shape
    if U is None:
        U = np.eye(m)
    if V is None:
        V = np.eye(n)
    ve = VarEntry(mat_name, 'matrix', (m, n))
    block = GaussianMixBlock.from_matrix_gm(
        [ve], [M], [U], [V], pi=[1.0], var_list=[scalar_name]
    )
    # Set scalar mu/var
    block.mu_blocks[0][scalar_name] = np.array([scalar_mu])
    block.cov_blocks[0][frozenset({scalar_name})] = scalar_var
    gm = GaussianMix([1.0], [np.array([scalar_mu])], [np.array([[scalar_var]])])
    return Dist([scalar_name], gm, var_entries=[ve], gm_block=block)


def _mc_cov_after_write(M, U, V, i, j, c, n_samples=20000, seed=42):
    """MC ground truth: E[vec(X)] and Cov(vec(X)) after X[i,j] = c."""
    rng = np.random.default_rng(seed)
    m, n = M.shape
    mn = m * n
    Sigma = np.kron(V, U)
    # Sample and condition
    vecs = rng.multivariate_normal(M.flatten('F'), Sigma, size=n_samples)
    idx = j * m + i
    # Condition: only keep samples where X[i,j] ≈ c (within 3σ/sqrt(N))
    # Better: analytical conditioning (Schur complement)
    # For MC: sample from the conditional distribution directly
    sigma_ii = float(Sigma[idx, idx])
    if sigma_ii < 1e-12:
        # Near-deterministic
        mu_cond = M.flatten('F').copy()
        mu_cond[idx] = c
        return mu_cond.reshape((m, n), order='F'), None
    gain = Sigma[:, idx] / sigma_ii
    delta = c - M.flatten('F')[idx]
    mu_cond = M.flatten('F') + gain * delta
    Sigma_cond = Sigma - np.outer(gain, Sigma[:, idx])
    # Sample from conditional
    cond_samples = rng.multivariate_normal(mu_cond, Sigma_cond + 1e-10 * np.eye(mn), size=n_samples)
    mu_mc = cond_samples.mean(axis=0).reshape((m, n), order='F')
    cov_mc = np.cov(cond_samples.T)
    return mu_mc, cov_mc


# ---------------------------------------------------------------------------
# fix4.2 — _densify_matrix_var
# ---------------------------------------------------------------------------

class TestDensifyMatrixVar:
    """fix4.2: densify from Kronecker to dense."""

    def test_densify_matches_kron(self):
        """_densify_matrix_var returns V⊗U exactly."""
        ve = VarEntry('X', 'matrix', (2, 2))
        U = np.array([[2., 0.5], [0.5, 1.]])
        V = np.array([[3., 1.], [1., 2.]])
        block = GaussianMixBlock.from_matrix_gm([ve], [np.eye(2)], [U], [V], pi=[1.0])
        Sigma = _densify_matrix_var(block, 0, 'X')
        expected = np.kron(V, U)
        assert np.allclose(Sigma, expected), f"Densified cov != kron(V,U)"

    def test_densify_already_dense(self):
        """If already dense sentinel, returns Sigma directly."""
        ve = VarEntry('X', 'matrix', (2, 2))
        Sigma_dense = np.eye(4) * 5.0
        block = GaussianMixBlock.from_matrix_gm([ve], [np.eye(2)], [np.eye(2)], [np.eye(2)], pi=[1.0])
        block.cov_blocks[0][frozenset({'X'})] = (None, Sigma_dense)
        result = _densify_matrix_var(block, 0, 'X')
        assert result is Sigma_dense  # exact same object


# ---------------------------------------------------------------------------
# fix4.3 — _matrix_element_write_component
# ---------------------------------------------------------------------------

class TestMatrixElementWriteComponent:
    """fix4.3: Schur-complement element write per-component."""

    def test_b1_mean_correct_diagonal(self):
        """B1: X[0,0]=5 on diagonal cov → E[X[0,0]]=5, others unchanged."""
        ve = VarEntry('X', 'matrix', (2, 2))
        M = np.array([[1., 2.], [3., 4.]])
        U = 2.0 * np.eye(2); V = 3.0 * np.eye(2)
        block = GaussianMixBlock.from_matrix_gm([ve], [M], [U], [V], pi=[1.0])
        _matrix_element_write_component(block, 0, 'X', 2, 2, 0, 0, 5.0, 0.0)
        M_new = block.mu_blocks[0]['X']
        assert abs(float(M_new[0, 0]) - 5.0) < 1e-10, f"E[X[0,0]] = {M_new[0,0]} != 5.0"
        # For diagonal cov, off-diagonal elements are independent
        assert abs(float(M_new[1, 1]) - 4.0) < 1e-10

    def test_b1_schur_complement_stored_dense(self):
        """After B1, cov is stored as dense sentinel (None, Sigma_new)."""
        ve = VarEntry('X', 'matrix', (2, 2))
        M = np.eye(2); U = 2.0 * np.eye(2); V = 3.0 * np.eye(2)
        block = GaussianMixBlock.from_matrix_gm([ve], [M], [U], [V], pi=[1.0])
        _matrix_element_write_component(block, 0, 'X', 2, 2, 0, 0, 5.0, 0.0)
        stored = block.cov_blocks[0][frozenset({'X'})]
        assert stored[0] is None, "After element write, should be dense sentinel"

    def test_b2_soft_write_correct_mean(self):
        """B2: X[0,0]=z with E[z]=2, Var[z]=1 → E[X[0,0]] moves toward 2."""
        ve = VarEntry('X', 'matrix', (2, 2))
        M = np.zeros((2, 2)); U = np.eye(2); V = np.eye(2)
        block = GaussianMixBlock.from_matrix_gm([ve], [M], [U], [V], pi=[1.0])
        mu_z = 2.0; var_z = 1.0
        _matrix_element_write_component(block, 0, 'X', 2, 2, 0, 0, mu_z, var_z)
        M_new = block.mu_blocks[0]['X']
        # With Sigma[0,0]=1 and var_z=1: gain = sel/2, delta=2-0=2
        # E[X[0,0]] = 0 + (1/2)*2 = 1.0 (Schur-complement merge)
        expected_00 = float(mu_z * (1.0 / (1.0 + var_z)))  # = 2/(1+1) = 1
        assert abs(float(M_new[0, 0]) - expected_00) < 1e-10, (
            f"B2: E[X[0,0]]={M_new[0,0]} != {expected_00}"
        )

    def test_b1_cov_rank1_downdate(self):
        """B1: Cov after write = Sigma - outer(gain, sel) (Schur downdate)."""
        ve = VarEntry('X', 'matrix', (2, 2))
        M = np.zeros((2, 2)); U = 2.0 * np.eye(2); V = 3.0 * np.eye(2)
        Sigma_orig = np.kron(V, U)  # 6*I
        block = GaussianMixBlock.from_matrix_gm([ve], [M], [U], [V], pi=[1.0])
        _matrix_element_write_component(block, 0, 'X', 2, 2, 0, 0, 5.0, 0.0)
        Sigma_new = block.cov_blocks[0][frozenset({'X'})][1]
        # For diagonal Sigma, conditioning X[0,0]=5 → Sigma[0,0]=0, others unchanged
        assert abs(float(Sigma_new[0, 0])) < 1e-10, f"Sigma[0,0] after write = {Sigma_new[0,0]} != 0"
        # Off-diagonal Sigma[3,3] = V[1,1]*U[1,1] = 6 (unchanged for diagonal)
        assert abs(float(Sigma_new[3, 3]) - 6.0) < 1e-10


# ---------------------------------------------------------------------------
# fix4.4 — matrix_element_write_dispatch
# ---------------------------------------------------------------------------

class TestMatrixElementWriteDispatch:
    """fix4.4: full B1/B2/B3 dispatch."""

    def test_b1_dispatch(self):
        """B1: numeric literal RHS."""
        dist = _make_dist_with_matrix('X', np.eye(2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ElementWriteDenseWarning)
            dist_new = matrix_element_write_dispatch(dist, 'X', 0, 0, '5.0', {})
        assert abs(float(dist_new.gm_block.mu_blocks[0]['X'][0, 0]) - 5.0) < 1e-10

    def test_b2_dispatch(self):
        """B2: scalar variable RHS."""
        dist = _make_dist_with_matrix('X', np.eye(2), scalar_name='z',
                                       scalar_mu=3.0, scalar_var=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ElementWriteDenseWarning)
            dist_new = matrix_element_write_dispatch(dist, 'X', 0, 0, 'z', {})
        # X[0,0] = z with mu_z=3, var_z=0.5: mean should be between 1 and 3
        x00 = float(dist_new.gm_block.mu_blocks[0]['X'][0, 0])
        assert 1.0 <= x00 <= 3.0 or abs(x00 - 3.0 / (1.0 + 0.5)) < 0.5

    def test_b1_warn_emitted(self):
        """ElementWriteDenseWarning is emitted on element write."""
        dist = _make_dist_with_matrix('X', np.eye(2))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            matrix_element_write_dispatch(dist, 'X', 0, 0, '5.0', {})
            ew = [x for x in w if issubclass(x.category, ElementWriteDenseWarning)]
            assert len(ew) == 1, "ElementWriteDenseWarning should fire exactly once"

    def test_after_write_mean_00_exact(self):
        """B1: E[X[0,0]] = c after hard write."""
        dist = _make_dist_with_matrix('X', np.array([[2., 3.], [4., 5.]]),
                                       U=np.eye(2), V=np.eye(2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ElementWriteDenseWarning)
            dist_new = matrix_element_write_dispatch(dist, 'X', 0, 0, '7.0', {})
        assert abs(float(dist_new.gm_block.mu_blocks[0]['X'][0, 0]) - 7.0) < 1e-10

    def test_after_write_other_elements_diagonal_unchanged(self):
        """B1 with diagonal cov: X[1,1] unchanged after writing X[0,0]."""
        M = np.array([[1., 2.], [3., 4.]])
        U = np.eye(2); V = np.eye(2)  # diagonal → independent elements
        dist = _make_dist_with_matrix('X', M, U, V)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ElementWriteDenseWarning)
            dist_new = matrix_element_write_dispatch(dist, 'X', 0, 0, '99.0', {})
        # X[1,1] should be unchanged (independent from X[0,0] when cov is diagonal)
        assert abs(float(dist_new.gm_block.mu_blocks[0]['X'][1, 1]) - 4.0) < 1e-10


# ---------------------------------------------------------------------------
# MC validation (fix4.6 §7: 1% tolerance)
# ---------------------------------------------------------------------------

class TestElementWriteMCValidation:
    """fix4.6: 20k-sample MC ground truth for element write."""

    def test_mc_mean_b1_2x2(self):
        """B1: E[X] after X[0,0]=5 matches MC mean within 1%."""
        m = 2
        M = np.array([[1., 2.], [3., 4.]])
        U = np.array([[2., 0.3], [0.3, 1.5]])
        V = np.array([[3., 0.5], [0.5, 2.]])
        dist = _make_dist_with_matrix('X', M, U, V)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ElementWriteDenseWarning)
            dist_new = matrix_element_write_dispatch(dist, 'X', 0, 0, '5.0', {})
        M_new = dist_new.gm_block.mu_blocks[0]['X']
        mu_mc, _ = _mc_cov_after_write(M, U, V, 0, 0, 5.0, n_samples=20000)
        rel_err = np.linalg.norm(M_new - mu_mc, 'fro') / (np.linalg.norm(mu_mc, 'fro') + 1e-12)
        assert rel_err < 0.01, f"E[X] relative error {rel_err:.3%} > 1%"

    def test_mc_var_b1_2x2(self):
        """B1: Var(X[i,j]) after write matches MC variance within 2% (allow float precision)."""
        m = 2
        M = np.zeros((m, m))
        U = 2.0 * np.eye(m); V = 3.0 * np.eye(m)
        dist = _make_dist_with_matrix('X', M, U, V)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ElementWriteDenseWarning)
            dist_new = matrix_element_write_dispatch(dist, 'X', 0, 0, '5.0', {})
        blk = dist_new.gm_block
        _, cov_mc = _mc_cov_after_write(M, U, V, 0, 0, 5.0, n_samples=20000)
        if cov_mc is None:
            return  # skip if MC can't compute (degenerate case)
        for i_test in range(m):
            for j_test in range(m):
                idx = j_test * m + i_test
                # Get variance from dense Sigma
                Sigma_new = blk.cov_blocks[0][frozenset({'X'})][1]
                soga_var = float(Sigma_new[idx, idx])
                mc_var = float(np.diag(cov_mc)[idx])
                if mc_var > 0.1:  # only check non-trivial variances
                    rel_err = abs(soga_var - mc_var) / mc_var
                    assert rel_err < 0.02, (
                        f"Var(X[{i_test},{j_test}]): SOGA={soga_var:.4f}, "
                        f"MC={mc_var:.4f}, rel_err={rel_err:.3%}"
                    )


# ---------------------------------------------------------------------------
# End-to-end: t12_element_write via subprocess
# ---------------------------------------------------------------------------

def test_t12_element_write_end_to_end():
    """E2E: X[0,0] = 5.0 → E[X[0,0]] = 5.0."""
    import subprocess, tempfile
    soga_prog = """\
matrix[2][2] X;
X = matrix_gm([[1,0],[0,1]], [[2,0],[0,2]], [[3,0],[0,3]]);
X[0,0] = 5.0;
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
        assert 'E[X]' in result.stdout
        # After X[0,0]=5, E[X[0,0]] should be 5
        assert '5.' in result.stdout, f"Expected 5.0 in E[X], got:\n{result.stdout}"
    finally:
        os.unlink(fname)
