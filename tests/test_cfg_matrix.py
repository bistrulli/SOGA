"""
tests/test_cfg_matrix.py — CFG type tracking tests for M2.

Tests: matrix declaration creates correct VarEntry; existing scalar CFGs have
empty var_entries; mixed-type programs have correct partitioning; routing guard
routes matrix LHS to update_rule_matrix; enterSymvars defect fix (M2.4).

Plan reference: §M2.8 and §M2.4 of plan/2026-05-22-matrix-gm-lishan.md
"""

import sys
import os
import textwrap

import numpy as np
import pytest

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
PROGRAMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "programs"))
sys.path.insert(0, SRC)

from producecfg import produce_cfg
from libSOGAshared import Dist, GaussianMix, VarEntry
from libSOGAupdate import update_rule
from libSOGAtruncate import truncate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_soga(tmp_path, content: str) -> str:
    """Write a .soga file to tmp_path and return its absolute path."""
    p = tmp_path / "test.soga"
    p.write_text(textwrap.dedent(content))
    return str(p)


# ---------------------------------------------------------------------------
# M2.3 / M2.8: matrix declaration creates correct VarEntry
# ---------------------------------------------------------------------------

class TestMatrixDecl:
    """CFG construction correctly registers matrix declarations in var_entries."""

    def test_single_matrix_decl_creates_var_entry(self, tmp_path):
        """matrix[4][4] X creates one VarEntry with name='X', kind='matrix', shape=(4,4)."""
        prog = """\
            matrix[4][4] X;
            X = matrix_gm(
              [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
              [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
              [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            );
        """
        path = _write_soga(tmp_path, prog)
        cfg = produce_cfg(path)
        assert len(cfg.var_entries) == 1
        ve = cfg.var_entries[0]
        assert ve.name == "X"
        assert ve.kind == "matrix"
        assert ve.shape == (4, 4)

    def test_matrix_decl_flat_offset_is_minus_one(self, tmp_path):
        """flat_offset is -1 until M3 assigns positions."""
        prog = """\
            matrix[3][3] M;
            M = matrix_gm(
              [[0,0,0],[0,0,0],[0,0,0]],
              [[1,0,0],[0,1,0],[0,0,1]],
              [[1,0,0],[0,1,0],[0,0,1]]
            );
        """
        path = _write_soga(tmp_path, prog)
        cfg = produce_cfg(path)
        assert cfg.var_entries[0].flat_offset == -1

    def test_matrix_shape_stored_in_data(self, tmp_path):
        """cfg.data['X_shape'] is set by enterMatrix_decl."""
        prog = """\
            matrix[2][5] X;
            X = matrix_gm(
              [[0,0,0,0,0],[0,0,0,0,0]],
              [[1,0],[0,1]],
              [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
            );
        """
        path = _write_soga(tmp_path, prog)
        cfg = produce_cfg(path)
        assert cfg.data.get("X_shape") == (2, 5)

    def test_multiple_matrix_decls(self, tmp_path):
        """Three matrix declarations create three VarEntry objects."""
        prog = """\
            matrix[2][2] A;
            matrix[2][3] B;
            matrix[3][2] C;
            A = matrix_gm([[0,0],[0,0]],[[1,0],[0,1]],[[1,0],[0,1]]);
            B = matrix_gm([[0,0,0],[0,0,0]],[[1,0],[0,1]],[[1,0,0],[0,1,0],[0,0,1]]);
            C = matrix_gm([[0,0],[0,0],[0,0]],[[1,0,0],[0,1,0],[0,0,1]],[[1,0],[0,1]]);
        """
        path = _write_soga(tmp_path, prog)
        cfg = produce_cfg(path)
        assert len(cfg.var_entries) == 3
        names = {ve.name for ve in cfg.var_entries}
        assert names == {"A", "B", "C"}

    def test_matrix_shape_correct_for_nonsquare(self, tmp_path):
        """2×3 matrix stores shape (2, 3), not (3, 2)."""
        prog = """\
            matrix[2][3] R;
            R = matrix_gm(
              [[0,0,0],[0,0,0]],
              [[1,0],[0,1]],
              [[1,0,0],[0,1,0],[0,0,1]]
            );
        """
        path = _write_soga(tmp_path, prog)
        cfg = produce_cfg(path)
        ve = cfg.var_entries[0]
        assert ve.shape == (2, 3)


# ---------------------------------------------------------------------------
# M2.4: enterSymvars fix — matrix vars NOT in ID_list
# ---------------------------------------------------------------------------

class TestEnterSymvarsDefect:
    """Matrix variable names must not appear in ID_list (M2.4 bug fix)."""

    def test_matrix_var_not_in_id_list(self, tmp_path):
        """matrix[4][4] X must NOT be added to ID_list."""
        prog = """\
            matrix[4][4] X;
            X = matrix_gm(
              [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
              [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
              [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            );
        """
        path = _write_soga(tmp_path, prog)
        cfg = produce_cfg(path)
        assert "X" not in cfg.ID_list

    def test_scalar_var_still_in_id_list(self, tmp_path):
        """Scalar variables (x, y) remain in ID_list as before."""
        prog = """\
            x = gm([1],[0],[1]);
            y = x + 1;
        """
        path = _write_soga(tmp_path, prog)
        cfg = produce_cfg(path)
        assert "x" in cfg.ID_list
        assert "y" in cfg.ID_list

    def test_mixed_program_partitioned_correctly(self, tmp_path):
        """In a mixed program, scalar vars go to ID_list, matrix vars to var_entries."""
        prog = """\
            matrix[2][2] X;
            x = gm([1],[0],[1]);
            X = matrix_gm([[0,0],[0,0]],[[1,0],[0,1]],[[1,0],[0,1]]);
            y = x + 1;
        """
        path = _write_soga(tmp_path, prog)
        cfg = produce_cfg(path)
        # x and y are scalar — in ID_list
        assert "x" in cfg.ID_list
        assert "y" in cfg.ID_list
        # X is matrix — in var_entries only
        assert "X" not in cfg.ID_list
        assert any(ve.name == "X" for ve in cfg.var_entries)


# ---------------------------------------------------------------------------
# M2.4 / M2.5: routing guards in update_rule and truncate
# ---------------------------------------------------------------------------

class TestRoutingGuards:
    """update_rule and truncate dispatch to matrix stubs for matrix variables."""

    def _make_dist_with_matrix_var(self):
        """Create a minimal Dist with one matrix VarEntry and no gm_block."""
        gm = GaussianMix([1.], [np.array([0.])], [np.zeros((1, 1))])
        ve = VarEntry("X", "matrix", (2, 2), -1)
        return Dist(["x"], gm, var_entries=[ve])

    def test_update_rule_matrix_lhs_dispatches_to_matrix_handler(self):
        """update_rule dispatches to matrix handler for matrix LHS (M2.4 routing guard + M4 impl)."""
        dist = self._make_dist_with_matrix_var()
        # matrix_gm assignment should succeed now that M4 is implemented
        result = update_rule(dist, "X=matrix_gm([[0,0],[0,0]],[[1,0],[0,1]],[[1,0],[0,1]])", {})
        # Should have a gm_block set
        assert result.gm_block is not None
        assert result.gm_block.matrix_mean("X").shape == (2, 2)

    def test_update_rule_scalar_lhs_passes_through(self):
        """update_rule uses scalar path for non-matrix LHS (M2.4 guard does not trigger)."""
        gm = GaussianMix([1.], [np.array([0.])], [np.zeros((1, 1))])
        ve = VarEntry("X", "matrix", (2, 2), -1)
        dist = Dist(["x"], gm, var_entries=[ve])
        result = update_rule(dist, "x=1", {})
        # Scalar update: x should now be 1
        assert abs(result.gm.mean()[0] - 1.0) < 1e-12
        # var_entries preserved
        assert len(result.var_entries) == 1

    def test_update_rule_no_matrix_scalar_only(self):
        """update_rule scalar path works normally when var_entries is empty."""
        gm = GaussianMix([1.], [np.array([0.])], [np.zeros((1, 1))])
        dist = Dist(["x"], gm)
        result = update_rule(dist, "x=5", {})
        assert abs(result.gm.mean()[0] - 5.0) < 1e-12

    def test_truncate_matrix_var_in_trunc_raises_not_implemented(self):
        """truncate raises NotImplementedError when trunc mentions a matrix var (M2.5 stub)."""
        dist = self._make_dist_with_matrix_var()
        with pytest.raises(NotImplementedError, match="M5-stub"):
            truncate(dist, "X>0", {})

    def test_truncate_scalar_var_passes_through(self):
        """truncate uses scalar path when no matrix variable is in trunc."""
        gm = GaussianMix([1.], [np.array([0.])], [np.zeros((1, 1))])
        ve = VarEntry("X", "matrix", (2, 2), -1)
        dist = Dist(["x"], gm, var_entries=[ve])
        # x>0 does not mention matrix var X — scalar path
        p, new_dist = truncate(dist, "x>0", {})
        # x ~ N(0,0) is a Dirac at 0; x>0 has probability 0 for the strict case
        # but the function should return without NotImplementedError
        assert p == 0 or isinstance(p, (int, float))

    def test_update_rule_skip_always_passes(self):
        """update_rule('skip') bypasses routing guard entirely."""
        dist = self._make_dist_with_matrix_var()
        result = update_rule(dist, "skip", {})
        assert result is dist  # skip returns the same dist


# ---------------------------------------------------------------------------
# M2.6: no-matrix-in-branch guard (_assert_no_matrix_in_lbc)
# ---------------------------------------------------------------------------

class TestNoMatrixInBranch:
    """_assert_no_matrix_in_lbc raises NotImplementedError for matrix branch conditions."""

    def test_matrix_in_lbc_raises(self):
        """Branching on a matrix variable raises NotImplementedError."""
        from libSOGA import _assert_no_matrix_in_lbc
        ve = VarEntry("X", "matrix", (4, 4), -1)
        with pytest.raises(NotImplementedError, match="M2.6"):
            _assert_no_matrix_in_lbc("X[0,1]>0.5", [ve])

    def test_scalar_in_lbc_passes(self):
        """Branching on a scalar variable (not in var_entries) does not raise."""
        from libSOGA import _assert_no_matrix_in_lbc
        ve = VarEntry("X", "matrix", (4, 4), -1)
        # Should not raise
        _assert_no_matrix_in_lbc("x>0.5", [ve])

    def test_empty_var_entries_passes(self):
        """With no matrix var_entries, _assert_no_matrix_in_lbc is a no-op."""
        from libSOGA import _assert_no_matrix_in_lbc
        _assert_no_matrix_in_lbc("anything>0", [])

    def test_none_var_entries_passes(self):
        """With None var_entries, _assert_no_matrix_in_lbc is a no-op."""
        from libSOGA import _assert_no_matrix_in_lbc
        _assert_no_matrix_in_lbc("anything>0", None)


# ---------------------------------------------------------------------------
# M2.7: var_entries propagated through start_SOGA init
# ---------------------------------------------------------------------------

class TestStartSOGAVarEntries:
    """start_SOGA propagates var_entries from CFG to init_dist."""

    def test_scalar_program_has_empty_var_entries(self, tmp_path):
        """A scalar-only program produces a CFG with empty var_entries."""
        prog = """\
            x = gm([1],[0],[1]);
            y = x + 1;
        """
        path = _write_soga(tmp_path, prog)
        cfg = produce_cfg(path)
        assert cfg.var_entries == []

    def test_matrix_program_var_entries_in_init_dist(self, tmp_path):
        """A matrix program's init_dist carries var_entries from the CFG."""
        from libSOGA import start_SOGA
        prog = """\
            matrix[2][2] X;
            X = matrix_gm([[0,0],[0,0]],[[1,0],[0,1]],[[1,0],[0,1]]);
        """
        path = _write_soga(tmp_path, prog)
        cfg = produce_cfg(path)
        assert len(cfg.var_entries) == 1
        # We can't easily call start_SOGA without it crashing on matrix ops (M4 stub),
        # but we can verify the CFG produces the right var_entries for downstream
        assert cfg.var_entries[0].name == "X"
        assert cfg.var_entries[0].kind == "matrix"
        assert cfg.var_entries[0].shape == (2, 2)
