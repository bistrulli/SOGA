"""
Tests for matrix-GM grammar extensions — M1.7 of matrix-gm-lishan plan.

Covers parsing of the three grammar additions:
- SOGA.g4: matrix_decl, matrix_gm constructor, mlist row-by-row literal
- ASGMT.g4: MATMUL '@', transp(X), X[i,j] two-arg indexing
- TRUNC.g4: X[i,j] two-arg indexing, row_sum(X, i), col_sum(X, j)

Test categories:
1. Positive parses — new constructs accepted (~20 tests)
2. Negative parses — malformed inputs rejected (~6 tests)
3. Scalar regression — existing scalar syntax unchanged (~4 tests)

Run with: .venv/bin/python -m pytest tests/test_grammar_matrix.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

# Add src/ to path so generated parsers can be imported with their bare names
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from SOGALexer import SOGALexer
from SOGAParser import SOGAParser
from ASGMTLexer import ASGMTLexer
from ASGMTParser import ASGMTParser
from TRUNCLexer import TRUNCLexer
from TRUNCParser import TRUNCParser


class _CountingErrorListener(ErrorListener):
    """ANTLR error listener that counts syntax errors instead of printing them."""

    def __init__(self):
        super().__init__()
        self.errors: list[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")


def _parse(text: str, lexer_cls, parser_cls, entry_rule: str):
    """Parse `text` using `lexer_cls`/`parser_cls`, invoke `entry_rule`, return (tree, errors)."""
    listener = _CountingErrorListener()
    lexer = lexer_cls(InputStream(text))
    lexer.removeErrorListeners()
    lexer.addErrorListener(listener)
    stream = CommonTokenStream(lexer)
    parser = parser_cls(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(listener)
    tree = getattr(parser, entry_rule)()
    return tree, listener.errors


def parse_soga(text: str):
    return _parse(text, SOGALexer, SOGAParser, "progr")


def parse_asgmt(text: str):
    return _parse(text, ASGMTLexer, ASGMTParser, "assignment")


def parse_trunc(text: str):
    return _parse(text, TRUNCLexer, TRUNCParser, "trunc")


def assert_ok(tree_errors):
    tree, errors = tree_errors
    assert errors == [], f"expected zero parse errors, got: {errors}"
    assert tree is not None


def assert_rejected(tree_errors):
    _, errors = tree_errors
    assert errors, "expected at least one parse error, got zero"


# ---------------------------------------------------------------------------
# 1. POSITIVE — SOGA.g4 matrix_decl and matrix_gm
# ---------------------------------------------------------------------------


class TestSOGAMatrixDecl:
    def test_matrix_decl_4x4(self):
        assert_ok(parse_soga("matrix[4][4] X;"))

    def test_matrix_decl_1x1(self):
        assert_ok(parse_soga("matrix[1][1] X;"))

    def test_matrix_decl_32x32(self):
        assert_ok(parse_soga("matrix[32][32] X;"))

    def test_matrix_decl_non_square(self):
        assert_ok(parse_soga("matrix[3][7] M;"))

    def test_multiple_matrix_decls(self):
        assert_ok(parse_soga("matrix[2][2] X; matrix[2][2] Y;"))


class TestSOGAMatrixGM:
    def test_matrix_gm_2x2_zero_iso(self):
        src = (
            "matrix[2][2] X;"
            " X = matrix_gm([[0,0],[0,0]], [[1,0],[0,1]], [[1,0],[0,1]]);"
        )
        assert_ok(parse_soga(src))

    def test_matrix_gm_2x3_non_square(self):
        src = (
            "matrix[2][3] X;"
            " X = matrix_gm([[1,2,3],[4,5,6]], [[1,0],[0,1]], [[1,0,0],[0,1,0],[0,0,1]]);"
        )
        assert_ok(parse_soga(src))

    def test_matrix_gm_1x1_degenerate(self):
        assert_ok(parse_soga("matrix[1][1] X; X = matrix_gm([[0]], [[1]], [[1]]);"))

    def test_matrix_gm_negative_literals(self):
        src = (
            "matrix[2][2] X;"
            " X = matrix_gm([[-1,-2],[3,4]], [[1,0],[0,1]], [[1,0],[0,1]]);"
        )
        assert_ok(parse_soga(src))


class TestSOGAMixedScalarMatrix:
    def test_data_scalar_plus_matrix_decl(self):
        src = (
            "data theta = [1.0];"
            " matrix[2][2] X;"
            " X = matrix_gm([[0,0],[0,0]], [[1,0],[0,1]], [[1,0],[0,1]]);"
        )
        assert_ok(parse_soga(src))

    def test_scalar_array_plus_matrix(self):
        src = "array[3] a; matrix[3][3] M;"
        assert_ok(parse_soga(src))


# ---------------------------------------------------------------------------
# 2. POSITIVE — ASGMT.g4 matrix operators
# ---------------------------------------------------------------------------


class TestASGMTMatrixOps:
    def test_matmul_left(self):
        assert_ok(parse_asgmt("Y = A @ X"))

    def test_matmul_right(self):
        assert_ok(parse_asgmt("Y = X @ B"))

    def test_matmul_chained(self):
        assert_ok(parse_asgmt("Y = A @ X @ B"))

    def test_matrix_add(self):
        assert_ok(parse_asgmt("Y = X + N"))

    def test_matmul_plus_add_precedence(self):
        # @ binds tighter than + per alternative-order convention
        assert_ok(parse_asgmt("Y = X @ X + N"))

    def test_transp(self):
        assert_ok(parse_asgmt("Y = transp(X)"))

    def test_transp_in_matmul(self):
        assert_ok(parse_asgmt("Y = transp(X) @ X"))

    def test_two_arg_index_to_scalar(self):
        assert_ok(parse_asgmt("y = X[1,2]"))

    def test_two_arg_index_var(self):
        assert_ok(parse_asgmt("y = X[i,j]"))


# ---------------------------------------------------------------------------
# 3. POSITIVE — TRUNC.g4 matrix observe constraints
# ---------------------------------------------------------------------------


class TestTRUNCMatrixObserve:
    def test_element_inequality(self):
        assert_ok(parse_trunc("X[1,2] > 0"))

    def test_element_with_coefficient(self):
        assert_ok(parse_trunc("2 * X[1,2] < 10"))

    def test_row_sum_inequality(self):
        assert_ok(parse_trunc("row_sum(X, 0) <= 10"))

    def test_col_sum_inequality(self):
        assert_ok(parse_trunc("col_sum(X, 3) >= -5"))

    def test_combined_linear_constraint(self):
        assert_ok(parse_trunc("2 * X[1,2] + 3 * row_sum(X, 0) < 100"))


# ---------------------------------------------------------------------------
# 4. NEGATIVE — malformed inputs must be rejected
# ---------------------------------------------------------------------------


class TestGrammarNegative:
    def test_mlist_flat_list_rejected(self):
        # mlist requires list-of-lists, a flat list must fail
        assert_rejected(parse_soga("matrix[2][2] X; X = matrix_gm([0,0], [1], [1]);"))

    def test_matrix_gm_missing_args(self):
        # matrix_gm requires exactly 3 mlist arguments
        assert_rejected(parse_soga("matrix[2][2] X; X = matrix_gm([[0,0],[0,0]]);"))

    def test_matrix_decl_missing_shape(self):
        # `matrix X;` without [m][n] must fail
        assert_rejected(parse_soga("matrix X;"))

    def test_trace_in_observe_rejected(self):
        # trace(X) deferred to v2 — must not parse as a TRUNC monom
        assert_rejected(parse_trunc("trace(X) > 0"))


# ---------------------------------------------------------------------------
# 5. SCALAR REGRESSION — existing programs must parse unchanged
# ---------------------------------------------------------------------------


_PROGRAMS_DIR = os.path.join(os.path.dirname(__file__), "..", "programs", "SOGA")


def _read_program(name: str) -> str:
    path = os.path.join(_PROGRAMS_DIR, name)
    if not os.path.isfile(path):
        pytest.skip(f"benchmark file not present: {name}")
    with open(path) as f:
        return f.read()


class TestScalarRegression:
    """Verify M1 grammar changes do not break parsing of canonical scalar programs."""

    # Note: scalar regression at the SOGA-parser level requires programs that
    # do NOT use preprocessor primitives (beta, laplace, exprnd, bern). Those
    # primitives are expanded into GM by sogaPreprocessor BEFORE the parser
    # sees them, so a parser-only test must avoid them. Programs like
    # CoinBias.soga (uses beta(...)) are validated end-to-end by /audit-grammar
    # (T13), not by this unit test.

    def test_bernoulli_soga_parses(self):
        text = _read_program("Bernoulli.soga")
        assert_ok(parse_soga(text))

    def test_burglar_soga_parses(self):
        text = _read_program("Burglar.soga")
        assert_ok(parse_soga(text))

    def test_bayes_point_machine_soga_parses(self):
        text = _read_program("BayesPointMachine.soga")
        assert_ok(parse_soga(text))

    def test_inline_scalar_gm_parses(self):
        # Inline smoke for scalar gm constructor
        assert_ok(
            parse_soga("x = gm([1.0], [0.5], [1.0]); observe(x > 0);")
        )

    def test_inline_scalar_observe_inequality(self):
        assert_ok(parse_trunc("2 * x + 3 * y < 10"))

    def test_inline_scalar_observe_equality(self):
        assert_ok(parse_trunc("x == 5"))
