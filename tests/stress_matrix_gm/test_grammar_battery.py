"""
test_grammar_battery.py — 50-test grammar stress battery for matrix-GM DSL.

Categories:
    A. Positive (32 tests across 7 groups A–G)
       Group A: matrix_decl variants (5)
       Group B: matrix_gm constructor (6)
       Group C: matrix_gm_full constructor (3)
       Group D: ASGMT matrix operators (10)
       Group E: TRUNC observe constraints (4)
       Group F: mixed scalar+matrix programs (2)
       Group G: keyword/IDV regression (2)

    B. Negative (8 tests N1–N8): genuine parse-time errors
       Every negative is annotated with the ANTLR production/token that rejects it.

    C. Edge cases (10 tests E1–E10): empirically determined outcomes

Total: 50 test nodes (parametrized).

Run:
    pytest tests/stress_matrix_gm/test_grammar_battery.py -v
    pytest tests/stress_matrix_gm/test_grammar_battery.py -k "positive" -v   -> 32
    pytest tests/stress_matrix_gm/test_grammar_battery.py -k "negative" -v   ->  8
    pytest tests/stress_matrix_gm/test_grammar_battery.py -k "edge" -v       -> 10
"""

from __future__ import annotations

import sys
import os

import pytest

# ---------------------------------------------------------------------------
# Path setup and imports (use conftest _CountingErrorListener via the package)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from antlr4 import CommonTokenStream, InputStream

from SOGALexer import SOGALexer
from SOGAParser import SOGAParser
from ASGMTLexer import ASGMTLexer
from ASGMTParser import ASGMTParser
from TRUNCLexer import TRUNCLexer
from TRUNCParser import TRUNCParser

from .conftest import _CountingErrorListener


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _parse_soga(text: str) -> list[str]:
    """Parse text as a SOGA progr. Return list of error strings."""
    listener = _CountingErrorListener()
    lexer = SOGALexer(InputStream(text))
    lexer.removeErrorListeners()
    lexer.addErrorListener(listener)
    stream = CommonTokenStream(lexer)
    parser = SOGAParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(listener)
    parser.progr()
    return listener.errors


def _parse_asgmt(text: str) -> list[str]:
    """Parse text as an ASGMT assignment. Return list of error strings."""
    listener = _CountingErrorListener()
    lexer = ASGMTLexer(InputStream(text))
    lexer.removeErrorListeners()
    lexer.addErrorListener(listener)
    stream = CommonTokenStream(lexer)
    parser = ASGMTParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(listener)
    parser.assignment()
    return listener.errors


def _parse_trunc(text: str) -> list[str]:
    """Parse text as a TRUNC trunc. Return list of error strings."""
    listener = _CountingErrorListener()
    lexer = TRUNCLexer(InputStream(text))
    lexer.removeErrorListeners()
    lexer.addErrorListener(listener)
    stream = CommonTokenStream(lexer)
    parser = TRUNCParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(listener)
    parser.trunc()
    return listener.errors


# ---------------------------------------------------------------------------
# Group A — matrix_decl variants (5 positive)
# ---------------------------------------------------------------------------

_POSITIVE_GROUP_A = [
    ("A1", "soga", "matrix[2][2] X;", "2x2 square matrix declaration"),
    ("A2", "soga", "matrix[1][1] S;", "1x1 degenerate matrix declaration"),
    ("A3", "soga", "matrix[3][2] R;", "3x2 rectangular matrix declaration"),
    ("A4", "soga", "matrix[2][3] R2;", "2x3 rectangular matrix declaration"),
    ("A5", "soga", "matrix[8][8] L; matrix[2][2] S;", "multiple matrix declarations"),
]

# ---------------------------------------------------------------------------
# Group B — matrix_gm constructor (6 positive)
# ---------------------------------------------------------------------------

_POSITIVE_GROUP_B = [
    (
        "B1",
        "soga",
        "matrix[2][2] X; X = matrix_gm([[1,2],[3,4]], [[1,0],[0,1]], [[1,0],[0,1]]);",
        "2x2 matrix_gm iso covariance",
    ),
    (
        "B2",
        "soga",
        "matrix[1][1] X; X = matrix_gm([[0]], [[1]], [[1]]);",
        "1x1 degenerate matrix_gm",
    ),
    (
        "B3",
        "soga",
        "matrix[2][3] X; X = matrix_gm([[1,2,3],[4,5,6]], [[1,0],[0,1]], [[1,0,0],[0,1,0],[0,0,1]]);",
        "2x3 non-square matrix_gm",
    ),
    (
        "B4",
        "soga",
        "matrix[3][2] X; X = matrix_gm([[1,2],[3,4],[5,6]], [[1,0,0],[0,1,0],[0,0,1]], [[1,0],[0,1]]);",
        "3x2 non-square matrix_gm",
    ),
    (
        "B5",
        "soga",
        (
            "matrix[2][2] X;"
            " X = matrix_gm([[-1,-2],[3,4]], [[1,0],[0,1]], [[1,0],[0,1]]);"
        ),
        "matrix_gm with negative mean entries",
    ),
    (
        "B6",
        "soga",
        (
            "matrix[2][2] X;"
            " X = matrix_gm([[0,0],[0,0]], [[0.5,0],[0,0.5]], [[2,1],[1,2]]);"
        ),
        "matrix_gm with non-isotropic covariance",
    ),
]

# ---------------------------------------------------------------------------
# Group C — matrix_gm_full constructor (3 positive)
# ---------------------------------------------------------------------------

_POSITIVE_GROUP_C = [
    (
        "C1",
        "soga",
        (
            "matrix[2][2] X;"
            " X = matrix_gm_full([[1,2],[3,4]],"
            " [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);"
        ),
        "matrix_gm_full 2x2 with I_4 covariance",
    ),
    (
        "C2",
        "soga",
        (
            "matrix[2][2] X;"
            " X = matrix_gm_full([[0,0],[0,0]],"
            " [[2,1,0,0],[1,2,0,0],[0,0,2,1],[0,0,1,2]]);"
        ),
        "matrix_gm_full 2x2 with block-diagonal covariance",
    ),
    (
        "C3",
        "soga",
        (
            "matrix[1][1] S;"
            " S = matrix_gm_full([[5]], [[4]]);"
        ),
        "matrix_gm_full 1x1 scalar degenerate",
    ),
]

# ---------------------------------------------------------------------------
# Group D — ASGMT matrix operators (10 positive)
# ---------------------------------------------------------------------------

_POSITIVE_GROUP_D = [
    ("D1", "asgmt", "Y = A @ X", "left matmul Y = A @ X"),
    ("D2", "asgmt", "Y = X @ B", "right matmul Y = X @ B"),
    ("D3", "asgmt", "Z = A @ X @ B", "chained matmul Z = A @ X @ B"),
    ("D4", "asgmt", "Y = X + N", "matrix add Y = X + N"),
    ("D5", "asgmt", "Y = transp(X)", "transpose Y = transp(X)"),
    ("D6", "asgmt", "y = X[0,0]", "extract scalar y = X[0,0]"),
    ("D7", "asgmt", "y = X[i,j]", "extract scalar y = X[i,j] (var indices)"),
    (
        "D8",
        "asgmt",
        "Y = matrix_gm([[1,2],[3,4]], [[1,0],[0,1]], [[1,0],[0,1]])",
        "matrix_gm as RHS of assignment",
    ),
    (
        "D9",
        "asgmt",
        (
            "Y = matrix_gm_full([[1,2],[3,4]],"
            " [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])"
        ),
        "matrix_gm_full as RHS of assignment",
    ),
    ("D10", "asgmt", "Y = X @ X + N", "matmul + add precedence"),
]

# ---------------------------------------------------------------------------
# Group E — TRUNC observe constraints (4 positive)
# ---------------------------------------------------------------------------

_POSITIVE_GROUP_E = [
    ("E_T1", "trunc", "X[1,2] > 0", "element inequality X[i,j] > c"),
    ("E_T2", "trunc", "2 * X[1,2] < 10", "scaled element inequality"),
    ("E_T3", "trunc", "row_sum(X, 0) <= 5", "row_sum inequality"),
    ("E_T4", "trunc", "X[0,0] + X[1,1] > 0", "linear combo of matrix elements"),
]

# ---------------------------------------------------------------------------
# Group F — mixed scalar+matrix programs (2 positive)
# ---------------------------------------------------------------------------

_POSITIVE_GROUP_F = [
    (
        "F1",
        "soga",
        (
            "data A = [[2,0],[0,1]];"
            " matrix[2][2] X;"
            " X = matrix_gm([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]);"
        ),
        "data decl + matrix_gm (data before matrix_decl)",
    ),
    (
        "F2",
        "soga",
        (
            "x = gm([0.5, 0.5], [-1.0, 1.0], [0.1, 0.1]);"
            " matrix[2][2] X;"
            " X = matrix_gm([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]);"
        ),
        "scalar gm + matrix_gm in same program",
    ),
]

# ---------------------------------------------------------------------------
# Group G — keyword/IDV regression (2 positive)
# ---------------------------------------------------------------------------

_POSITIVE_GROUP_G = [
    (
        "G1",
        "soga",
        "row_sum = 5;",
        "row_sum as IDV (not reserved when used as lhs)",
    ),
    (
        "G2",
        "soga",
        "matrixVar = 1;",
        "matrixVar as IDV (starts with 'matrix' but no underscore — ALPHA-only IDV)",
    ),
]

# ---------------------------------------------------------------------------
# Combined positive list (32 total)
# ---------------------------------------------------------------------------

_ALL_POSITIVE = (
    _POSITIVE_GROUP_A
    + _POSITIVE_GROUP_B
    + _POSITIVE_GROUP_C
    + _POSITIVE_GROUP_D
    + _POSITIVE_GROUP_E
    + _POSITIVE_GROUP_F
    + _POSITIVE_GROUP_G
)
assert len(_ALL_POSITIVE) == 32, f"Expected 32 positives, got {len(_ALL_POSITIVE)}"


# ---------------------------------------------------------------------------
# Negative tests N1–N8 — genuine parse-time errors
#
# Each entry: (neg_id, grammar, snippet, production_that_fails, description)
# ---------------------------------------------------------------------------

_NEGATIVES = [
    (
        "N1",
        "soga",
        "x = 1.2.3;",
        "NUM lexer rule",
        "bad float literal 1.2.3 — token recognition error at second '.'",
    ),
    (
        "N2",
        "soga",
        "x = 1",
        "';' token in stat rule",
        "missing ';' at EOF — parser expects ';' after assignment",
    ),
    (
        "N3",
        "asgmt",
        "X = transp(X;",
        "')' in transp rule",
        "missing ')' in transp call — token recognition error at ';'",
    ),
    (
        "N4",
        "asgmt",
        "X = A @;",
        "mat_atom rule",
        "missing mat_atom after '@' — token recognition error at ';'",
    ),
    (
        "N5",
        "asgmt",
        "X = [1,2;",
        "']' in mlist rule",
        "unclosed '[' in mlist — token recognition error at ';'",
    ),
    (
        "N6",
        "asgmt",
        "X = X @ @ X",
        "mat_atom rule between consecutive '@'",
        "double '@' operator — extraneous '@' where mat_atom expected",
    ),
    (
        "N7",
        "soga",
        "matrix[2][2] X; X = matrix_gm([[1,2],[3,4]], [[1,0],[0,1]], [[1,0],[0,1]], [[0,0],[0,0]]);",
        "matrix_gm rule (exactly 3 mlist args)",
        "matrix_gm with 4 arguments — 4th mlist causes no-viable-alternative",
    ),
    (
        "N8",
        "soga",
        "x = (1 + ;",
        "expr rule (RHS missing after '+')",
        "open expression (1 + ; — extraneous '(' then missing operand for '+'",
    ),
]
assert len(_NEGATIVES) == 8, f"Expected 8 negatives, got {len(_NEGATIVES)}"


# ---------------------------------------------------------------------------
# Edge cases E1–E10 — empirically determined outcomes
#
# Each entry: (edge_id, grammar, snippet, expected_ok, description)
# Note: expected_ok=True means parse accepted; False means rejected.
# ---------------------------------------------------------------------------

_EDGE_CASES = [
    (
        "E1",
        "asgmt",
        "Y = X",
        True,
        "bare IDV assignment Y=X — no ambiguity with mat_expr path",
    ),
    (
        "E2",
        "soga",
        "matrix[2][2] X; y = X[-1,0];",
        True,
        "X[-1,0] with NUM allowing negatives — accepted by grammar (NUM : '-'? DIGIT+)",
    ),
    (
        "E3",
        "soga",
        "y = [[1]];",
        False,
        "bare mlist as RHS is not a valid assignment in SOGA progr (no matrix_gm wrapping)",
    ),
    (
        "E4",
        "asgmt",
        "Y = X@Y",
        True,
        "matmul X@Y without spaces — tokenizer handles @ robustly",
    ),
    (
        "E5",
        "soga",
        (
            "matrix[2][2] Y;"
            " Y = 2*matrix_gm_full([[1,2],[3,4]],"
            " [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);"
        ),
        True,
        "2*matrix_gm_full(...) as scaled assignment — matches add_term rule in SOGA",
    ),
    (
        "E6",
        "soga",
        "matrix[1][1] X; X = matrix_gm([[-1]], [[1]], [[1]]);",
        True,
        "[[-1]] in mlist — negative literal accepted (NUM allows '-')",
    ),
    (
        "E7",
        "soga",
        "",
        True,
        "empty program — SOGA grammar accepts empty progr (star-quantified rules)",
    ),
    (
        "E8a",
        "trunc",
        "X[1] > 0",
        True,
        "X[1] single-arg idd in TRUNC — accepted (idd: IDV '[' (NUM|IDV) ']')",
    ),
    (
        "E8b",
        "trunc",
        "X[1,2] > 0",
        True,
        "X[1,2] two-arg mat_idd in TRUNC — accepted",
    ),
    (
        "E9",
        "soga",
        "row_sum = 5;",
        True,
        "row_sum used as an IDV (not reserved at SOGA prog level)",
    ),
    # NOTE: 10 edge cases total; E8a and E8b count as separate entries for clarity
]
# We have E1,E2,E3,E4,E5,E6,E7,E8a,E8b,E9 = 10 items
assert len(_EDGE_CASES) == 10, f"Expected 10 edge cases, got {len(_EDGE_CASES)}"


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

def _parse(grammar: str, snippet: str) -> list[str]:
    """Dispatch to correct parser. grammar: 'soga' | 'asgmt' | 'trunc'."""
    if grammar == "soga":
        return _parse_soga(snippet)
    elif grammar == "asgmt":
        return _parse_asgmt(snippet)
    elif grammar == "trunc":
        return _parse_trunc(snippet)
    else:
        raise ValueError(f"Unknown grammar: {grammar!r}")


# ---------------------------------------------------------------------------
# POSITIVE tests (32)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "test_id, grammar, snippet, description",
    _ALL_POSITIVE,
    ids=[t[0] for t in _ALL_POSITIVE],
)
def test_positive(test_id: str, grammar: str, snippet: str, description: str):
    """Positive parse: snippet must be accepted with zero parse errors."""
    errors = _parse(grammar, snippet)
    assert errors == [], (
        f"[{test_id}] {description!r}: expected 0 parse errors, got: {errors}"
    )


# ---------------------------------------------------------------------------
# NEGATIVE tests (8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "neg_id, grammar, snippet, production, description",
    _NEGATIVES,
    ids=[t[0] for t in _NEGATIVES],
)
def test_negative(
    neg_id: str,
    grammar: str,
    snippet: str,
    production: str,
    description: str,
):
    """Negative parse: snippet must be rejected (>0 parse errors).

    Each negative is annotated with the ANTLR production/token that fails.
    """
    errors = _parse(grammar, snippet)
    assert errors, (
        f"[{neg_id}] {description!r}: expected >=1 parse error ({production}), "
        f"got zero errors"
    )


# ---------------------------------------------------------------------------
# EDGE CASE tests (10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "edge_id, grammar, snippet, expected_ok, description",
    _EDGE_CASES,
    ids=[t[0] for t in _EDGE_CASES],
)
def test_edge(
    edge_id: str,
    grammar: str,
    snippet: str,
    expected_ok: bool,
    description: str,
):
    """Edge case: outcome (accept/reject) determined empirically.

    expected_ok=True → zero parse errors expected.
    expected_ok=False → at least one parse error expected.
    """
    errors = _parse(grammar, snippet)
    if expected_ok:
        assert errors == [], (
            f"[{edge_id}] {description!r}: expected accepted (0 errors), "
            f"got: {errors}"
        )
    else:
        assert errors, (
            f"[{edge_id}] {description!r}: expected rejected (>=1 error), "
            f"got zero errors"
        )
