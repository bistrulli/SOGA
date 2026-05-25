"""
conftest.py — shared fixtures and helpers for tests/stress_matrix_gm/.

SEED = 42 (master seed, deterministic cross-platform where possible).
Fixtures: rng (function-scoped numpy Generator).
Path setup: inserts src/ into sys.path.
Shared helpers: _CountingErrorListener (ANTLR syntax-error counter).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path setup — expose src/ for all tests in this package
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Master seed
# ---------------------------------------------------------------------------
SEED: int = 42


# ---------------------------------------------------------------------------
# rng fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def rng() -> np.random.Generator:
    """Function-scoped numpy default_rng seeded with SEED=42."""
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# ANTLR error listener
# ---------------------------------------------------------------------------
from antlr4.error.ErrorListener import ErrorListener  # noqa: E402


class _CountingErrorListener(ErrorListener):
    """ANTLR error listener that accumulates syntax errors without printing."""

    def __init__(self):
        super().__init__()
        self.errors: list[str] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.errors.append(f"line {line}:{column} {msg}")


# FIXED_NON_KRON_4x4 lives in analytical_ground_truth.py
# (a constant, not a pytest fixture — imported directly by tests).
