"""
conftest.py — shared fixtures and helpers for tests/stress_matrix_gm/.

SEED = 42 (master seed, deterministic cross-platform where possible).
Fixtures: rng (function-scoped numpy Generator).
Path setup: inserts src/ into sys.path.
Shared helpers: _CountingErrorListener (ANTLR syntax-error counter).
FIXED_NON_KRON_4x4: a certified non-Kronecker SPD 4x4 matrix.
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


# ---------------------------------------------------------------------------
# FIXED_NON_KRON_4x4 — certified non-Kronecker SPD 4×4 matrix
#
# A Kronecker product V ⊗ U with U (2×2) and V (2×2) satisfies the rank-1
# condition on the rearrangement R[Sigma].  The necessary condition is:
#   Sigma[0,3] * Sigma[1,2] == Sigma[0,2] * Sigma[1,3]
#
# We choose a matrix that violates this:
#   abs(Sigma[0,3]*Sigma[1,2] - Sigma[0,2]*Sigma[1,3]) > 0.1
# and is SPD (all eigenvalues > 0).
# ---------------------------------------------------------------------------
FIXED_NON_KRON_4x4 = np.array([
    [4.0, 1.5, 0.8, 0.3],
    [1.5, 3.0, 0.6, 0.9],
    [0.8, 0.6, 2.5, 1.1],
    [0.3, 0.9, 1.1, 2.0],
], dtype=float)

# Verify SPD at module load time
_eigs = np.linalg.eigvalsh(FIXED_NON_KRON_4x4)
assert np.all(_eigs > 0), (
    f"FIXED_NON_KRON_4x4 is not SPD: eigenvalues = {_eigs}"
)

# Verify non-Kronecker: |Sigma[0,3]*Sigma[1,2] - Sigma[0,2]*Sigma[1,3]| > 0.1
_kron_violation = abs(
    FIXED_NON_KRON_4x4[0, 3] * FIXED_NON_KRON_4x4[1, 2]
    - FIXED_NON_KRON_4x4[0, 2] * FIXED_NON_KRON_4x4[1, 3]
)
assert _kron_violation > 0.1, (
    f"FIXED_NON_KRON_4x4 may be Kronecker: violation = {_kron_violation:.4f} (need > 0.1)"
)
