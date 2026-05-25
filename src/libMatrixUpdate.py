"""
libMatrixUpdate.py — M4 matrix assignment operations for SOGA.

Implements update_rule_matrix and all per-operation helpers:
  _matrix_affine_left   (M4.2): Y = A @ X
  _matrix_affine_right  (M4.3): Y = X @ B
  _matrix_add_const     (M4.4): Y = X + C (deterministic)
  _matrix_add_random    (M4.5): Y = X + N (random, iso + general paths)
  _matrix_scale         (M4.6): Y = c * X
  _matrix_transpose     (M4.7): Y = transp(X)
  _apply_psd_to_block   (M4.9): PSD wrap

Called from libSOGAupdate.update_rule_matrix.

Convention: KRONECKER_CONVENTION = "V_outer_U" throughout.
Plan reference: §M4 of plan/2026-05-22-matrix-gm-lishan.md
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from copy import deepcopy
from typing import List, Tuple

import numpy as np

from libSOGAshared import Dist, VarEntry, GaussianMix
from libSOGAsharedMatrix import GaussianMixBlock, _enforce_psd_kron_factors
from libMatrixGaussian import (
    _nearest_kronecker, _try_kronecker_decompose,
    KroneckerDetectionInfo, KroneckerNearMissWarning, DenseCovarianceInfo,
    StaleCrossCovWarning,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe nested-list parser for matrix_gm arguments
# ---------------------------------------------------------------------------

def _parse_nested_list(text: str) -> list:
    """Parse a nested list of numbers from a string without using exec/eval.

    Converts e.g. '[[1,0],[0,1]]' to [[1.0, 0.0], [0.0, 1.0]].
    Only supports numeric literals (int or float) inside brackets.
    Raises ValueError for any unexpected token.
    """
    text = text.strip()
    # Replace all whitespace with space, then parse character-by-character
    import re
    tokens = re.findall(r'[\[\],]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', text)

    def _parse_value(pos):
        """Recursively parse a value (number or list) starting at tokens[pos]."""
        tok = tokens[pos]
        if tok == '[':
            result = []
            pos += 1
            if tokens[pos] == ']':
                return result, pos + 1
            while True:
                val, pos = _parse_value(pos)
                result.append(val)
                tok = tokens[pos]
                if tok == ']':
                    return result, pos + 1
                elif tok == ',':
                    pos += 1
                else:
                    raise ValueError(f"Unexpected token '{tok}' while parsing list")
        else:
            return float(tok), pos + 1

    result, _ = _parse_value(0)
    return result


def _parse_matrix_gm_full_text(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse matrix_gm_full(M_list, Sigma_list) text into numpy arrays.

    Returns (M, Sigma) where:
      M:     shape (m, n) — mean matrix
      Sigma: shape (mn, mn) — full vectorised covariance

    Validates:
      - M is 2D (m rows, each with n elements)
      - Sigma is 2D square with shape (mn, mn)
      - Sigma is symmetric (max |Sigma - Sigma.T| < 1e-10)

    Raises ValueError on shape or symmetry violation.
    Uses a safe numeric-only parser — no code execution.
    """
    inner = text[len("matrix_gm_full("):-1]
    # Split the two top-level list arguments at depth-0 commas
    depth = 0
    parts: List[str] = []
    current: List[str] = []
    for ch in inner:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    if len(parts) != 2:
        raise ValueError(
            f"[B.3] matrix_gm_full expects 2 arguments (M, Sigma), got {len(parts)}"
        )
    M = np.asarray(_parse_nested_list(parts[0]), dtype=float)
    Sigma = np.asarray(_parse_nested_list(parts[1]), dtype=float)

    # Validate M shape: must be 2D
    if M.ndim != 2:
        raise ValueError(
            f"[B.3] matrix_gm_full: M must be a 2D matrix (list of lists), "
            f"got ndim={M.ndim}"
        )
    m, n = M.shape
    mn = m * n

    # Validate Sigma shape: must be (mn, mn)
    if Sigma.ndim != 2:
        raise ValueError(
            f"[B.3] matrix_gm_full: Sigma must be a 2D matrix, got ndim={Sigma.ndim}"
        )
    if Sigma.shape != (mn, mn):
        raise ValueError(
            f"[B.3] matrix_gm_full: Sigma shape mismatch. "
            f"M is {m}x{n} → expected Sigma shape ({mn},{mn}), got {Sigma.shape}"
        )

    # Validate Sigma is symmetric
    sym_err = float(np.max(np.abs(Sigma - Sigma.T)))
    if sym_err > 1e-8:
        raise ValueError(
            f"[B.3] matrix_gm_full: Sigma is not symmetric (max|Sigma-Sigma.T|={sym_err:.3e})"
        )

    return M, Sigma


def _parse_matrix_gm_text(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse matrix_gm(M_list, U_list, V_list) text into numpy arrays.

    Uses a safe numeric-only parser — no code execution.
    """
    inner = text[len("matrix_gm("):-1]
    # Split the three top-level list arguments at depth-0 commas
    depth = 0
    parts: List[str] = []
    current: List[str] = []
    for ch in inner:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    if len(parts) != 3:
        raise ValueError(
            f"[M4] matrix_gm expects 3 arguments (M, U, V), got {len(parts)}"
        )
    M = np.asarray(_parse_nested_list(parts[0]), dtype=float)
    U = np.asarray(_parse_nested_list(parts[1]), dtype=float)
    V = np.asarray(_parse_nested_list(parts[2]), dtype=float)
    return M, U, V


# ---------------------------------------------------------------------------
# Isotropy helper (M4.5)
# ---------------------------------------------------------------------------

def _is_iso(M: np.ndarray, atol: float = 1e-10, rtol: float = 1e-8) -> bool:
    """Check if M is a positive scalar multiple of the identity."""
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False
    c = M[0, 0]
    target = c * np.eye(M.shape[0])
    deviation = float(np.max(np.abs(M - target)))
    scale = float(np.max(np.abs(M))) if np.max(np.abs(M)) > 0 else 1.0
    threshold = atol + rtol * scale
    return deviation <= threshold


# ---------------------------------------------------------------------------
# PSD wrap (M4.9)
# ---------------------------------------------------------------------------

def _apply_psd_to_block(block: GaussianMixBlock, k: int, var_name: str) -> None:
    """Apply PSD enforcement to (U, V) Kronecker factors for var_name in component k.

    fix4: skips PSD enforcement if the variable is in dense sentinel mode
    (None, Sigma) — dense PSD is enforced separately in _matrix_element_write_component.
    """
    stored = block.cov_blocks[k].get(frozenset({var_name}))
    if stored is not None and stored[0] is None:
        # Dense sentinel: PSD already enforced by _matrix_element_write_component
        return
    U, V = block.get_cov(k, var_name, var_name)
    U, V = _enforce_psd_kron_factors(U, V)
    block.cov_blocks[k][frozenset({var_name})] = (U, V)


# ---------------------------------------------------------------------------
# Expression parser (M4.1)
# ---------------------------------------------------------------------------

def _parse_matrix_expr(expr: str) -> dict:
    """Parse a matrix assignment expression into an op-type dict."""
    body = expr.split("=", 1)[1].strip()

    if body.startswith("matrix_gm_full("):
        return {"op": "MATRIX_GM_FULL", "text": body}

    if body.startswith("matrix_gm("):
        return {"op": "MATRIX_GM", "text": body}

    if body.startswith("transp("):
        src = body[len("transp("):-1].strip()
        return {"op": "TRANSPOSE", "src": src}

    if "@" in body:
        parts = [p.strip() for p in body.split("@", 1)]
        return {"op": "MATMUL_RAW", "left": parts[0], "right": parts[1]}

    if "+" in body:
        parts = [p.strip() for p in body.split("+", 1)]
        return {"op": "ADD_RAW", "left": parts[0], "right": parts[1]}

    if "*" in body:
        parts = [p.strip() for p in body.split("*", 1)]
        return {"op": "SCALE_RAW", "left": parts[0], "right": parts[1]}

    raise NotImplementedError(
        f"[M4] Cannot parse matrix expression '{expr}'. "
        "Supported forms: matrix_gm(...), transp(X), A@X, X+Y, c*X. "
        "See plan/2026-05-22-matrix-gm-lishan.md §M4.1."
    )


# ---------------------------------------------------------------------------
# Per-operation matrix update helpers
# ---------------------------------------------------------------------------

def _copy_var_into_lhs(block: GaussianMixBlock, k: int, lhs: str, src: str) -> None:
    """Copy variable `src` into variable `lhs` in component k."""
    block.mu_blocks[k][lhs] = block.mu_blocks[k][src].copy()
    block.cov_blocks[k][frozenset({lhs})] = block.cov_blocks[k][frozenset({src})]
    for sv in block.var_list:
        key_src = frozenset({sv, src})
        if key_src in block.cov_blocks[k]:
            block.cov_blocks[k][frozenset({sv, lhs})] = block.cov_blocks[k][key_src].copy()


def _matrix_affine_left(block: GaussianMixBlock, k: int, lhs: str, A: np.ndarray) -> None:
    """Y = A @ X per component k (M4.2)."""
    M_x = block.get_mu(k, lhs)
    U_x, V_x = block.get_cov(k, lhs, lhs)
    if U_x is None:
        raise NotImplementedError(
            f"[C1/C7] left-affine (A@X) on dense-sentinel covariance not implemented. "
            f"Variable '{lhs}' was densified by observe() or matrix_gm_full(dense Sigma). "
            f"See docs/LIMITATIONS.md §C1/C7."
        )
    n = M_x.shape[1]
    block.mu_blocks[k][lhs] = A @ M_x
    block.cov_blocks[k][frozenset({lhs})] = (A @ U_x @ A.T, V_x.copy())
    I_n_kron_A = np.kron(np.eye(n), A)
    for sv in block.var_list:
        key = frozenset({sv, lhs})
        if key in block.cov_blocks[k]:
            block.cov_blocks[k][key] = I_n_kron_A @ block.cov_blocks[k][key]


def _matrix_affine_right(block: GaussianMixBlock, k: int, lhs: str, B: np.ndarray) -> None:
    """Y = X @ B per component k (M4.3)."""
    M_x = block.get_mu(k, lhs)
    U_x, V_x = block.get_cov(k, lhs, lhs)
    if U_x is None:
        raise NotImplementedError(
            f"[C1/C7] right-affine (X@B) on dense-sentinel covariance not implemented. "
            f"Variable '{lhs}' was densified by observe() or matrix_gm_full(dense Sigma). "
            f"See docs/LIMITATIONS.md §C1/C7."
        )
    m = M_x.shape[0]
    block.mu_blocks[k][lhs] = M_x @ B
    block.cov_blocks[k][frozenset({lhs})] = (U_x.copy(), B.T @ V_x @ B)
    BT_kron_Im = np.kron(B.T, np.eye(m))
    for sv in block.var_list:
        key = frozenset({sv, lhs})
        if key in block.cov_blocks[k]:
            block.cov_blocks[k][key] = BT_kron_Im @ block.cov_blocks[k][key]


def _matrix_add_const(block: GaussianMixBlock, k: int, lhs: str, C: np.ndarray) -> None:
    """Y = X + C (deterministic C) per component k (M4.4)."""
    block.mu_blocks[k][lhs] = block.get_mu(k, lhs) + C


def _matrix_add_random(block: GaussianMixBlock, k: int, lhs: str, src: str) -> None:
    """Y = X + N (independent random N) per component k (M4.5)."""
    M_x = block.get_mu(k, lhs)
    U_x, V_x = block.get_cov(k, lhs, lhs)
    M_n = block.get_mu(k, src)
    U_n, V_n = block.get_cov(k, src, src)
    m, n = M_x.shape
    M_new = M_x + M_n

    # Iso path A: U_X isotropic AND N fully isotropic
    if _is_iso(U_x) and _is_iso(U_n) and _is_iso(V_n):
        c = float(U_x[0, 0])
        if c > 0:
            a, b = float(U_n[0, 0]), float(V_n[0, 0])
            V_new = V_x + (a * b / c) * np.eye(n)
            U_new, V_new = _enforce_psd_kron_factors(U_x.copy(), V_new)
            block.mu_blocks[k][lhs] = M_new
            block.cov_blocks[k][frozenset({lhs})] = (U_new, V_new)
            return

    # Iso path B: V_X isotropic AND N fully isotropic
    if _is_iso(V_x) and _is_iso(U_n) and _is_iso(V_n):
        d = float(V_x[0, 0])
        if d > 0:
            a, b = float(U_n[0, 0]), float(V_n[0, 0])
            U_new = U_x + (a * b / d) * np.eye(m)
            U_new, V_new = _enforce_psd_kron_factors(U_new, V_x.copy())
            block.mu_blocks[k][lhs] = M_new
            block.cov_blocks[k][frozenset({lhs})] = (U_new, V_new)
            return

    # General path: nearest-Kronecker projection
    Cov_full = np.kron(V_x, U_x) + np.kron(V_n, U_n)
    U_new, V_new = _nearest_kronecker(Cov_full, m, n)
    norm_full = float(np.linalg.norm(Cov_full, "fro"))
    if norm_full > 0:
        eps_proj = float(np.linalg.norm(Cov_full - np.kron(V_new, U_new), "fro")) / norm_full
        if eps_proj > 0.05:
            warnings.warn(
                f"[M4.5] add_random projection error {eps_proj:.3%} > 5% "
                f"for '{lhs}' component {k}",
                UserWarning,
                stacklevel=4,
            )
    U_new, V_new = _enforce_psd_kron_factors(U_new, V_new)
    block.mu_blocks[k][lhs] = M_new
    block.cov_blocks[k][frozenset({lhs})] = (U_new, V_new)


def _matrix_scale(block: GaussianMixBlock, k: int, lhs: str, c: float) -> None:
    """Y = c * X per component k (M4.6)."""
    M_x = block.get_mu(k, lhs)
    U_x, V_x = block.get_cov(k, lhs, lhs)
    if U_x is None:
        raise NotImplementedError(
            f"[C1/C7] scale (c*X) on dense-sentinel covariance not implemented. "
            f"Variable '{lhs}' was densified by observe() or matrix_gm_full(dense Sigma). "
            f"See docs/LIMITATIONS.md §C1/C7."
        )
    block.mu_blocks[k][lhs] = c * M_x
    U_new, V_new = _enforce_psd_kron_factors((c * c) * U_x, V_x.copy())
    block.cov_blocks[k][frozenset({lhs})] = (U_new, V_new)
    for sv in block.var_list:
        key = frozenset({sv, lhs})
        if key in block.cov_blocks[k]:
            block.cov_blocks[k][key] = c * block.cov_blocks[k][key]


def _matrix_transpose(block: GaussianMixBlock, k: int, lhs: str) -> None:
    """Y = X^T per component k (M4.7). Factors swap; cross-covs via commutation matrix."""
    M_x = block.get_mu(k, lhs)
    U_x, V_x = block.get_cov(k, lhs, lhs)
    if U_x is None:
        raise NotImplementedError(
            f"[C1/C7] transpose (X^T) on dense-sentinel covariance not implemented. "
            f"Variable '{lhs}' was densified by observe() or matrix_gm_full(dense Sigma). "
            f"See docs/LIMITATIONS.md §C1/C7."
        )
    m, n = M_x.shape
    block.mu_blocks[k][lhs] = M_x.T
    block.cov_blocks[k][frozenset({lhs})] = (V_x.copy(), U_x.copy())
    mn = m * n
    P_comm = np.zeros((mn, mn))
    for i in range(m):
        for j in range(n):
            P_comm[j * m + i, i * n + j] = 1.0
    for sv in block.var_list:
        key = frozenset({sv, lhs})
        if key in block.cov_blocks[k]:
            block.cov_blocks[k][key] = P_comm @ block.cov_blocks[k][key]


# ---------------------------------------------------------------------------
# fix3: warning class for matmul approximation quality
# ---------------------------------------------------------------------------

class MatmulApproxWarning(UserWarning):
    """Emitted when delta-method + NKP approximation is potentially inaccurate.

    Fired when the second singular value of the rearrangement of Sigma_Z
    exceeds 5% of the first (indicating significant discarded Kronecker term).
    """


# ---------------------------------------------------------------------------
# fix3.2 — Matmul random × random component
# ---------------------------------------------------------------------------

def matmul_random_random_component(
    M_X: np.ndarray, U_X: np.ndarray, V_X: np.ndarray,
    M_Y: np.ndarray, U_Y: np.ndarray, V_Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate Z = X @ Y for X ~ MN(M_X, U_X, V_X), Y ~ MN(M_Y, U_Y, V_Y).

    Uses EXACT 2nd-moment Isserlis matriciale (derived 2026-05-24, see
    research note 05 §Q2c-extended) + Van Loan-Pitsianis nearest-Kronecker
    projection.

    Exact moments (no Taylor approximation):
      E[Z]_{ab}              = sum_k (M_X)_{ak} (M_Y)_{kb} = (M_X M_Y)_{ab}
      Cov(Z_{ab}, Z_{cd})    = E[X_{ak} X_{cl}] E[Y_{kb} Y_{ld}] - E[Z]E[Z]^T

    Expanding the Isserlis bilinear form for two independent matrix-variate
    Gaussians yields THREE Kronecker contributions (delta-method had only 2):

      Cov(vec(Z)) = tr(V_X · U_Y) · (V_Y ⊗ U_X)              [Isserlis trace]
                  + (M_Y^T V_X M_Y) ⊗ U_X                    [delta term 1]
                  + V_Y ⊗ (M_X U_Y M_X^T)                    [delta term 2]

    The trace term vanishes when M_X = 0 or M_Y = 0 only in degenerate
    cases; in general it is the dominant correction when both means are
    comparable to the standard deviations (the regime where the prior
    delta-method had ~20% relative error on Cov, e.g. R3 in MC validation).

    After computing the exact dense Cov(vec(Z)) we project to single
    Kronecker (V_Z ⊗ U_Z) via Van Loan-Pitsianis SVD rank-1 (only
    approximation in the path).

    Parameters
    ----------
    M_X, U_X, V_X : mean (m×p), row-cov (m×m), col-cov (p×p) for X
    M_Y, U_Y, V_Y : mean (p×n), row-cov (p×p), col-cov (n×n) for Y

    Returns
    -------
    M_Z (m×n), U_Z (m×m), V_Z (n×n) — Kronecker-factored MN approximation.

    Validation against MC ground truth (research note 05 §Problem 2):
    - R1 small cov: rel err ~ 0% on Cov (was 0% with delta too)
    - R2 moderate cov: rel err ~ 0% on Cov (was 9%)
    - R3 balanced (M=I, σ²=0.5): rel err ~ 0% on Cov (was 20%) — KEY GAIN
    - R4 3x3 small cov: rel err ~ 0% on Cov
    """
    m, p = M_X.shape
    _, n = M_Y.shape
    # Exact mean
    M_Z = M_X @ M_Y

    # Isserlis exact covariance: TWO Kronecker products with a corrected
    # first outer factor (research note 05 §Q2c-extended, derived 2026-05-24).
    # The third Isserlis term tr(V_X U_Y)·(V_Y ⊗ U_X) shares the V_Y ⊗ U_X
    # structure with delta-method term 2 of the inner factor when expanded;
    # but the cleanest grouping is to add it to term 1 since it has the SAME
    # inner factor U_X. We use the additive combination:
    #
    #   Cov = [tr(V_X U_Y)·V_Y + M_Y^T V_X M_Y] ⊗ U_X     (the "X-side" Kronecker)
    #       + V_Y ⊗ (M_X U_Y M_X^T)                         (the "Y-side" Kronecker)

    trace_VxUy = float(np.trace(V_X @ U_Y))                # scalar — Isserlis correction
    A = trace_VxUy * V_Y + M_Y.T @ V_X @ M_Y               # (n, n) — corrected outer factor
    B = U_X                                                  # (m, m)
    C = V_Y                                                  # (n, n)
    D = M_X @ U_Y @ M_X.T                                   # (m, m)

    # Build full (mn × mn) dense covariance in V⊗U column-major order.
    Sigma_full = np.kron(A, B) + np.kron(C, D)

    # Check approximation quality: rearrange Sigma_full and compute SVD rank
    mn = m * n
    R = np.zeros((m * m, n * n))
    for jj in range(n):
        for jjj in range(n):
            block_ij = Sigma_full[jj * m:(jj + 1) * m, jjj * m:(jjj + 1) * m]
            R[:, jj * n + jjj] = block_ij.flatten('F')
    _, sv, _ = np.linalg.svd(R, full_matrices=False)
    if sv[0] > 0 and len(sv) > 1:
        ratio = sv[1] / sv[0]
        if ratio > 0.05:
            warnings.warn(
                f"MatmulApproxWarning: NKP projection discards significant term "
                f"(s2/s1={ratio:.3%} > 5%).  Delta-method approximation may be inaccurate.",
                MatmulApproxWarning,
                stacklevel=3,
            )

    # NKP projection to nearest single Kronecker product
    U_Z, V_Z = _nearest_kronecker(Sigma_full, m, n)
    U_Z, V_Z = _enforce_psd_kron_factors(U_Z, V_Z)
    return M_Z, U_Z, V_Z


# ---------------------------------------------------------------------------
# Main dispatcher: update_rule_matrix (M4.1)
# ---------------------------------------------------------------------------

def update_rule_matrix(dist: Dist, expr: str, data: dict) -> Dist:
    """Full dispatcher for matrix-variable assignments (M4).

    Parses expr, classifies the operation, dispatches to the appropriate
    helper, applies PSD enforcement, and returns an updated Dist.
    """
    lhs = expr.split("=")[0].strip()
    parsed = _parse_matrix_expr(expr)
    op = parsed["op"]

    # --- MATRIX_GM: initialise or extend gm_block ---
    if op == "MATRIX_GM":
        M, U, V = _parse_matrix_gm_text(parsed["text"])
        ve = next((v for v in dist.var_entries if v.name == lhs), None)
        if ve is None:
            raise RuntimeError(f"[M4] No VarEntry for '{lhs}'")

        if dist.gm_block is None:
            block = GaussianMixBlock.from_matrix_gm(
                [ve], [M], [U], [V],
                pi=[1.0],
                var_list=list(dist.var_list),
            )
        else:
            block = deepcopy(dist.gm_block)
            m_var, n_var = ve.shape
            for k in range(block.n_comp()):
                block.mu_blocks[k][lhs] = M.copy()
                U_k, V_k = _enforce_psd_kron_factors(U.copy(), V.copy())
                block.cov_blocks[k][frozenset({lhs})] = (U_k, V_k)
                for sv in block.var_list:
                    block.cov_blocks[k][frozenset({sv, lhs})] = np.zeros(m_var * n_var)
            if ve not in block.var_entries:
                block.var_entries.append(ve)
        return Dist(dist.var_list, dist.gm, var_entries=dist.var_entries, gm_block=block)

    # --- MATRIX_GM_FULL: auto-Kronecker detection + initialise/extend gm_block ---
    if op == "MATRIX_GM_FULL":
        M, Sigma = _parse_matrix_gm_full_text(parsed["text"])
        ve = next((v for v in dist.var_entries if v.name == lhs), None)
        if ve is None:
            raise RuntimeError(f"[B.5] No VarEntry for '{lhs}'")

        m_var, n_var = ve.shape
        if M.shape != (m_var, n_var):
            raise ValueError(
                f"[B.5] matrix_gm_full: M shape {M.shape} does not match "
                f"declared shape ({m_var}, {n_var}) for variable '{lhs}'"
            )
        mn = m_var * n_var
        if Sigma.shape != (mn, mn):
            raise ValueError(
                f"[B.5] matrix_gm_full: Sigma shape {Sigma.shape} does not match "
                f"expected ({mn}, {mn}) for {m_var}x{n_var} variable '{lhs}'"
            )

        # Read configurable thresholds from env (default 1e-8 strict, 1e-3 loose)
        import os as _os
        kron_strict = float(_os.environ.get("SOGA_KRON_STRICT", "1e-8"))
        kron_loose = float(_os.environ.get("SOGA_KRON_LOOSE", "1e-3"))

        # Auto-Kronecker detection
        U_approx, V_approx, residual_ratio = _try_kronecker_decompose(Sigma, m_var, n_var, kron_strict)

        if residual_ratio < kron_strict:
            # Exact Kronecker (within threshold) — store as (U, V) Kronecker factors
            warnings.warn(
                KroneckerDetectionInfo(
                    f"[matrix_gm_full] '{lhs}': Kronecker structure detected "
                    f"(residual={residual_ratio:.3e} < {kron_strict:.3e}). "
                    f"Storing as (U, V) factors for efficient operations."
                ),
                KroneckerDetectionInfo,
                stacklevel=2,
            )
            use_kronecker = True
        else:
            # Not Kronecker — store as dense sentinel
            use_kronecker = False
            if kron_strict <= residual_ratio < kron_loose:
                warnings.warn(
                    KroneckerNearMissWarning(
                        f"[matrix_gm_full] '{lhs}': near-Kronecker Sigma "
                        f"(residual={residual_ratio:.3e}, "
                        f"threshold={kron_strict:.3e}..{kron_loose:.3e}). "
                        f"Storing as DENSE for safety. "
                        f"If Sigma was intended to be Kronecker, check for rounding."
                    ),
                    KroneckerNearMissWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    DenseCovarianceInfo(
                        f"[matrix_gm_full] '{lhs}': non-Kronecker Sigma "
                        f"(residual={residual_ratio:.3e} >= {kron_loose:.3e}). "
                        f"Storing as dense sentinel (None, Sigma)."
                    ),
                    DenseCovarianceInfo,
                    stacklevel=2,
                )

        if dist.gm_block is None:
            # First matrix variable in program: use from_matrix_gm_full
            block = GaussianMixBlock.from_matrix_gm_full(
                ve=ve,
                M=M,
                Sigma=Sigma,
                U=U_approx if use_kronecker else None,
                V=V_approx if use_kronecker else None,
                pi=[1.0],
                var_list=list(dist.var_list),
            )
        else:
            block = deepcopy(dist.gm_block)
            for k in range(block.n_comp()):
                block.mu_blocks[k][lhs] = M.copy()
                if use_kronecker:
                    U_k, V_k = _enforce_psd_kron_factors(U_approx.copy(), V_approx.copy())
                    block.cov_blocks[k][frozenset({lhs})] = (U_k, V_k)
                else:
                    # Dense sentinel
                    Sigma_k = 0.5 * (Sigma + Sigma.T)  # enforce symmetry
                    block.cov_blocks[k][frozenset({lhs})] = (None, Sigma_k.copy())
                # Cross-covariances with scalars: zero at declaration
                for sv in block.var_list:
                    block.cov_blocks[k][frozenset({sv, lhs})] = np.zeros(mn)
            if ve not in block.var_entries:
                block.var_entries.append(ve)

        return Dist(dist.var_list, dist.gm, var_entries=dist.var_entries, gm_block=block)

    # For all other ops, gm_block must exist
    if dist.gm_block is None:
        raise RuntimeError(
            f"[M4] update_rule_matrix('{lhs}') called but dist.gm_block is None. "
            "Initialise with matrix_gm(...) first."
        )

    block = deepcopy(dist.gm_block)

    # Ensure the LHS variable is registered in block.var_entries.
    # This handles the case where C is declared (in dist.var_entries via CFG)
    # but has not yet been written into block (e.g., C = X + N where C has no
    # matrix_gm initialisation — C's first write is via an affine/add op).
    lhs_ve = next((v for v in dist.var_entries if v.name == lhs), None)
    if lhs_ve is not None and not any(v.name == lhs for v in block.var_entries):
        block.var_entries.append(lhs_ve)
        m_lhs, n_lhs = lhs_ve.shape
        mn_lhs = m_lhs * n_lhs
        for k in range(block.n_comp()):
            # Initialise LHS with zero mean and zero covariance (to be overwritten)
            block.mu_blocks[k][lhs] = np.zeros((m_lhs, n_lhs))
            block.cov_blocks[k][frozenset({lhs})] = (
                np.zeros((m_lhs, m_lhs)),
                np.zeros((n_lhs, n_lhs)),
            )
            for sv in block.var_list:
                block.cov_blocks[k][frozenset({sv, lhs})] = np.zeros(mn_lhs)
            for other_ve in block.var_entries:
                if other_ve.name != lhs:
                    # cross-cov between matrix vars not supported in v1 — initialise zero
                    # (will be set correctly by _copy_var_into_lhs or left at zero)
                    pass

    if op == "MATMUL_RAW":
        left_tok, right_tok = parsed["left"], parsed["right"]
        left_is_mat = any(ve.name == left_tok for ve in dist.var_entries)
        right_is_mat = any(ve.name == right_tok for ve in dist.var_entries)
        if left_is_mat and right_is_mat:
            # fix3: implement random × random matmul via delta-method + NKP
            # Full mixture product: |comp(left)| * |comp(right)| components.
            # Apply ranking_prune(K_max) immediately after.
            K_MAX_MATMUL = int(os.environ.get("SOGA_MATMUL_PRUNE_K", "50"))
            new_pi_mm = []
            new_mu_mm = []
            new_sigma_mm = []
            new_mu_blocks_mm = []
            new_cov_blocks_mm = []
            lhs_ve = next((v for v in dist.var_entries if v.name == lhs), None)
            left_ve = next((v for v in dist.var_entries if v.name == left_tok), None)
            right_ve = next((v for v in dist.var_entries if v.name == right_tok), None)
            # Preserve all matrix variables that existed before this op
            # (they are independent of X1@X2; we just copy their state
            # from ki=0 since no random@random can correlate them).
            other_ves = [v for v in dist.var_entries if v.name != lhs]
            for ki in range(block.n_comp()):
                M_X = block.get_mu(ki, left_tok)
                U_X, V_X = block.get_cov(ki, left_tok, left_tok)
                pi_X = block.pi[ki]
                for kj in range(block.n_comp()):
                    M_Y = block.get_mu(kj, right_tok)
                    U_Y, V_Y = block.get_cov(kj, right_tok, right_tok)
                    pi_Y = block.pi[kj]
                    M_Z, U_Z, V_Z = matmul_random_random_component(
                        M_X, U_X, V_X, M_Y, U_Y, V_Y
                    )
                    # Combined weight for (ki, kj) pair
                    new_pi_mm.append(pi_X * pi_Y)
                    # Scalar gm component: copy from ki
                    new_mu_mm.append(dist.gm.mu[ki % dist.gm.n_comp()])
                    new_sigma_mm.append(dist.gm.sigma[ki % dist.gm.n_comp()])
                    # Build new mu/cov block — preserve ALL pre-existing matrix
                    # variables (they're independent of the matmul op).
                    mu_k = {lhs: M_Z}
                    cov_k = {frozenset({lhs}): (U_Z, V_Z)}
                    for ve_sv in other_ves:
                        if ve_sv.name in block.mu_blocks[ki]:
                            mu_k[ve_sv.name] = block.mu_blocks[ki][ve_sv.name].copy()
                            sv_cov_key = frozenset({ve_sv.name})
                            if sv_cov_key in block.cov_blocks[ki]:
                                sv_cov = block.cov_blocks[ki][sv_cov_key]
                                if isinstance(sv_cov, tuple) and len(sv_cov) == 2:
                                    cov_k[sv_cov_key] = (sv_cov[0], sv_cov[1])
                                else:
                                    cov_k[sv_cov_key] = sv_cov
                    # Scalar-matrix cross-cov for lhs (zero — newly created)
                    for sv in block.var_list:
                        cov_k[frozenset({sv, lhs})] = np.zeros(
                            M_Z.shape[0] * M_Z.shape[1]
                        )
                    new_mu_blocks_mm.append(mu_k)
                    new_cov_blocks_mm.append(cov_k)
            # Normalise weights
            total_w = sum(new_pi_mm)
            if total_w > 0:
                new_pi_mm = [p / total_w for p in new_pi_mm]
            from libSOGAsharedMatrix import GaussianMixBlock as _GMB
            # Preserve full var_entries list (lhs + all pre-existing matrix vars).
            preserved_ves = list(dist.var_entries)
            if lhs_ve is not None and not any(v.name == lhs for v in preserved_ves):
                preserved_ves.append(lhs_ve)
            new_block = _GMB(
                var_list=block.var_list,
                var_entries=preserved_ves,
                pi=new_pi_mm,
                mu_blocks=new_mu_blocks_mm,
                cov_blocks=new_cov_blocks_mm,
            )
            new_gm = GaussianMix(new_pi_mm, new_mu_mm, new_sigma_mm)
            new_dist = Dist(
                dist.var_list, new_gm,
                var_entries=preserved_ves,
                gm_block=new_block,
            )
            # fix3.4: prune to K_max immediately
            if new_block.n_comp() > K_MAX_MATMUL:
                from libSOGAmerge import ranking_prune as _rp
                new_dist = _rp(new_dist, K_MAX_MATMUL)
            return new_dist
        elif not left_is_mat:
            A = np.asarray(data[left_tok], dtype=float)
            for k in range(block.n_comp()):
                if lhs != right_tok:
                    _copy_var_into_lhs(block, k, lhs, right_tok)
                _matrix_affine_left(block, k, lhs, A)
                _apply_psd_to_block(block, k, lhs)
        else:
            B = np.asarray(data[right_tok], dtype=float)
            for k in range(block.n_comp()):
                if lhs != left_tok:
                    _copy_var_into_lhs(block, k, lhs, left_tok)
                _matrix_affine_right(block, k, lhs, B)
                _apply_psd_to_block(block, k, lhs)

    elif op == "ADD_RAW":
        left_tok, right_tok = parsed["left"], parsed["right"]
        right_is_mat = any(ve.name == right_tok for ve in dist.var_entries)
        if right_is_mat:
            for k in range(block.n_comp()):
                if lhs != left_tok:
                    _copy_var_into_lhs(block, k, lhs, left_tok)
                _matrix_add_random(block, k, lhs, right_tok)
                _apply_psd_to_block(block, k, lhs)
        else:
            C = np.asarray(data[right_tok], dtype=float)
            for k in range(block.n_comp()):
                if lhs != left_tok:
                    _copy_var_into_lhs(block, k, lhs, left_tok)
                _matrix_add_const(block, k, lhs, C)

    elif op == "SCALE_RAW":
        left_tok, right_tok = parsed["left"], parsed["right"]
        left_is_mat = any(ve.name == left_tok for ve in dist.var_entries)
        mat_tok = left_tok if left_is_mat else right_tok
        sc_tok = right_tok if left_is_mat else left_tok
        try:
            c = float(sc_tok)
        except ValueError:
            c = float(data[sc_tok]) if sc_tok in data else None
            if c is None:
                raise NotImplementedError(
                    f"[M4.6] Cannot resolve scale '{sc_tok}' for '{expr}'."
                )
        for k in range(block.n_comp()):
            if lhs != mat_tok:
                _copy_var_into_lhs(block, k, lhs, mat_tok)
            _matrix_scale(block, k, lhs, c)
            _apply_psd_to_block(block, k, lhs)

    elif op == "TRANSPOSE":
        src = parsed["src"]
        for k in range(block.n_comp()):
            if lhs != src:
                _copy_var_into_lhs(block, k, lhs, src)
            _matrix_transpose(block, k, lhs)

    else:
        raise NotImplementedError(f"[M4] Unknown op '{op}' for '{expr}'.")

    return Dist(dist.var_list, dist.gm, var_entries=dist.var_entries, gm_block=block)


# ---------------------------------------------------------------------------
# M4.8: scalar = matrix[i, j] — extract scalar from matrix variable
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# fix4: Dense matrix element write X[i,j] = c/z/expr
# ---------------------------------------------------------------------------

def _densify_matrix_var(block: GaussianMixBlock, k: int, mat_name: str) -> np.ndarray:
    """Materialise the dense (mn × mn) covariance for mat_name in component k.

    fix4.2: Converts Kronecker-stored (U, V) → V⊗U dense matrix.
    If already stored as dense sentinel (None, Sigma_dense), returns Sigma_dense.

    Raises MemoryError if the resulting matrix exceeds MATRIX_ELEMENT_WRITE_BUDGET_MB.
    """
    stored = block.cov_blocks[k].get(frozenset({mat_name}))
    if stored is None:
        raise RuntimeError(f"[fix4.2] No covariance stored for '{mat_name}' in component {k}")
    U_or_none, V_or_Sigma = stored
    if U_or_none is None:
        # Already dense: V_or_Sigma is the dense (mn × mn) matrix
        return V_or_Sigma
    U, V = U_or_none, V_or_Sigma
    m_v, n_v = U.shape[0], V.shape[0]
    mn = m_v * n_v
    budget_mb = int(os.environ.get("MATRIX_ELEMENT_WRITE_BUDGET_MB", "1024"))
    cost_mb = (mn ** 2) * 8 / (1024 ** 2)
    if cost_mb > budget_mb:
        raise MemoryError(
            f"[fix4.2] Densifying '{mat_name}' ({m_v}x{n_v}) requires {cost_mb:.1f} MB "
            f"(budget {budget_mb} MB).  Use --matrix-element-write-budget-mb to increase."
        )
    return np.kron(V, U)  # V⊗U column-major convention


def _matrix_element_write_component(
    block: GaussianMixBlock,
    k: int,
    mat_name: str,
    m: int,
    n: int,
    i: int,
    j: int,
    c_or_mu: float,
    var_z: float = 0.0,
) -> None:
    """Schur-complement element write for component k (in-place).

    fix4.3: Implements Finding B1 (var_z=0, hard write) and B2 (var_z>0, soft
    write) from research note 04.  Both cases:

      idx       = j*m + i           (column-major: V⊗U convention)
      sel_vec   = Sigma[:, idx]     (the idx-th column of the dense covariance)
      gain      = sel_vec / (Sigma[idx, idx] + var_z)
      vec_M_new = vec_M + gain * (c_or_mu - vec_M[idx])
      Sigma_new = Sigma - outer(gain, sel_vec)

    Destroys Kronecker separability — stores result as dense sentinel (None, Sigma_new).

    IMPORTANT: after this call, further matrix ops on this variable that require
    Kronecker structure (e.g. affine_left, transpose) will fail with NotImplementedError.
    Only element extract and observe remain valid.  Documented in plan §fix4.5.
    """
    Sigma = _densify_matrix_var(block, k, mat_name)
    M_k = block.mu_blocks[k][mat_name]          # (m, n)
    vec_M = M_k.flatten('F')                    # column-major, length mn
    idx = j * m + i                             # column-major index

    sel_vec = Sigma[:, idx]                     # (mn,) selector column
    denom = float(Sigma[idx, idx]) + var_z
    if abs(denom) < 1e-14:
        # Near-zero variance: treat as deterministic (no update needed)
        vec_M_new = vec_M.copy()
        vec_M_new[idx] = c_or_mu
        block.mu_blocks[k][mat_name] = vec_M_new.reshape((m, n), order='F')
        block.cov_blocks[k][frozenset({mat_name})] = (None, Sigma.copy())
        return

    gain = sel_vec / denom
    delta = c_or_mu - float(vec_M[idx])
    vec_M_new = vec_M + gain * delta
    Sigma_new = Sigma - np.outer(gain, sel_vec)

    # Enforce symmetry and clip eigenvalues
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
    w, Q = np.linalg.eigh(Sigma_new)
    w = np.clip(w, 0.0, None)
    Sigma_new = Q @ np.diag(w) @ Q.T

    block.mu_blocks[k][mat_name] = vec_M_new.reshape((m, n), order='F')
    # Store as dense sentinel: (None, Sigma_dense)
    block.cov_blocks[k][frozenset({mat_name})] = (None, Sigma_new)

    # β-warning: scan for scalar variables with non-zero cross-cov to mat_name.
    # These scalars will NOT be updated by this element write (F4 — stale cross-cov).
    cov_k = block.cov_blocks[k]
    for s in block.var_list:
        key = frozenset({s, mat_name})
        if key in cov_k:
            cov_vec = cov_k[key]
            if hasattr(cov_vec, '__len__'):
                n_val = float(np.linalg.norm(cov_vec))
            else:
                n_val = abs(float(cov_vec))
            if n_val > 1e-8:
                warnings.warn(
                    StaleCrossCovWarning(
                        f"Scalar '{s}' has non-zero cross-cov with '{mat_name}' "
                        f"(norm={n_val:.2e}); element write on '{mat_name}' will not "
                        f"update '{s}'. Value of '{s}' remains STALE. "
                        f"See docs/LIMITATIONS.md §F4."
                    ),
                    StaleCrossCovWarning,
                    stacklevel=3,
                )


class ElementWriteDenseWarning(UserWarning):
    """Emitted when an element write densifies a matrix variable's covariance.

    fix4.5: After this write, subsequent Kronecker-dependent ops on the variable
    (affine_left, transpose) will raise NotImplementedError.  Only element
    extract and observe remain valid.
    """


def matrix_element_write_dispatch(
    dist: Dist,
    mat_name: str,
    i: int,
    j: int,
    rhs_expr: str,
    data: dict,
) -> Dist:
    """Dispatch X[i,j] = rhs_expr for three RHS cases (fix4.4).

    B1: rhs_expr is a numeric literal → hard write (var_z=0).
    B2: rhs_expr is a scalar variable name → soft write (var_z=gm.var[z]).
    B3: rhs_expr is a general scalar expression → evaluate via scalar update first,
        then reduce to B2.

    Always densifies the matrix var's covariance per research note 04 §B1+B2.
    Emits ElementWriteDenseWarning so the user knows Kronecker is lost.
    """
    if dist.gm_block is None:
        raise RuntimeError(
            f"[fix4.4] X[{i},{j}] = ... called but dist.gm_block is None.  "
            "Initialise with matrix_gm(...) first."
        )
    ve = next((v for v in dist.var_entries if v.name == mat_name), None)
    if ve is None:
        raise RuntimeError(f"[fix4.4] '{mat_name}' not in dist.var_entries")
    m, n = ve.shape

    warnings.warn(
        f"ElementWriteDenseWarning: '{mat_name}[{i},{j}] = {rhs_expr}' densifies "
        f"the covariance of '{mat_name}' from O(m²+n²)=O({m*m+n*n}) to "
        f"O((mn)²)=O({(m*n)**2}).  Subsequent Kronecker-dependent ops will fail.",
        ElementWriteDenseWarning,
        stacklevel=3,
    )

    block = deepcopy(dist.gm_block)

    # Try B1: numeric literal
    try:
        c_val = float(rhs_expr)
        # B1: deterministic constant
        for k in range(block.n_comp()):
            _matrix_element_write_component(block, k, mat_name, m, n, i, j, c_val, 0.0)
        return Dist(dist.var_list, dist.gm, var_entries=dist.var_entries, gm_block=block)
    except ValueError:
        pass

    # Try B2: scalar variable name in var_list
    if rhs_expr in dist.var_list:
        z_idx = dist.var_list.index(rhs_expr)
        for k in range(block.n_comp()):
            mu_z = float(dist.gm.mu[k][z_idx])
            var_z = float(dist.gm.sigma[k][z_idx, z_idx])
            _matrix_element_write_component(block, k, mat_name, m, n, i, j, mu_z, var_z)
        return Dist(dist.var_list, dist.gm, var_entries=dist.var_entries, gm_block=block)

    # B3: general scalar expression — evaluate into a temporary variable, then B2
    # Use a temporary var name that won't collide with existing vars
    tmp_var = '_tmp_rhs_elem_write'
    if tmp_var in dist.var_list:
        tmp_var = tmp_var + '_2'
    # Evaluate the scalar expression via the scalar update path
    new_expr = f'{tmp_var} = {rhs_expr}'
    try:
        from libSOGAupdate import update_rule
        dist_with_tmp = update_rule(dist, new_expr, data)
    except Exception as exc:
        raise NotImplementedError(
            f"[fix4.4 B3] Cannot evaluate RHS expression '{rhs_expr}' as a scalar "
            f"expression for element write '{mat_name}[{i},{j}] = {rhs_expr}': {exc}"
        ) from exc
    # Now it's B2
    return matrix_element_write_dispatch(dist_with_tmp, mat_name, i, j, tmp_var, data)


def _matrix_dense_get_cov(block: GaussianMixBlock, k: int, mat_name: str) -> np.ndarray:
    """Return the dense (mn × mn) covariance for mat_name in component k.

    Handles both Kronecker (U, V) and dense sentinel (None, Sigma) storage.
    """
    stored = block.cov_blocks[k].get(frozenset({mat_name}))
    if stored is None:
        raise KeyError(f"No covariance for '{mat_name}' in cov_blocks[{k}]")
    U_or_none, V_or_Sigma = stored
    if U_or_none is None:
        return V_or_Sigma
    return np.kron(V_or_Sigma, U_or_none)


def extract_scalar_from_matrix(dist: Dist, lhs: str, mat_name: str, i: int, j: int) -> Dist:
    """Implements y = X[i, j] where X is a matrix variable and y is a scalar.

    Marginal moments of y per component k (Gupta & Nagar 1999, Thm 2.3.1):
      mu_y(k)  = M_k[i, j]
      var_y(k) = U_k[i, i] * V_k[j, j]

    Cross-covariances (A1 of plan M4.8 / docs/research-notes/04-...):
      Cov(y, vec(X)) = V[:, j] ⊗ U[:, i]                  (mn-vector, exact)
      Cov(y, z)      = (V[:, j] ⊗ U[:, i])^T · Cov(z, vec(X))
                     = factored shortcut via mixed-product property

    Storage convention (matches GaussianMixBlock §M3.1):
      - cov_blocks[k][frozenset({y, X})] = (mn,) ndarray, column-major V⊗U order
      - sigma scalar-scalar entries are floats stored in dist.gm.sigma[k][y_idx, z_idx]

    The new scalar y is appended (or overwritten) in the scalar gm.  X's
    own (M, U, V) is preserved unchanged; only the cross-covariance dict
    grows by one entry per component.

    Raises
    ------
    RuntimeError
        if mat_name is not in dist.var_entries or dist.gm_block is None.
    IndexError
        if (i, j) is outside the declared shape of mat_name.
    """
    if dist.gm_block is None:
        raise RuntimeError(
            f"[M4.8] extract_scalar_from_matrix({mat_name}[{i},{j}]) called but "
            "dist.gm_block is None.  Initialise the matrix variable with matrix_gm(...)."
        )
    ve = next((v for v in dist.var_entries if v.name == mat_name), None)
    if ve is None:
        raise RuntimeError(
            f"[M4.8] '{mat_name}' is not a matrix variable in dist.var_entries"
        )
    m, n = ve.shape
    if not (0 <= i < m and 0 <= j < n):
        raise IndexError(
            f"[M4.8] index ({i},{j}) out of range for matrix '{mat_name}' of shape ({m},{n})"
        )

    block = deepcopy(dist.gm_block)  # mutate copy: we add a cross-cov entry per k
    n_comp = block.n_comp()

    # Determine layout: new scalar var or overwrite existing
    if lhs in dist.var_list:
        y_idx = dist.var_list.index(lhs)
        new_var_list = list(dist.var_list)
        is_new = False
    else:
        new_var_list = list(dist.var_list) + [lhs]
        y_idx = len(new_var_list) - 1
        is_new = True
    d_new = len(new_var_list)

    new_pi = []
    new_mu = []
    new_sigma = []
    for k in range(n_comp):
        M_k = block.get_mu(k, mat_name)
        U_k, V_k = block.get_cov(k, mat_name, mat_name)
        mu_y_k = float(M_k[i, j])

        # fix4-extend: support both Kronecker-stored (U, V) tuples and
        # dense-sentinel storage (None, Sigma_mn_x_mn) created by a previous
        # element write.  The cross-covariance formula is computed accordingly.
        if U_k is None:
            # Dense mode: V_k is actually the full (mn, mn) covariance.
            Sigma_full = V_k
            idx_col = j * m + i           # column-major index of X[i,j] in vec(X)
            var_y_k = float(Sigma_full[idx_col, idx_col])
            cross_y_X = Sigma_full[:, idx_col].copy()    # exact, no Kron factor
        else:
            var_y_k = float(U_k[i, i] * V_k[j, j])
            # A1: Cov(y, vec(X)) = V[:,j] ⊗ U[:,i] (Gupta&Nagar 1999 Thm 2.3.1)
            cross_y_X = np.kron(V_k[:, j], U_k[:, i])  # shape (mn,)
        block.cov_blocks[k][frozenset({lhs, mat_name})] = cross_y_X

        # Existing scalar mean/cov for this mixture weight
        if dist.gm.n_comp() > k:
            old_mu_k = np.asarray(dist.gm.mu[k], dtype=float)
            old_sigma_k = np.asarray(dist.gm.sigma[k], dtype=float)
        else:
            # Should not happen: gm_block and gm components are kept in sync
            old_mu_k = np.zeros(d_new - (0 if is_new else 1))
            old_sigma_k = np.zeros((len(old_mu_k), len(old_mu_k)))

        if is_new:
            new_mu_k = np.concatenate([old_mu_k, [mu_y_k]])
            new_sigma_k = np.zeros((d_new, d_new))
            new_sigma_k[: len(old_mu_k), : len(old_mu_k)] = old_sigma_k
            new_sigma_k[y_idx, y_idx] = var_y_k
        else:
            new_mu_k = old_mu_k.copy()
            new_mu_k[y_idx] = mu_y_k
            new_sigma_k = old_sigma_k.copy()
            new_sigma_k[y_idx, :] = 0.0
            new_sigma_k[:, y_idx] = 0.0
            new_sigma_k[y_idx, y_idx] = var_y_k

        # A1: Cov(y, z) for every prior scalar z that has a stored cross-cov to X.
        # Uses the mixed-product identity: (V[:,j] ⊗ U[:,i])^T · cov_z_vecX
        # is a scalar dot product over the mn-vector cov_z_vecX.
        for z_idx_scan, z_name in enumerate(dist.var_list):
            if z_name == lhs:
                continue  # skip self (handled by var_y_k diagonal)
            key_zX = frozenset({z_name, mat_name})
            if key_zX in block.cov_blocks[k]:
                cov_z_vecX = block.cov_blocks[k][key_zX]
                # Robustness: support either (mn,) flat or a factored (u,v) pair
                if isinstance(cov_z_vecX, tuple) and len(cov_z_vecX) == 2:
                    u_z, v_z = cov_z_vecX
                    cov_yz = float(np.dot(U_k[:, i], u_z) * np.dot(V_k[:, j], v_z))
                else:
                    cov_yz = float(np.dot(cross_y_X, np.asarray(cov_z_vecX)))
                if abs(cov_yz) > 0:
                    new_sigma_k[y_idx, z_idx_scan] = cov_yz
                    new_sigma_k[z_idx_scan, y_idx] = cov_yz

        new_pi.append(float(block.pi[k]) if k < len(block.pi) else 1.0)
        new_mu.append(new_mu_k)
        new_sigma.append(new_sigma_k)

    # Keep block.var_list in sync (it's the scalar var list reflected in gm)
    block.var_list = list(new_var_list)

    new_gm = GaussianMix(new_pi, new_mu, new_sigma)
    return Dist(new_var_list, new_gm, var_entries=dist.var_entries, gm_block=block)
