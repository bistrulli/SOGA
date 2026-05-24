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
import warnings
from copy import deepcopy
from typing import List, Tuple

import numpy as np

from libSOGAshared import Dist, VarEntry
from libSOGAsharedMatrix import GaussianMixBlock, _enforce_psd_kron_factors
from libMatrixGaussian import _nearest_kronecker

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
    """Apply PSD enforcement to (U, V) Kronecker factors for var_name in component k."""
    U, V = block.get_cov(k, var_name, var_name)
    U, V = _enforce_psd_kron_factors(U, V)
    block.cov_blocks[k][frozenset({var_name})] = (U, V)


# ---------------------------------------------------------------------------
# Expression parser (M4.1)
# ---------------------------------------------------------------------------

def _parse_matrix_expr(expr: str) -> dict:
    """Parse a matrix assignment expression into an op-type dict."""
    body = expr.split("=", 1)[1].strip()

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
            raise NotImplementedError(
                f"[M4.1] Random x random matmul not supported in v1. See plan §M4.1."
            )
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
