"""
libMatrixTruncate.py — M5 matrix observe / truncation operations for SOGA.

Implements truncate_matrix and all per-constraint handlers:
  _truncate_matrix_element_ineq   (M5.2): X[i,j] op c
  _truncate_matrix_row_sum_ineq   (M5.3): row_sum(X,i) op c  (sum_j X[i,j] op c)
  _truncate_matrix_col_sum_ineq   (M5.4): col_sum(X,j) op c  (sum_i X[i,j] op c)

Strategy: DENSE (default for v1).
  Materialise the (mn x mn) dense covariance, apply the standard rank-1
  conditional Gaussian update, then optionally re-project to Kronecker via
  nearest-Kronecker if requested.  For Lishan-class programs (rare observe) the
  DENSE strategy is exact and acceptable.

Memory budget guard:
  If K * (mn)^2 * 8 bytes would exceed --matrix-dense-budget-mb (default 1024),
  auto-downgrade to PROJECT and emit KroneckerDenseMemoryWarning.

Log-weight update (M5.5):
  log_pi[k] += log(P) after each truncation step; renormalise at output time.
  Prevents underflow at large mn * many observes.

Plan reference: §M5 of plan/2026-05-22-matrix-gm-lishan.md
"""

from __future__ import annotations

import logging
import math
import os
import re
import warnings
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from libSOGAshared import Dist, GaussianMix, VarEntry, NumericalError, prob_tol
from libSOGAsharedMatrix import GaussianMixBlock, _enforce_psd_kron_factors
from libMatrixGaussian import _nearest_kronecker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constant: default memory budget for DENSE materialisation
# ---------------------------------------------------------------------------

MATRIX_DENSE_BUDGET_MB: int = int(os.environ.get("MATRIX_DENSE_BUDGET_MB", "1024"))


# ---------------------------------------------------------------------------
# Warning classes
# ---------------------------------------------------------------------------

class KroneckerDenseMemoryWarning(UserWarning):
    """Emitted when DENSE materialisation would exceed the memory budget.

    The system automatically falls back to PROJECT strategy and logs this
    warning with the requested vs budget values.
    """


class KroneckerProjectWarning(UserWarning):
    """Emitted after PROJECT strategy is applied, reporting the projection error."""


# ---------------------------------------------------------------------------
# Constraint classifier
# ---------------------------------------------------------------------------

# Patterns:
#   ELEMENT_INEQ  : X[i,j]  op  c
#   ROW_SUM_INEQ  : row_sum(X, i)  op  c
#   COL_SUM_INEQ  : col_sum(X, j)  op  c

_RE_ELEMENT = re.compile(
    r'^\s*(\w+)\s*\[\s*(\d+|\w+)\s*,\s*(\d+|\w+)\s*\]\s*(>=|<=|>|<)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$'
)
_RE_ROW_SUM = re.compile(
    r'^\s*row_sum\s*\(\s*(\w+)\s*,\s*(\d+|\w+)\s*\)\s*(>=|<=|>|<)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$'
)
_RE_COL_SUM = re.compile(
    r'^\s*col_sum\s*\(\s*(\w+)\s*,\s*(\d+|\w+)\s*\)\s*(>=|<=|>|<)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$'
)
_RE_TRACE = re.compile(
    r'^\s*trace\s*\(\s*(\w+)\s*\)\s*(>=|<=|>|<)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$'
)


def _resolve_index(tok: str, data: Optional[dict] = None) -> int:
    """Resolve an index token to an integer.

    fix1.2: accepts both numeric strings (e.g. '0', '3') and identifier
    strings that refer to loop-counter variables resolved via data[tok][0].
    This matches SOGA's loop-counter convention: data[name] = [current_value].

    Parameters
    ----------
    tok  : str — the raw token from the regex group (digit string or identifier)
    data : dict | None — the program data dict (required for identifier tokens)

    Raises
    ------
    KeyError if tok is an identifier not present in data (or data is None).
    """
    if tok.lstrip('-').isdigit():
        return int(tok)
    # Identifier: resolve via data dict (loop-counter convention)
    if data is not None and tok in data and data[tok][0] is not None:
        return int(data[tok][0])
    raise KeyError(
        f"[fix1.2] Cannot resolve index token '{tok}': not a numeric literal "
        f"and not found in data dict (data={'<None>' if data is None else list(data.keys())})."
    )


def _classify_constraint(trunc: str, mat_var: str, data: Optional[dict] = None):
    """Classify a matrix constraint string.

    Returns a dict with keys:
        type  : 'ELEMENT_INEQ' | 'ROW_SUM_INEQ' | 'COL_SUM_INEQ'
        ...   : type-specific fields (i, j, direction, threshold)
    Raises NotImplementedError for TRACE_INEQ or unrecognised forms.

    fix1.2: data parameter added to allow resolving loop-variable indices via
    _resolve_index (e.g. X[i,j] op c where i,j are loop counters).
    """
    m = _RE_ELEMENT.match(trunc)
    if m:
        var, i_s, j_s, op, c_s = m.groups()
        return {"type": "ELEMENT_INEQ", "var": var,
                "i": _resolve_index(i_s, data), "j": _resolve_index(j_s, data),
                "direction": op, "threshold": float(c_s)}

    m = _RE_ROW_SUM.match(trunc)
    if m:
        var, i_s, op, c_s = m.groups()
        return {"type": "ROW_SUM_INEQ", "var": var,
                "i": _resolve_index(i_s, data),
                "direction": op, "threshold": float(c_s)}

    m = _RE_COL_SUM.match(trunc)
    if m:
        var, j_s, op, c_s = m.groups()
        return {"type": "COL_SUM_INEQ", "var": var,
                "j": _resolve_index(j_s, data),
                "direction": op, "threshold": float(c_s)}

    m = _RE_TRACE.match(trunc)
    if m:
        raise NotImplementedError(
            f"[M5] trace(X) observe is not supported in v1.  "
            "See plan/2026-05-22-matrix-gm-lishan.md §M5.1: "
            "'TRACE_INEQ (raise NotImplementedError in v1)'."
        )

    raise NotImplementedError(
        f"[M5] Unrecognised matrix constraint pattern: '{trunc}'.  "
        "Supported forms: X[i,j] op c, row_sum(X,i) op c, col_sum(X,j) op c.  "
        "See plan/2026-05-22-matrix-gm-lishan.md §M5.1."
    )


# ---------------------------------------------------------------------------
# 1D truncated normal moments helper (re-exports from libSOGAtruncate)
# ---------------------------------------------------------------------------

def _tnorm1d(mu_s: float, var_s: float, c: float, direction: str):
    """Wrapper around libSOGAtruncate._truncated_normal_moments_1d.

    Returns (m_hat, v_hat, P).  Imported lazily to avoid circular import.
    """
    from libSOGAtruncate import _truncated_normal_moments_1d
    return _truncated_normal_moments_1d(mu_s, var_s, c, direction)


# ---------------------------------------------------------------------------
# Memory budget guard
# ---------------------------------------------------------------------------

def _check_memory_budget(m: int, n: int, n_comp: int, budget_mb: int) -> bool:
    """Return True if DENSE strategy is within budget, False → use PROJECT.

    Emits KroneckerDenseMemoryWarning if budget would be exceeded.
    """
    cost_bytes = n_comp * (m * n) ** 2 * 8
    cost_mb = cost_bytes / (1024 ** 2)
    if cost_mb > budget_mb:
        warnings.warn(
            f"KroneckerDenseMemoryWarning: DENSE materialisation would require "
            f"{cost_mb:.1f} MB (budget {budget_mb} MB) for {n_comp} components "
            f"of a {m}x{n} matrix.  Auto-downgrading to PROJECT strategy.",
            KroneckerDenseMemoryWarning,
            stacklevel=4,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Core rank-1 conditional Gaussian update on dense joint
# ---------------------------------------------------------------------------

def _rank1_cond_update(
    M_vec: np.ndarray,
    Sigma: np.ndarray,
    a_vec: np.ndarray,
    c: float,
    direction: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rank-1 conditional Gaussian update for a linear constraint a^T x op c.

    Standard formulas (see libSOGAtruncate._ineq_func_sparse for derivation):
        mu_s  = a^T M_vec
        var_s = a^T Sigma a
        g     = Sigma a
        m_hat, v_hat, P = 1D truncated normal moments
        M_new = M_vec + g * (m_hat - mu_s) / var_s
        S_new = Sigma - outer(g, g) * (1 - v_hat/var_s) / var_s

    Parameters
    ----------
    M_vec   : vec(M) in V⊗U column-major order, shape (mn,)
    Sigma   : dense covariance (mn, mn)
    a_vec   : selector vector, shape (mn,)
    c       : threshold scalar
    direction : one of '>', '>=', '<', '<='

    Returns
    -------
    M_new   : updated mean vector (mn,)
    S_new   : updated covariance (mn, mn)
    P       : tail probability (0 → component should be dropped)
    """
    mu_s = float(a_vec @ M_vec)
    g = Sigma @ a_vec                  # shape (mn,)
    var_s = float(a_vec @ g)

    m_hat, v_hat, P = _tnorm1d(mu_s, var_s, c, direction)

    if P < prob_tol:
        return M_vec.copy(), Sigma.copy(), float(P)

    delta = m_hat - mu_s
    M_new = M_vec + g * (delta / var_s)
    S_new = Sigma - np.outer(g, g) * ((1.0 - v_hat / var_s) / var_s)
    return M_new, S_new, float(P)


# ---------------------------------------------------------------------------
# Strategy: extract dense cov, update, return (M_new, Sigma_new)
# ---------------------------------------------------------------------------

def _dense_update_component(
    block: GaussianMixBlock,
    k: int,
    mat_var: str,
    a_vec: np.ndarray,
    c: float,
    direction: str,
    m: int,
    n: int,
    use_project: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:
    """Apply rank-1 update to one mixture component.

    Returns
    -------
    M_new_mat  : updated mean matrix (m, n)
    U_new      : updated U factor (m, m) — None if dense stored
    V_new      : updated V factor (n, n) — None if dense stored
    Sigma_new  : updated dense covariance (mn, mn) if DENSE strategy, else None
    P          : tail probability

    Handles both Kronecker-stored (U, V) and already-dense-stored (None, Sigma)
    components.  A dense-stored component was created by a previous DENSE
    truncation step; it is identified by the sentinel (None, Sigma_dense) tuple
    stored in cov_blocks.
    """
    stored = block.get_cov(k, mat_var, mat_var)
    U_k, V_k = stored

    if U_k is None:
        # Already densified by a previous DENSE truncation: V_k holds Sigma
        Sigma = V_k
    else:
        Sigma = np.kron(V_k, U_k)                    # (mn, mn) dense, V⊗U

    M_k = block.mu_blocks[k][mat_var]                # (m, n)
    M_vec = M_k.flatten("F")                         # column-major, V⊗U

    M_new_vec, Sigma_new, P = _rank1_cond_update(M_vec, Sigma, a_vec, c, direction)

    M_new_mat = M_new_vec.reshape((m, n), order="F")

    if use_project:
        # Nearest-Kronecker re-projection
        U_new, V_new = _nearest_kronecker(Sigma_new, m, n)
        U_new, V_new = _enforce_psd_kron_factors(U_new, V_new)
        return M_new_mat, U_new, V_new, None, P
    else:
        # Store dense — return Sigma_new, mark U/V as None sentinel
        return M_new_mat, None, None, Sigma_new, P


# ---------------------------------------------------------------------------
# M5.2 — Element inequality: X[i,j] op c
# ---------------------------------------------------------------------------

def _truncate_matrix_element_ineq(
    block: GaussianMixBlock,
    mat_var: str,
    m: int,
    n: int,
    i: int,
    j: int,
    c: float,
    direction: str,
    use_project: bool,
) -> Tuple[GaussianMixBlock, float]:
    """Rank-1 conditional update for X[i,j] op c.

    Selector vector a: in column-major (V⊗U) ordering,
        index(i, j) = j * m + i
    so a = e_{j*m + i} in R^{mn}.

    Mean update (numerically verified in plan §M5.2):
        delta/var_s * outer(U[:,i], V[:,j]) — reshaped from the dense formula.

    Returns updated GaussianMixBlock and norm_factor (sum of surviving weights).
    """
    mn = m * n
    a_vec = np.zeros(mn)
    a_vec[j * m + i] = 1.0            # column-major indexing

    use_dense = _check_memory_budget(m, n, block.n_comp(), MATRIX_DENSE_BUDGET_MB)
    if use_dense and not use_project:
        pass  # use DENSE
    elif not use_dense:
        use_project = True

    new_pi = []
    new_log_pi = [] if block.log_pi is not None else None
    new_mu_blocks = []
    new_cov_blocks = []

    for k in range(block.n_comp()):
        M_new_mat, U_new, V_new, Sigma_dense, P = _dense_update_component(
            block, k, mat_var, a_vec, c, direction, m, n, use_project
        )

        if P < prob_tol:
            logger.debug(
                "Component %d dropped (P=%.2e < prob_tol) in element_ineq %s[%d,%d]",
                k, P, mat_var, i, j,
            )
            continue

        new_pi.append(block.pi[k] * P)
        if new_log_pi is not None:
            new_log_pi.append(block.log_pi[k] + math.log(P))

        mu_k = deepcopy(block.mu_blocks[k])
        mu_k[mat_var] = M_new_mat
        cov_k = deepcopy(block.cov_blocks[k])

        if use_project or U_new is not None:
            cov_k[frozenset({mat_var})] = (U_new, V_new)
        else:
            # DENSE: store as "dense_block" sentinel tuple (None, Sigma_dense)
            cov_k[frozenset({mat_var})] = (None, Sigma_dense)

        new_mu_blocks.append(mu_k)
        new_cov_blocks.append(cov_k)

    norm_factor = sum(new_pi)
    if norm_factor > prob_tol:
        norm_pi = [p / norm_factor for p in new_pi]
    else:
        # Degenerate: keep first component with zero weight
        norm_pi = [0.0]
        if not new_mu_blocks:
            new_mu_blocks = [deepcopy(block.mu_blocks[0])]
            new_cov_blocks = [deepcopy(block.cov_blocks[0])]
        if new_log_pi is not None:
            new_log_pi = [float("-inf")]

    result = GaussianMixBlock(
        var_list=block.var_list,
        var_entries=block.var_entries,
        pi=norm_pi,
        mu_blocks=new_mu_blocks,
        cov_blocks=new_cov_blocks,
        log_pi=new_log_pi,
    )
    return result, norm_factor


# ---------------------------------------------------------------------------
# M5.3 — Row sum inequality: sum_j X[i,j] op c
# ---------------------------------------------------------------------------

def _truncate_matrix_row_sum_ineq(
    block: GaussianMixBlock,
    mat_var: str,
    m: int,
    n: int,
    i: int,
    c: float,
    direction: str,
    use_project: bool,
) -> Tuple[GaussianMixBlock, float]:
    """Rank-1 conditional update for row_sum(X, i) op c.

    The linear functional: s = sum_j X[i,j] = (e_i ⊗ 1_n)^T vec(X) in row-major,
    or equivalently s = (1_n^T ⊗ e_i^T) vec(X).
    In column-major (V⊗U convention): the mn-vector a has
        a[j*m + i] = 1  for all j = 0..n-1

    mu_s = (M @ 1_n)[i] = sum_j M[i,j]
    var_s = U[i,i] * (1_n^T V 1_n)

    Correct mean update (verified, plan §M5.3):
        M_new = M + (delta/var_s) * outer(U[:,i], V @ 1_n)
    """
    mn = m * n
    a_vec = np.zeros(mn)
    for j_idx in range(n):
        a_vec[j_idx * m + i] = 1.0   # column-major: index(i, j) = j*m + i

    use_dense = _check_memory_budget(m, n, block.n_comp(), MATRIX_DENSE_BUDGET_MB)
    if not use_dense:
        use_project = True

    new_pi = []
    new_log_pi = [] if block.log_pi is not None else None
    new_mu_blocks = []
    new_cov_blocks = []

    for k in range(block.n_comp()):
        M_new_mat, U_new, V_new, Sigma_dense, P = _dense_update_component(
            block, k, mat_var, a_vec, c, direction, m, n, use_project
        )

        if P < prob_tol:
            logger.debug(
                "Component %d dropped (P=%.2e < prob_tol) in row_sum_ineq %s[%d,:]",
                k, P, mat_var, i,
            )
            continue

        new_pi.append(block.pi[k] * P)
        if new_log_pi is not None:
            new_log_pi.append(block.log_pi[k] + math.log(P))

        mu_k = deepcopy(block.mu_blocks[k])
        mu_k[mat_var] = M_new_mat
        cov_k = deepcopy(block.cov_blocks[k])

        if use_project or U_new is not None:
            cov_k[frozenset({mat_var})] = (U_new, V_new)
        else:
            cov_k[frozenset({mat_var})] = (None, Sigma_dense)

        new_mu_blocks.append(mu_k)
        new_cov_blocks.append(cov_k)

    norm_factor = sum(new_pi)
    if norm_factor > prob_tol:
        norm_pi = [p / norm_factor for p in new_pi]
    else:
        norm_pi = [0.0]
        if not new_mu_blocks:
            new_mu_blocks = [deepcopy(block.mu_blocks[0])]
            new_cov_blocks = [deepcopy(block.cov_blocks[0])]
        if new_log_pi is not None:
            new_log_pi = [float("-inf")]

    result = GaussianMixBlock(
        var_list=block.var_list,
        var_entries=block.var_entries,
        pi=norm_pi,
        mu_blocks=new_mu_blocks,
        cov_blocks=new_cov_blocks,
        log_pi=new_log_pi,
    )
    return result, norm_factor


# ---------------------------------------------------------------------------
# M5.4 — Column sum inequality: sum_i X[i,j] op c
# ---------------------------------------------------------------------------

def _truncate_matrix_col_sum_ineq(
    block: GaussianMixBlock,
    mat_var: str,
    m: int,
    n: int,
    j: int,
    c: float,
    direction: str,
    use_project: bool,
) -> Tuple[GaussianMixBlock, float]:
    """Rank-1 conditional update for col_sum(X, j) op c.

    The linear functional: s = sum_i X[i,j] = (1_m^T ⊗ e_j^T) vec(X).
    Wait — in column-major (V⊗U): the mn-vector a has
        a[j*m + i] = 1  for all i = 0..m-1

    mu_s = (1_m^T @ M)[j] = sum_i M[i,j]
    var_s = (1_m^T U 1_m) * V[j,j]

    Correct mean update (verified, plan §M5.4):
        M_new = M + (delta/var_s) * outer(U @ 1_m, V[:,j])
    """
    mn = m * n
    a_vec = np.zeros(mn)
    for i_idx in range(m):
        a_vec[j * m + i_idx] = 1.0   # column-major: index(i, j) = j*m + i

    use_dense = _check_memory_budget(m, n, block.n_comp(), MATRIX_DENSE_BUDGET_MB)
    if not use_dense:
        use_project = True

    new_pi = []
    new_log_pi = [] if block.log_pi is not None else None
    new_mu_blocks = []
    new_cov_blocks = []

    for k in range(block.n_comp()):
        M_new_mat, U_new, V_new, Sigma_dense, P = _dense_update_component(
            block, k, mat_var, a_vec, c, direction, m, n, use_project
        )

        if P < prob_tol:
            logger.debug(
                "Component %d dropped (P=%.2e < prob_tol) in col_sum_ineq %s[:,%d]",
                k, P, mat_var, j,
            )
            continue

        new_pi.append(block.pi[k] * P)
        if new_log_pi is not None:
            new_log_pi.append(block.log_pi[k] + math.log(P))

        mu_k = deepcopy(block.mu_blocks[k])
        mu_k[mat_var] = M_new_mat
        cov_k = deepcopy(block.cov_blocks[k])

        if use_project or U_new is not None:
            cov_k[frozenset({mat_var})] = (U_new, V_new)
        else:
            cov_k[frozenset({mat_var})] = (None, Sigma_dense)

        new_mu_blocks.append(mu_k)
        new_cov_blocks.append(cov_k)

    norm_factor = sum(new_pi)
    if norm_factor > prob_tol:
        norm_pi = [p / norm_factor for p in new_pi]
    else:
        norm_pi = [0.0]
        if not new_mu_blocks:
            new_mu_blocks = [deepcopy(block.mu_blocks[0])]
            new_cov_blocks = [deepcopy(block.cov_blocks[0])]
        if new_log_pi is not None:
            new_log_pi = [float("-inf")]

    result = GaussianMixBlock(
        var_list=block.var_list,
        var_entries=block.var_entries,
        pi=norm_pi,
        mu_blocks=new_mu_blocks,
        cov_blocks=new_cov_blocks,
        log_pi=new_log_pi,
    )
    return result, norm_factor


# ---------------------------------------------------------------------------
# M5.1 — Main dispatcher: truncate_matrix
# ---------------------------------------------------------------------------

def truncate_matrix(
    dist: Dist,
    trunc: str,
    data: dict,
    mat_var: str,
    use_project: bool = False,
) -> Tuple[float, Dist]:
    """Full dispatcher for matrix-variable truncation (observe/condition).

    Classifies the constraint type and delegates to the appropriate handler.
    Initialises log-weight infrastructure (M5.5) on first call if not present.

    Parameters
    ----------
    dist : Dist
        Current joint distribution with non-empty var_entries / gm_block.
    trunc : str
        Raw truncation expression string.
    data : dict
        Program data dictionary.
    mat_var : str
        Name of the matrix variable referenced in trunc.
    use_project : bool
        If True, use PROJECT strategy (nearest-Kronecker re-projection after
        dense update).  Default False → DENSE strategy.

    Returns
    -------
    (norm_factor, new_dist) matching the scalar truncate() API.
    """
    if dist.gm_block is None:
        raise ValueError(
            f"[M5] truncate_matrix called but dist.gm_block is None.  "
            "The distribution must have been initialised with a GaussianMixBlock."
        )

    # Find the matrix variable entry for shape
    ve = next((v for v in dist.var_entries if v.name == mat_var), None)
    if ve is None:
        raise KeyError(f"[M5] Matrix variable '{mat_var}' not found in dist.var_entries")
    m, n = ve.shape

    block = dist.gm_block

    # M5.5: initialise log-weight infrastructure if not yet present
    if block.log_pi is None:
        block.log_pi = [
            math.log(p) if p > 0 else float("-inf")
            for p in block.pi
        ]

    # Classify constraint — pass data so loop-variable indices can be resolved (fix1.2)
    constraint = _classify_constraint(trunc, mat_var, data)
    ct = constraint["type"]

    if ct == "ELEMENT_INEQ":
        i, j = constraint["i"], constraint["j"]
        direction = constraint["direction"]
        threshold = constraint["threshold"]
        new_block, norm_factor = _truncate_matrix_element_ineq(
            block, mat_var, m, n, i, j, threshold, direction, use_project
        )

    elif ct == "ROW_SUM_INEQ":
        i = constraint["i"]
        direction = constraint["direction"]
        threshold = constraint["threshold"]
        new_block, norm_factor = _truncate_matrix_row_sum_ineq(
            block, mat_var, m, n, i, threshold, direction, use_project
        )

    elif ct == "COL_SUM_INEQ":
        j = constraint["j"]
        direction = constraint["direction"]
        threshold = constraint["threshold"]
        new_block, norm_factor = _truncate_matrix_col_sum_ineq(
            block, mat_var, m, n, j, threshold, direction, use_project
        )

    else:
        raise NotImplementedError(
            f"[M5] Constraint type '{ct}' not handled in v1.  "
            "See plan/2026-05-22-matrix-gm-lishan.md §M5.1."
        )

    new_dist = Dist(
        dist.var_list,
        dist.gm,               # scalar GM unchanged (observe is on matrix var)
        var_entries=dist.var_entries,
        gm_block=new_block,
    )
    return norm_factor, new_dist
