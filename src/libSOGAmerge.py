# Contains the functions for computing the resulting distribution when a merge is invoked according to the following dependencies.

# SOGA (defined in SOGA.py)
# |- merge
#     |- prune
#        |- classic_prune
#        |   |- compute_matrix_mean
#        |   |- dist
#        |   |- merge_comp
#        |- ranking_prune


from copy import deepcopy as _deepcopy
from libSOGAshared import *

import warnings as _warnings
import logging as _logging
import os as _os

_logger = _logging.getLogger(__name__)

# Memory budget for merged matrix-GM blocks (MB). Merge can multiply component
# count (K_then + K_else), so we guard before the concatenation.
MATRIX_MERGE_BUDGET_MB: int = int(_os.environ.get("MATRIX_MERGE_BUDGET_MB", "1024"))


class MatrixMergeMemoryWarning(UserWarning):
    """Emitted when merged component count would exceed the memory budget."""


def _merge_matrix_gm_blocks(list_dist, list_p, current_p):
    """Concatenate gm_block components from N branches with weight rescaling.

    Implements fix2 per research note 05 §Problem 1:
    - Each branch i with weight p_i and dist_i has K_i components.
    - Appended with weights p_i · pi_k (renormalised by current_p at the end).
    - mu_blocks and cov_blocks are concatenated.
    - Variables present in only one branch are supplemented from the first
      branch where the variable exists (prior-snapshot rule, fix2.2).

    Returns the merged GaussianMixBlock.
    """
    from libSOGAsharedMatrix import GaussianMixBlock

    merged_pi = []
    merged_mu_blocks = []
    merged_cov_blocks = []

    # Collect the union of all var_entries across branches
    all_ve_names = {}
    for dist_i in list_dist:
        if dist_i.gm_block is not None:
            for ve in dist_i.gm_block.var_entries:
                if ve.name not in all_ve_names:
                    all_ve_names[ve.name] = ve
    all_var_entries = list(all_ve_names.values())

    # Collect union of scalar var_list
    all_scalar_vars = list(list_dist[0].var_list) if list_dist else []

    for dist_i, p_i in zip(list_dist, list_p):
        if p_i <= 0:
            continue
        block_i = dist_i.gm_block
        if block_i is None:
            # Branch has no matrix state — skip matrix contribution
            continue
        n_comp_i = block_i.n_comp()
        for k in range(n_comp_i):
            merged_pi.append(p_i * block_i.pi[k])
            mu_k = _deepcopy(block_i.mu_blocks[k])
            cov_k = _deepcopy(block_i.cov_blocks[k])

            # fix2.2: for any matrix var absent in this branch but present in
            # another, inject a zero-state placeholder so downstream ops don't
            # crash. The semantics: "variable in only one branch → carry prior"
            # — in this case we carry zeros as a conservative placeholder and
            # log a warning, since the true prior would require back-tracking.
            for ve in all_var_entries:
                if ve.name not in mu_k:
                    _logger.debug(
                        "[fix2.2] Matrix var '%s' absent in branch; injecting zero placeholder.",
                        ve.name,
                    )
                    m_v, n_v = ve.shape
                    mu_k[ve.name] = np.zeros((m_v, n_v))
                    cov_k[frozenset({ve.name})] = (np.zeros((m_v, m_v)), np.zeros((n_v, n_v)))
                    for sv in all_scalar_vars:
                        cov_k[frozenset({sv, ve.name})] = np.zeros(m_v * n_v)

            merged_mu_blocks.append(mu_k)
            merged_cov_blocks.append(cov_k)

    if not merged_pi:
        # Degenerate: return first branch block as-is
        return list_dist[0].gm_block

    # Renormalise weights
    total = sum(merged_pi)
    if total > 0:
        merged_pi = [p / total for p in merged_pi]

    # Memory budget check (fix2.4)
    if all_var_entries:
        m_v, n_v = all_var_entries[0].shape
        n_comp_merged = len(merged_pi)
        cost_bytes = n_comp_merged * (m_v * n_v) ** 2 * 8
        cost_mb = cost_bytes / (1024 ** 2)
        if cost_mb > MATRIX_MERGE_BUDGET_MB:
            _warnings.warn(
                f"MatrixMergeMemoryWarning: merged {n_comp_merged} components "
                f"for {m_v}x{n_v} matrix would use {cost_mb:.1f} MB "
                f"(budget {MATRIX_MERGE_BUDGET_MB} MB).  "
                "Trigger ranking_prune to reduce.",
                MatrixMergeMemoryWarning,
                stacklevel=4,
            )

    # Use var_list and var_entries from first non-None block
    ref_block = next((d.gm_block for d in list_dist if d.gm_block is not None), None)
    if ref_block is None:
        return None

    return GaussianMixBlock(
        var_list=ref_block.var_list,
        var_entries=all_var_entries,
        pi=merged_pi,
        mu_blocks=merged_mu_blocks,
        cov_blocks=merged_cov_blocks,
    )


def merge(list_dist):
    """
    Given a list of couples (p,dist), where each dist is a GaussianMix object,
    computes a couple (current_p, current_dist), in which current_pi is the sum
    of p and current_dist is a single GaussianMix object.

    fix2: Lifts the NotImplementedError [M3.6] for matrix-variable distributions.
    Implements component concatenation per research note 05 §Problem 1:
      - Concatenate components from all branches with weight rescaling p_i · pi_k.
      - Each component retains its per-matrix-var (M_k, U_k, V_k).
      - Variables present in only one branch get zero-state placeholder (fix2.2).
      - After concat: ranking_prune is triggered if budget exceeded (fix2.4).
    Scalar-only merge is unchanged.
    """
    # Single-element pass-through: return dist directly preserving gm_block
    if len(list_dist) == 1:
        p, d = list_dist[0]
        return p, d

    final_pi = []
    final_mu = []
    final_sigma = []
    current_p = 0

    # Check if any branch has matrix state
    has_matrix = any(d.var_entries for _, d in list_dist)

    # Build list_p for matrix block merger (normalised p_i per branch)
    raw_ps = [p for p, _ in list_dist if p > 0]
    total_raw = sum(raw_ps)

    ## creates a single mixture
    for (p, dist) in list_dist:
        if p > 0:
            current_p += p
            final_pi = final_pi + list(p*np.array(dist.gm.pi))
            final_mu = final_mu + list(dist.gm.mu)
            final_sigma = final_sigma + list(dist.gm.sigma)

    if len(final_pi) == 0:
        d = len(list_dist[0][1].gm.mu[0])
        #print('no components found')
        return 0, Dist(list_dist[0][1].var_list, GaussianMix([0], [np.array([0]*d)], [np.zeros((d,d))]))

    final_pi = list(np.array(final_pi)/current_p)

    # deletes components with probability less than tol
    zero_list = np.where(np.array(final_pi) < prob_tol)[0]
    if len(zero_list)>0:
        if len(zero_list) == len(final_pi):
            d = len(dist.gm.mu[0])
            #print('no components found')
            return 0, Dist(list_dist[0][1].var_list, GaussianMix([0], [np.array([0]*d)], [np.zeros((d,d))]))
        else:
            for index in sorted(zero_list, reverse=True):
                del final_pi[index]
                del final_mu[index]
                del final_sigma[index]

    ref_dist = list_dist[0][1]

    # fix2: merge matrix gm_block if any branch carries matrix state
    merged_gm_block = None
    if has_matrix and total_raw > 0:
        list_dists_only = [d for p, d in list_dist if p > 0]
        list_ps = [p / total_raw for p, _ in list_dist if p > 0]
        merged_gm_block = _merge_matrix_gm_blocks(list_dists_only, list_ps, current_p)

    current_dist = Dist(
        ref_dist.var_list,
        GaussianMix(final_pi, final_mu, final_sigma),
        var_entries=ref_dist.var_entries,
        gm_block=merged_gm_block,
    )
    return current_p, current_dist


def prune(current_dist, pruning, Kmax):
    if pruning == 'classic':
        current_dist = classic_prune(current_dist, Kmax)
    elif pruning == 'ranking':
        current_dist = ranking_prune(current_dist, Kmax)
    return current_dist


def ranking_prune(current_dist, Kmax):
    """Keeps only the Kmax component with higher prob.

    fix2.3: When gm_block is present, prune its mu_blocks/cov_blocks/pi in
    sync with the scalar gm, so that both structures remain aligned after
    any merge that multiplied the component count.
    """
    if current_dist.gm.n_comp() > Kmax:
        rank = np.argsort(current_dist.gm.pi)[::-1]
        # Keep top-Kmax indices (in descending weight order)
        kept = list(rank[:Kmax])
        current_dist.gm.pi = list(np.array(current_dist.gm.pi)[kept])
        current_dist.gm.mu = list(np.array(current_dist.gm.mu)[kept])
        current_dist.gm.sigma = list(np.array(current_dist.gm.sigma)[kept])
        current_dist.gm.pi = list(np.array(current_dist.gm.pi)/sum(current_dist.gm.pi))

        # fix2.3: synchronise gm_block component lists if present
        if current_dist.gm_block is not None:
            blk = current_dist.gm_block
            blk.mu_blocks = [blk.mu_blocks[i] for i in kept]
            blk.cov_blocks = [blk.cov_blocks[i] for i in kept]
            blk.pi = list(current_dist.gm.pi)  # already renormalised above
            if blk.log_pi is not None:
                import math
                blk.log_pi = [math.log(p) if p > 0 else float('-inf') for p in blk.pi]

    return current_dist

def compute_matrix_mean(current_dist):
    s = len(current_dist.gm.pi)
    pi = np.array(current_dist.gm.pi)
    pmu = np.array(pi).reshape(-1,1)*np.array(current_dist.gm.mu)
    sums = pmu[:,None] + pmu
    pis = (pi + pi[:,None]).reshape(s,s,1)
    return sums/pis
        
def classic_prune(current_dist, Kmax):
    """Merges components with optimal cost (Salmond/Runnalls criterion).

    M3.7: asserts that current_dist.gm_block is None — matrix programs must
    use ranking_prune (top-K by weight) only, since distance-based merging
    on dense vector means doesn't make sense across Kronecker block representations.
    """
    assert current_dist.gm_block is None, (
        "[M3.7] classic_prune invoked on a Dist with gm_block set. "
        "Matrix-variable programs must use ranking_prune (top-K by weight).  "
        "Call prune(dist, 'ranking', Kmax) instead.  "
        "See plan/2026-05-22-matrix-gm-lishan.md §M3.7."
    )
    if current_dist.gm.n_comp() > Kmax:
        n = current_dist.gm.n_comp()
        #computes a matrix containing the weighted means
        matrix_mu = compute_matrix_mean(current_dist)
        # At the first iteration computes the whole matrix cot
        matrixcost = np.ones((n,n))*np.inf
        for i in range(n):
            for j in range(i):
                matrixcost[i,j] = current_dist.gm.pi[i]*dist(current_dist.gm.mu[i],matrix_mu[i,j]) + current_dist.gm.pi[j]*dist(current_dist.gm.mu[j],matrix_mu[i,j])
       
        while n > Kmax:
            # Computes indices of components with minimal cost
            min_idx = np.where(matrixcost == np.min(matrixcost))
            i, j = min_idx
            i = i[0]
            j = j[0]      
            # Merges components
            current_dist = merge_comp(current_dist, i, j, matrix_mu[i,j])
            # Updates 
            n = current_dist.gm.n_comp()
            matrix_mu = compute_matrix_mean(current_dist)
            matrixcost = matrix_delete(matrixcost, i, j)
            # If number of components still too high computes the new matrix cost adding just one row
            if n > Kmax:
                for j in range(n-1):
                    matrixcost[n-1,j] = current_dist.gm.pi[n-1]*dist(current_dist.gm.mu[n-1],matrix_mu[n-1,j]) + current_dist.gm.pi[j]*dist(current_dist.gm.mu[j],matrix_mu[n-1,j])
    return current_dist


def matrix_delete(matrix, i, j):
    matrix = np.delete(matrix, max(i,j), axis=0)
    matrix = np.delete(matrix, max(i,j), axis=1)
    matrix = np.delete(matrix, min(i,j), axis=0)
    matrix = np.delete(matrix, min(i,j), axis=1)
    matrix = np.vstack([matrix, np.ones((1, len(matrix)))*np.inf])
    matrix = np.hstack([matrix, np.ones((matrix.shape[0],1))*np.inf])
    return matrix

def dist(vec1, vec2):
    return sum((np.array(vec1)-np.array(vec2))**2)


def merge_comp(current_dist, i, j, tot_mean):
    pii, pij = current_dist.gm.pi[i], current_dist.gm.pi[j]
    compi, compj = current_dist.gm.comp(i), current_dist.gm.comp(j)
    # deletes component to be merged from the current dist
    idx_list = [i,j]
    idx_list.sort(reverse=True)
    for idx in idx_list:
        del current_dist.gm.pi[idx]
        del current_dist.gm.mu[idx]
        del current_dist.gm.sigma[idx]
    # computes statistics of the merged component
    tot_p = pii + pij
    v = np.array(np.array([compi.mu[0], compj.mu[0]])) - tot_mean
    tot_cov = np.tensordot(np.array([pii, pij])/tot_p, np.array([compi.sigma[0], compj.sigma[0]]), axes=1) + np.transpose(v).dot(((np.array([pii,pij])/tot_p).reshape(-1,1)*v))
    # updates distribution
    current_dist.gm.pi.append(tot_p)
    current_dist.gm.mu.append(tot_mean)
    current_dist.gm.sigma.append(tot_cov)
    return current_dist


