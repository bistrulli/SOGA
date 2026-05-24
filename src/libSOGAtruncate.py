# Contains the functions for computing the resulting distribution when a truncation occurs in conditional or observe instructions according to the following dependencies.

from libSOGAshared import *
from TRUNCLexer import *
from TRUNCParser import *
from TRUNCListener import *
import parser_extensions  # noqa: F401 — restores Context helper methods on import (CLAUDE.md §8: keeps autogen files pure)
import timing
import multiprocessing as mp
from scipy.stats import norm as _scipy_norm_1d

pool=None

# Sparse-aware truncate toggle. When True, ineq_func dispatches to the
# rank-1 conditional Gaussian update (_ineq_func_sparse). When False (default),
# it uses the classic SVD-rotation path (_ineq_func_classic). Set via
# set_sparse_truncate() from start_SOGA based on the --sparse-truncate CLI flag.
USE_SPARSE_TRUNCATE = False

# Vectorize-truncate toggle. When True, truncate() bypasses the per-component
# Python loop entirely and operates on stacked (n_comp, d, d) tensors. Implies
# sparse-aware semantics (the math is the rank-1 update, just batched). Set
# from --vectorize-truncate CLI flag via start_SOGA.
USE_VECTORIZE_TRUNCATE = False


def set_sparse_truncate(enabled):
    """Toggle the sparse-aware truncate optimization at runtime."""
    global USE_SPARSE_TRUNCATE
    USE_SPARSE_TRUNCATE = bool(enabled)


def set_vectorize_truncate(enabled):
    """Toggle the vectorized batch truncate optimization at runtime."""
    global USE_VECTORIZE_TRUNCATE
    USE_VECTORIZE_TRUNCATE = bool(enabled)


def _truncated_normal_moments_1d(mu_s, var_s, c, direction):
    """First two moments and tail probability of a 1D Gaussian N(mu_s, var_s)
    truncated by `direction` at threshold `c`.

    direction must be one of '>', '>=', '<', '<='. Returns (m_hat, v_hat, P).
    When var_s is 0, treats the variable as a Dirac at mu_s.
    """
    if var_s <= 0:
        if direction in ('>', '>='):
            return mu_s, 0.0, 1.0 if mu_s > c else 0.0
        return mu_s, 0.0, 1.0 if mu_s < c else 0.0
    sigma = np.sqrt(var_s)
    z = (c - mu_s) / sigma
    if direction in ('>', '>='):
        denom = 1.0 - _scipy_norm_1d.cdf(z)
        if denom < prob_tol:
            return mu_s, var_s, 0.0
        lam = _scipy_norm_1d.pdf(z) / denom
    else:
        denom = _scipy_norm_1d.cdf(z)
        if denom < prob_tol:
            return mu_s, var_s, 0.0
        lam = -_scipy_norm_1d.pdf(z) / denom
    m_hat = mu_s + sigma * lam
    v_hat = var_s * (1.0 - lam * (lam - z))
    return m_hat, v_hat, denom


def _truncated_normal_moments_1d_batched(MU_s, VAR_s, C, direction):
    """Vectorized version of _truncated_normal_moments_1d. All inputs may be
    scalars or arrays of compatible shape; outputs (M_HAT, V_HAT, P) match the
    broadcasted shape.

    Edge cases:
    - VAR_s <= 0: treat as Dirac at MU_s; P = 1 if mu satisfies the inequality, else 0.
    - tail-prob < prob_tol: P = 0, leave moments unchanged at (MU_s, VAR_s).
    """
    MU_s = np.asarray(MU_s, dtype=float)
    VAR_s = np.asarray(VAR_s, dtype=float)
    C = np.asarray(C, dtype=float)
    shape = np.broadcast_shapes(MU_s.shape, VAR_s.shape, C.shape)
    MU_s = np.broadcast_to(MU_s, shape).astype(float, copy=True)
    VAR_s = np.broadcast_to(VAR_s, shape).astype(float, copy=True)
    C = np.broadcast_to(C, shape).astype(float, copy=True)

    valid_var = VAR_s > 0
    SIGMA = np.sqrt(np.where(valid_var, VAR_s, 1.0))
    Z = (C - MU_s) / SIGMA

    if direction in ('>', '>='):
        denom = 1.0 - _scipy_norm_1d.cdf(Z)
    elif direction in ('<', '<='):
        denom = _scipy_norm_1d.cdf(Z)
    else:
        raise ValueError(f"unknown direction: {direction}")

    valid_denom = (denom >= prob_tol) & valid_var
    safe_denom = np.where(valid_denom, denom, 1.0)

    if direction in ('>', '>='):
        LAM = _scipy_norm_1d.pdf(Z) / safe_denom
    else:
        LAM = -_scipy_norm_1d.pdf(Z) / safe_denom

    M_HAT = MU_s + SIGMA * LAM
    V_HAT = VAR_s * (1.0 - LAM * (LAM - Z))
    P = denom

    # Where VAR_s <= 0: deterministic check on mu vs c
    if direction in ('>', '>='):
        det_P = (MU_s > C).astype(float)
    else:
        det_P = (MU_s < C).astype(float)
    not_valid_var = ~valid_var
    M_HAT = np.where(not_valid_var, MU_s, M_HAT)
    V_HAT = np.where(not_valid_var, 0.0, V_HAT)
    P = np.where(not_valid_var, det_P, P)

    # Where tail-prob < prob_tol (but var was valid): leave moments untouched, P = 0
    bad_denom = ~valid_denom & valid_var
    M_HAT = np.where(bad_denom, MU_s, M_HAT)
    V_HAT = np.where(bad_denom, VAR_s, V_HAT)
    P = np.where(bad_denom, 0.0, P)

    return M_HAT, V_HAT, P


def ineq_func(self, comp):
    """Dispatcher: picks classic (default) or sparse-aware truncate based on
    the USE_SPARSE_TRUNCATE module flag. (Single-component path; the
    USE_VECTORIZE_TRUNCATE batch path is handled inside truncate() itself.)"""
    if USE_SPARSE_TRUNCATE:
        return _ineq_func_sparse(self, comp)
    return _ineq_func_classic(self, comp)


def _ineq_func_classic(self,comp):
    mu = comp.gm.mu[0]
    sigma = comp.gm.sigma[0]
    final_pi = []
    final_mu = []
    final_sigma = []
    for part in product(*[range(len(mean)) for mean in self.aux_means]):
        # for a given combination of components of the auxiliary variables, creates a new component extending comp
        aux_pi = 1
        aux_mean = list(copy(mu))
        aux_sigma = []
        ineq_coeff = np.array(copy(self.coeff))
        ineq_const = self.const
        for p,q in zip(range(len(self.aux_means)), part):
            aux_pi = aux_pi*self.aux_pis[p][q]
            aux_mean.append(self.aux_means[p][q])
            aux_sigma.append(self.aux_covs[p][q])
        aux_mean = np.array(aux_mean)
        aux_sigma = np.diag(aux_sigma)
        aux_cov = np.block([[sigma, np.zeros((len(sigma), len(aux_sigma)))], [np.zeros((len(aux_sigma), len(sigma))), aux_sigma]])
        # substitute deltas
        delta_idx = np.where(np.diag(aux_cov) < delta_tol)[0]
        ineq_const -= np.array(self.coeff)[delta_idx].dot(aux_mean[delta_idx])
        ineq_coeff[delta_idx] = np.zeros(len(delta_idx))
        # if all variables were deltas return
        if np.all(np.array(ineq_coeff) == 0):
            if (self.type == '>' and ineq_const < 0) or (self.type == '>=' and ineq_const <= 0) or (self.type == '<' and ineq_const > 0) or (self.type == '<=' and ineq_const >= 0):
                new_P = 1.
            else:
                new_P = 0.
            new_mu = mu
            new_sigma = sigma
        # else compute truncated distribution
        else:
            # STEP 1: change variables
            norm = np.linalg.norm(ineq_coeff)
            ineq_coeff = np.array(ineq_coeff)/norm
            ineq_const = ineq_const/norm
            A = find_basis(ineq_coeff)           # maybe instead of A a vector can be used to improve scalability (?)
            transl_mu = A.dot(aux_mean)
            transl_sigma = A.dot(aux_cov).dot(A.transpose())
            # STEP 2: finds the indices of the components that needs to be transformed
            transl_alpha = np.zeros(len(transl_mu))
            transl_alpha[0] = 1
            indices = select_indices(transl_alpha, transl_sigma)
            # STEP 3: creates reduced vectors taking into account only the coordinates that need to be transformed
            red_transl_alpha = reduce_indices(transl_alpha, indices)
            red_transl_mu = reduce_indices(transl_mu, indices)
            red_transl_sigma = reduce_indices(transl_sigma, indices) 
            # STEP 4: creates the hyper-rectangle to integrate on
            a = np.ones(len(red_transl_alpha))*(-np.inf)
            b = np.ones(len(red_transl_alpha))*(np.inf)
            if self.type=='>' or self.type=='>=':
                a[0] = ineq_const
            if self.type=='<' or self.type=='<=':
                b[0] = ineq_const   
            # STEP 5: compute moments in the transformed coordinates
            new_P, new_red_transl_mu, new_red_transl_sigma = compute_moments(red_transl_mu, red_transl_sigma, a, b)
            # STEP 6: recreates extended vectors
            new_transl_mu = extend_indices(new_red_transl_mu, transl_mu, indices)
            new_transl_sigma = extend_indices(new_red_transl_sigma, transl_sigma, indices)
            # STEP 7: goes back to older coordinates
            d = len(comp.var_list)
            A_inv = np.linalg.inv(A)
            new_mu = A_inv.dot(new_transl_mu)[:d]
            new_sigma = A_inv.dot(new_transl_sigma).dot(A_inv.transpose())[:d,:d]
            end = time()
        # append new values
        final_pi.append(aux_pi*new_P)
        final_mu.append(new_mu)
        final_sigma.append(new_sigma)
    return GaussianMix(final_pi, final_mu, final_sigma)


def _ineq_func_sparse(self, comp):
    """Sparse-aware rank-1 update version of _ineq_func_classic.

    Mathematically equivalent to the classic SVD-rotation path, verified to
    machine epsilon on 7200 random samples in
    experiments/sparse_truncate_prototype_2026-05-20/.

    Replaces the O(d^3) rotation A * Sigma * A.T + inv(A) + back-projection
    with O(d^2) rank-1 conditional Gaussian update:

        g     = Sigma @ alpha_orig                            # O(d^2)
        mu_s  = alpha_orig @ mu  + alpha_aux @ aux_mean       # scalar
        var_s = alpha_orig @ g   + sum(alpha_aux^2 * aux_var) # scalar
        m_hat, v_hat, P = 1D truncated normal moments
        mu'    = mu    + g * (m_hat - mu_s) / var_s
        Sigma' = Sigma - outer(g, g) * (1 - v_hat/var_s) / var_s

    Auxiliary gm() variables inside the LBC are handled scalar-wise
    (no extension of the d x d Sigma block needed).
    """
    mu = comp.gm.mu[0]
    sigma = comp.gm.sigma[0]
    d = len(mu)
    final_pi = []
    final_mu = []
    final_sigma = []
    n_aux = len(self.aux_means)

    for part in product(*[range(len(mean)) for mean in self.aux_means]):
        # Combine the aux components selected in this iteration
        aux_pi = 1.0
        aux_mean_combo = np.zeros(n_aux)
        aux_var_combo = np.zeros(n_aux)
        for p, q in zip(range(n_aux), part):
            aux_pi = aux_pi * self.aux_pis[p][q]
            aux_mean_combo[p] = self.aux_means[p][q]
            aux_var_combo[p] = self.aux_covs[p][q]

        # Split self.coeff into the original d entries and the n_aux extension
        coeff = np.array(self.coeff, dtype=float)
        coeff_orig = coeff[:d].copy()
        coeff_aux = coeff[d:].copy() if n_aux > 0 else np.zeros(0)
        ineq_const = float(self.const)

        # Substitute delta variables (those with variance below delta_tol):
        # their value collapses to the mean and folds into the constant term.
        delta_idx = np.where(np.diag(sigma) < delta_tol)[0]
        if len(delta_idx) > 0:
            ineq_const -= float(coeff_orig[delta_idx].dot(mu[delta_idx]))
            coeff_orig[delta_idx] = 0.0
        if n_aux > 0:
            aux_delta_mask = aux_var_combo < delta_tol
            if np.any(aux_delta_mask):
                ineq_const -= float(coeff_aux[aux_delta_mask].dot(aux_mean_combo[aux_delta_mask]))
                coeff_aux[aux_delta_mask] = 0.0

        # Degenerate case: all coefficients vanished -> truth depends only on the constant
        if np.all(coeff_orig == 0) and (n_aux == 0 or np.all(coeff_aux == 0)):
            if (self.type == '>' and ineq_const < 0) or \
               (self.type == '>=' and ineq_const <= 0) or \
               (self.type == '<' and ineq_const > 0) or \
               (self.type == '<=' and ineq_const >= 0):
                new_P = 1.0
            else:
                new_P = 0.0
            new_mu = mu
            new_sigma = sigma
        else:
            # Scalar projection s = alpha @ x_extended
            mu_s = float(coeff_orig.dot(mu))
            if n_aux > 0:
                mu_s += float(coeff_aux.dot(aux_mean_combo))

            # Cross-covariance of x_orig with s (independent of aux dims by construction)
            g_orig = sigma.dot(coeff_orig)

            var_s = float(coeff_orig.dot(g_orig))
            if n_aux > 0:
                var_s += float(np.sum(coeff_aux * coeff_aux * aux_var_combo))

            m_hat, v_hat, new_P = _truncated_normal_moments_1d(
                mu_s, var_s, ineq_const, self.type
            )

            if new_P < prob_tol:
                new_mu = mu
                new_sigma = sigma
            else:
                # Rank-1 conditional Gaussian update in the original d-dim space.
                # Both formulas are exact (law of total expectation / variance).
                new_mu = mu + g_orig * ((m_hat - mu_s) / var_s)
                new_sigma = sigma - np.outer(g_orig, g_orig) * ((1.0 - v_hat / var_s) / var_s)

        final_pi.append(aux_pi * new_P)
        final_mu.append(new_mu)
        final_sigma.append(new_sigma)
    return GaussianMix(final_pi, final_mu, final_sigma)


def eq_func(self,comp):
    mu = comp.gm.mu[0]
    sigma = comp.gm.sigma[0]
    final_pi = []
    final_mu = []
    final_sigma = []
    eq_coeff = copy(self.coeff)
    eq_const = self.const
    # check if delta
    i = np.where(np.array(self.coeff) != 0)[0][0]
    if sigma[i,i] < delta_tol:
        eq_const = eq_const - self.coeff[i]*mu[i]
        eq_coeff[i] = 0.
    # if delta return
    if np.all(np.array(eq_coeff) == 0):
        if (self.type == '==' and eq_const == 0) or (self.type == '!=' and eq_const != 0):
            new_P = 1.
        else:
            new_P = 0.
        new_mu = mu
        new_sigma = sigma
    else:
        # STEP 1: selects indices to condition
        indices = select_indices(eq_coeff, sigma)
        if len(indices) == 1:
            new_P = 0.
            new_mu = mu
            new_sigma = sigma
        else:
            # STEP 2: creates reduced vectors
            red_mu = reduce_indices(mu, indices)
            red_sigma = reduce_indices(sigma, indices) 
            red_alpha = reduce_indices(eq_coeff, indices)
            red_obs_idx = int(list(np.where(np.array(red_alpha)!=0))[0][0])
            # STEP 3: computes cond_sigma (select is a mask containing the index of the conditioned variables)
            select = (np.arange(len(red_mu))!=red_obs_idx)
            cond_sigma = red_sigma[select,:][:,select]
            cond_sigma = cond_sigma - (1/red_sigma[red_obs_idx,red_obs_idx])*(red_sigma[select,red_obs_idx].reshape(len(select)-1,1)).dot(red_sigma[red_obs_idx,select].reshape(1,len(select)-1))
            # STEP 4: computes cond_mu
            cond_mu = red_mu[select] + (1/red_sigma[red_obs_idx,red_obs_idx])*(eq_const-red_mu[red_obs_idx])*red_sigma[select,red_obs_idx]
            # if conditioned matrix is Null, it is equivalent to observing a single independent component
            if np.all(cond_sigma == 0):  
                new_P = 0.
            else:
                new_P = 1.
            # STEP 5: adds value for the observed variable (now a delta)
            cond_mu, cond_sigma = insert_value(eq_const, red_obs_idx, cond_mu, cond_sigma)
            # STEP 6: returns to the original set of variables
            new_sigma = extend_indices(cond_sigma, sigma, indices)
            new_mu = extend_indices(cond_mu, mu, indices) 
    return GaussianMix([new_P], [new_mu], [new_sigma])

def negate(trunc):
    """ Produces a string which is the logic negation of trunc """
    if '<' in trunc:
        if '<=' in trunc:
            trunc = trunc.replace('<=', '>')
        else:
            trunc = trunc.replace('<', '>=')
    elif '>' in trunc:
        if '>=' in trunc:
            trunc = trunc.replace('>=', '<')
        else:
            trunc = trunc.replace('>', '<=')
    elif '==' in trunc:
        trunc = trunc.replace('==', '!=')
    elif '!=' in turnc:
        trunc = trunc.replace('!=', '==')
    return trunc

def split_trunc(trunc):
    """ When trunc is 'x != c' returns 'x > c' and 'x < c' """
    assert '!=' in trunc
    trunc1 = trunc.replace('!=','>')
    trunc2 = trunc.replace('!=','<')
    return trunc1, trunc2

class TruncRule(TRUNCListener):
    
    def __init__(self, var_list, data):
        self.var_list = var_list
        self.data = data
        self.type = None
        self.coeff = [0.]*len(var_list)
        self.const = 0
        self.func = None
        
        self.aux_pis = []
        self.aux_means = []
        self.aux_covs = []
        
    def enterIneq(self, ctx):
        self.type = ctx.inop().getText()
        if not ctx.const().NUM() is None:
            self.const = float(ctx.const().NUM().getText())
        elif not ctx.const().idd() is None:
            self.const = ctx.const().idd().getValue(self.data)
                
    
    def enterLexpr(self, ctx):
        self.flag_sign = 1.

            
    def exitLexpr(self, ctx):
        self.func = partial(ineq_func,self)
        
        
    def enterMonom(self,ctx):
        if ctx.var().gm() is None:
            # monom in the form const? '*' (IDV | idd)
            ID = ctx.var()._getText(self.data)
            if not ctx.const() is None:
                if not ctx.const().NUM() is None:
                    coeff = self.flag_sign*float(ctx.const().NUM().getText())
                elif not ctx.const().idd() is None:
                    coeff = self.flag_sign*ctx.const().idd().getValue(self.data)
            else:
                coeff = self.flag_sign
            idx = self.var_list.index(ID)
            self.coeff[idx] = coeff
        # monom in the form const? '*' gm
        else:
            self.aux_pis.append(eval(ctx.var().gm().list_()[0].getText()))
            self.aux_means.append(eval(ctx.var().gm().list_()[1].getText()))
            self.aux_covs.append(np.array(eval(ctx.var().gm().list_()[2].getText()))**2)
            if not ctx.const() is None:
                if not ctx.const().NUM() is None:
                    coeff = self.flag_sign*float(ctx.const().NUM().getText())
                elif not ctx.const().idd() is None:
                    coeff = self.flag_sign*ctx.const().idd().getValue(self.data)
            else:
                coeff = self.flag_sign 
            self.coeff.append(coeff)            
            
    def enterSub(self, ctx):
        self.flag_sign = -1.
        
    def enterSum(self, ctx):
        self.flag_sign = 1.
        
    def enterEq(self, ctx):
        self.type = ctx.eqop().getText()
        idx = self.var_list.index(ctx.var()._getText(self.data))
        self.coeff[idx] = 1.
        if not ctx.const() is None:
            if not ctx.const().NUM() is None:
                self.const = float(ctx.const().NUM().getText())
            elif not ctx.const().idd() is None:
                self.const = ctx.const().idd().getValue(self.data)
            
        self.func = partial(eq_func,self)


def _ineq_truncate_vectorized_impl(dist, trunc_rule):
    """Batched rank-1 inequality truncate over all GM components at once.

    Returns (norm_factor, new_dist) directly — bypasses the outer truncate()
    aggregation loop. Mathematically equivalent to running _ineq_func_sparse
    on each component and then aggregating, but with the per-component loop
    replaced by numpy batched ops.
    """
    n_comp = dist.gm.n_comp()
    d = len(dist.gm.mu[0])
    MU = np.array(dist.gm.mu, dtype=float)        # (n_comp, d)
    SIGMA = np.array(dist.gm.sigma, dtype=float)  # (n_comp, d, d)
    PI = np.array(dist.gm.pi, dtype=float)        # (n_comp,)

    n_aux = len(trunc_rule.aux_means)
    coeff_full = np.array(trunc_rule.coeff, dtype=float)
    base_coeff_orig = coeff_full[:d]
    coeff_aux_base = coeff_full[d:] if n_aux > 0 else np.zeros(0)
    base_const = float(trunc_rule.const)
    trunc_type = trunc_rule.type

    # Per-component delta detection (constant across aux combinations)
    delta_mask = np.diagonal(SIGMA, axis1=1, axis2=2) < delta_tol  # (n_comp, d)

    # Per-component effective coefficient (zero out delta positions)
    COEFF = np.broadcast_to(base_coeff_orig, (n_comp, d)).copy()
    COEFF[delta_mask] = 0.0
    # Delta contribution folded into the constant: -= sum(coeff * mu) over delta positions
    delta_contrib = (delta_mask * base_coeff_orig * MU).sum(-1)  # (n_comp,)

    # Constants that do not depend on aux combination
    G = np.einsum('kij,kj->ki', SIGMA, COEFF)  # (n_comp, d) cross-cov
    VAR_s_orig = (COEFF * G).sum(-1)            # (n_comp,)
    MU_s_orig = (COEFF * MU).sum(-1)            # (n_comp,)
    all_zero_orig = np.all(COEFF == 0, axis=-1)  # (n_comp,)

    final_pi = []
    final_mu = []
    final_sigma = []

    aux_iter = product(*[range(len(m)) for m in trunc_rule.aux_means]) if n_aux > 0 else [()]
    for part in aux_iter:
        # Aux combination: accumulate scalar contributions, fold delta auxs into const
        aux_pi = 1.0
        aux_mu_contrib = 0.0
        aux_var_contrib = 0.0
        aux_const_shift = 0.0
        for p, q in zip(range(n_aux), part):
            pi_pq = trunc_rule.aux_pis[p][q]
            mean_pq = trunc_rule.aux_means[p][q]
            var_pq = trunc_rule.aux_covs[p][q]
            aux_pi *= pi_pq
            cp = float(coeff_aux_base[p])
            if var_pq < delta_tol:
                aux_const_shift += cp * mean_pq
            else:
                aux_mu_contrib += cp * mean_pq
                aux_var_contrib += cp * cp * float(var_pq)

        MU_s = MU_s_orig + aux_mu_contrib
        VAR_s = VAR_s_orig + aux_var_contrib
        INEQ_CONST = (base_const - aux_const_shift) - delta_contrib  # (n_comp,)

        # Degenerate components: all coeff_orig zero AND no aux variance/mean contribution
        aux_inactive = (aux_var_contrib == 0.0) and (aux_mu_contrib == 0.0)
        degenerate = all_zero_orig & aux_inactive  # (n_comp,)

        # 1D truncated moments
        M_HAT, V_HAT, P = _truncated_normal_moments_1d_batched(MU_s, VAR_s, INEQ_CONST, trunc_type)

        # Override P for degenerate components
        if np.any(degenerate):
            if trunc_type == '>':
                det_P = (INEQ_CONST < 0).astype(float)
            elif trunc_type == '>=':
                det_P = (INEQ_CONST <= 0).astype(float)
            elif trunc_type == '<':
                det_P = (INEQ_CONST > 0).astype(float)
            else:  # '<='
                det_P = (INEQ_CONST >= 0).astype(float)
            P = np.where(degenerate, det_P, P)

        # Where to apply rank-1 update vs keep original
        good = (P >= prob_tol) & (VAR_s > 0) & (~degenerate)

        safe_var = np.where(VAR_s > 0, VAR_s, 1.0)
        factor_mu_arr = np.where(good, (M_HAT - MU_s) / safe_var, 0.0)
        factor_cov_arr = np.where(good, (1.0 - V_HAT / safe_var) / safe_var, 0.0)

        NEW_MU = MU + G * factor_mu_arr[:, None]
        NEW_SIGMA = SIGMA - np.einsum('ki,kj->kij', G, G) * factor_cov_arr[:, None, None]

        # Restore original mu/sigma where update was skipped (factor was zero so this
        # is mostly a no-op already, but be explicit to match the per-component path)
        skip = ~good
        if np.any(skip):
            NEW_MU = np.where(skip[:, None], MU, NEW_MU)
            NEW_SIGMA = np.where(skip[:, None, None], SIGMA, NEW_SIGMA)

        WEIGHTS = PI * aux_pi * P  # (n_comp,)
        keep = WEIGHTS > prob_tol
        for k in np.where(keep)[0]:
            final_pi.append(float(WEIGHTS[k]))
            final_mu.append(NEW_MU[k])
            final_sigma.append(NEW_SIGMA[k])

    norm_factor = float(sum(final_pi))
    if norm_factor > prob_tol:
        normalized = [p / norm_factor for p in final_pi]
        new_dist = Dist(dist.var_list, GaussianMix(normalized, final_mu, final_sigma))
    else:
        # All mass below tolerance: return a degenerate sentinel (mirror outer truncate())
        new_dist = Dist(dist.var_list,
                        GaussianMix([0.0], [np.array([0.0] * d)], [np.zeros((d, d))]))
    return norm_factor, new_dist


def _eq_truncate_vectorized_impl(dist, trunc_rule):
    """Batched equality conditioning over all GM components at once.

    The eq grammar enforces a single non-zero coefficient (LBC is `var == const`),
    so we condition every component on x_i = const/coeff_i via Schur complement,
    then collapse x_i to a delta. Handles both '==' and '!=' types.
    Returns (norm_factor, new_dist).
    """
    n_comp = dist.gm.n_comp()
    d = len(dist.gm.mu[0])
    MU = np.array(dist.gm.mu, dtype=float)
    SIGMA = np.array(dist.gm.sigma, dtype=float)
    PI = np.array(dist.gm.pi, dtype=float)

    coeff = np.array(trunc_rule.coeff[:d], dtype=float)
    eq_const = float(trunc_rule.const)
    nonzero = np.where(coeff != 0)[0]
    if len(nonzero) == 0:
        # No active variable: deterministic on constant alone
        if trunc_rule.type == '==':
            P_scalar = 1.0 if eq_const == 0 else 0.0
        else:
            P_scalar = 0.0 if eq_const == 0 else 1.0
        if P_scalar < prob_tol:
            return 0.0, Dist(dist.var_list,
                             GaussianMix([0.0], [np.array([0.0] * d)], [np.zeros((d, d))]))
        return 1.0, dist
    i = int(nonzero[0])
    coeff_i = float(coeff[i])
    target_val = eq_const / coeff_i  # value of x_i if equality holds

    is_delta = SIGMA[:, i, i] < delta_tol  # (n_comp,) per-component delta check on x_i

    # Delta components: P depends on whether mu[i] == target_val
    delta_residual = MU[:, i] - target_val
    if trunc_rule.type == '==':
        P_delta = (np.abs(delta_residual) < 1e-10).astype(float)
    else:  # '!='
        P_delta = (np.abs(delta_residual) >= 1e-10).astype(float)

    # Non-delta components: Schur-complement conditional Gaussian update
    sigma_col_i = SIGMA[:, :, i]  # (n_comp, d) — column i across all components
    sigma_ii = SIGMA[:, i, i]     # (n_comp,)
    safe_sigma_ii = np.where(sigma_ii > 0, sigma_ii, 1.0)
    factor_mu_nd = (target_val - MU[:, i]) / safe_sigma_ii  # (n_comp,)

    NEW_MU_nd = MU + sigma_col_i * factor_mu_nd[:, None]
    NEW_SIGMA_nd = SIGMA - np.einsum('ki,kj->kij', sigma_col_i, sigma_col_i) / safe_sigma_ii[:, None, None]
    # Collapse x_i to its delta value
    NEW_MU_nd[:, i] = target_val
    NEW_SIGMA_nd[:, i, :] = 0.0
    NEW_SIGMA_nd[:, :, i] = 0.0

    # If the conditioned cov (excluding i) is all-zero, original Gaussian had x_i
    # fully determined by the other components — singular case → P = 0
    mask_keep = np.ones(d, dtype=bool)
    mask_keep[i] = False
    sub_sigma = NEW_SIGMA_nd[:, mask_keep, :][:, :, mask_keep]  # (n_comp, d-1, d-1)
    cond_singular = np.all(np.abs(sub_sigma) < 1e-15, axis=(-2, -1))  # (n_comp,)
    P_nd = np.where(cond_singular, 0.0, 1.0)

    # Combine delta vs non-delta cases
    P = np.where(is_delta, P_delta, P_nd)
    NEW_MU = np.where(is_delta[:, None], MU, NEW_MU_nd)
    NEW_SIGMA = np.where(is_delta[:, None, None], SIGMA, NEW_SIGMA_nd)

    WEIGHTS = PI * P

    # Hard-observe filtering for '==': if any delta component matches the equality,
    # ONLY those contribute (mirror outer truncate() logic).
    if trunc_rule.type == '==':
        hard_set = is_delta & (P_delta > prob_tol)
        if np.any(hard_set):
            keep_mask = hard_set & (WEIGHTS > prob_tol)
        else:
            keep_mask = WEIGHTS > prob_tol
    else:
        keep_mask = WEIGHTS > prob_tol

    final_pi = []
    final_mu = []
    final_sigma = []
    for k in np.where(keep_mask)[0]:
        final_pi.append(float(WEIGHTS[k]))
        final_mu.append(NEW_MU[k])
        final_sigma.append(NEW_SIGMA[k])

    norm_factor = float(sum(final_pi))
    if norm_factor > prob_tol:
        normalized = [p / norm_factor for p in final_pi]
        new_dist = Dist(dist.var_list, GaussianMix(normalized, final_mu, final_sigma))
    else:
        new_dist = Dist(dist.var_list,
                        GaussianMix([0.0], [np.array([0.0] * d)], [np.zeros((d, d))]))
    return norm_factor, new_dist


def _truncate_vectorized(dist, trunc_rule):
    """Entry point for the vectorized truncate path. Dispatches to ineq or eq."""
    if trunc_rule.type in ('==', '!='):
        return _eq_truncate_vectorized_impl(dist, trunc_rule)
    return _ineq_truncate_vectorized_impl(dist, trunc_rule)


def truncate_matrix(dist, trunc, data, mat_var):
    """M5: Full dispatcher for matrix-variable truncation (observe/condition).

    Delegates to libMatrixTruncate.truncate_matrix which implements:
      ELEMENT_INEQ (X[i,j] op c), ROW_SUM_INEQ, COL_SUM_INEQ.

    See libMatrixTruncate.py for full docstring and plan §M5.
    """
    from libMatrixTruncate import truncate_matrix as _trm
    return _trm(dist, trunc, data, mat_var)


def _backprop_scalar_truncate_to_matrices(prior_dist, new_dist):
    """A2 (research-note 04): after a scalar truncate updates dist.gm, propagate
    the change in scalar means back to any matrix variables via the stored
    scalar↔matrix cross-covariance.

    Only the MEAN of each matrix var (block.mu_blocks[k][mat_name]) is updated.
    The matrix's (U, V) Kronecker factors are NOT touched here — that would
    break the Kronecker structure (research-note 04 Finding B1).  Covariance
    back-prop requires densification (M4.8 Opt-2 / plan §M5.2 DENSE strategy).

    Only applied when prior and posterior have a 1:1 component mapping (i.e.,
    no truncate-driven component split or drop).  In other cases the back-prop
    is skipped silently; a future iteration can handle split components by
    duplicating cov_blocks entries per sub-component.

    Kalman gain per component k, per scalar z that has cross-cov to matrix X:
        Δmu_X = reshape(cov_zX[k], (m, n), 'F') · (mu_z_post - mu_z_prior) / var_z_prior

    where cov_zX[k] is the stored (mn,) ndarray in V⊗U column-major order.
    """
    block = new_dist.gm_block
    if block is None or not new_dist.var_entries:
        return  # no matrix state to back-prop into
    if new_dist.gm.n_comp() != prior_dist.gm.n_comp():
        return  # 1:1 component mapping broken (split/drop) — skip in v1

    n_comp = new_dist.gm.n_comp()
    var_list = prior_dist.var_list
    for k in range(n_comp):
        mu_prior = np.asarray(prior_dist.gm.mu[k], dtype=float)
        sigma_prior = np.asarray(prior_dist.gm.sigma[k], dtype=float)
        mu_post = np.asarray(new_dist.gm.mu[k], dtype=float)
        for z_idx, z_name in enumerate(var_list):
            var_z = float(sigma_prior[z_idx, z_idx]) if z_idx < sigma_prior.shape[0] else 0.0
            if var_z <= 0:
                continue
            delta_z = float(mu_post[z_idx] - mu_prior[z_idx])
            if abs(delta_z) <= 0:
                continue
            gain = delta_z / var_z
            for ve in new_dist.var_entries:
                key = frozenset({z_name, ve.name})
                if key in block.cov_blocks[k]:
                    cov_z_vecX = block.cov_blocks[k][key]
                    m, n = ve.shape
                    if isinstance(cov_z_vecX, tuple) and len(cov_z_vecX) == 2:
                        u_z, v_z = cov_z_vecX
                        delta_M = np.outer(u_z, v_z) * gain
                    else:
                        delta_M = np.asarray(cov_z_vecX, dtype=float).reshape(
                            (m, n), order='F'
                        ) * gain
                    block.mu_blocks[k][ve.name] = block.mu_blocks[k][ve.name] + delta_M


def truncate(dist, trunc, data):
    """ Given a distribution dist computes its truncation to trunc. Returns a pair norm_factor, new_dist where norm_factor is the probability mass of the original distribution dist on trunc and new_dist is a Dist object representing the (approximated) truncated distribution.

    M2.5 routing guard: if any token in trunc matches a matrix variable name from
    dist.var_entries, dispatch to truncate_matrix (M5 stub for now).  The guard
    is a simple string-contains check against each matrix variable name.
    The existing scalar path is unchanged (zero overhead for scalar programs).

    A2 back-prop (research-note 04): after the scalar path completes, any change
    in scalar mean is propagated back to matrix-variable means via the stored
    cross-covariance.  Covariance back-prop deferred (would densify Kronecker
    factors; tracked as Opt-2 in research-note 04).
    """
    if trunc == 'true':
        return 1., dist
    elif trunc == 'false':
        return 0., dist

    # M2.5: routing guard — check if trunc references any matrix variable
    if dist.var_entries:
        for ve in dist.var_entries:
            if ve.name in trunc:
                return truncate_matrix(dist, trunc, data, ve.name)

    # Scalar path (unchanged) — reached when no matrix variable is referenced in trunc
    trunc_rule = trunc_parse(dist.var_list, trunc, data)
    if USE_VECTORIZE_TRUNCATE:
        return _truncate_vectorized(dist, trunc_rule)
    trunc_func = trunc_rule.func
    trunc_type = trunc_rule.type
    trunc_idx = np.where(np.array(trunc_rule.coeff) != 0)[0][0]
    hard = []
    new_dist = Dist(dist.var_list, GaussianMix([],[],[]),
                    var_entries=dist.var_entries, gm_block=dist.gm_block)
    new_pi = []
    trans_comp = []
    for k in range(dist.gm.n_comp()):
        comp = Dist(dist.var_list, dist.gm.comp(k))
        trans_comp.append(trunc_func(comp))
    for k in range(dist.gm.n_comp()):
        if trunc_type == '==' and dist.gm.sigma[k][trunc_idx,trunc_idx] < delta_tol and sum(trans_comp[k].pi) > 0:
            hard.append(k)
    if len(hard) == 0:
        for k in range(dist.gm.n_comp()):
            new_mix = trans_comp[k]
            for h in range(new_mix.n_comp()):
                if new_mix.pi[h] > prob_tol:
                    new_dist.gm.mu.append(new_mix.mu[h])
                    new_dist.gm.sigma.append(new_mix.sigma[h])
                    new_pi.append(dist.gm.pi[k]*new_mix.pi[h])
    else:
        for k in hard:
            new_mix = trans_comp[k]
            for h in range(new_mix.n_comp()):
                if new_mix.pi[h] > prob_tol:
                    new_dist.gm.mu.append(new_mix.mu[h])
                    new_dist.gm.sigma.append(new_mix.sigma[h])
                    new_pi.append(dist.gm.pi[k]*new_mix.pi[h])
    norm_factor = sum(np.array(new_pi))
    if norm_factor > prob_tol:
        new_dist.gm.pi = list(np.array(new_pi)/norm_factor)
    else:
        new_dist.gm.pi = [0.]
        new_dist.gm.mu = [dist.gm.mu[0]]
        new_dist.gm.sigma = [dist.gm.sigma[0]]

    # A2 back-prop: propagate scalar mean change to matrix var means via cross-cov.
    # Deep-copy gm_block so we don't mutate the caller's state.
    if new_dist.gm_block is not None and new_dist.var_entries:
        from copy import deepcopy as _deepcopy
        new_dist.gm_block = _deepcopy(new_dist.gm_block)
        _backprop_scalar_truncate_to_matrices(dist, new_dist)

    return norm_factor, new_dist

# parallel implementation
def parallel_truncate(dist, trunc, data,nproc):
    global pool
    gst=time()
    if(pool is None):
        print("creating pool")
        #pool=ThreadPoolExecutor(max_workers=nproc)
        pool=mp.Pool(nproc)

    """ Given a distribution dist computes its truncation to trunc. Returns a pair norm_factor, new_dist where norm_factor is the probability mass of the original distribution dist on trunc and new_dist is a Dist object representing the (approximated) truncated distribution. """
    if trunc == 'true':
        return 1., dist
    elif trunc == 'false':
        return 0., dist
    else:
        #print(f"##### ncomp={dist.gm.n_comp()}")
        st=time()
        trunc_rule = trunc_parse(dist.var_list, trunc, data)
        #print(f"trunc_parse:{time()-st}")
        trunc_func = trunc_rule.func
        trunc_type = trunc_rule.type
        trunc_idx = np.where(np.array(trunc_rule.coeff) != 0)[0][0]
        hard = []
        new_dist = Dist(dist.var_list, GaussianMix([],[],[]))
        new_pi = []
        comp_list = []

        st=time()
        for k in range(dist.gm.n_comp()):
            comp = Dist(dist.var_list, dist.gm.comp(k))
            comp_list.append(comp)
        #print(f"loop time:{time()-st}")

        st=time()

        trans_comp = list(pool.map(trunc_func, comp_list))
        #print(f"map time:{time()-st}")

        st=time()
        for k in range(dist.gm.n_comp()):
            if trunc_type == '==' and dist.gm.sigma[k][trunc_idx,trunc_idx] < delta_tol and sum(trans_comp[k].pi) > prob_tol:
                hard.append(k)
        #print(f"second loop time:{time()-st}")

        if len(hard) == 0:
            for k in range(dist.gm.n_comp()):
                new_mix = trans_comp[k]
                for h in range(new_mix.n_comp()):
                    if new_mix.pi[h] > prob_tol:
                        new_dist.gm.mu.append(new_mix.mu[h])
                        new_dist.gm.sigma.append(new_mix.sigma[h])
                        new_pi.append(dist.gm.pi[k]*new_mix.pi[h])
        else:
            for k in hard:
                new_mix = trans_comp[k]
                for h in range(new_mix.n_comp()):
                    if new_mix.pi[h] > prob_tol:
                        new_dist.gm.mu.append(new_mix.mu[h])
                        new_dist.gm.sigma.append(new_mix.sigma[h])
                        new_pi.append(dist.gm.pi[k]*new_mix.pi[h])
        norm_factor = sum(np.array(new_pi))
        if norm_factor > prob_tol:
            new_dist.gm.pi = list(np.array(new_pi)/norm_factor)
        #print(f"total time:{time()-gst}")
        return norm_factor, new_dist
    
    
def trunc_parse(var_list, trunc, data):
    """ Parses trunc using ANTLR4. Returns a function """
    lexer = TRUNCLexer(InputStream(trunc))
    stream = CommonTokenStream(lexer)
    parser = TRUNCParser(stream)
    tree = parser.trunc()
    trunc_rule = TruncRule(var_list, data)
    walker = ParseTreeWalker()
    walker.walk(trunc_rule, tree) 
    return trunc_rule


def find_basis(alpha):
    """
    Given alpha (vector of the truncation) returns a matrix A giving the change of variable necessary to make alpha one of the axis
    """
    alpha = np.array(alpha)
    u, s, v = np.linalg.svd([alpha])
    alpha1 = v[:,1:]
    A = np.vstack((alpha.reshape(1,alpha.shape[0]), alpha1.transpose()))
    return A


#def select_indices(alpha, sigma):
#    """
#    Finds the indices of the components that needs to be transformed based on the vector representation of the truncation (alpha)
#    and the covariance matrix (sigma)
#    """
#    
#    def enlarge_set(index_set):
#        total_set = index_set
#        for i in index_set:
#            i_indices = list(np.where(sigma[i,:] != 0)[0])
#            total_set = list(set(total_set + i_indices))
#        return total_set
#    
#    init_set = list(np.where(np.array(alpha)!=0)[0])
#    new_set = enlarge_set(init_set)
#    while set(init_set) != set(new_set):
#        init_set = new_set
#        new_set = enlarge_set(init_set)   
#       
#    return np.sort(new_set)  

def select_indices(alpha, aux_cov):
    alpha = np.array(alpha)
    no_zeros = np.where(alpha != 0)[0]
    coeff_set = list(copy(no_zeros))
    for idx in no_zeros:
        new_set = np.where(aux_cov[idx,:] != 0)[0]
        coeff_set += list(new_set)
    
    coeff_set = np.sort(list(set(coeff_set)))
    return coeff_set


def reduce_indices(vec, indices):
    """
    Extracts subvector/submatrix indexed by indices
    """
    try:
        vec = np.array(vec)
    except np.ComplexWarning:
        print(vec)    
    if len(vec.shape) == 1:
        red_vec = vec[indices]
    if len(vec.shape) == 2:
        red_vec = vec[indices][:,indices]
    return red_vec


def extend_indices(red_vec, old_vec, indices):
    """
    puts red_vec in the indices of old_vec
    """
    red_vec = np.array(red_vec)
    old_vec = np.array(old_vec)
    if len(old_vec.shape) == 1:
        red_vec = red_vec.reshape(len(red_vec),)
        old_vec[indices] = red_vec
    if len(old_vec.shape) == 2:
        C = old_vec[indices]
        C[:,indices] = red_vec
        old_vec[indices] = C
    return old_vec


### compute moments functions

def partitionfunc(n,k,l=0):
    """
    n is the integer to partition, k is the length of partitions, l is the min partition element size
    """
    if k < 1:
        return
    if k == 1:
        if n >= l:
            yield (n,)
        return
    for i in range(l,n+1):
        for result in partitionfunc(n-i,k-1):
            yield (i,)+result

def _prob(mu, sigma, a, b):
    """
    Computes the mass probability of the normal distribution with mean mu and covariance matrix sigma in the 
    hyper-rectangle [a,b].
    Even for one-dimensional distributions, mu, sigma, a, b must be vectors.
    """
    #n = len(mu)
    #P = 0
    #for i_list in product(*[[0,1]]*n):
    #    x = np.zeros(n)
    #    for i, idx in enumerate(i_list):
    #        if idx==0:
    #            x[i] = a[i]
    #        else:
    #            x[i] = b[i]
    #    try:
    #        p = mvnorm.cdf(x,mean=mu,cov=sigma,allow_singular=True)
    #    except ValueError:
    #        sigma = make_psd(sigma)
    #        p = mvnorm.cdf(x,mean=mu,cov=sigma,allow_singular=True)
    #    if np.isnan(p):
    #        # due to a bug in scipy (https://github.com/scipy/scipy/issues/7669), when applied to two dimensional vectors mvnorm.cdf can return nan. The problem is solvable by adding a third variable, indipendent from the others (does not affect the computed probability).
    #        new_x = list(x) + [0]
    #        new_mu = list(mu) + [0]
    #        new_sigma = list(sigma)
    #        for i in range(len(sigma)):
    #            new_sigma[i] = list(sigma[i]) + [0]
    #        new_sigma.append([0]*(len(sigma)+1))
    #        p = mvnorm.cdf(new_x, mean=new_mu, cov=new_sigma, allow_singular=True)
    #    P = P + ((-1)**(n-sum(i_list)))*p
    P = mvnorm.cdf(b[0], mean=mu[0], cov=sigma[0,0], allow_singular=True) - mvnorm.cdf(a[0], mean=mu[0], cov=sigma[0,0], allow_singular=True) 
    return P
    

#def compute_lower_mom(mu, sigma, a, b, trunc_idx, trunc):
#    """
#    Given a normal with mean mu and cov matrix sigma,  truncated to [a,b] (where a[i] = -inf and b[i] = inf except
#    for a[trunc_idx] (if trunc = low) or b[trunc_idx] (if trunc=up)), computes the first two orders moments of a 
#    (n-1) dimensional normal distribution with mean \tilde(mu), \tilde(sigma) (as defined in Kan-Robotti).
#    """
#    n = len(mu)
#    c = np.delete(a, trunc_idx)
#    d = np.delete(b, trunc_idx)
#    # computes the new mean
#    if trunc == 'low':
#        muj = np.delete(mu, trunc_idx) + ((a[trunc_idx]-mu[trunc_idx])/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx]
#    elif trunc == 'up':
#        muj = np.delete(mu, trunc_idx) + ((b[trunc_idx]-mu[trunc_idx])/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx]
#    # computes the new covariance matrix
#    sigmaj = np.delete(np.delete(sigma, trunc_idx, axis=0), trunc_idx, axis=1)
#    sigmaj = sigmaj - (1/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx].reshape(len(sigma)-1,1) @         np.delete(sigma, trunc_idx, axis=1)[trunc_idx,:].reshape(1,len(sigma)-1)  
#    # saves the moments in a dictionary
#    dict_mom_lower = {}
#    for k in range(3):
#        for part in partitionfunc(k, n-1):
#            if sum(part) == 0:
#                dict_mom_lower[part] = 1
#            if sum(part) == 1:
#                idx = np.where(np.array(part) == 1)[0][0]
#                dict_mom_lower[part] = muj[idx]
#            if sum(part) == 2:
#                idx_list = np.where(np.array(part)!=0)[0]
#                if len(idx_list) == 2:
#                    idx1, idx2 = idx_list
#                    dict_mom_lower[part] = sigmaj[idx1, idx2] + muj[idx1]*muj[idx2]
#                elif len(idx_list) == 1:
#                    idx = idx_list[0]
#                    dict_mom_lower[part] = sigmaj[idx, idx] + muj[idx]**2
#    return dict_mom_lower

def compute_lower_mom(mu, sigma, a, b, trunc_idx, trunc):
    """
    Given a normal with mean mu and cov matrix sigma,  truncated to [a,b] (where a[i] = -inf and b[i] = inf except
    for a[trunc_idx] (if trunc = low) or b[trunc_idx] (if trunc=up)), computes the first two orders moments of a 
    (n-1) dimensional normal distribution with mean \tilde(mu), \tilde(sigma) (as defined in Kan-Robotti).
    """
    n = len(mu)
    c = np.delete(a, trunc_idx)
    d = np.delete(b, trunc_idx)
    # computes the new mean
    if trunc == 'low':
        muj = np.delete(mu, trunc_idx) + ((a[trunc_idx]-mu[trunc_idx])/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx]
    elif trunc == 'up':
        muj = np.delete(mu, trunc_idx) + ((b[trunc_idx]-mu[trunc_idx])/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx]
    # computes the new covariance matrix
    #sigmaj = np.delete(np.delete(sigma, trunc_idx, axis=0), trunc_idx, axis=1)
    #sigmaj = sigmaj - (1/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx].reshape(len(sigma)-1,1) @         np.delete(sigma, trunc_idx, axis=1)[trunc_idx,:].reshape(1,len(sigma)-1)  
    # saves the moments in a dictionary
    #sigmaj = sigmaj + muj.reshape(-1,1).dot(muj.reshape(1,-1))
    return muj


#def _compute_mom1(n, k, mu, sigma, a, b, trunc_idx, trunc, dict_mom):
#    c = np.zeros(n)
#    idx = np.where(np.array(k)==1)[0][0]
#    if trunc == 'low':
#        c[trunc_idx] = norm.pdf(a[trunc_idx], loc=mu[trunc_idx], scale=np.sqrt(sigma[trunc_idx,trunc_idx]))
#    elif trunc == 'up':
#        c[trunc_idx] = -norm.pdf(b[trunc_idx], loc=mu[trunc_idx], scale=np.sqrt(sigma[trunc_idx,trunc_idx]))
#    return mu[idx]*dict_mom[tuple(n*[0])] + np.array(k).dot(sigma).dot(c)           

def compute_mom1(n, mu, sigma, a, b, trunc_idx, trunc, P):
    c = np.zeros(n)
    if trunc == 'low':
        c[trunc_idx] = norm.pdf(a[trunc_idx], loc=mu[trunc_idx], scale=np.sqrt(sigma[trunc_idx,trunc_idx]))
    elif trunc == 'up':
        c[trunc_idx] = -norm.pdf(b[trunc_idx], loc=mu[trunc_idx], scale=np.sqrt(sigma[trunc_idx,trunc_idx]))
    return mu + sigma.dot(c)/P     


#def _compute_mom2(n, k, mu, sigma, a, b, trunc_idx, trunc, dict_mom, dict_mom_lower):
#    c = np.zeros(n)
#    index_list = np.where(np.array(k)!=0)[0]
#    if len(index_list) == 2:
#        idxk, idxe = index_list
#        ek = np.zeros(n)
#        ek[idxk] = 1
#        e = np.zeros(n)
#        e[idxe] = 1
#        for i in range(n):
#            if i == idxk:
#                c[i] = dict_mom[tuple(n*[0])]
#            if i == trunc_idx:
#                if trunc == 'low':
#                    c[i] = c[i] + (a[i]**ek[i])*norm.pdf(a[i], loc=mu[i], scale=np.sqrt(sigma[i,i]))*dict_mom_lower[tuple(np.delete(ek,i))]
#                elif trunc == 'up':
#                    c[i] = c[i] - (b[i]**ek[i])*norm.pdf(b[i], loc=mu[i], scale=np.sqrt(sigma[i,i]))*dict_mom_lower[tuple(np.delete(ek,i))]
#        return mu[idxe]*dict_mom[tuple(ek)] + e.dot(sigma).dot(c)   
#    elif len(index_list) == 1:
#        idx = index_list[0]
#        e = np.zeros(n)
#        e[idx] = 1
#        for i in range(n):
#            if i == idx:
#                c[i] = dict_mom[tuple(n*[0])]
#            if i == trunc_idx:
#                if trunc == 'low':
#                    c[i] = c[i] + (a[i]**e[i])*norm.pdf(a[i], loc=mu[i], scale=np.sqrt(sigma[i,i]))*dict_mom_lower[tuple(np.delete(e,i))]
#                elif trunc == 'up':
#                    c[i] = c[i] - (b[i]**e[i])*norm.pdf(b[i], loc=mu[i], scale=np.sqrt(sigma[i,i]))*dict_mom_lower[tuple(np.delete(e,i))]
#        return mu[idx]*dict_mom[tuple(e)] + e.dot(sigma).dot(c) 
    
def compute_mom2(n, mu, sigma, a, b, trunc_idx, trunc, new_P, new_mu, muj):
    e0 = np.zeros(n)
    e0[0] = 1
    C = new_P*np.eye(n)
    if trunc == 'low':
        C[:,0] += norm.pdf(a[0], mu[0], np.sqrt(sigma[0,0]))*(a[0]**e0)*np.array([1]+list(muj))
    elif trunc == 'up':
        C[:,0] += -norm.pdf(b[0], mu[0], np.sqrt(sigma[0,0]))*(b[0]**e0)*np.array([1]+list(muj))
    new_sigma = new_P*mu.reshape(-1,1).dot(new_mu.reshape(1,-1)) + sigma.dot(np.transpose(C))
    new_sigma = new_sigma/new_P - new_mu.reshape(-1,1).dot(new_mu.reshape(1,-1))
    return new_sigma  
    

def compute_moments(mu, sigma, a, b):
    """
    Given a normal distribution with mean mu and covariance matrix sigma, truncated to [a,b], where all a_i=-np.inf and
    all b_i=np.inf except at most one a_i or one b_i, computes exactly the mean and the covariance matrix of the 
    truncated distribution
    """        
    a = np.array(a)
    b = np.array(b)
    n = len(a)   
    # truncation in one dimension
    if n==1:
        new_P = norm.cdf(b[0], loc=mu[0], scale=np.sqrt(sigma[0,0])) - norm.cdf(a[0], loc=mu[0], scale=np.sqrt(sigma[0,0]))
        new_mu, new_sigma = truncnorm.stats(loc=mu[0], scale=np.sqrt(sigma[0,0]), a=(a[0]-mu[0])/np.sqrt(sigma[0,0]), b=(b[0]-mu[0])/np.sqrt(sigma[0,0]), moments='mv')
        new_mu = np.array([new_mu])
        new_sigma = np.array([[new_sigma]])
        return new_P, new_mu, new_sigma
    # if in more dimensions applies Kan-Robotti formulas
    # first determines if the truncation is 'low' (i.e. x > c) or 'up' (i.e. x < c)
    trunc_idx = 0
    if a[0] > -np.inf:
        trunc = 'low'
    else:
        trunc = 'up'  
    # returns the moments for the distribution of dimension n-1, in which the trunc_idx component has been removed
    #dict_mom_lower = compute_lower_mom(mu, sigma, a, b, trunc_idx, trunc)  
    muj = compute_lower_mom(mu, sigma, a, b, trunc_idx, trunc)
    # computes first two order moments using the recurrence formulas of Kan-Robotti and stores them in a dictionary
    new_P = _prob(mu, sigma, a, b)
    new_mu = compute_mom1(n, mu, sigma, a, b, trunc_idx, trunc, new_P)
    new_sigma = compute_mom2(n, mu, sigma, a, b, trunc_idx, trunc, new_P, new_mu, muj)
    #dict_mom = {}
    #for k in range(3):
    #    for part in partitionfunc(k, n): 
    #        if sum(part) == 0:
    #            dict_mom[part] = _prob(mu, sigma, a, b)
    #            if dict_mom[part] < prob_tol:
    #                return 0, mu, sigma
    #        if sum(part) == 1:
    #            dict_mom[part] = _compute_mom1(n, part, mu, sigma, a, b, trunc_idx, trunc, dict_mom)
    #        if sum(part) == 2:
    #            dict_mom[part] = _compute_mom2(n, part, mu, sigma, a, b, trunc_idx, trunc, dict_mom, dict_mom_lower)   
    ## assembles the dictionaries result in new_P, new_mu, new_sigma
    #new_P = dict_mom[tuple(n*[0])]
    #new_mu = np.zeros(n)
    #new_sigma = np.zeros((n,n))
    #for i in range(n):
    #    e = np.zeros(n)
    #    e[i] = 1
    #    new_mu[i] = dict_mom[tuple(e)]/new_P
    #    new_sigma[i,i] = dict_mom[tuple(2*e)]/new_P - (dict_mom[tuple(e)]/new_P)**2
    #    for j in range(i):
    #        f = np.zeros(n)
    #        f[j] = 1
    #        new_sigma[i,j] = new_sigma[j,i] = dict_mom[tuple(e+f)]/new_P - (dict_mom[tuple(e)]/new_P)*(dict_mom[tuple(f)]/new_P)
    return new_P, new_mu, new_sigma

### conditioning to zero probability events

def insert_value(val, idx, mu, sigma):
    """ Extends mu and sigma by adding val in corresponding to the idx position (for sigma the other row- and column-entries are 0) """
    d = len(mu)
    new_mu = np.array(list(mu[:idx]) + [val] + list(mu[idx:]))
    new_sigma = np.block([[sigma[:idx,:idx], np.zeros((idx,1)), sigma[:idx,idx:]], 
          [np.zeros((1,d+1))],
          [sigma[idx:,:idx], np.zeros((d-idx,1)), sigma[idx:,idx:]]])
    return new_mu, new_sigma
    
    
    