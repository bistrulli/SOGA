# Contains some general purpose classes, functions and variables used by the SOGA Python Libraries. 
# In particular it contains:
# - import statement for auxiliary Python libraries;
# - definition of an handle used for debugging;
# - definition of tolerance parameters used by various functions;
# - classes definition for representing distributions and Gaussian Mixtures;
# - function definitions for numerical stability of the covariance matrices;
# - function definitions invoked by multiple functions in different libraries.

# TO DO:
# -  add controls on the attributes of GaussianMix (lenghts of pi, mu, sigma, dimensions of mu and sigma)

# AUXILIARY LIBRARIES 

from copy import deepcopy, copy
from sympy import *
import re
import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import multivariate_normal as mvnorm
from itertools import product, chain
from functools import partial
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any


from time import time

### DEBUGGING METHOD

import functools

def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})\n")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}\n")           # 4
        return value
    return wrapper_debug 

### TOLERANCE PARAMETERS

delta_tol = 1e-10 # if the 1-norm of a covariance matrix is <= delta_tol the corresponding Gaussian component is treated as a delta
prob_tol = 1e-10 # probability below prob_tol are treated as zero
eig_tol = 1e-4

### CUSTOM EXCEPTIONS

class NumericalError(Exception):
    """Raised when a numerical operation fails after exhausting retries.

    Attributes:
        message: human-readable description
        kwargs: context metadata (e.g. min_eig, n_retries)
    """
    def __init__(self, message: str, **kwargs):
        super().__init__(message)
        self.context = kwargs

    def __str__(self):
        base = super().__str__()
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{base} ({ctx_str})"
        return base

### CLASSES FOR DISTRIBUTIONS AND GAUSSIAN MIXTURES

class GaussianMix():
    """ A Gaussian Mixtures is represented by a list of mixing coefficients (stored in pi), a list of means (stored in mu) and a list of covariance matrices (stored in sigma)."""
    
    def __init__(self, pi, mu, sigma):
        self.pi = list(pi)         # pi is a list of scalars whose sum is 1
        self.mu = list(mu)         # mu is a list, with len(mu)==len(pi) and each element is an array
        self.sigma = list(sigma)   # sigma is a list, with len(sigma)==len(pi), and each element is a covariance matrix
    
    def n_comp(self):
        return len(self.pi)
    
    def n_dim(self):
        return len(self.mu[0])
    
    def __repr__(self):
        str_repr = 'pi: ' + str(self.pi) + ' mu: ' + str(self.mu) + ' sigma: ' + str(self.sigma)
        return str_repr
    
    def comp(self, k):
        return GaussianMix([1.], [self.mu[k]], [self.sigma[k]])
        
    # Pdfs and Cdfs 
    def comp_pdf(self, x, k):
        if np.sum(abs(self.sigma[k])) > delta_tol:
            return mvnorm.pdf(x, mean=self.mu[k], cov=self.sigma[k], allow_singular=True)
        else:
            if np.all(x == self.mu[k]):
                return 1.0
            else:
                return 0.0
            
    def marg_comp_pdf(self, x, k, idx):
        if np.sum(abs(self.sigma[k])) > delta_tol:
            return norm.pdf(x, loc=self.mu[k][idx], scale=np.sqrt(self.sigma[k][idx,idx]))
        else:
            if np.all(x == self.mu[k][idx]):
                return 1.0
            else:
                return 0.0
    
    def pdf(self, x):
        return sum([self.pi[k]*self.comp_pdf(x,k) for k in range(self.n_comp())])
    
    def marg_pdf(self, x, idx):
        return sum([self.pi[k]*self.marg_comp_pdf(x,k,idx) for k in range(self.n_comp())])
    
    def comp_cdf(self, x, k):
        return mvnorm.cdf(x, mean=self.mu[k], cov=self.sigma[k], allow_singular=True)
    
    def marg_comp_cdf(self, x, k, idx):
        if np.sum(abs(self.sigma[k])) > delta_tol:
            return norm.cdf(x, loc=self.mu[k][idx], scale=np.sqrt(self.sigma[k][idx,idx]))
        
    def cdf(self, x):
        return sum([self.pi[k]*self.comp_cdf(x,k) for k in range(self.n_comp())])
    
    def marg_cdf(self, x, idx):
        return sum([self.pi[k]*self.marg_comp_cdf(x,k,idx) for k in range(self.n_comp())])
      
    
    # Moments of mixtures
    def mean(self):
        return np.array(self.pi).dot(np.array(self.mu))
    
    def cov(self):
        v = np.array(self.mu) - self.mean()
        cov = np.tensordot(np.array(self.pi), np.array(self.sigma), axes=1) + np.transpose(v).dot((np.array(self.pi).reshape(-1,1)*v))
        return cov


@dataclass
class VarEntry:
    """Type-tracking entry for a variable in a Dist.

    Added in M2.1 to enable scalar vs matrix dispatch in libSOGAupdate /
    libSOGAtruncate without parsing the LHS string at every assignment.
    Populated by producecfg's enterMatrix listener (M2.3) for each matrix
    declaration. Scalar variables are not yet tracked here in M2; they may
    be added in a later milestone if needed for symmetry.

    Fields
    ------
    name : str
        Variable identifier as it appears in the .soga source.
    kind : str
        'scalar' (placeholder, not currently populated) or 'matrix'.
    shape : tuple
        () for scalar, (m, n) for matrix variables.
    flat_offset : int
        Position of the variable in the flat vec(...) representation used
        by the joint covariance. -1 if not yet placed (M3 fills this in).
    """
    name: str
    kind: str
    shape: Tuple[int, ...]
    flat_offset: int = -1


class Dist():
    """ A distribution is given by a ordered list of variable names, stored in var_list, and a Gaussian Mixture, stored in gm, describing the joint distribution over the variable vector.

    M2.2 extension: optional var_entries (list of VarEntry, set by producecfg.enterMatrix
    listener — see plan M2.3) and gm_block (GaussianMixBlock, created in M3.1; the
    type is intentionally `Optional[Any]` here because libSOGAsharedMatrix does not
    exist yet at M2 — added as forward-compatible field placeholder). Existing
    positional signature `(var_list, gm)` is unchanged so all M0/M1 scalar code paths
    continue to construct Dist via positional args without touching the new fields.
    """
    def __init__(self, var_list, gm, var_entries: Optional[List[VarEntry]] = None, gm_block: Optional[Any] = None):
        self.var_list = var_list
        self.gm = gm
        self.var_entries = var_entries if var_entries is not None else []
        self.gm_block = gm_block

    def __str__(self):
        return 'Dist<{},{}>'.format(self.var_list, self.gm)

    def __repr__(self):
        return str(self)
        
### FUNCTIONS FOR NUMERICAL STABILITY OF COVARIANCE MATRICES

def make_psd(sigma):
    """Make sigma positive semi-definite with a conditional clip strategy.

    Strategy: clip only negative eigenvalues to a small positive jitter derived
    from the matrix trace (not an unconditional global shift, which would perturb
    already-PSD inputs). Retries up to N=3 times with halved jitter if residual
    negative eigenvalues remain after recomposition round-off.

    Raises NumericalError if PSD cannot be achieved after N=3 retries.
    """
    _MAX_RETRIES = 3
    new_sigma = make_sym(sigma)
    d = new_sigma.shape[0]
    eig, Q = np.linalg.eigh(new_sigma)

    if np.all(eig > 1e-15):
        # Already PSD — return immediately without perturbation
        return new_sigma

    # Compute adaptive jitter from trace
    trace_val = float(np.trace(new_sigma))
    base_jitter = max(1e-8, 1e-10 * trace_val / d)

    jitter = base_jitter
    for retry in range(_MAX_RETRIES + 1):
        # Clip only negative eigenvalues; leave positive ones untouched
        eig_clipped = np.where(eig <= 1e-15, jitter, eig)
        new_sigma = Q.dot(np.diag(eig_clipped)).dot(Q.T)
        eig, Q = np.linalg.eigh(new_sigma)
        if np.all(eig > 1e-15):
            break
        jitter = jitter / 2.0
    else:
        min_eig_val = float(np.min(eig))
        raise NumericalError(
            "make_psd failed after N=3 clips",
            min_eig=min_eig_val,
            n_retries=_MAX_RETRIES,
        )

    rel_err = float(np.sum(np.abs(new_sigma - sigma)))
    if rel_err > eig_tol:
        print(f"Warning: make_psd eigenvalue clip led to an error of: {rel_err:.4g}")
    return new_sigma

def make_sym(sigma):
    """ 
    Reassigns sigma to make it symmetric. For sigma[i,j]!=sigma[j,i] substitutes both with the average value. If the substitution leads to an error above a certain threshold prints an error message.
    """
    for i in range(len(sigma)):
        for j in range(i+1,len(sigma)):
            if sigma[i,j] != sigma[j,i]:
                v = (sigma[i,j] + sigma[j,i])/2
                if abs(v-sigma[i,j]) > eig_tol: 
                    print('substituting {} with {}'.format(sigma[i,j], v))
                if abs(v-sigma[i,j]) > eig_tol: 
                    print('substituting {} with {}'.format(sigma[j,i], v))
                sigma[i,j] = sigma[j,i] = v
    return sigma

        
### SHARED FUNCTIONS 

def extract_aux(dist, trunc):
    """ Parses a string trunc to check for any gm(pi, mu, sigma) variable and adds it to dist in the form of an auxialiary variable with suitable parameters """
    groups = [m.group() for m in re.finditer('gm\(.*?\)', trunc)]
    aux_dist = deepcopy(dist)
    aux_trunc = trunc
    # for each gm(pi, mu, sigma) a new variable is added
    for n_aux, group in enumerate(groups):
        new_pi = []
        new_mu = []
        new_sigma = []
        aux_name = 'aux{}'.format(n_aux)
        aux_trunc = aux_trunc.replace(group, aux_name)
        pi_list, mu_list, sigma_list = [eval(m.group()) for m in re.finditer('\[.*?\]', group)]
        aux_dist.var_list.append(aux_name)
        # for each component of the original distribution dist and for each component of a variable gm(pi, mu, sigma) a new Gaussian component is generated with mixing coefficient dist.pi[i]*pi[i].
        for k in range(aux_dist.gm.n_comp()):
            for j in range(len(pi_list)):
                new_pi.append(aux_dist.gm.pi[k]*pi_list[j])
                new_mu.append(np.hstack((aux_dist.gm.mu[k], mu_list[j])))
                old_sigma = aux_dist.gm.sigma[k]
                d = len(old_sigma)
                aux_sigma = np.zeros((d+1,d+1))
                aux_sigma[:d,:d] = old_sigma
                aux_sigma[-1,-1] = sigma_list[j]**2
                new_sigma.append(aux_sigma)
        aux_dist.gm = GaussianMix(new_pi, new_mu, new_sigma)
    return aux_dist, aux_trunc

def substitute_deltas(dist, trunc):
    """ Substitutes variables in trunc which are Dirac Delta """
    mu = dist.gm.mu[0]
    sigma = dist.gm.sigma[0]
    for i in range(len(sigma)):
        if sigma[i,i] < delta_tol:
            trunc = trunc.subs({dist.var_list[i]:mu[i]})
    return trunc
