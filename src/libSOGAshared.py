# Contains some general purpose classes, functions and variables used by the SOGA Python Libraries. 
# In particular it contains:
# - import statement for auxiliary Python libraries;
# - definition of tolerance parameters used by various functions;
# - classes definition for representing distributions and Gaussian Mixtures;
# - function definitions for numerical stability of the covariance matrices;
# - function definitions invoked by multiple functions in different libraries.

# TO DO:
# -  add controls on the attributes of GaussianMix (lenghts of pi, mu, sigma, dimensions of mu and sigma)

# AUXILIARY LIBRARIES 
import torch
import torch.distributions as distributions
import botorch.utils.probability.mvnxpb as mvn
from copy import deepcopy, copy

#from sympy import *
#import re
import numpy as np
#from scipy.stats import norm
#from scipy.stats import truncnorm
#from scipy.stats import multivariate_normal as mvnorm
from itertools import product
from functools import partial


### TOLERANCE PARAMETERS 

EPS = 1e-5              # used on the diagonal of the initial distribution
#DELTA_EIG = 1e-8        # when making the cov matrix psd increases eigenvalues by this
TOL_EIG = 1e-5          # eigenvalues below this value are considered zero
TOL_PROB = 1e-4         # probability below prob_tol are treated as zero
TOL_ERR = 5e-3          # error tolerance (print an error message if error is above)
INFTY = 1e10            # infinity


#delta_tol = 1e-10 # if the 1-norm of a covariance matrix is <= delta_tol the corresponding Gaussian component is treated as a delta
#prob_tol = 1e-10 # probability below prob_tol are treated as zero
#eig_tol = 1e-4

### CLASSES FOR DISTRIBUTIONS AND GAUSSIAN MIXTURES

class GaussianMix():
    """ A Gaussian Mixtures is represented by a list of mixing coefficients (stored in pi), a list of means (stored in mu) and a list of covariance matrices (stored in sigma)."""
    
    def __init__(self, pi, mu, sigma):
        self.pi = list(pi)         # pi is a list of scalar tensors whose sum is 1
        self.mu = list(mu)         # mu is a list, with len(mu)==len(pi) and each element is a tensor
        self.sigma = list(sigma)   # sigma is a list, with len(sigma)==len(pi), and each element is a covariance matrix (tensor)
    
    def n_comp(self):
        return len(self.pi)
    
    def n_dim(self):
        return len(self.mu[0])
    
    def __repr__(self):
        str_repr = 'pi: ' + str(self.pi) + '  mu: ' + str(self.mu) + ' sigma: ' + str(self.sigma)
        return str_repr
    
    def comp(self, k):
        return GaussianMix([torch.tensor(1.)], [torch.clone(self.mu[k])], [torch.clone(self.sigma[k])])
        
    # Pdfs 
    def comp_pdf(self, x, k):
        if self.n_dim() > 1:
            try:
                return torch.exp(distributions.MultivariateNormal(self.mu[k], covariance_matrix=self.sigma[k]).log_prob(x))
            except ValueError:
                #print('Degenerate covariance matrix, I will proceed to correct it')
                sigma = self.sigma[k]
                eigs, _ = torch.linalg.eigh(sigma)
                is_psd = torch.all(eigs > 0)
                is_sym = torch.all(sigma == sigma.T)
                if not is_psd:
                    print(eigs)
                    print('matrix is not psd!')
                    print(sigma)
                    raise Error
                if not is_sym:
                    #print('matrix is not sym!')                    
                    self.sigma[k] = make_sym(self.sigma[k])
                return torch.exp(distributions.MultivariateNormal(self.mu[k], covariance_matrix=self.sigma[k]).log_prob(x))
        else:
            return torch.exp(distributions.Normal(self.mu[k], torch.sqrt(self.sigma[k])).log_prob(x)).reshape(x.shape)
            
    def marg_comp_pdf(self, x, k, idx):
        if isinstance(idx, list):
            cov_submatrix = torch.clone(self.sigma[k][torch.tensor(idx).unsqueeze(1), torch.tensor(idx)])
            try:
                return torch.exp(distributions.MultivariateNormal(self.mu[k][idx], cov_submatrix).log_prob(x))
            except ValueError as e:
                # make the covariance matrix symmetric
                cov_submatrix = make_psd(cov_submatrix)
                cov_submatrix = make_sym(cov_submatrix)
                return torch.exp(distributions.MultivariateNormal(self.mu[k][idx], cov_submatrix).log_prob(x))
        else:
            return torch.exp(distributions.Normal(self.mu[k][idx], torch.sqrt(self.sigma[k][idx,idx])).log_prob(x))

    
    def pdf(self, x):
        pdf = torch.zeros(x.shape)
        for k in range(self.n_comp()):
            pdf += self.pi[k]*self.comp_pdf(x,k)
        return pdf 
        
    def marg_pdf(self, x, idx):
        marg = torch.zeros(x.shape)
        for k in range(self.n_comp()):
            marg += self.pi[k]*self.marg_comp_pdf(x,k,idx) 
        return marg
        
    # Cdfs
    def comp_cdf(self, x, k):
        if self.n_dim() > 1:
            return mvncdf(x, self.mu[k], self.sigma[k])
        else:
            return distributions.Normal(self.mu[k], torch.sqrt(self.sigma[k])).cdf(x)
            
    def marg_comp_cdf(self, x, k, idx):
        return distributions.Normal(self.mu[k][idx], torch.sqrt(self.sigma[k][idx,idx])).cdf(x)
        
    def cdf(self, x):
        cdf = torch.zeros(x.shape)
        for k in range(self.n_comp()):
            cdf += self.pi[k]*self.comp_cdf(x,k) 
        return cdf
    
    def marg_cdf(self, x, idx):
        marg = torch.zeros(x.shape)
        for k in range(self.n_comp()):
            marg += self.pi[k]*self.marg_comp_cdf(x,k,idx) 
        return marg      
    
    # Moments of mixtures
    def mean(self):
        return torch.tensordot(torch.hstack(self.pi), torch.vstack(self.mu), dims=1)
    
    def cov(self):
        v = torch.vstack(self.mu) - self.mean()
        cov = torch.tensordot(torch.hstack(self.pi), torch.stack(self.sigma), dims=1) + torch.mm(v.t(), torch.vstack(self.pi)*v)
        return cov


class Dist():
    """ A distribution is given by a ordered list of variable names, stored in var_list, and a Gaussian Mixture, stored in gm, describing the joint distribution over the variable vector"""
    def __init__(self, var_list, gm):
        self.var_list = var_list
        self.gm = gm
        
    def __str__(self):
        return 'Dist<{},{}>'.format(self.var_list, self.gm)
    
    def __repr__(self):
        return str(self)

### CDF FUNCTION OF MULTIVARIATE GAUSSIAN

def mvncdf(x, mean, cov):
    bounds = torch.tensor([[-torch.inf, x[i] - mean[i]] for i in range(len(x))])
    res = torch.exp(mvn.MVNXPB(covariance_matrix=cov, bounds=bounds).solve())
    if res.isnan():
        cov = make_psd(cov)
        res = torch.exp(mvn.MVNXPB(covariance_matrix=cov, bounds=bounds).solve())
    return res

### CUSTOM CLASS FOR UNIVARIATE TRUNCATED NORMAL

class TruncatedNormal():
    """ Univariate Truncated Normal distribution. Helps computing moments using torch utils."""

    def __init__(self, loc, scale, a, b):
        self.loc = loc
        self.scale = scale
        self.low_bound = a
        self.up_bound = b

        # auxiliary normal
        self.norm = distributions.Normal(self.loc, self.scale)

        # rescaled bounds
        self.alpha = (self.low_bound - self.loc)/(self.scale)
        self.beta = (self.up_bound - self.loc)/(self.scale)
        self.phi_alpha = self.norm.log_prob(self.alpha).exp()
        self.phi_beta = self.norm.log_prob(self.beta)
        self.phi_beta = self.phi_beta.exp()

        # normalization constant
        self.norm_const = self.norm.cdf(self.up_bound) - self.norm.cdf(self.low_bound)

    # mean
    def mean(self):
        return self.loc + self.scale*(self.phi_alpha - self.phi_beta)/self.norm_const

    # variance
    def var(self):
        
        if self.phi_beta != 0:
            prod_beta = self.beta * self.phi_beta
        else:
            prod_beta = torch.tensor(0.)
        if self.phi_alpha != 0:
            prod_alpha = self.alpha * self.phi_alpha
        else:
            prod_alpha = torch.tensor(0.)   
        return self.scale**2*(torch.tensor(1.) - (prod_beta - prod_alpha)/self.norm_const - ((self.phi_alpha - self.phi_beta)/self.norm_const)**2)
        
### FUNCTIONS FOR NUMERICAL STABILITY OF COVARIANCE MATRICES

# this is commented because it breaks the gradient, it is better not to cause non-psd matrices (use smoothing instead)
#def make_psd(sigma):
#    """
#    Triggered when sigma is not positive semidefinite. Sets to 1e-10 negative eigenvalues of sigma. If the eigenvalues or the total error in the substitution are above a certain threshold prints an error message.
#    """
#    eig, M = torch.linalg.eigh(sigma)
#    add = 0
#    c_it = 0
#    if torch.all(eig > TOL_EIG):
#        #print('not correcting')
#        return sigma
#    while not torch.all(eig > TOL_EIG):
#    #while True:
#        c_it+=1
#        add += DELTA_EIG
#        for i, e in enumerate(eig):
#            if e <= TOL_EIG:
#                eig[i] = add
#        new_sigma = torch.mm(torch.mm(M, torch.diag(eig)), M.t())
#        eig, M = torch.linalg.eigh(new_sigma)
#    rel_err = torch.sum(torch.abs(new_sigma-sigma))
#    if rel_err > TOL_ERR:
#        print('Warning: eigenvalue substitution led to an error of: {}'.format(rel_err))
#    #print('corrected output', new_sigma)
#    return new_sigma

def make_sym(sigma, eig_tol=1e-3):
    """
    Makes a 2D PyTorch tensor symmetric by averaging mismatched elements.
    Prints an error message if the error exceeds eig_tol.
    """
    # Ensure sigma is a PyTorch tensor
    sigma = sigma.clone()  # To avoid modifying the input tensor
    # Average the matrix with its transpose
    symmetric_sigma = (sigma + sigma.T) / 2
    # Compute the difference introduced by the symmetrization
    diff = torch.abs(symmetric_sigma - sigma)
    # Find indices where the difference exceeds the tolerance
    indices = torch.nonzero(diff > TOL_ERR, as_tuple=False)
    for i, j in indices:
        print(f"Substituting {sigma[i, j].item()} with {symmetric_sigma[i, j].item()}")
    return symmetric_sigma

        
### SHARED FUNCTIONS 

#def extract_aux(dist, trunc):
#    """ Parses a string trunc to check for any gm(pi, mu, sigma) variable and adds it to dist in the form of an auxialiary variable with suitable parameters """
#    groups = [m.group() for m in re.finditer('gm\(.*?\)', trunc)]
#    aux_dist = deepcopy(dist)
#    aux_trunc = trunc
#    # for each gm(pi, mu, sigma) a new variable is added
#    for n_aux, group in enumerate(groups):
#        new_pi = []
#        new_mu = []
#        new_sigma = []
#        aux_name = 'aux{}'.format(n_aux)
#        aux_trunc = aux_trunc.replace(group, aux_name)
#        pi_list, mu_list, sigma_list = [eval(m.group()) for m in re.finditer('\[.*?\]', group)]
#        aux_dist.var_list.append(aux_name)
#        # for each component of the original distribution dist and for each component of a variable gm(pi, mu, sigma) a new Gaussian component is generated with mixing coefficient dist.pi[i]*pi[i].
#        for k in range(aux_dist.gm.n_comp()):
#            for j in range(len(pi_list)):
#                new_pi.append(aux_dist.gm.pi[k]*pi_list[j])
#                new_mu.append(np.hstack((aux_dist.gm.mu[k], mu_list[j])))
#                old_sigma = aux_dist.gm.sigma[k]
#                d = len(old_sigma)
#                aux_sigma = np.zeros((d+1,d+1))
#                aux_sigma[:d,:d] = old_sigma
#                aux_sigma[-1,-1] = sigma_list[j]**2
#                new_sigma.append(aux_sigma)
#        aux_dist.gm = GaussianMix(new_pi, new_mu, new_sigma)
#    return aux_dist, aux_trunc
#
#def substitute_deltas(dist, trunc):
#    """ Substitutes variables in trunc which are Dirac Delta """
#    mu = dist.gm.mu[0]
#    sigma = dist.gm.sigma[0]
#    for i in range(len(sigma)):
#        if sigma[i,i] < delta_tol:
#            trunc = trunc.subs({dist.var_list[i]:mu[i]})
#    return trunc
