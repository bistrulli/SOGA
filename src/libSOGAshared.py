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
#from copy import deepcopy, copy

#from sympy import *
import re
#import numpy as np
#from scipy.stats import norm
#from scipy.stats import truncnorm
#from scipy.stats import multivariate_normal as mvnorm
from itertools import product
from functools import partial


### TOLERANCE PARAMETERS 

EPS = 1e-5              # used on the diagonal of the initial distribution
#DELTA_EIG = 1e-8        # when making the cov matrix psd increases eigenvalues by this
TOL_EIG = 1e-15          # eigenvalues below this value are considered zero
TOL_PROB = 1e-10# 1e-4         # probability below prob_tol are treated as zero
TOL_ERR = 5e-3          # error tolerance (print an error message if error is above)
INFTY = 1e10            # infinity
# SMOOTHING PARAMETERS
SMOOTH_EPS = 1e-5    # starting noise for smoothing
SMOOTH_DELTA = 1e-5  # addition to gaussian noise for smoothing


#delta_tol = 1e-10 # if the 1-norm of a covariance matrix is <= delta_tol the corresponding Gaussian component is treated as a delta
#prob_tol = 1e-10 # probability below prob_tol are treated as zero
#eig_tol = 1e-4

### CLASSES FOR DISTRIBUTIONS AND GAUSSIAN MIXTURES

class GaussianMix():
    """ A Gaussian Mixtures is represented by a list of mixing coefficients (stored in pi), a list of means (stored in mu) and a list of covariance matrices (stored in sigma)."""
    
    def __init__(self, pi, mu, sigma):
        self.pi = pi         # pi is a tensor (c, 1) where c is the number of components
        self.mu = mu         # mu is a tensor (c, d) where d is the dimension of the space
        self.sigma = sigma   # sigma is a tensor (c, d, d) where d is the dimension of the space
    
    def n_comp(self):
        return self.pi.shape[0]
    
    def n_dim(self):
        return self.mu.shape[1]
    
    def __repr__(self):
        str_repr = 'pi: ' + str(self.pi) + '\nmu: ' + str(self.mu) + '\nsigma: ' + str(self.sigma)
        return str_repr
    
    def comp(self, k):
        return GaussianMix(torch.tensor([[1.]]), torch.clone(self.mu[:,k]), torch.clone(self.sigma[:,:,k]))
        
    # Pdfs 
    def comp_pdf(self, x, k):
        if self.n_dim() > 1:
            try:
                return torch.exp(distributions.MultivariateNormal(self.mu[k], covariance_matrix=self.sigma[k]).log_prob(x))
            except ValueError:
                sigma = self.sigma[k]
                eigs, _ = torch.linalg.eigh(sigma)
                is_psd = torch.all(eigs > 0)
                is_sym = torch.all(sigma == sigma.T)
                if not is_psd:
                    print(eigs)
                    print('matrix is not psd!')
                    print(sigma)
                    raise 
                if not is_sym:
                    self.sigma[k] = make_sym(self.sigma[k])
                return torch.exp(distributions.MultivariateNormal(self.mu[k], covariance_matrix=self.sigma[k]).log_prob(x))
        else:
            return torch.exp(distributions.Normal(self.mu[:,k], torch.sqrt(self.sigma[:,:,k])).log_prob(x)).reshape(x.shape)
            
    def marg_comp_pdf(self, x, k, idx):
        if isinstance(idx, list):
            cov_submatrix = torch.clone(self.sigma[k][torch.tensor(idx).unsqueeze(1), torch.tensor(idx)])
            try:
                return torch.exp(distributions.MultivariateNormal(self.mu[k][idx], cov_submatrix).log_prob(x))
            except ValueError:
                eigs, _ = torch.linalg.eigh(cov_submatrix)
                is_psd = torch.all(eigs > 0)
                is_sym = torch.all(cov_submatrix == cov_submatrix.T)
                if not is_psd:
                    print(eigs)
                    print('matrix k={} is not psd!'.format(k))
                    print(cov_submatrix)
                    raise 
                if not is_sym:
                    print('matrix k={} is not symmetric!'.format(k))
                    self.sigma[k][torch.tensor(idx).unsqueeze(1), torch.tensor(idx)] = new_cov_submatrix = make_sym(cov_submatrix)
                return torch.exp(distributions.MultivariateNormal(self.mu[k][idx], covariance_matrix=new_cov_submatrix).log_prob(x))
        else:
            return torch.exp(distributions.Normal(self.mu[k][idx], torch.sqrt(self.sigma[k][idx,idx])).log_prob(x))

    
    def pdf(self, x):
        #comp_pdfs = torch.stack([self.comp_pdf(x, k) for k in range(self.n_comp())], dim=1)
        comp_pdfs = torch.vstack([self.comp_pdf(x, k) for k in range(self.n_comp())])
        pdf = torch.matmul(comp_pdfs, self.pi.view(-1, 1))
        return pdf
        
    
    def marg_pdf(self, x, idx):
        comp_pdfs = torch.stack([self.marg_comp_pdf(x, k,idx) for k in range(self.n_comp())], dim=1)
        pdf = comp_pdfs*self.pi.view(-1,1)
        return torch.sum(pdf, dim=1)
        
    # Cdfs
    
    def comp_cdf(self, x, k):
        if self.n_dim() > 1:
            return mvncdf(x, self.mu[k], self.sigma[k])
        else:
            return distributions.Normal(self.mu[k], torch.sqrt(self.sigma[k])).cdf(x)
            
    def marg_comp_cdf(self, x, k, idx):
        if isinstance(idx, list):
            cov_submatrix = torch.clone(self.sigma[k][torch.tensor(idx).unsqueeze(1), torch.tensor(idx)])
            return mvncdf(x, self.mu[k][idx], cov_submatrix)
        else:
            return distributions.Normal(self.mu[k][idx], torch.sqrt(self.sigma[k][idx,idx])).cdf(x)
        
    
    def cdf(self, x):
        comp_cdfs = torch.stack([self.comp_cdf(x, k) for k in range(self.n_comp())], dim=1)
        cdf = torch.matmul(comp_cdfs, self.pi.view(-1, 1))
        return cdf
    
    def marg_cdf(self, x, idx):
        comp_cdfs = torch.stack([self.marg_comp_cdf(x, k, idx) for k in range(self.n_comp())], dim=1)
        cdf = torch.matmul(comp_cdfs, self.pi.view(-1, 1))
        return cdf
      
    
    # Moments of mixtures
    def mean(self):
        return torch.sum(self.pi * self.mu, dim=0)
    
    def cov(self):
        pi = self.pi.view(-1, 1, 1)  
        v = self.mu - self.mean()
        cov = (pi * self.sigma).sum(dim=0) + torch.mm(v.t(), self.pi*v)
        return cov
    
    # utilities 

    def delete_zeros(self):
        indexes = torch.where(self.pi >= TOL_PROB)
        self.pi = self.pi[indexes].reshape(-1,1)
        self.pi = self.pi/torch.sum(self.pi)
        self.mu = self.mu[indexes[0], :]
        self.sigma = self.sigma[indexes[0], :, :]


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
    # Ensure x has a batch dimension
    if x.dim() == 1:
        x = x.unsqueeze(0)
    batch_size = x.shape[0]
    dim = x.shape[1]
    # Compute bounds for each x in the batch
    bounds = torch.stack([torch.tensor([[-torch.inf, x[i, j] - mean[j]] for j in range(dim)]) for i in range(batch_size)])
    # Initialize result tensor
    res = torch.zeros(batch_size)
    for i in range(batch_size):
        result = torch.exp(mvn.MVNXPB(covariance_matrix=cov, bounds=bounds[i]).solve())
        if result.isnan():
            cov = make_sym(cov)
            result = torch.exp(mvn.MVNXPB(covariance_matrix=cov, bounds=bounds[i]).solve())
        res[i] = result
    return res

# THIS VERSION ONLY WORKS FOR SINGLE SAMPLES x
#def mvncdf(x, mean, cov):
#    bounds = torch.tensor([[-torch.inf, x[i] - mean[i]] for i in range(len(x))])
#    res = torch.exp(mvn.MVNXPB(covariance_matrix=cov, bounds=bounds).solve())
#    if res.isnan():
#        cov = make_sym(cov)
#        res = torch.exp(mvn.MVNXPB(covariance_matrix=cov, bounds=bounds).solve())
#    return res

### CUSTOM CLASS FOR UNIVARIATE TRUNCATED NORMAL

class TruncatedNormal():
    """ Univariate Truncated Normal distribution. Helps computing moments using torch utils."""

    def __init__(self, loc, scale, a, b):
        self.loc = loc
        self.scale = scale
        self.low_bound = a
        self.up_bound = b

        # auxiliary normal
        self.norm = distributions.Normal(torch.zeros(self.loc.shape), torch.ones(self.scale.squeeze(2).shape))

        # rescaled bounds
        self.alpha = (self.low_bound - self.loc)/(self.scale.squeeze(2))
        self.beta = (self.up_bound - self.loc)/(self.scale.squeeze(2))
        self.phi_alpha = self.norm.log_prob(self.alpha).exp()
        self.phi_beta = self.norm.log_prob(self.beta).exp()

        # normalization constant
        self.norm_const = self.norm.cdf(self.beta) - self.norm.cdf(self.alpha)
    # mean
    def mean(self):
        return self.loc + self.scale.squeeze(2)*(self.phi_alpha - self.phi_beta)/self.norm_const

    # variance
    def var(self):
        prod_beta = self.beta * self.phi_beta
        prod_alpha = self.alpha * self.phi_alpha  
        return (self.scale.squeeze(2)**2*(torch.tensor(1.) - (prod_beta - prod_alpha)/self.norm_const - ((self.phi_alpha - self.phi_beta)/self.norm_const)**2)).unsqueeze(2)
    
### FUNCTIONS FOR PARSING

def extend_dist(self, dist):
    """ Extends the current distribution with the auxiliary variables. Returns a new GaussianMix object."""

    if len(self.aux_pis) > 0:
        old_dim = dist.gm.n_dim()
        new_dim = old_dim + len(self.aux_pis)

        new_pis = torch.empty((0,1))
        new_mus = torch.empty((0,new_dim))
        new_sigmas = torch.empty((0,new_dim,new_dim))
        for part in product(*[range(len(mean)) for mean in self.aux_means]):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            # for each combination multiplies the original weight by the weights of the combination
            aux_pi = torch.prod(torch.tensor([self.aux_pis[i][part[i]] for i in range(len(part))]))
            new_pis = torch.vstack([new_pis, (aux_pi*dist.gm.pi)]) 
            # for each combination creates new means
            aux_mu = torch.hstack([self.aux_means[i][part[i]] for i in range(len(part))])
            new_mus = torch.vstack([new_mus, torch.cat([dist.gm.mu, aux_mu.expand(dist.gm.n_comp(), len(aux_mu))], dim=1)])
            # for each combination creates new covs
            aux_sigma = torch.diag(torch.hstack([self.aux_covs[i][part[i]] for i in range(len(part))]))
            aux_sigmas = torch.zeros((dist.gm.n_comp(), new_dim, new_dim))
            aux_sigmas[:, :old_dim, :old_dim] = dist.gm.sigma
            aux_sigmas[:, old_dim:, old_dim:] = aux_sigma
            new_sigmas = torch.vstack([new_sigmas, aux_sigmas])

        extended_gm = GaussianMix(new_pis, new_mus, new_sigmas)
        extended_gm.delete_zeros()

        return extended_gm
    else:
        return dist.gm

        
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