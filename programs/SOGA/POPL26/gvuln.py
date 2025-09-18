import torch
import sys
sys.path.append('../../../src')

from sogaPreprocessor import *
from producecfg import *
from libSOGA import *

from time import time

from torch.distributions import Normal, Laplace, Gamma, MultivariateNormal

#################################################################
# G-VULNERABILITY METRICS
#################################################################

### V_{g_\Delta} 

def vdelta_bivariate_gaussian(mu, sigma, idx_o=0):
    """
    Computes V_g_Delta for a bivariate Gaussian distribution with mean mu and covariance sigma analytically
    idx_o is the index of the output variable.
    """
    return torch.sqrt(sigma[idx_o,idx_o]/(torch.det(sigma)*2*torch.pi))

def vdelta_mvariate_gaussian(mu, sigma, idx_o):
    """
    Computes V_g_Delta for a multivariate Gaussian distribution with mean mu and covariance sigma analytically
    idx_o is a list of indices of the output variables. All other variables are considered secrets
    """
    # extract indices of output and secrete variables
    idx_s = [i for i in range(len(mu)) if i not in idx_o]
    # computes the conditional covariance matrix
    d = len(idx_o)
    sigma_ss = sigma[idx_s, :][:, idx_s] 
    sigma_so = sigma[idx_s, :][:, idx_o]
    sigma_oo = sigma[idx_o, :][:, idx_o]
    inv_sigma_oo = torch.linalg.inv(sigma_oo)
    cond_sigma = sigma_ss - sigma_so @ inv_sigma_oo @ sigma_so.transpose(-1, -2)
    return 1/((2*torch.pi)**(d/2) * torch.sqrt(torch.linalg.det(cond_sigma)))

def vdelta_lower_bound(dist, idx_o=0, mvariate=False):
    """ 
    Computes the lower bound on the V_{g_Delta} for dist Gaussian Mixture.
    If mvariate is True, idx_o can be a list of indices of output variables
    """
    bounds = []
    for i in range(dist.gm.n_comp()):
        if mvariate:
            bounds.append(dist.gm.pi[i]*vdelta_mvariate_gaussian(dist.gm.mu[i], dist.gm.sigma[i], idx_o))
        else:
            bounds.append(dist.gm.pi[i]*vdelta_bivariate_gaussian(dist.gm.mu[i], dist.gm.sigma[i], idx_o))
    return torch.max(torch.stack(bounds))

def vdelta_upper_bound(dist, idx_o=0, mvariate=False):
    """ 
    Computes the upper bound on the V_{g_Delta} for dist Gaussian Mixture
    If mvariate is True, idx_o can be a list of indices of output variables
    """
    bounds = []
    for i in range(dist.gm.n_comp()):
        if mvariate:
            bounds.append(dist.gm.pi[i]*vdelta_mvariate_gaussian(dist.gm.mu[i], dist.gm.sigma[i], idx_o))
        else:
            bounds.append(dist.gm.pi[i]*vdelta_bivariate_gaussian(dist.gm.mu[i], dist.gm.sigma[i], idx_o))
    return torch.sum(torch.stack(bounds))

### V_{g_gauss}

def vgauss_bivariate_gaussian(mu, sigma, eps, idx_o=0):
    """
    Computes the g-Vulnerability of a bivariate gaussian distribution with mean mu and cov matrix sigma
    with respect to the gain function g(s,w) = exp(-0.5*(w-s)**2/eps).
    idx_o is the index of the output variable.
    """
    return torch.sqrt(sigma[idx_o,idx_o]/((1/eps)*torch.det(sigma)+sigma[idx_o,idx_o]))

def vgauss_mvariate_gaussian(mu, sigma, eps, idx_o):
    """
    Computes V_g_N for a multivariate Gaussian distribution with mean mu and covariance sigma analytically 
    with respect to the gain function g(s,w) = exp(-0.5*(w-s)**2/eps).
    idxs_o is a list of indices of the output variables. All other variables are considered secrets
    """
    # extract indices of output and secrete variables
    idx_s = [i for i in range(len(mu)) if i not in idx_o]
    # computes the conditional covariance matrix
    d = len(idx_o)
    sigma_ss = sigma[idx_s, :][:, idx_s] 
    sigma_so = sigma[idx_s, :][:, idx_o]
    sigma_oo = sigma[idx_o, :][:, idx_o]
    inv_sigma_oo = torch.linalg.inv(sigma_oo)
    cond_sigma = sigma_ss - sigma_so @ inv_sigma_oo @ sigma_so.transpose(-1, -2)
    # create sigma_eps
    sigma_eps = eps*torch.eye(d)
    return  torch.sqrt(torch.linalg.det(sigma_eps @ torch.linalg.inv(cond_sigma + sigma_eps)))

def vgauss_lower_bound(dist,  eps=1, idx_o=0, mvariate=False):
    """
    Computes the lower bound on the V_{g_N} for dist Gaussian Mixture
    eps = float parameter for the gain function
    """ 
    bounds = []
    for i in range(dist.gm.n_comp()):
        if mvariate:
            bounds.append(dist.gm.pi[i]*vgauss_mvariate_gaussian(dist.gm.mu[i], dist.gm.sigma[i], eps, idx_o))
        else:
            bounds.append(dist.gm.pi[i]*vgauss_bivariate_gaussian(dist.gm.mu[i], dist.gm.sigma[i], eps, idx_o))
    return torch.max(torch.stack(bounds))

def vgauss_upper_bound(dist, eps=1, idx_o=0, mvariate=False):
    """
    Computes the lower bound on the V_{g_N} for dist Gaussian Mixture
    eps = float parameter for the gain function
    """ 
    bounds = []
    for i in range(dist.gm.n_comp()):
        if mvariate:
            bounds.append(dist.gm.pi[i]*vgauss_mvariate_gaussian(dist.gm.mu[i], dist.gm.sigma[i], eps, idx_o))
        else:
            bounds.append(dist.gm.pi[i]*vgauss_bivariate_gaussian(dist.gm.mu[i], dist.gm.sigma[i], eps, idx_o))
    return torch.sum(torch.stack(bounds))

# Fitting Laplace

def fit_laplace(mu, b):
    """
    Finds the parameters of a two-component Gaussian mixture that approximates a Laplace distribution
    using moment matching constraints.
    """
    
    # Target distribution 
    target_dist = Laplace(mu, b)

    # Grid for evaluation
    x = torch.linspace(mu-2*torch.sqrt(2*b**2), mu+2*torch.sqrt(2*b**2), 200)

    # Initial parameters for the mixture - only optimize alpha
    alpha = torch.tensor(0.5, requires_grad=True)

    def lap_mix_log_pdf(x, alpha, mu, b):
        # Compute mixture params from moment matching constraints
        pi = torch.clamp(8/(alpha**2-4*alpha+12), min = 0.01, max=0.99)
        mu1 = mu2 = mu
        sigma1 = torch.sqrt(torch.clamp(alpha*b**2, min = 0.01))
        sigma2 = torch.sqrt(torch.clamp((2-alpha*pi)*b**2/(1-pi), min = 0.01))
        return torch.log(pi*Normal(mu1, sigma1).log_prob(x).exp() + (1-pi)*Normal(mu2, sigma2).log_prob(x).exp()), pi, mu1, mu2, sigma1, sigma2

    optimizer = torch.optim.Adam([alpha], lr=0.01)

    for step in range(1000):
        
        optimizer.zero_grad()
        # Compute the mixture and the target log densities
        p, pi, mu1, mu2, sigma1, sigma2 = lap_mix_log_pdf(x, alpha, mu, b)
        q = target_dist.log_prob(x)
        # Computes KL
        kl = torch.sum(torch.kl_div(p, q, log_target=True))
        # Combined loss with normalization
        total_loss = kl
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        #if step % 100 == 0:
        #    print(f"Step {step}, KL: {kl.item():.4f}")

    # Compute final mu2 and sigma2 for display
    with torch.no_grad():
        pi_final = torch.clamp(pi, min=0.01, max=0.99)
        sigma1_final = torch.clamp(sigma1, min=0.01)
        sigma2_final = torch.clamp(sigma2, min=0.01)

    return pi_final.item(), mu1.item(), mu2.item(), sigma1_final.item(), sigma2_final.item(), x

### Sampling Univariate



def compute_conditioned_stats(dist, value_to_cond, idx_to_cond):
    """"
    Given a dist computes the conditional means and variances, when conditioning with 
    respect to the idx_to_cond coordinate.
    Returns the conditioned distribution
    """
    d = dist.gm.n_dim()
    c = dist.gm.n_comp()
    idxs = [i for i in range(d) if i != idx_to_cond]   
    cond_mus = []
    cond_sigmas = []
    for k in range(c):
        mu = dist.gm.mu[k].clone()
        sigma = dist.gm.sigma[k].clone()
        # Partition mean and covariance
        mu_1 = mu[idxs]  # shape (d-1,)
        mu_2 = mu[idx_to_cond]  # scalar
        sigma_11 = sigma[idxs][:, idxs]  # (d-1, d-1)
        sigma_12 = sigma[idxs, idx_to_cond].unsqueeze(1)  # (d-1, 1)
        sigma_21 = sigma[idx_to_cond, idxs].unsqueeze(0)  # (1, d-1)
        sigma_22 = sigma[idx_to_cond, idx_to_cond].unsqueeze(0).unsqueeze(0)  # (1, 1)
        # Compute conditional mean and covariance (for x2 = 0)
        sigma_22_inv = 1.0 / sigma_22  # (1, 1)
        cond_mu = mu_1 - (sigma_12 @ sigma_22_inv).squeeze() * (value_to_cond - mu_2)
        cond_mu = torch.hstack([cond_mu, mu_2.reshape(1,)])
        cond_mus.append(cond_mu)
        cond_sigma = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_21
        new_sigma = torch.zeros((d, d))
        new_sigma[:d-1, :d-1] = cond_sigma
        new_sigma[-1, -1] = torch.tensor(0.001)
        cond_sigmas.append(new_sigma)
    cond_mus = torch.stack(cond_mus)  # (c, d-1)
    cond_sigmas = torch.stack(cond_sigmas)  # (c, d-1, d-1)
    return Dist(dist.var_list, GaussianMix(dist.gm.pi, cond_mus, cond_sigmas))

def gamma_gauss(w, cond_dist, eps):
    """
    V_{g_N} = \\mathbb{E}_{p(o)} [ \\sup_w gamma_gauss(w)] 
    NOTE: cond_dist is a univariate mixture conditioned to a value of the output.
    """
    value = 0.0
    for i in range(cond_dist.gm.n_comp()):
        value += cond_dist.gm.pi[i]*torch.sqrt(eps/(eps + cond_dist.gm.sigma[i]))*torch.exp(-0.5*(w - cond_dist.gm.mu[i])**2/(eps + cond_dist.gm.sigma[i])) 
    return value.reshape([])

def gamma_delta(w, cond_dist):
    """
    V_{g_Delta} = \\mathbb{E}_{p(o)} [ \\sup_w gamma_delta(w)] 
    NOTE: cond_dist is a univariate mixture conditioned to a value of the output.
    """
    value = 0.0
    for i in range(cond_dist.gm.n_comp()):
        value += cond_dist.gm.pi[i]*(1/torch.sqrt(2*torch.pi*cond_dist.gm.sigma[i]))*torch.exp(-0.5*(w - cond_dist.gm.mu[i])**2/cond_dist.gm.sigma[i])
    return value.reshape([])

def sample_gvuln(dist, n_samples, secret_var, output_var, gain_type, tol=1e-3, min_samples=10, num_steps=100, lr=0.05):
    """
    Approximates the g-vulnerability of a mixture of bivariate Gaussian distributions by sampling.
    secret_var: name of the secrete variable
    output_var: name of the output variable
    gain_type: 'gauss' for g_N or 'delta' for g_Delta
    tol = 1e-3 sampling stops when abs(old_estimate - new_estimate) < tol
    min_samples = minimum number of samples before stopping criterion is checked
    num_steps = number of gradient ascent steps to find the maximum of gamma
    lr = learning rate for gradient ascent
    """

    output_idx = dist.var_list.index(output_var)
    out_marg = extract_marginal(dist, [output_var])
    samples = sample_univariate_gm(out_marg, n_samples)  # Sample output coordinate

    total = 0.0

    for i in range(len(samples)):

        o = samples[i]

        cond_dist = compute_conditioned_stats(dist, o, output_idx)
        marg_cond_dist = extract_marginal(cond_dist, [secret_var])
        marg_cond_dist.gm.mu = marg_cond_dist.gm.mu.detach()
        marg_cond_dist.gm.sigma = marg_cond_dist.gm.sigma.detach()

        if gain_type == 'delta':
            f = lambda w : gamma_delta(w, marg_cond_dist)
        elif gain_type == 'gauss':
            f = lambda w : gamma_gauss(w, marg_cond_dist, 1.)

        # this is added to avoid optimizations too long
        rank = torch.argsort(marg_cond_dist.gm.pi, dim=0, descending=True)
        starts = marg_cond_dist.gm.mu[rank].squeeze(1)[:10].flatten()
        max_value, _ = find_global_maximum(f, starts, num_steps=num_steps, lr=lr)

        curr_total = total + max_value
        
        old_estimate = total / i if i > 0 else 0.0
        new_estimate = curr_total / (i + 1)


        if abs(new_estimate - old_estimate) < tol and i >= min_samples:
            return new_estimate
        else:
            total = curr_total

    print("Convergence not reached for MCMC.")
    return new_estimate

def find_global_maximum(f, starts, num_steps=500, lr=0.05, tol=1e-4):
    """
    Finds the maximum value of f by running gradient ascent from multiple random starting points.
    Args:
        f: function to maximize, takes a tensor of shape (dim,)
        starts: tensor of starting points
        num_steps: gradient ascent steps per start
        lr: learning rate
        tol: optimization stops when the change in the function is below tol
    Returns:
        max_value: the maximum value found
        max_point: the point where the maximum was found
    """
    all_local_maxima = []
    for start in starts:
        # Random starting point
        w = start.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([w], lr=lr)
        prev_value = None
        for i in range(num_steps):
            optimizer.zero_grad()
            gamma_value = -f(w)  # Reverting to minimize -f for maximization
            # Gradient ascent: maximize f
            gamma_value.backward()
            #if i%10 == 0:
            #    print('Gamma value: ', gamma_value)
            optimizer.step()
            curr_value = f(w.detach()).item()
            if prev_value is not None and abs(curr_value - prev_value) < 1e-6:
                print('Stopped before')
                break
        # Store local maximum
        local_max = curr_value
        all_local_maxima.append((local_max, w.detach().clone()))
    # Find the global maximum among local maxima
    max_value, max_point = max(all_local_maxima, key=lambda x: x[0])
    #print('Returning max value:', max_value)
    return max_value, max_point

## Geometric Mechanism

def discrete_V1(dist, secret_var, output_var):
    """
    Computes Bayes risk for a discrete distribution
    """
    tot = 0.0
    marg_dist = extract_marginal(dist, [secret_var, output_var])
    secret_idx = marg_dist.var_list.index(secret_var)
    output_idx = marg_dist.var_list.index(output_var)
    o_supp = []
    s_supp = []
    # finds the support of the output and the secret
    s_supp = torch.unique(marg_dist.gm.mu[:, secret_idx])
    o_supp = torch.unique(marg_dist.gm.mu[:, output_idx])
    # cycles on o values
    for o in o_supp:
        # selects components for given value of o
        indices = marg_dist.gm.mu[:, output_idx] == o
        conditional_pi = marg_dist.gm.pi[indices]
        conditional_mu = marg_dist.gm.mu[indices]
        # probability of o 
        po = torch.sum(conditional_pi)
        # cycles on s values computing the prob of s given o
        ps = []
        for s in s_supp:
            ps.append(torch.sum(conditional_pi[conditional_mu[:, secret_idx] == s]))
        ps = torch.stack(ps)
        ps = ps/torch.sum(ps)  # normalization of the conditional probabilities
        # computes the contribution to the total risk
        tot = tot + po*torch.max(ps)
    return tot

def create_gm_from_p(p, tol=1e-6):
    """
    Returns a string representing a gm in SOGA language, approximating the geometric with parameter p.
    """
    # support and prob mass of the geometric
    x = torch.arange(0, 50)
    y = (1 - p)**x*p
    # truncates to significant prob mass
    y = y[y > tol]
    # creates the string
    pi_list = '['
    mean_list = '['
    cov_list = '['
    for i in range(len(y)):
        pi_list += '{:.5f},'.format(y[i].item())
        mean_list += '{:.5f},'.format(x[i].item())
        cov_list += '0.001,'
    pi_list = pi_list[:-1] + ']'
    mean_list = mean_list[:-1] + ']'
    cov_list = cov_list[:-1] + ']'
    gm_str = 'gm({}, {}, {})'.format(pi_list, mean_list, cov_list)
    return gm_str

## Geo-indistinguishability

def fit_gamma(mean, scale):
    """
    Finds the parameters of a two-component Gaussian mixture that approximates a Gamma distribution
    using moment-matching and squared error minimization.
    """
    # Second central moment of Gamma (needed for moment-matching)
    second = mean*scale**2*(1 + mean)
    
    # Target gamma distribution 
    target_dist = Gamma(torch.tensor([mean]), torch.tensor([1.0/scale]))

    # Grid for evaluation
    lb = torch.tensor(0.01)
    ub = torch.max(torch.tensor([2*scale**2, 1]))
    x = torch.linspace(lb, ub, 200)

    # Initial parameters for the mixture - only optimize pi, sigma1 
    # The optimization will start by putting most of the mass in correspondence of the mode of the Gamma
    pi1 = torch.tensor(0.9, requires_grad=True)
    sigma1 = torch.tensor(1.0, requires_grad=True)

    def mix_log_pdf(x, pi1, sigma1, mean, scale):
        second = mean*scale**2*(1 + mean)
        # Compute mu2 and sigma2 from moment matching constraints
        pi2 = 1 - pi1
        # clamping the means to keep them in the support of the Gamma
        mu1 = torch.clamp((mean - 1)*scale, min=lb, max=ub)  # one component fits the mode of the Gamma
        mu2 = torch.clamp((mean*scale - pi1*mu1)/(1 - pi1), min=lb, max=ub)
        sigma2_sq = (second - pi1*(mu1**2 + sigma1**2))/pi2 - mu2**2
        sigma2 = torch.sqrt(torch.clamp(sigma2_sq, min=0.01))  # Clamp to avoid negative values
        #print('pi1:', pi1.item(), ', mu1:', mu1.item(), ', sigma1:', sigma1.item(), ', mu2:', mu2.item(), ', sigma2:', sigma2.item())
        return pi1*Normal(mu1, sigma1).log_prob(x).exp() + pi2*Normal(mu2, sigma2).log_prob(x).exp(), pi2, mu1, mu2, sigma2

    optimizer = torch.optim.Adam([pi1, sigma1], lr=0.05)

    for step in range(1500):
        optimizer.zero_grad()
        # Clamp pi to [0,1] and sigma1 to positive values for stability
        pi_clamped = torch.clamp(pi1, 0.01, 0.99)
        sigma1_clamped = torch.clamp(sigma1, 0.1)
    
        p, pi2, mu1, mu2, sigma2 = mix_log_pdf(x, pi_clamped, sigma1_clamped, mean, scale)
        q = target_dist.log_prob(x).exp()
    
        loss = torch.sum((p - q)**2)
    
        # Combined loss with normalization
        total_loss = loss# + torch.abs(pi1*(mu1**3 + 3*mu1*sigma1_clamped**2) + pi2*(mu2**3 + 3*mu2*sigma2**2) - third)
    
        total_loss.backward()
        optimizer.step()
        #if step % 10 == 0:
        #    print(f"Step {step}, Loss: {loss.item():.4f}")

    # Compute final mu2 and sigma2 for display
    with torch.no_grad():
        pi1 = torch.clamp(pi1, 0.01, 0.99)
        sigma1 = torch.clamp(sigma1, 0.1)
        pi2 = 1 - pi1
        mu1 = torch.clamp((mean - 1)*scale, min=lb, max=ub)
        mu2 = torch.clamp((mean - pi1*mu1)/(1 - pi1), min=lb, max=ub)
        sigma2_sq = (second - pi1*(mu1**2 + sigma1**2))/pi2 - mu2**2
        sigma2 = torch.sqrt(torch.clamp(sigma2_sq, min=0.01))  # Clamp to avoid negative values

    return (x, p), pi1, pi2, mu1, mu2, sigma1, sigma2

## Sample multivariate

def compute_conditioned_stats_multivariate(dist, value_to_cond, idxs_to_cond):
    """"
    Given a dist computes the conditional means and variances, when conditioning with 
    respect to the idx_to_cond coordinate.
    Returns the conditioned distribution
    """
    d = dist.gm.n_dim()
    c = dist.gm.n_comp()
    idxs = [i for i in range(d) if i not in idxs_to_cond] 
    sigma_ss = dist.gm.sigma[:, idxs, :][:, :, idxs]
    sigma_so = dist.gm.sigma[:, idxs, :][:, :, idxs_to_cond]
    sigma_oo = dist.gm.sigma[:, idxs_to_cond, :][:, :, idxs_to_cond]
    inv_sigma_oo = torch.linalg.inv(sigma_oo)
    # Computes conditioned mu
    cond_mu = (dist.gm.mu[:, idxs].unsqueeze(2) + sigma_so @ inv_sigma_oo @ (value_to_cond - dist.gm.mu[:, idxs_to_cond]).unsqueeze(2)).reshape((c, len(idxs)))
    # Computes conditioned sigma
    cond_sigma = sigma_ss - sigma_so @ inv_sigma_oo @ sigma_so.transpose(-1, -2)    
    return Dist([dist.var_list[i] for i in idxs], GaussianMix(dist.gm.pi, cond_mu, cond_sigma))

def gamma_delta_multivariate(w, cond_dist):
    """
    V_{g_Delta} = \\mathbb{E}_{p(o)} [ \\sup_w gamma_delta(w)] 
    NOTE: cond_dist is a univariate mixture conditioned to a value of the output.
    """
    value = 0.0
    for i in range(cond_dist.gm.n_comp()):
        value += cond_dist.gm.pi[i]*MultivariateNormal(cond_dist.gm.mu[i].detach(), cond_dist.gm.sigma[i].detach()).log_prob(w).exp()
    return value.reshape([])

def gamma_gauss_multivariate(w, cond_dist, eps):
    """
    V_{g_N} = \\mathbb{E}_{p(o)} [ \\sup_w gamma_gauss(w)] 
    NOTE: cond_dist is a multivariate mixture conditioned to a value of the output.
    """
    value = 0.0
    sigma_eps = eps*torch.eye(len(w))
    for i in range(cond_dist.gm.n_comp()):
        sigma_sum = cond_dist.gm.sigma[i].detach() + sigma_eps
        inv_sigma_sum = torch.linalg.inv(sigma_sum)
        value += cond_dist.gm.pi[i]*torch.sqrt(torch.linalg.det(sigma_eps)/torch.linalg.det(sigma_sum))*torch.exp(-0.5*(w - cond_dist.gm.mu[i].detach()) @ inv_sigma_sum @ (w - cond_dist.gm.mu[i].detach()).reshape(-1, 1)) 
    return value.reshape([])

def sample_gvuln_multivariate(dist, n_samples, secret_var, output_var, gain_type, tol=1e-3, min_samples=5, num_steps=100, lr=0.05):
    """
    Approximates the g-vulnerability of a mixture of bivariate Gaussian distributions by sampling.
    secret_var: name of the secret variable
    output_var: name of the output variable
    gain_type: 'gauss' for g_N or 'delta' for g_Delta
    tol = 1e-3 sampling stops when abs(old_estimate - new_estimate) < tol
    min_samples = minimum number of samples before stopping criterion is checked
    num_steps = number of gradient ascent steps to find the maximum of gamma
    lr = learning rate for gradient ascent
    """

    output_idxs = []
    for output in output_var:
        output_idxs.append(dist.var_list.index(output))

    marg_dist = extract_marginal(dist, output_var)
    samples = sample_from_gm(marg_dist.gm, n_samples)  # Sample output coordinate

    total = 0.0

    cond_time = 0.
    max_time = 0.

    for i in range(len(samples)):

        o = samples[i]
        
        start = time()
        cond_dist = compute_conditioned_stats_multivariate(dist, o, output_idxs)
        end = time()
        cond_time += end - start 

        if gain_type == 'delta':
            f = lambda w : gamma_delta_multivariate(w, cond_dist)
        elif gain_type == 'gauss':
            f = lambda w : gamma_gauss_multivariate(w, cond_dist, 1.)

        # select only K starting points with higher initial gamma
        initial_f = torch.stack([f(cond_dist.gm.mu[i]) for i in range(cond_dist.gm.mu.shape[0])])
        rank = torch.argsort(initial_f, dim=0, descending=True)
        starts = cond_dist.gm.mu[rank].squeeze(1)[:5]
        start = time()
        max_value, _ = find_global_maximum(f, starts, num_steps=num_steps, lr=lr)
        end = time()
        max_time += end - start
        #if i%10 == 0:
        #    print(f"Sample {i}, max_value: {max_value}")

        curr_total = total + max_value
        
        old_estimate = total / i if i > 0 else -10.
        new_estimate = curr_total / (i + 1)

        if i >= min_samples and abs(new_estimate - old_estimate) < tol:
            print('Returning after {} samples'.format(i))
            print('Total time for conditioning: {:.2f}s'.format(cond_time))
            print('Total time for maximizing: {:.2f}s'.format(max_time))
            return new_estimate
        else:
            total = curr_total
    

    print("Convergence not reached for MCMC.")
    return new_estimate

###############################################################
# ENTROPY RELATED METRICS
###############################################################

# Entropy 

def entropy_gaussian(mu, sigma):
    """
    Computes the differential entropy of a Gaussian distribution with mean mu and covariance sigma.
    """
    d = mu.shape[0]
    return 0.5 * torch.log((torch.tensor(2.)*torch.pi*torch.e)**d*torch.det(sigma))

def entropy_ub(dist):
    """
    Computes the lower bound on the entropy of a Gaussian Mixture distribution.
    """
    if dist.gm.n_comp() == 1:
        return entropy_gaussian(dist.gm.mu[0], dist.gm.sigma[0]).item()
    lb = torch.tensor([0.])
    for c in range(dist.gm.n_comp()):
        lb += dist.gm.pi[c]*(- torch.log(dist.gm.pi[c]) + entropy_gaussian(dist.gm.mu[c], dist.gm.sigma[c]))
    return lb.item()

def entropy_lb(dist):
    """
    Computes the upper bound on the entropy of a Gaussian Mixture distribution.
    """
    if dist.gm.n_comp() == 1:
        return entropy_gaussian(dist.gm.mu[0], dist.gm.sigma[0]).item()
    lb = torch.tensor([0.])
    c = dist.gm.n_comp()
    for i in range(c):
        sum_zij = torch.tensor([0.])
        for j in range(c):
            sum_zij += dist.gm.pi[j]*MultivariateNormal(dist.gm.mu[j], dist.gm.sigma[j]+dist.gm.sigma[i]).log_prob(dist.gm.mu[i]).exp().squeeze(0)
        lb += dist.gm.pi[i]*torch.log(sum_zij)
    return -lb.item()

def discrete_entropy(dist):
    """
    Computes the discrete entropy of a distribution.
    """
    # Extracts the support of the distribution
    s_supp = torch.unique(dist.gm.mu)
    value = torch.tensor([0.])
    # cycles on support of the secret
    for s in s_supp:
        mask = (dist.gm.mu == s).all(dim=1)
        p_s = dist.gm.pi[mask].sum()
        value += p_s * torch.log(p_s)
    return -value.item()


def sample_entropy(dist, n_samples=1000):
    """
    Computes the entropy of dist using MCMC integration
    dist is an object Dist as defined in libSOGAshared
    """
    samples = sample_from_gm(dist.gm, n_samples)
    log_probs = torch.log(dist.gm.pdf(samples))
    return -torch.mean(log_probs).item()


# Conditional Entropy

def cond_entropy_gaussian(mu, sigma, idx_o):
    """ 
    Computes the conditional entropy of a Gaussian distribution with mean mu and covariance sigma, 
    where the conditioning variables have indices idx_0
    """
    joint_entropy = entropy_gaussian(mu, sigma)
    marg_mu = mu[idx_o]
    marg_sigma = sigma[idx_o, :][:, idx_o]
    marg_entropy = entropy_gaussian(marg_mu, marg_sigma)
    return joint_entropy - marg_entropy

def cond_entropy_lb(dist, idx_o):
    """
    Computes the lower bound on the conditional entropy of a Gaussian Mixture distribution.
    idx_o is a list of the the indices of the output variable.
    """
    var_names = [dist.var_list[i] for i in idx_o]
    marg_dist = extract_marginal(dist, var_names)
    marg_dist = aggregate_mixture(marg_dist)
    return entropy_lb(dist) - entropy_ub(marg_dist)

def cond_entropy_ub(dist, idx_o):
    """
    Computes the upper bound on the conditional entropy of a Gaussian Mixture distribution.
    idx_o is a list of the the indices of the output variable.
    """
    var_names = [dist.var_list[i] for i in idx_o]
    marg_dist = extract_marginal(dist, var_names)
    marg_dist = aggregate_mixture(marg_dist)
    return entropy_ub(dist) - entropy_lb(marg_dist)

def discrete_cond_entropy(dist, secret_var, output_var):
    """
    Computes Conditional Entropy of secret given output for a discrete distribution
    """
    joint_dist = extract_marginal(dist, [secret_var, output_var])
    output_dist = extract_marginal(dist, [output_var])
    secret_idx = joint_dist.var_list.index(secret_var)
    output_idx = joint_dist.var_list.index(output_var)
    # finds the support of the output and the secret
    o_supp = torch.unique(joint_dist.gm.mu[:, output_idx])
    s_supp = torch.unique(joint_dist.gm.mu[:, secret_idx])
    value = torch.tensor([0.])
    # cycles on support of the joint
    for o in o_supp:
        for s in s_supp:
            # computes probability mass for (s, o)
            mask = (joint_dist.gm.mu == torch.tensor([s, o])).all(dim=1)
            p_xy = torch.sum(joint_dist.gm.pi[mask])
            # computes probability mass for o
            mask = (output_dist.gm.mu == torch.tensor([o])).all(dim=1)
            p_y = torch.sum(output_dist.gm.pi[mask])
            # sums to total
            if p_xy == 0.:
                continue
            else:
                value += p_xy * torch.log(p_xy / p_y)
    return - value.item()

def sample_cond_entropy(dist, idx_o, n_samples):
    """
    Estimates conditional entropy using MCMC integration.
    """
    out_marg = extract_marginal(dist, [dist.var_list[i] for i in idx_o])
    out_marg = aggregate_mixture(out_marg)
    joint_entropy = sample_entropy(dist, n_samples)
    marg_entropy = sample_entropy(out_marg, n_samples)
    return joint_entropy - marg_entropy

# Mutual Information

def mi_gaussian(mu, sigma, idx_o):
    """
    I(X; Y) = H(X) - H(X | Y)
    """
    idx_s = [i for i in range(len(mu)) if i not in idx_o]
    mu_s = mu[idx_s]
    sigma_s = sigma[idx_s][:, idx_s]
    s_entropy = entropy_gaussian(mu_s, sigma_s)
    cond_entropy = cond_entropy_gaussian(mu, sigma, idx_o)
    return (s_entropy - cond_entropy).item()

def mi_lb(dist, idx_o):
    """
    Computes the lower bound on the mutual information of a Gaussian Mixture distribution.
    idx_o is a list of the the indices of the output variable.
    """
    idx_s = [i for i in range(dist.gm.n_dim()) if i not in idx_o]
    s_marg = extract_marginal(dist, [dist.var_list[i] for i in idx_s])
    s_marg = aggregate_mixture(s_marg)
    return max(entropy_lb(s_marg) - cond_entropy_ub(dist, idx_o), 0.)

def mi_ub(dist, idx_o):
    """
    Computes the upper bound on the mutual information of a Gaussian Mixture distribution.
    idx_o is a list of the the indices of the output variable.
    """
    idx_s = [i for i in range(dist.gm.n_dim()) if i not in idx_o]
    s_marg = extract_marginal(dist, [dist.var_list[i] for i in idx_s])
    s_marg = aggregate_mixture(s_marg)
    return entropy_ub(s_marg) - cond_entropy_lb(dist, idx_o)

def discrete_mi(dist, secret_var, output_var):
    secret_dist = extract_marginal(dist, [secret_var])
    Hx = discrete_entropy(secret_dist)
    Hxy = discrete_cond_entropy(dist, secret_var, output_var)
    return Hx - Hxy

def sample_mi(dist, idx_o, n_samples):
    """
    Estimates mutual information using MCMC integration.
    """
    idx_s = [i for i in range(dist.gm.n_dim()) if i not in idx_o]
    s_marg = extract_marginal(dist, [dist.var_list[i] for i in idx_s])
    s_marg = aggregate_mixture(s_marg)
    return sample_entropy(s_marg, n_samples) - sample_cond_entropy(dist, idx_o, n_samples)

# KL Divergence

def kl_div_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Computes the KL divergence between two Gaussian distributions with means mu1, mu2 and covariances sigma1, sigma2.
    """
    d = mu1.shape[0]
    inv_sigma2 = torch.linalg.inv(sigma2)
    term1 = torch.log(torch.det(sigma2) / torch.det(sigma1))
    term2 = torch.trace(inv_sigma2 @ sigma1)
    term3 = (mu2 - mu1).reshape(1, d) @ inv_sigma2 @ (mu2 - mu1).reshape(d, 1)
    return 0.5 * (term1 + term2 - d + term3).item()

def L_gaussian(mu1, sigma1, mu2, sigma2):
    """ 
    Computes the term \\int p_1(x) log(p_2(x)) dx where p_i(x) are Normal densities with means mu_i and
    covariances sigma_i.
    """
    return - entropy_gaussian(mu1, sigma1) - kl_div_gaussian(mu1, sigma1, mu2, sigma2)

def kl_div_lb(dist1, dist2):
    """
    Computes the lower bound on the KL divergence between two Gaussian Mixture distributions.
    """
    return max(- L_ub(dist1, dist2) - entropy_ub(dist1), 0.)

def kl_div_ub(dist1, dist2):
    """
    Computes the upper bound on the KL divergence between two Gaussian Mixture distributions.
    """
    return - L_lb(dist1, dist2) - entropy_lb(dist1)

def L_ub(dist1, dist2):
    """
    Computes the upper bound on the term \\int p_1(x) log(p_2(x)) dx where p_i(x) are Gaussian Mixture densities.
    """
    Na = dist1.gm.n_comp()
    Nb = dist2.gm.n_comp()
    value = torch.tensor([0.])
    for a in range(Na):
        log_arg = torch.tensor([0.])
        for b in range(Nb):
            sigma = dist1.gm.sigma[a] + dist2.gm.sigma[b]
            z_ab = MultivariateNormal(dist2.gm.mu[b], sigma).log_prob(dist1.gm.mu[a]).exp().squeeze(0)
            log_arg += dist2.gm.pi[b] * z_ab
        value += dist1.gm.pi[a] * torch.log(log_arg)
    return value.item()

def L_lb(dist1, dist2):
    """
    Computes the lower bound on the term \\int p_1(x) log(p_2(x)) dx where p_i(x) are Gaussian Mixture densities.
    """
    Na = dist1.gm.n_comp()
    Nb = dist2.gm.n_comp()
    value = torch.tensor([0.])
    for a in range(Na):
        phi_list = torch.zeros(Nb)
        L_list = torch.zeros(Nb)
        for b in range(Nb):
            phi_list[b] = dist2.gm.pi[b] * torch.exp(-torch.tensor(kl_div_gaussian(dist1.gm.mu[a], dist1.gm.sigma[a], dist2.gm.mu[b], dist2.gm.sigma[b])))
            L_list[b] = L_gaussian(dist1.gm.mu[a], dist1.gm.sigma[a], dist2.gm.mu[b], dist2.gm.sigma[b])
        phi_list = phi_list / torch.sum(phi_list) 
        a_sum = torch.sum(phi_list * (torch.log(dist2.gm.pi.reshape(Nb,)) - torch.log(phi_list) + L_list))
        value += dist1.gm.pi[a] * a_sum
    return value.item()

def discrete_kl(dist1, dist2, var):
    """
    Computes the discrete KL divergence between two distributions.
    """
    marg1 = extract_marginal(dist1, [var])
    marg2 = extract_marginal(dist2, [var])
    
    kl_value = torch.tensor([0.])

    supp1 = torch.unique(marg1.gm.mu[:, 0])
    supp2 = torch.unique(marg2.gm.mu[:, 0])
    supp = torch.unique(torch.cat((supp1, supp2)))
    
    for x in supp:
        mask1 = (marg1.gm.mu[:, 0] == x)
        p1 = marg1.gm.pi[mask1].sum()
        mask2 = (marg2.gm.mu[:, 0] == x)
        p2 = marg2.gm.pi[mask2].sum()
        if p1 > 0 and p2 > 0:
            kl_value += p1 * torch.log(p1 / p2)
    
    return kl_value.item()

def sample_kl(dist1, dist2, n_samples):
    """
    Estimates the KL divergence between two distributions using MCMC integration.
    """
    samples = sample_from_gm(dist1.gm, n_samples)
    
    pdfs1 = dist1.gm.pdf(samples)
    mask = pdfs1 == 0.
    log_probs1 = torch.empty(pdfs1.shape)
    log_probs1[mask] = 0.0  # Avoid log(0) issues
    log_probs1[~mask] = torch.log(pdfs1[~mask])

    log_probs2 = torch.log(dist2.gm.pdf(samples))
    
    kl_value = torch.mean(log_probs1 - log_probs2).item()
    
    return kl_value

#################################################################
# UTILS
#################################################################

def sample_univariate_gm(dist, n_samples):
    """
    Samples coordinate idx from dist
    Returns:
    - samples: tensor of shape (n_samples, ) containing the samples from the specified coordinate
    """
    gm = dist.gm
    # Ensure the number of components matches the mixture weights
    n_components = gm.n_comp()
    # Sample component indices based on mixture weights
    component_indices = (torch.multinomial(gm.pi.flatten(), n_samples, replacement=True))
    # Initialize an empty tensor for samples
    samples = torch.empty(n_samples, gm.n_dim())
    # Generate samples for each component
    for i in range(n_components):
        # Get the indices of samples belonging to the current component
        mask = component_indices == i
        # Sample from the Gaussian distribution of the current component
        n_component_samples = mask.sum().item()
        if n_component_samples > 0:
            component_samples = Normal(gm.mu[i], torch.sqrt(gm.sigma[i])).sample((n_component_samples,)).reshape(samples[mask, :].shape)
            samples[mask, :] = component_samples
    return samples

def sample_from_gm(gm, n_samples):
    """
    Samples from a Gaussian Mixture object
    """
    # Ensure the number of components matches the mixture weights
    n_components = gm.n_comp()

    # Sample component indices based on mixture weights
    component_indices = (torch.multinomial(gm.pi.flatten(), n_samples, replacement=True))

    # Initialize an empty tensor for samples
    samples = torch.empty(n_samples, gm.n_dim())

    # Generate samples for each component
    for i in range(n_components):
        # Get the indices of samples belonging to the current component
        mask = component_indices == i

        # Sample from the Gaussian distribution of the current component
        n_component_samples = mask.sum().item()
        if n_component_samples > 0:
            component_samples = MultivariateNormal(gm.mu[i], gm.sigma[i]).sample((n_component_samples,)).reshape(samples[mask, :].shape)
            samples[mask, :] = component_samples

    return samples

def extract_marginal(dist, var_list):
    """
    Extract the marginal of the variables specified in var_list from the distribution dist
    """

    indices = []
    for var in var_list:
        indices.append(dist.var_list.index(var))
    
    marg_pi = dist.gm.pi
    marg_mu = dist.gm.mu[:, indices]
    marg_sigma = dist.gm.sigma[:, :, indices][:, indices, :]
    return Dist(var_list, GaussianMix(marg_pi, marg_mu, marg_sigma))


def aggregate_mixture(dist):
    """
    Aggregates together components with same mean and variance
    """
    mu_sigma_list = [torch.vstack([dist.gm.mu[i], dist.gm.sigma[i]]) for i in range(dist.gm.mu.shape[0])]
    unique_tensors = []
    for tensor in mu_sigma_list:
        if not any(torch.equal(tensor, unique) for unique in unique_tensors):
            unique_tensors.append(tensor)

    new_pis = torch.zeros((len(unique_tensors), 1))
    new_mus = torch.zeros((len(unique_tensors), dist.gm.mu.shape[1]))
    new_sigmas = torch.zeros((len(unique_tensors), dist.gm.sigma.shape[1], dist.gm.sigma.shape[2]))

    for i, tensor in enumerate(unique_tensors):
        # this condition expresses that mean and variance are the same
        condition = torch.all(dist.gm.mu == tensor[0], dim=1) * torch.all(dist.gm.sigma == tensor[1:], dim=(1,2))
        new_pis[i] = torch.sum(dist.gm.pi[condition])
        new_mus[i] = tensor[0]
        new_sigmas[i] = tensor[1:]

    return Dist(dist.var_list, GaussianMix(new_pis, new_mus, new_sigmas))

def ranking_prune(current_dist, Kmax):
    """ Keeps only the Kmax component with higher prob"""
    if current_dist.gm.n_comp() > Kmax:
        rank = torch.argsort(current_dist.gm.pi, dim=0, descending=True)
        current_dist.gm.pi = current_dist.gm.pi[rank].squeeze(1)[:Kmax]
        current_dist.gm.mu = current_dist.gm.mu[rank].squeeze(1)[:Kmax]
        current_dist.gm.sigma = current_dist.gm.sigma[rank].squeeze(1)[:Kmax]
        current_dist.gm.pi = current_dist.gm.pi/torch.sum(current_dist.gm.pi)
    return current_dist

def from_gm_to_string(gm):
    """
    Converts a univariate GaussianMix object into its string representation in SOGA language.
    """
    pi_list = '['
    mean_list = '['
    cov_list = '['
    for i in range(gm.n_comp()):
        pi_list += '{:.5f},'.format(gm.pi[i][0].item())
        mean_list += '{:.5f},'.format(gm.mu[i][0].item())
        cov_list += '{:.5f},'.format(gm.sigma[i][0][0].item())
    pi_list = pi_list[:-1] + ']'
    mean_list = mean_list[:-1] + ']'
    cov_list = cov_list[:-1] + ']'
    gm_str = 'gm({}, {}, {})'.format(pi_list, mean_list, cov_list)
    return gm_str

