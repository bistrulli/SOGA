# Contains the functions for computing the resulting distribution when a truncation occurs in conditional or observe instructions according to the following dependencies.

from libSOGAshared import *
from TRUNCLexer import *
from TRUNCParser import *
from TRUNCListener import *

import multiprocessing as mp

pool=None

def normalize_weights(pi):
    """
    Normalized weights and returns the normalization factor
    """
    norm_fact = torch.sum(pi)
    if norm_fact > TOL_PROB:
        new_pi = pi/norm_fact
    else:
        new_pi = pi
        norm_fact = torch.tensor(0.)
    return norm_fact, new_pi


def ineq_func(self, dist):
    """ Invoked when any inequality expression is encountered """

    ineq_coeff = self.coeff
    ineq_const = self.const
    
    # creates extended distribution
    extended_gm = extend_dist(self, dist)
        
    # here there was a part to deal with deltas, but we removed it because in torch everything is differentiable

    # STEP 1: change variables
    norm = torch.linalg.norm(ineq_coeff)
    ineq_coeff = ineq_coeff/norm
    ineq_const = ineq_const/norm
    A = find_basis(ineq_coeff)           
    transl_mu = torch.matmul(A, extended_gm.mu.unsqueeze(2)).squeeze(2)
    transl_sigma = torch.matmul(torch.matmul(A, extended_gm.sigma), A.t())
    transl_alpha = torch.zeros(len(ineq_coeff))
    transl_alpha[0] = 1

    # I suppressed the parts in which we truncate only some variables
        
    # STEP 2: creates the hyper-rectangle to integrate on
    a = torch.ones(len(transl_alpha))*(-INFTY)
    b = torch.ones(len(transl_alpha))*(INFTY)
    if self.type=='>' or self.type=='>=':
        a[0] = ineq_const
    if self.type=='<' or self.type=='<=':
        b[0] = ineq_const   
        
    # STEP 3: compute moments in the transformed coordinates
    # some components might have 0 prob in the truncation
    # indexes contains the indexes of the components that have non-zero probability
    new_P, new_transl_mu, new_transl_sigma, indexes = compute_moments(transl_mu, transl_sigma, a, b)
    # if the whole distribution has zero prob in the truncation
    if len(indexes) == 0:
        return torch.tensor(0.), dist
        
    # STEP 4: goes back to older coordinates
    old_dim = len(dist.var_list)
    A_inv = torch.linalg.inv(A)
    new_mu = torch.matmul(A_inv, new_transl_mu.unsqueeze(2)).squeeze(2)[:, :old_dim]
    new_sigma = torch.matmul( torch.matmul(A_inv, new_transl_sigma), A_inv.t())[:, :old_dim,:old_dim]
    
    # STEP 5: weights normalization
    new_pi = extended_gm.pi[indexes]*new_P.view(-1,1)
    norm_fact, norm_new_pi = normalize_weights(new_pi)

    return norm_fact, Dist(dist.var_list, GaussianMix(norm_new_pi, new_mu, new_sigma))


def eq_func(self, dist):
    """ Invoked when observe(var == c) """

    eq_coeff = self.coeff
    eq_const = self.const
    
    # here there was a part to deal with deltas, but we removed it because in torch everything is differentiable
    # I suppressed the parts in which we truncate only some variables
    
    # observed and non-observed variables
    obs_idx = int(list(torch.where(eq_coeff!=0))[0][0])
    select = (torch.arange(dist.gm.n_dim())!=obs_idx)
    # computes conditional cov and mean
    cond_sigma = torch.clone(dist.gm.sigma[:, select, :][:, :, select])
    cond_sigma = cond_sigma - (1/dist.gm.sigma[:,obs_idx,obs_idx])*torch.bmm(dist.gm.sigma[:,select,obs_idx].unsqueeze(2), dist.gm.sigma[:,obs_idx,select].unsqueeze(1))
    cond_mu = dist.gm.mu[:,select] + (eq_const-dist.gm.mu[:,obs_idx]).view(dist.gm.sigma.shape[0],1)*dist.gm.sigma[:,select,obs_idx]
    # if conditioned matrix is Null, it is equivalent to observing a single independent component
    all_zeros = torch.all(cond_sigma == 0, dim=(1,2))
    new_pi = torch.where(all_zeros, 0., dist.gm.pi.flatten()).view(-1,1)
    # normalizes weights
    norm_fact, norm_new_pi = normalize_weights(new_pi)
    # excludes observed varibles
    new_var_list = [dist.var_list[i] for i in range(len(dist.var_list)) if i!= obs_idx]
        
    return norm_fact, Dist(new_var_list, GaussianMix(norm_new_pi, cond_mu, cond_sigma))


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


class TruncRule(TRUNCListener):
    """ Class to parse truncation rules. The result of the parsing is a function stored in TruncRule.func that can be applied to distributions to truncate them."""
    
    def __init__(self, var_list, data, params_dict):
        # variables, data, parameters
        self.var_list = var_list
        self.data = data
        self.params = params_dict
        # parameters of the inequality
        self.type = None                             # <, <=, ==, !=, >=, >
        self.coeff = torch.zeros(len(var_list))      # coefficients of the linear inequality
        self.const = torch.tensor(0.)                # constant r.h.s. of the inequality
        #additional random variables (cannot use a tensors here because different a.r.v.s can have different numbers of components)
        self.aux_pis = []             # stores the weights of auxiliary variables
        self.aux_means = []           # stores the means of auxiliary variables
        self.aux_covs = []            # stores the cov matrices of auxiliary variables
        # truncation function
        self.func = None


    def parse_const(self, ctx):
        """ Parses the constant at r.h.s. of an (in)equality"""

        if not ctx.const().NUM() is None:
            const = torch.tensor(float(ctx.const().NUM().getText()))
        elif not ctx.const().idd() is None:
            const = ctx.const().idd().getValue(self.data)
        elif not ctx.const().par() is None:
            const = ctx.const().par().getValue(self.params)
        return const
    

    def parse_monom_const(self, ctx):
        """ Parses the constant in front of a monomial"""
        
        if not ctx.const() is None:
            if not ctx.const().NUM() is None:
                coeff = self.flag_sign*torch.tensor(float(ctx.const().NUM().getText()))
            elif not ctx.const().idd() is None:
                coeff = self.flag_sign*torch.tensor(ctx.const().idd().getValue(self.data))
            elif not ctx.const().par() is None:
                coeff = self.flag_sign*ctx.const().par().getValue(self.params)
        else:
            coeff = self.flag_sign
        return coeff
    
    def unpack_rvs(self, term):
        """ Unpacks the different terms when unpacking gm([pi],[mu],[sigma]) """

        self.aux_pis.append(term.gm().list_()[0].unpack(self.params))
        self.aux_means.append(term.gm().list_()[1].unpack(self.params))
        self.aux_covs.append(torch.pow(term.gm().list_()[2].unpack(self.params),2))
    

    def enterIneq(self, ctx):
        self.type = ctx.inop().getText()
        self.const = self.parse_const(ctx)
    

    def enterLexpr(self, ctx):
        self.flag_sign = torch.tensor(1.)


    def exitLexpr(self, ctx):
        self.func = partial(ineq_func,self)
        

    def enterMonom(self,ctx):
        if ctx.var().gm() is None:
            # monom in the form const? '*' (IDV | idd)
            ID = ctx.var()._getText(self.data)
            coeff = self.parse_monom_const(ctx)
            idx = self.var_list.index(ID)
            self.coeff[idx] = coeff
        else:
            # monom in the form const? '*' gm
            self.unpack_rvs(ctx.var())
            coeff = self.parse_monom_const(ctx)
            self.coeff = torch.hstack([self.coeff, coeff])            
            
    def enterSub(self, ctx):
        self.flag_sign = torch.tensor(-1.)
        
    def enterSum(self, ctx):
        self.flag_sign = torch.tensor(1.)
        
    def enterEq(self, ctx):
        self.type = ctx.eqop().getText()
        idx = self.var_list.index(ctx.var()._getText(self.data))
        self.coeff[idx] = torch.tensor(1.)
        self.const = self.parse_const(ctx)
        self.func = partial(eq_func,self)


def truncate(dist, trunc, data, params_dict):
    """ Given a distribution dist computes its truncation to trunc. Returns a pair norm_factor, new_dist where norm_factor is the probability mass of the original distribution dist on trunc and new_dist is a Dist object representing the (approximated) truncated distribution. """
    
    if trunc == 'true':
        return torch.tensor(1.), dist
    elif trunc == 'false':
        return torch.tensor(0.), dist
    else:
        trunc_rule = trunc_parse(dist.var_list, trunc, data, params_dict)
        trunc_func = trunc_rule.func
        norm_fact, new_dist = trunc_func(dist)
        return norm_fact, new_dist

    
def trunc_parse(var_list, trunc, data, params_dict):
    """ Parses trunc using ANTLR4. Returns a function """
    
    lexer = TRUNCLexer(InputStream(trunc))
    stream = CommonTokenStream(lexer)
    parser = TRUNCParser(stream)
    tree = parser.trunc()
    trunc_rule = TruncRule(var_list, data, params_dict)
    walker = ParseTreeWalker()
    walker.walk(trunc_rule, tree) 
    return trunc_rule


def find_basis(alpha):
    """
    Given alpha (vector of the truncation) returns a matrix A giving the change of variable necessary to make alpha one of the axis
    """
    
    u, s, v = torch.linalg.svd(alpha.reshape(1,alpha.shape[0]))
    alpha1 = v[:,1:]
    A = torch.vstack((alpha.reshape(1,alpha.shape[0]), alpha1.t()))
    return A


def compute_moments(mu, sigma, a, b):
    """
    Given a normal distribution with mean mu and covariance matrix sigma, truncated to [a,b], where all a_i=-np.inf and
    all b_i=np.inf except at most one a_i or one b_i, computes exactly the mean and the covariance matrix of the 
    truncated distribution
    """        
    n = len(a)
    # truncation in one dimension
    if n==1:
        tn = TruncatedNormal(mu, torch.sqrt(sigma), a, b)
        new_P = tn.norm_const
        # excluding truncated components with probability 0
        indexes = torch.where(new_P > TOL_PROB)[0]
        if len(indexes) == 0:
            return new_P, mu, sigma
        # keeping only components with non-zero prob
        new_tn = TruncatedNormal(mu[indexes], torch.sqrt(sigma[indexes]), a, b)
        new_mu = new_tn.mean()
        new_sigma = new_tn.var()
        return new_P[indexes], new_mu, new_sigma, indexes
    
    # if in more dimensions applies Kan-Robotti formulas
    # first determines if the truncation is 'low' (i.e. x > c) or 'up' (i.e. x < c)
    if a[0] > -INFTY:
        trunc = 'low'
    else:
        trunc = 'up'  
    
    new_P = prob(mu, sigma, a, b)
    # excluding truncated components with probability 0
    indexes = torch.where(new_P > TOL_PROB)[0]
    if len(indexes) == 0:
        return new_P, mu, sigma, indexes
    # keeping only components with non-zero probability
    # returns the moments for the distribution of dimension n-1, in which the trunc_idx component has been removed
    muj = compute_lower_mom(mu[indexes], sigma[indexes], a, b, trunc)
    # computes first two order moments using the recurrence formulas of Kan-Robotti and stores them in a dictionary
    new_mu = compute_mom1(mu[indexes], sigma[indexes], a, b, trunc, new_P[indexes])
    new_sigma = compute_mom2(mu[indexes], sigma[indexes], a, b, trunc, new_P[indexes], new_mu, muj)
    return new_P[indexes], new_mu, new_sigma, indexes


def compute_lower_mom(mu, sigma, a, b, trunc):
    """
    Given a normal with mean mu and cov matrix sigma,  truncated to [a,b] (where a[i] = -inf and b[i] = inf except
    for a[0] (if trunc = low) or b[0] (if trunc=up)), computes the first two orders moments of a 
    (n-1) dimensional normal distribution with mean \tilde(mu), \tilde(sigma) (as defined in Kan-Robotti).
    """
    # computes the new mean
    if trunc == 'low':
        muj = mu[:,1:] + ((a[0]-mu[:,0])/sigma[:,0,0]).view(-1,1)*sigma[:,1:][:,:,0]
    elif trunc == 'up':
        muj =  mu[:,1:] + ((b[0]-mu[:,0])/sigma[:,0,0]).view(-1,1)*sigma[:,1:][:,:,0]
    return muj


def prob(mu, sigma, a, b):
    """
    Computes the mass probability of the normal distribution with mean mu and covariance matrix sigma in the 
    hyper-rectangle [a,b].
    Even for one-dimensional distributions, mu, sigma, a, b must be vectors.
    """ 
    
    norm = distributions.Normal(loc=mu[:,0], scale=torch.sqrt(sigma[:,0,0]))
    P = norm.cdf(b[0]) - norm.cdf(a[0])
    return P


def compute_mom1(mu, sigma, a, b, trunc, P):

    c = torch.zeros(mu.shape)
    norm = distributions.Normal(mu[:,0], scale=torch.sqrt(sigma[:,0,0]))
    if trunc == 'low':
        c[:,0] = norm.log_prob(a[0]).exp()
    elif trunc == 'up':
        c[:,0] = -norm.log_prob(b[0]).exp()
    return mu + torch.matmul(sigma, c.unsqueeze(2)).squeeze(2)/P.view(-1,1)


def compute_mom2(mu, sigma, a, b, trunc, new_P, new_mu, muj):
    # vector dimensions
    n = len(a)    # number of variables
    c = mu.shape[0] # number of components
    # creates auxialiary vectors
    e0 = torch.zeros((c,n))
    e0[:,0] = torch.ones(c)
    C = (new_P.view(c,1,1))*torch.eye(n).unsqueeze(0).expand(c,-1,-1)
    norm = distributions.Normal(loc=mu[:,0], scale=torch.sqrt(sigma[:,0,0]))
    if trunc == 'low':
        C[:,:,0] += norm.log_prob(a[0]).exp().view(-1,1)*(a[0]**e0)*torch.hstack((torch.ones((c,1)), muj))
    elif trunc == 'up':
        C[:,:,0] += -norm.log_prob(b[0]).exp().view(-1,1)*(b[0]**e0)*torch.hstack((torch.ones((c,1)), muj))
    # computes the new matrix
    new_sigma = new_P.view(c,1,1)*torch.matmul(mu.unsqueeze(2), new_mu.unsqueeze(1)) + torch.matmul(sigma, C.transpose(1,2))
    new_sigma = new_sigma/new_P.view(c,1,1) - torch.matmul(new_mu.unsqueeze(2), new_mu.unsqueeze(1))
    return new_sigma 

#def insert_value(val, idx, mu, sigma):
#    """ Extends mu and sigma by adding val in corresponding to the idx position (for sigma the other row- and column-entries are 0) """
#    d = len(mu)
#    new_mu = np.array(list(mu[:idx]) + [val] + list(mu[idx:]))
#    new_sigma = np.block([[sigma[:idx,:idx], np.zeros((idx,1)), sigma[:idx,idx:]], 
#          [np.zeros((1,d+1))],
#          [sigma[idx:,:idx], np.zeros((d-idx,1)), sigma[idx:,idx:]]])
#    return new_mu, new_sigma
#    

#def select_indices(alpha, aux_cov):
#    no_zeros = torch.where(alpha != 0)[0].numpy()
#    coeff_set = list(no_zeros)
#    for idx in no_zeros:
#        new_set = torch.where(aux_cov[idx,:] != 0)[0].numpy()
#        coeff_set += list(new_set)
#    # removed np.sort 
#    coeff_set = list(set(coeff_set))
#    return coeff_set


#def reduce_indices(vec, indices):
#    """
#    Extracts subvector/submatrix indexed by indices
#    """
#    if len(vec.shape) == 1:
#        red_vec = vec[indices]
#    if len(vec.shape) == 2:
#        red_vec = vec[indices][:,indices]
#    return red_vec

#def extend_indices(red_vec, old_vec, indices):
#    """
#    puts red_vec in the indices of old_vec
#    """
#    if len(old_vec.shape) == 1:
#        red_vec = red_vec.reshape(len(red_vec),)
#        old_vec[indices] = red_vec
#    if len(old_vec.shape) == 2:
#        C = old_vec[indices]
#        C[:,indices] = red_vec
#        old_vec[indices] = C
#    return old_vec


### compute moments functions

#def partitionfunc(n,k,l=0):
#    """
#    n is the integer to partition, k is the length of partitions, l is the min partition element size
#    """
#    if k < 1:
#        return
#    if k == 1:
#        if n >= l:
#            yield (n,)
#        return
#    for i in range(l,n+1):
#        for result in partitionfunc(n-i,k-1):
#            yield (i,)+result


# parallel implementation
#def parallel_truncate(dist, trunc, data,nproc):
#    global pool
#    gst=time()
#    if(pool is None):
#        print("creating pool")
#        #pool=ThreadPoolExecutor(max_workers=nproc)
#        pool=mp.Pool(nproc)
#
#    """ Given a distribution dist computes its truncation to trunc. Returns a pair norm_factor, new_dist where norm_factor is the probability mass of the original distribution dist on trunc and new_dist is a Dist object representing the (approximated) truncated distribution. """
#    if trunc == 'true':
#        return 1., dist
#    elif trunc == 'false':
#        return 0., dist
#    else:
#        #print(f"##### ncomp={dist.gm.n_comp()}")
#        st=time()
#        trunc_rule = trunc_parse(dist.var_list, trunc, data)
#        #print(f"trunc_parse:{time()-st}")
#        trunc_func = trunc_rule.func
#        trunc_type = trunc_rule.type
#        trunc_idx = np.where(np.array(trunc_rule.coeff) != 0)[0][0]
#        hard = []
#        new_dist = Dist(dist.var_list, GaussianMix([],[],[]))
#        new_pi = []
#        comp_list = []
#
#        st=time()
#        for k in range(dist.gm.n_comp()):
#            comp = Dist(dist.var_list, dist.gm.comp(k))
#            comp_list.append(comp)
#        #print(f"loop time:{time()-st}")
#
#        st=time()
#
#        trans_comp = list(pool.map(trunc_func, comp_list))
#        #print(f"map time:{time()-st}")
#
#        st=time()
#        for k in range(dist.gm.n_comp()):
#            if trunc_type == '==' and dist.gm.sigma[k][trunc_idx,trunc_idx] < delta_tol and sum(trans_comp[k].pi) > prob_tol:
#                hard.append(k)
#        #print(f"second loop time:{time()-st}")
#
#        if len(hard) == 0:
#            for k in range(dist.gm.n_comp()):
#                new_mix = trans_comp[k]
#                for h in range(new_mix.n_comp()):
#                    if new_mix.pi[h] > prob_tol:
#                        new_dist.gm.mu.append(new_mix.mu[h])
#                        new_dist.gm.sigma.append(new_mix.sigma[h])
#                        new_pi.append(dist.gm.pi[k]*new_mix.pi[h])
#        else:
#            for k in hard:
#                new_mix = trans_comp[k]
#                for h in range(new_mix.n_comp()):
#                    if new_mix.pi[h] > prob_tol:
#                        new_dist.gm.mu.append(new_mix.mu[h])
#                        new_dist.gm.sigma.append(new_mix.sigma[h])
#                        new_pi.append(dist.gm.pi[k]*new_mix.pi[h])
#        norm_factor = sum(np.array(new_pi))
#        if norm_factor > prob_tol:
#            new_dist.gm.pi = list(np.array(new_pi)/norm_factor)
#        #print(f"total time:{time()-gst}")
#        return norm_factor, new_dist

#def split_trunc(trunc):
#    """ When trunc is 'x != c' returns 'x > c' and 'x < c' """
#    assert '!=' in trunc
#    trunc1 = trunc.replace('!=','>')
#    trunc2 = trunc.replace('!=','<')
#    return trunc1, trunc2