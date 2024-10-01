# Contains the functions for computing the resulting distribution when a truncation occurs in conditional or observe instructions according to the following dependencies.

from libSOGAshared import *
from TRUNCLexer import *
from TRUNCParser import *
from TRUNCListener import *

import multiprocessing as mp
from functools import partial

pool=None

def ineq_func(self,comp):
    
    # mean, and variance of the component
    mu = comp.gm.mu[0]
    sigma = comp.gm.sigma[0]
    
    # list to store truncated moments
    final_pi = []
    final_mu = []
    final_sigma = []
    
    # cycles over all possible combinations of components of the auxiliary variables
    for part in product(*[range(len(mean)) for mean in self.aux_means]):

        # creates the extended vector of moments (with auxiliary variables)
        aux_pi = 1
        aux_mean = torch.clone(mu)
        aux_sigma = torch.tensor([])
        ineq_coeff = torch.clone(self.coeff)
        ineq_const = self.const
        for p,q in zip(range(len(self.aux_means)), part):
            aux_pi = aux_pi*self.aux_pis[p][q]
            aux_mean = torch.hstack([aux_mean, self.aux_means[p][q]])
            aux_sigma = torch.hstack([aux_sigma, self.aux_covs[p][q]])
        aux_sigma = torch.diag(aux_sigma)
        aux_cov = torch.vstack([torch.hstack([sigma, torch.zeros((len(sigma), len(aux_sigma)))]), torch.hstack([torch.zeros((len(aux_sigma), len(sigma))), aux_sigma])])
        
        # here there was a part to deal with deltas, but we removed it because in torch everything is differentiable
        
        # STEP 1: change variables
        norm = torch.linalg.norm(ineq_coeff)
        ineq_coeff = ineq_coeff/norm
        ineq_const = ineq_const/norm
        A = find_basis(ineq_coeff)           # maybe instead of A a vector can be used to improve scalability (?)
        transl_mu = torch.mm(A, aux_mean.reshape((len(aux_mean),1))).flatten()
        transl_sigma = torch.mm(torch.mm(A, aux_cov), A.t())
        
        # STEP 2: finds the indices of the components that needs to be transformed
        transl_alpha = torch.zeros(len(transl_mu))
        transl_alpha[0] = 1
        indices = select_indices(transl_alpha, transl_sigma)

        # STEP 3: creates reduced vectors taking into account only the coordinates that need to be transformed
        red_transl_alpha = reduce_indices(transl_alpha, indices)
        red_transl_mu = reduce_indices(transl_mu, indices)
        red_transl_sigma = reduce_indices(transl_sigma, indices) 
        
        # STEP 4: creates the hyper-rectangle to integrate on
        a = torch.ones(len(red_transl_alpha))*(-torch.inf)
        b = torch.ones(len(red_transl_alpha))*(torch.inf)
        if self.type=='>' or self.type=='>=':
            a[0] = ineq_const
        if self.type=='<' or self.type=='<=':
            b[0] = ineq_const   
        
        # STEP 5: compute moments in the transformed coordinates
        new_P, new_red_transl_mu, new_red_transl_sigma = compute_moments(red_transl_mu, red_transl_sigma, a, b)
        new_red_transl_mu = new_red_transl_mu.flatten()
        
        # STEP 6: recreates extended vectors
        new_transl_mu = extend_indices(new_red_transl_mu, transl_mu, indices)
        new_transl_sigma = extend_indices(new_red_transl_sigma, transl_sigma, indices)
        
        # STEP 7: goes back to older coordinates
        d = len(comp.var_list)
        A_inv = torch.linalg.inv(A)
        new_mu = torch.mm(A_inv, new_transl_mu.reshape((len(new_transl_mu), 1))).flatten()[:d]
        new_sigma = torch.mm(torch.mm(A_inv, new_transl_sigma), A_inv.t())[:d,:d]
        
        # append new values
        final_pi.append(aux_pi*new_P)   # contains original weight of the component * prob. of the component being in the trunc
        final_mu.append(new_mu)
        final_sigma.append(new_sigma)
        
    return Dist(comp.var_list, GaussianMix(final_pi, final_mu, final_sigma))

def eq_func(self,comp):
    mu = comp.gm.mu[0]
    sigma = comp.gm.sigma[0]
    final_pi = []
    final_mu = []
    final_sigma = []
    eq_coeff = copy(self.coeff)
    eq_const = self.const
   
    # here there was a part to deal with deltas, but we removed it because in torch everything is differentiable
    
    # STEP 1: selects indices to condition
    indices = select_indices(eq_coeff, sigma)
    # if single Gaussian raise error
    if len(indices) == 1:
        print('Conditioning to zero probability event!')
        raise
    else:
        
        # STEP 2: creates reduced vectors
        red_mu = reduce_indices(mu, indices)
        red_sigma = reduce_indices(sigma, indices) 
        red_alpha = reduce_indices(eq_coeff, indices)
        red_obs_idx = int(list(torch.where(red_alpha!=0))[0][0])
        
        # STEP 3: computes cond_sigma (select is a mask containing the index of the conditioned variables)
        select = (torch.arange(len(red_mu))!=red_obs_idx)
        cond_sigma = red_sigma[select,:][:,select]
        cond_sigma = cond_sigma - (1/red_sigma[red_obs_idx,red_obs_idx])*torch.mm(red_sigma[select,red_obs_idx].reshape(len(select)-1,1), red_sigma[red_obs_idx,select].reshape(1,len(select)-1))
        
        # STEP 4: computes cond_mu
        cond_mu = red_mu[select] + (1/red_sigma[red_obs_idx,red_obs_idx])*(eq_const-red_mu[red_obs_idx])*red_sigma[select,red_obs_idx]
        # if conditioned matrix is Null, it is equivalent to observing a single independent component
        if torch.all(cond_sigma == 0):  
            new_P = 0.
        else:
            new_P = 1.
        
        # STEP 5: returns to the original set of variables
        new_sigma = extend_indices(cond_sigma, sigma, indices)
        new_mu = extend_indices(cond_mu, mu, indices)

        # STEP 6: remove observed variable
        obs_idx = int(list(torch.where(eq_coeff!=0))[0][0])
        indices = [i for i in range(len(new_mu)) if i!= obs_idx]
        new_sigma = reduce_indices(new_sigma, indices)
        new_mu = reduce_indices(new_mu, indices)
        new_var_list = [comp.var_list[i] for i in indices]
        
    return Dist(new_var_list, GaussianMix([new_P], [new_mu], [new_sigma]))

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

#def split_trunc(trunc):
#    """ When trunc is 'x != c' returns 'x > c' and 'x < c' """
#    assert '!=' in trunc
#    trunc1 = trunc.replace('!=','>')
#    trunc2 = trunc.replace('!=','<')
#    return trunc1, trunc2

class TruncRule(TRUNCListener):
    
    def __init__(self, var_list, data):
        self.var_list = var_list
        self.data = data
        self.type = None
        self.coeff = torch.zeros(len(var_list))
        self.const = torch.tensor(0.)
        self.func = None
        
        self.aux_pis = []
        self.aux_means = []
        self.aux_covs = []
        
    def enterIneq(self, ctx):
        self.type = ctx.inop().getText()
        if not ctx.const().NUM() is None:
            self.const = torch.tensor(float(ctx.const().NUM().getText()))
        elif not ctx.const().idd() is None:
            self.const = ctx.const().idd().getValue(self.data)
                
    
    def enterLexpr(self, ctx):
        self.flag_sign = torch.tensor(1.)

            
    def exitLexpr(self, ctx):
        self.func = partial(ineq_func,self)
        
        
    def enterMonom(self,ctx):
        if ctx.var().gm() is None:
            # monom in the form const? '*' (IDV | idd)
            ID = ctx.var()._getText(self.data)
            if not ctx.const() is None:
                if not ctx.const().NUM() is None:
                    coeff = self.flag_sign*torch.tensor(float(ctx.const().NUM().getText()))
                elif not ctx.const().idd() is None:
                    coeff = self.flag_sign*torch.tensor(ctx.const().idd().getValue(self.data))
            else:
                coeff = self.flag_sign
            idx = self.var_list.index(ID)
            self.coeff[idx] = coeff
        # monom in the form const? '*' gm
        else:
            self.aux_pis.append(torch.tensor(eval(ctx.var().gm().list_()[0].getText())))
            self.aux_means.append(torch.tensor(eval(ctx.var().gm().list_()[1].getText())))
            self.aux_covs.append(torch.pow(torch.tensor(eval(ctx.var().gm().list_()[2].getText())),2))
            if not ctx.const() is None:
                if not ctx.const().NUM() is None:
                    coeff = self.flag_sign*torch.tensor(float(ctx.const().NUM().getText()))
                elif not ctx.const().idd() is None:
                    coeff = self.flag_sign*torch.tensor(ctx.const().idd().getValue(self.data))
            else:
                coeff = self.flag_sign 
            self.coeff = torch.hstack([self.coeff, coeff])            
            
    def enterSub(self, ctx):
        self.flag_sign = torch.tensor(-1.)
        
    def enterSum(self, ctx):
        self.flag_sign = torch.tensor(1.)
        
    def enterEq(self, ctx):
        self.type = ctx.eqop().getText()
        idx = self.var_list.index(ctx.var()._getText(self.data))
        self.coeff[idx] = torch.tensor(1.)
        if not ctx.const() is None:
            if not ctx.const().NUM() is None:
                self.const = torch.tensor(float(ctx.const().NUM().getText()))
            elif not ctx.const().idd() is None:
                self.const = torch.tensor(ctx.const().idd().getValue(self.data))
        self.func = partial(eq_func,self)


def truncate(dist, trunc, data):
    """ Given a distribution dist computes its truncation to trunc. Returns a pair norm_factor, new_dist where norm_factor is the probability mass of the original distribution dist on trunc and new_dist is a Dist object representing the (approximated) truncated distribution. """
    if trunc == 'true':
        return torch.tensor(1.), dist
    elif trunc == 'false':
        return torch.tensor(0.), dist
    else:
        trunc_rule = trunc_parse(dist.var_list, trunc, data)
        trunc_func = trunc_rule.func
        trunc_type = trunc_rule.type
        trunc_idx = torch.where(trunc_rule.coeff != 0)[0][0]
        new_dist = Dist(dist.var_list, GaussianMix([],[],[]))
        new_pi = [] 
        trans_comp = []
        for k in range(dist.gm.n_comp()):
            comp = Dist(dist.var_list, dist.gm.comp(k))  
            trans_comp.append(trunc_func(comp))
        for k in range(dist.gm.n_comp()):
            new_mix = trans_comp[k].gm
            for h in range(new_mix.n_comp()):
                if new_mix.pi[h] > TOL_PROB:
                    new_dist.gm.mu.append(new_mix.mu[h])
                    new_dist.gm.sigma.append(new_mix.sigma[h])
                    new_pi.append(dist.gm.pi[k]*new_mix.pi[h])
        # this is needed because observing that a variable is equal to a value it is removed
        new_dist.var_list = trans_comp[0].var_list       
        # renormalizing
        norm_factor = torch.sum(torch.tensor(new_pi))
        if norm_factor > TOL_PROB:
            new_dist.gm.pi = list(np.array(new_pi)/norm_factor)
        return norm_factor, new_dist


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
    u, s, v = torch.linalg.svd(alpha.reshape(1,alpha.shape[0]))
    alpha1 = v[:,1:]
    A = torch.vstack((alpha.reshape(1,alpha.shape[0]), alpha1.t()))
    return A

def select_indices(alpha, aux_cov):
    no_zeros = torch.where(alpha != 0)[0].numpy()
    coeff_set = list(no_zeros)
    for idx in no_zeros:
        new_set = torch.where(aux_cov[idx,:] != 0)[0].numpy()
        coeff_set += list(new_set)
    # removed np.sort 
    coeff_set = list(set(coeff_set))
    return coeff_set


def reduce_indices(vec, indices):
    """
    Extracts subvector/submatrix indexed by indices
    """
    if len(vec.shape) == 1:
        red_vec = vec[indices]
    if len(vec.shape) == 2:
        red_vec = vec[indices][:,indices]
    return red_vec

def extend_indices(red_vec, old_vec, indices):
    """
    puts red_vec in the indices of old_vec
    """
    if len(old_vec.shape) == 1:
        red_vec = red_vec.reshape(len(red_vec),)
        old_vec[indices] = red_vec
    if len(old_vec.shape) == 2:
        C = old_vec[indices]
        C[:,indices] = red_vec
        old_vec[indices] = C
    return old_vec


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

def _prob(mu, sigma, a, b):
    """
    Computes the mass probability of the normal distribution with mean mu and covariance matrix sigma in the 
    hyper-rectangle [a,b].
    Even for one-dimensional distributions, mu, sigma, a, b must be vectors.
    """ 
    norm = distributions.Normal(loc=mu[0], scale=torch.sqrt(sigma[0,0]))
    P = norm.cdf(b[0]) - norm.cdf(a[0])
    return P
    

def compute_lower_mom(mu, sigma, a, b, trunc_idx, trunc):
    """
    Given a normal with mean mu and cov matrix sigma,  truncated to [a,b] (where a[i] = -inf and b[i] = inf except
    for a[trunc_idx] (if trunc = low) or b[trunc_idx] (if trunc=up)), computes the first two orders moments of a 
    (n-1) dimensional normal distribution with mean \tilde(mu), \tilde(sigma) (as defined in Kan-Robotti).
    """
    n = len(mu)
    c = torch.hstack([a[:trunc_idx], a[trunc_idx+1:]])
    d = torch.hstack([b[:trunc_idx], b[trunc_idx+1:]])
    # computes the new mean
    if trunc == 'low':
        muj = torch.hstack([mu[:trunc_idx], mu[trunc_idx+1:]]) + ((a[trunc_idx]-mu[trunc_idx])/sigma[trunc_idx, trunc_idx])*torch.vstack([sigma[:trunc_idx], sigma[trunc_idx+1:]])[:,trunc_idx]
    elif trunc == 'up':
        muj =  torch.hstack([mu[:trunc_idx], mu[trunc_idx+1:]]) + ((b[trunc_idx]-mu[trunc_idx])/sigma[trunc_idx, trunc_idx])*torch.vstack([sigma[:trunc_idx], sigma[trunc_idx+1:]])[:,trunc_idx]
    return muj


def compute_mom1(n, mu, sigma, a, b, trunc_idx, trunc, P):
    c = torch.zeros((n,1))
    norm = distributions.Normal(mu[trunc_idx], scale=torch.sqrt(sigma[trunc_idx,trunc_idx]))
    if trunc == 'low':
        c[trunc_idx] = norm.log_prob(a[trunc_idx]).exp()
    elif trunc == 'up':
        c[trunc_idx] = -norm.log_prob(b[trunc_idx]).exp()
    return mu + torch.mm(sigma, c).t()/P      

    
def compute_mom2(n, mu, sigma, a, b, trunc_idx, trunc, new_P, new_mu, muj):
    e0 = torch.zeros(n)
    e0[0] = 1
    C = new_P*torch.eye(n)
    norm = distributions.Normal(loc=mu[0], scale=torch.sqrt(sigma[0,0]))
    if trunc == 'low':
        C[:,0] += norm.log_prob(a[0]).exp()*(a[0]**e0)*torch.hstack((torch.tensor([1.]), muj))
    elif trunc == 'up':
        C[:,0] += -norm.log_prob(b[0]).exp()*(b[0]**e0)*torch.hstack((torch.tensor([1.]), muj))
    new_sigma = new_P*torch.mm(mu.reshape(-1,1), new_mu.reshape(1,-1)) + torch.mm(sigma, C.t())
    new_sigma = new_sigma/new_P - torch.mm(new_mu.reshape(-1,1), new_mu.reshape(1,-1))
    return new_sigma 
    

def compute_moments(mu, sigma, a, b):
    """
    Given a normal distribution with mean mu and covariance matrix sigma, truncated to [a,b], where all a_i=-np.inf and
    all b_i=np.inf except at most one a_i or one b_i, computes exactly the mean and the covariance matrix of the 
    truncated distribution
    """        
    n = len(a)   
    # truncation in one dimension
    if n==1:
        tn = TruncatedNormal(mu[0], torch.sqrt(sigma[0,0]), a[0], b[0])
        new_P = tn.norm_const
        new_mu = tn.mean()
        new_sigma = tn.var()
        new_mu = torch.tensor([new_mu])
        new_sigma = torch.tensor([[new_sigma]])
        return new_P, new_mu, new_sigma
    # if in more dimensions applies Kan-Robotti formulas
    # first determines if the truncation is 'low' (i.e. x > c) or 'up' (i.e. x < c)
    trunc_idx = 0
    if a[0] > -torch.inf:
        trunc = 'low'
    else:
        trunc = 'up'  
    # returns the moments for the distribution of dimension n-1, in which the trunc_idx component has been removed
    muj = compute_lower_mom(mu, sigma, a, b, trunc_idx, trunc)
    # computes first two order moments using the recurrence formulas of Kan-Robotti and stores them in a dictionary
    new_P = _prob(mu, sigma, a, b)
    new_mu = compute_mom1(n, mu, sigma, a, b, trunc_idx, trunc, new_P)
    new_sigma = compute_mom2(n, mu, sigma, a, b, trunc_idx, trunc, new_P, new_mu, muj)
    return new_P, new_mu, new_sigma#

#def insert_value(val, idx, mu, sigma):
#    """ Extends mu and sigma by adding val in corresponding to the idx position (for sigma the other row- and column-entries are 0) """
#    d = len(mu)
#    new_mu = np.array(list(mu[:idx]) + [val] + list(mu[idx:]))
#    new_sigma = np.block([[sigma[:idx,:idx], np.zeros((idx,1)), sigma[:idx,idx:]], 
#          [np.zeros((1,d+1))],
#          [sigma[idx:,:idx], np.zeros((d-idx,1)), sigma[idx:,idx:]]])
#    return new_mu, new_sigma
#    
    
    