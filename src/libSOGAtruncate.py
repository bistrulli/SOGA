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

def and_func(self, dist):
    """ Truncates the distribution to var > self.low and var < self.up """

    ineq_coeff = self.coeff
    ineq_idx = torch.where(ineq_coeff != 0)[0][0]
    low_const = self.low
    up_const = self.up
    
    # This function only works for "var > self.low and var < self.up" so there is no need
    # to extend the distribution and change variables
        
    # STEP 1: creates the hyper-rectangle to integrate on
    a = torch.ones(len(ineq_coeff))*(-INFTY)
    b = torch.ones(len(ineq_coeff))*(INFTY)
    a[ineq_idx] = low_const
    b[ineq_idx] = up_const   
        
    # STEP 2: compute moments in the transformed coordinates
    # some components might have 0 prob in the truncation
    # indexes contains the indexes of the components that have non-zero probability
    new_P, new_mu, new_sigma, indexes = compute_moments(dist.gm.mu, dist.gm.sigma, a, b, ineq_idx)
    # if the whole distribution has zero prob in the truncation
    if len(indexes) == 0:
        return torch.tensor(0.), dist
    
    # STEP 3: weights normalization
    new_pi = dist.gm.pi[indexes]*new_P.view(-1,1)
    norm_fact, norm_new_pi = normalize_weights(new_pi)

    return norm_fact, Dist(dist.var_list, GaussianMix(norm_new_pi, new_mu, new_sigma))


def or_func(self, dist):
    """ Truncates the distribution to var > self.low or var < self.up """

    idx = torch.where(self.coeff != 0)[0][0]
    var = dist.var_list[idx]
    trunc_low = '{} < {:.10f}'.format(var,self.up)
    trunc_up = '{} > {:.10f}'.format(var,self.low)
    p_low, dist_low = truncate(dist, trunc_low, {}, {})
    p_up, dist_up = truncate(dist, trunc_up, {}, {})
    new_pi = torch.vstack([p_low*dist_low.gm.pi, p_up*dist_up.gm.pi])
    new_mu = torch.vstack([dist_low.gm.mu, dist_up.gm.mu])
    new_sigma = torch.vstack([dist_low.gm.sigma, dist_up.gm.sigma])
    norm_fact, norm_new_pi = normalize_weights(new_pi)
    return norm_fact, Dist(dist.var_list, GaussianMix(norm_new_pi, new_mu, new_sigma))

# version with no index selection
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
    # add again the part for deltas
    obs_idx = int(list(torch.where(eq_coeff!=0))[0][0])
    if torch.all(dist.gm.sigma[:, obs_idx, obs_idx] < 1e-5):
        mask = dist.gm.mu[:, obs_idx] == eq_const
        new_pi = dist.gm.pi[mask]
        norm_fact = torch.sum(new_pi) 
        norm_new_pi = new_pi/norm_fact
        new_mu = dist.gm.mu[mask]
        new_sigma = dist.gm.sigma[mask]
        return norm_fact, Dist(dist.var_list, GaussianMix(norm_new_pi, new_mu, new_sigma))
    
    # I suppressed the parts in which we truncate only some variables
    # observed and non-observed variables
    obs_idx = int(list(torch.where(eq_coeff!=0))[0][0])
    select = (torch.arange(dist.gm.n_dim())!=obs_idx)
    # computes conditional cov and mean
    cond_sigma = torch.clone(dist.gm.sigma[:, select, :][:, :, select])
    cond_sigma = cond_sigma - (1/dist.gm.sigma[:,obs_idx,obs_idx]).view(-1,1,1)*torch.bmm(dist.gm.sigma[:,select,obs_idx].unsqueeze(2), dist.gm.sigma[:,obs_idx,select].unsqueeze(1))
    cond_mu = dist.gm.mu[:,select] + (1/dist.gm.sigma[:,obs_idx,obs_idx]).view(-1,1)*(eq_const-dist.gm.mu[:,obs_idx]).view(dist.gm.sigma.shape[0],1)*dist.gm.sigma[:,select,obs_idx]
    # if conditioned matrix is Null, it is equivalent to observing a single independent component
    all_zeros = torch.all(cond_sigma == 0, dim=(1,2))
    new_pi = torch.where(all_zeros, 0., dist.gm.pi.flatten()).view(-1,1)
    # normalizes weights
    norm_fact, norm_new_pi = normalize_weights(new_pi)
    # extends cond mu and sigma with values for observed bar (puts small variance to the observed variable)
    new_cond_mu = torch.zeros((cond_mu.shape[0], cond_mu.shape[1]+1))
    new_cond_mu[:, :obs_idx] = cond_mu[:, :obs_idx]
    new_cond_mu[:, obs_idx] = torch.ones(cond_mu.shape[0])*eq_const
    new_cond_mu[:, obs_idx+1:] = cond_mu[:, obs_idx:]
    mask = torch.ones(cond_mu.shape[1] + 1, dtype=torch.bool)
    mask[obs_idx] = False
    new_cond_sigma = torch.zeros((cond_sigma.shape[0], cond_sigma.shape[1]+1, cond_sigma.shape[2]+1))
    C = new_cond_sigma[:, mask, :]
    C[:, :, mask] = cond_sigma
    new_cond_sigma[:, mask, :] = C
    new_cond_sigma[:, obs_idx, obs_idx] = torch.ones(cond_sigma.shape[0])*SMOOTH_EPS
    return norm_fact, Dist(dist.var_list, GaussianMix(norm_new_pi, new_cond_mu, new_cond_sigma))


def negate(trunc):
    """ Produces a string which is the logic negation of trunc """

    if ' and ' in trunc:
        trunc1, trunc2 = trunc.split(' and ')
        trunc1 = negate(trunc1)
        trunc2 = negate(trunc2)
        trunc = trunc1 + ' or ' + trunc2
    elif ' or ' in trunc:
        trunc1, trunc2 = trunc.split(' or ')
        trunc1 = negate(trunc1)
        trunc2 = negate(trunc2)
        trunc = trunc2 + ' and ' + trunc1
    else:
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
        elif '!=' in trunc:
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
        if isinstance(ctx, TRUNCParser.ConstContext):
            parse_ctx = ctx
            if not parse_ctx.num() is None:
                const = torch.tensor(float(parse_ctx.num().getText()))
            elif not parse_ctx.idd() is None:
                const = parse_ctx.idd().getValue(self.data)
            elif not parse_ctx.par() is None:
                const = parse_ctx.par().getValue(self.params)
        elif isinstance(ctx, TRUNCParser.Pos_constContext):
            parse_ctx = ctx
            if not parse_ctx.POSNUM() is None:
                const = torch.tensor(float(parse_ctx.POSNUM().getText()))
            elif not parse_ctx.idd() is None:
                const = parse_ctx.idd().getValue(self.data)
            elif not parse_ctx.par() is None:
                const = parse_ctx.par().getValue(self.params)
        return const
    
    def parse_const_expr(self, ctx):
        init_const = self.parse_const(ctx.const())
        if ctx.pos_const():
            for aop, pos_const in zip(ctx.aop(), ctx.pos_const()):
                if aop.getText() == '+':
                    init_const = init_const + self.parse_const(pos_const)
                elif aop.getText() == '-':
                    init_const = init_const - self.parse_const(pos_const)     
                elif aop.getText() == '*':
                    init_const = init_const * self.parse_const(pos_const)       
        return init_const


    #def parse_monom_const(self, ctx):
    #    """ Parses the constant in front of a monomial"""
    #    
    #    if ctx.const():
    #        if ctx.const().num():
    #            coeff = self.flag_sign*torch.tensor(float(ctx.const().num().getText()))
    #        elif ctx.const().idd():
    #            coeff = self.flag_sign*torch.tensor(ctx.const().idd().getValue(self.data))
    #        elif ctx.const().par():
    #            coeff = self.flag_sign*ctx.const().par().getValue(self.params)
    #    else:
    #        coeff = self.flag_sign
    #    return coeff
    
    def unpack_rvs(self, term):
        """ Unpacks the different terms when unpacking gm([pi],[mu],[sigma]) """

        self.aux_pis.append(term.gm().list_()[0].unpack(self.params))
        self.aux_means.append(term.gm().list_()[1].unpack(self.params))
        self.aux_covs.append(torch.pow(term.gm().list_()[2].unpack(self.params),2))
    

    def enterIneq(self, ctx):
        self.type = ctx.inop().getText()
        self.const = self.parse_const_expr(ctx.const_expr())


    def enterAnd_trunc(self, ctx):
        ID = ctx.IDV()[0].getText()
        idx = self.var_list.index(ID)
        self.coeff[idx] = torch.tensor(1.)
        inops = ctx.inop()
        const_exprs = ctx.const_expr()
        for i in range(2):
            const = self.parse_const_expr(const_exprs[i])
            inop = inops[i].getText()
            if '<' in inop:
                self.up = const
            elif '>' in inop:
                self.low = const
        self.func = partial(and_func,self)


    def enterOr_trunc(self, ctx):
        ID = ctx.IDV()[0].getText()
        idx = self.var_list.index(ID)
        self.coeff[idx] = torch.tensor(1.)
        inops = ctx.inop()
        const_exprs = ctx.const_expr()
        for i in range(2):
            const = self.parse_const_expr(const_exprs[i])
            inop = inops[i].getText()
            if '<' in inop:
                self.up = const
            elif '>' in inop:
                self.low = const
        self.func = partial(or_func,self)

    def enterLexpr(self, ctx):
        self.flag_sign = torch.tensor(1.)
        monom = ctx.monom()[0]
        self.parse_monom(monom)
        for aop, monom in zip(ctx.aop(), ctx.monom()[1:]):
            self.parse_monom(monom, aop)


    def exitLexpr(self, ctx):
        self.func = partial(ineq_func,self)
        

    def parse_monom(self,ctx,op=None):
        """ Parses a monomial, if op is not None, is the operator before the monomial"""
        if ctx.var().gm() is None:
            # monom in the form const? '*' (IDV | idd)
            ID = ctx.var()._getText(self.data)
            coeff = torch.tensor(1.)
            if ctx.const():
                coeff = self.parse_const(ctx)
            if op:
                if op.SUB():
                    coeff = -coeff
            idx = self.var_list.index(ID)
            self.coeff[idx] = coeff
        else:
            # monom in the form const? '*' gm
            self.unpack_rvs(ctx.var())
            coeff = torch.tensor(1.)
            if ctx.const():
                coeff = self.parse_const(ctx)
            if op:
                if op.SUB():
                    coeff = -coeff
            self.coeff = torch.hstack([self.coeff, coeff])            
            
    #def enterSub(self, ctx):
    #    self.flag_sign = torch.tensor(-1.)
        
    #def enterSum(self, ctx):
    #    self.flag_sign = torch.tensor(1.)
        
    def enterEq(self, ctx):
        self.type = ctx.eqop().getText()
        idx = self.var_list.index(ctx.var()._getText(self.data))
        self.coeff[idx] = torch.tensor(1.)
        self.const = self.parse_const_expr(ctx.const_expr())
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


def compute_moments(mu, sigma, a, b, idx=0):
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
            return new_P, mu, sigma, indexes
        # keeping only components with non-zero prob
        new_tn = TruncatedNormal(mu[indexes], torch.sqrt(sigma[indexes]), a, b)
        new_mu = new_tn.mean()
        new_sigma = new_tn.var()
        return new_P[indexes], new_mu, new_sigma, indexes
    
    # if in more dimensions applies Kan-Robotti formulas
    # first determines if the truncation is 'low' (i.e. x > c) or 'up' (i.e. x < c) or None if in and_func
    if a[idx] > -INFTY and b[idx] < INFTY:
        trunc = None
    elif a[0] > -INFTY:
        trunc = 'low'
    else:
        trunc = 'up'  
    
    new_P = prob(mu, sigma, a, b, idx)
    # excluding truncated components with probability 0
    indexes = torch.where(new_P > TOL_PROB)[0]
    if len(indexes) == 0:
        return new_P, mu, sigma, indexes
    # keeping only components with non-zero 
    # computes first two order moments using the recurrence formulas of Kan-Robotti and stores them in a dictionary
    new_mu = compute_mom1(mu[indexes], sigma[indexes], a, b, trunc, new_P[indexes], idx)
    if trunc:
        # returns the moments for the distribution of dimension n-1, in which the trunc_idx component has been removed
        muj = compute_lower_mom(mu[indexes], sigma[indexes], a, b, trunc)
        new_sigma = compute_mom2(mu[indexes], sigma[indexes], a, b, trunc, new_P[indexes], new_mu, muj)
    else:  # and_func
        new_sigma = compute_mom2_and(mu[indexes], sigma[indexes], a, b, new_P[indexes], new_mu, idx)
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


def prob(mu, sigma, a, b, idx=0):
    """
    Computes the mass probability of the normal distribution with mean mu and covariance matrix sigma in the 
    hyper-rectangle [a,b].
    Even for one-dimensional distributions, mu, sigma, a, b must be vectors.
    """ 
    
    norm = distributions.Normal(loc=mu[:,idx], scale=torch.sqrt(sigma[:,idx,idx]))
    P = norm.cdf(b[idx]) - norm.cdf(a[idx])
    return P


def compute_mom1(mu, sigma, a, b, trunc, P, idx=0):

    c = torch.zeros(mu.shape)
    norm = distributions.Normal(mu[:,idx], scale=torch.sqrt(sigma[:,idx,idx]))
    if trunc:
        if trunc == 'low':
            c[:,0] = norm.log_prob(a[0]).exp()
        elif trunc == 'up':
            c[:,0] = -norm.log_prob(b[0]).exp()
        return mu + torch.matmul(sigma, c.unsqueeze(2)).squeeze(2)/P.view(-1,1)
    else:   # for and_func
        c[:,idx] = norm.log_prob(a[idx]).exp() - norm.log_prob(b[idx]).exp()
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

# New functions to compute and/or truncations

def compute_mom2_and(mu, sigma, a, b, new_P, new_mu, idx):
    # vector dimensions
    n = len(a)    # number of variables
    c = mu.shape[0] # number of components
    # creates auxialiary vectors
    e0 = torch.zeros((c,n))
    e0[:,idx] = torch.ones(c)
    C = (new_P.view(c,1,1))*torch.eye(n).unsqueeze(0).expand(c,-1,-1)
    norm = distributions.Normal(loc=mu[:,idx], scale=torch.sqrt(sigma[:,idx,idx]))
    C[:, :, idx] += norm.log_prob(a[idx]).exp().view(-1,1)*(a[idx]**e0)*compute_muj(a, mu, sigma, idx) - norm.log_prob(b[idx]).exp().view(-1,1)*(b[idx]**e0)*compute_muj(b, mu, sigma, idx)
    # computes the new matrix
    new_sigma = new_P.view(c,1,1)*torch.matmul(mu.unsqueeze(2), new_mu.unsqueeze(1)) + torch.matmul(sigma, C.transpose(1,2))
    new_sigma = new_sigma/new_P.view(c,1,1) - torch.matmul(new_mu.unsqueeze(2), new_mu.unsqueeze(1))
    return new_sigma 

def compute_muj(a, mu, sigma, idx):
    # mask excluding index idx
    mask = torch.ones(len(a), dtype=torch.bool)
    mask[idx] = False
    
    mu_minusj = mu[:, mask]
    sigmaj = sigma[:, mask][:, :, idx]
    muj = mu_minusj + ((a[idx]-mu[:,idx])/sigma[:,idx,idx]).view(-1,1)*sigmaj

    new_muj = torch.ones(mu.shape)
    new_muj[:, :idx] = muj[:, :idx]
    new_muj[:, idx+1:] = muj[:, idx:]

    return new_muj



# versions that selects the indices to truncate (slower)
#def ineq_func(self, dist):
#    """ Invoked when any inequality expression in encountered """
#    
#    ineq_coeff = self.coeff
#    ineq_const = self.const
#    
#    # creates extended distribution
#    extended_gm = extend_dist(self, dist)
#        
#    #   here there was a part to deal with deltas, but we removed it because in torch everything is differentiable
#
#    # STEP 1: change variables
#    norm = torch.linalg.norm(ineq_coeff)
#    ineq_coeff = ineq_coeff/norm
#    ineq_const = ineq_const/norm
#    A = find_basis(ineq_coeff)           
#    transl_mu = torch.matmul(A, extended_gm.mu.unsqueeze(2)).squeeze(2)
#    transl_sigma = torch.matmul(torch.matmul(A, extended_gm.sigma), A.t())
#    transl_alpha = torch.zeros(len(ineq_coeff))
#    transl_alpha[0] = 1
#
#    # STEP 2: finds the indices of the components that needs to be transformed
#    indices = select_indices(transl_alpha, transl_sigma)
#
#    # STEP 3: creates reduced vectors taking into account only the coordinates that need to be transformed
#    red_transl_alpha = reduce_indices(transl_alpha, indices)
#    red_transl_mu = reduce_indices(transl_mu, indices)
#    red_transl_sigma = reduce_indices(transl_sigma, indices) 
#
#    # STEP 4: creates the hyper-rectangle to integrate on
#    a = torch.ones(len(red_transl_alpha))*(-INFTY)
#    b = torch.ones(len(red_transl_alpha))*(INFTY)
#    if self.type=='>' or self.type=='>=':
#        a[0] = ineq_const
#    if self.type=='<' or self.type=='<=':
#        b[0] = ineq_const   
#        
#    # STEP 5: compute moments in the transformed coordinates
#    # some components might have 0 prob in the truncation
#    # indexes contains the comp_idxs of the components that have non-zero probability
#    new_P, new_red_transl_mu, new_red_transl_sigma, comp_idxs = compute_moments(red_transl_mu, red_transl_sigma, a, b)
#
#    # if the whole distribution has zero prob in the truncation
#    if len(comp_idxs) == 0:
#        return torch.tensor(0.), dist
#
#    # STEP 6: recreates extended vectors
#    new_transl_mu = extend_indices(new_red_transl_mu, transl_mu, indices, comp_idxs)
#    new_transl_sigma = extend_indices(new_red_transl_sigma, transl_sigma, indices, comp_idxs)
#
#    # STEP 7: goes back to older coordinates
#    old_dim = len(dist.var_list)
#    A_inv = torch.linalg.inv(A)
#    new_mu = torch.matmul(A_inv, new_transl_mu.unsqueeze(2)).squeeze(2)[:, :old_dim]
#    new_sigma = torch.matmul( torch.matmul(A_inv, new_transl_sigma), A_inv.t())[:, :old_dim,:old_dim]
#    
#    # STEP 8: weights normalization
#    new_pi = extended_gm.pi[comp_idxs]*new_P.view(-1,1)
#    norm_fact, norm_new_pi = normalize_weights(new_pi)
#
#    return norm_fact, Dist(dist.var_list, GaussianMix(norm_new_pi, new_mu, new_sigma))

#def insert_value(val, idx, mu, sigma):
#    """ Extends mu and sigma by adding val in corresponding to the idx position (for sigma the other row- and column-entries are 0) """
#    d = len(mu)
#    new_mu = np.array(list(mu[:idx]) + [val] + list(mu[idx:]))
#    new_sigma = np.block([[sigma[:idx,:idx], np.zeros((idx,1)), sigma[:idx,idx:]], 
#          [np.zeros((1,d+1))],
#          [sigma[idx:,:idx], np.zeros((d-idx,1)), sigma[idx:,idx:]]])
#    return new_mu, new_sigma
#    

def select_indices(alpha, sigma):
    """ 
    Selects all indices i such that alpha[i] != 0 or j such that exists i such that sigma[i,j] != 0 and alpha[i] != 0 
    """
    queue = list(torch.where(alpha != 0)[0].numpy())
    coeff_set = []
    while len(queue) > 0:
        idx = queue.pop()
        coeff_set.append(idx)
        new_set = torch.where(sigma[:,idx] != 0)[1].numpy()
        for new_idx in set(new_set):
            if new_idx not in coeff_set:
                queue.append(new_idx)
    return list(set(coeff_set))


#def reduce_indices(vec, indices):
#    """
#    Extracts subvector/submatrix indexed by indices
#    """
#    if len(vec.shape) == 1:
#        red_vec = vec[indices]
#    if len(vec.shape) == 2:
#        red_vec = vec[:, indices]
#    if len(vec.shape) == 3:
#        red_vec = vec[:,indices][:,:,indices]
#    return red_vec


#def extend_indices(red_vec, old_vec, indices, comp_idxs):
#    """
#    Puts red_vec in the indices of old_vec, keeps only components indexed by comp_idxs
#    """
#    if len(old_vec.shape) == 2:
#        C = old_vec[comp_idxs, :]
#        C[:, indices] = red_vec
#        return C
#    if len(old_vec.shape) == 3:
#        C1 = old_vec[comp_idxs]
#        C2 = C1[:, indices] 
#        C2[:, :, indices] = red_vec
#        C1[:, indices] = C2
#        return C1

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