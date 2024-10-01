# Contains the functions for computing the resulting distribution when an assignment instruction is encountered (in state nodes).

# SOGA (defined in SOGA.py)
# |- update_rule
#    |- sym_expr
#    |- update_gaussian

from libSOGAshared import *
from ASGMTListener import *
from ASGMTParser import * 
from ASGMTLexer import *

def mul_func(self, comp):
    i = self.target
    j, k = self.mul_idx
    mu = comp.gm.mu[0]
    sigma = comp.gm.sigma[0]
    final_pi = []
    final_mu = []
    final_sigma = []
    
    for part in product(*[range(len(mean)) for mean in self.aux_means]):
        # STEP 1: for a given combination of components of the auxiliary variables, creates a new component extending comp
        # creates the extended vector of moments (with auxiliary variables)
        aux_pi = 1
        aux_mean = torch.clone(mu)
        aux_sigma = torch.tensor([])
        for p,q in zip(range(len(self.aux_means)), part):
            aux_pi = aux_pi*self.aux_pis[p][q]
            aux_mean = torch.hstack([aux_mean, self.aux_means[p][q]])
            aux_sigma = torch.hstack([aux_sigma, self.aux_covs[p][q]])
        aux_sigma = torch.diag(aux_sigma)
        aux_cov = torch.vstack([torch.hstack([sigma, torch.zeros((len(sigma), len(aux_sigma)))]), torch.hstack([torch.zeros((len(aux_sigma), len(sigma))), aux_sigma])])
        # STEP 2: computes mean and covariance matrix for the extended component
        new_mu = torch.clone(mu)
        new_mu[i] = (aux_cov[j,k] + aux_mean[j]*aux_mean[k])
        new_sigma = torch.clone(sigma)
        new_sigma[i,:] = new_sigma[:,i] = (aux_mean[j]*aux_cov[k,:] + aux_mean[k]*aux_cov[j,:])[:len(mu)] 
        new_sigma[i,i] = (aux_cov[j,k]**2 + 2*aux_cov[j,k]*aux_mean[j]*aux_mean[k] + aux_cov[j,j]*aux_cov[k,k] + aux_cov[j,j]*aux_mean[k]**2 + aux_cov[k,k]*aux_mean[j]**2)
        # STEP 3: appends weight, new mean and new covariance matrix to the final vectors
        final_pi.append(aux_pi)
        final_mu.append(new_mu)
        final_sigma.append(new_sigma)
    return Dist(comp.var_list, GaussianMix(final_pi, final_mu, final_sigma))

def add_func(self, comp):
    
    i = self.target
    mu = comp.gm.mu[0]
    sigma = comp.gm.sigma[0]
    final_pi = []
    final_mu = []
    final_sigma = []
    
    for part in product(*[range(len(mean)) for mean in self.aux_means]):
        
        # STEP 1: for a given combination of components of the auxiliary variables, creates a new component extending comp
        aux_pi = 1
        aux_mean = torch.clone(mu)
        aux_sigma = torch.tensor([])
        for p,q in zip(range(len(self.aux_means)), part):
            aux_pi = aux_pi*self.aux_pis[p][q]
            aux_mean = torch.hstack([aux_mean, self.aux_means[p][q]])
            aux_sigma = torch.hstack([aux_sigma, self.aux_covs[p][q]])
        aux_sigma = torch.diag(aux_sigma)
        aux_cov = torch.vstack([torch.hstack([sigma, torch.zeros((len(sigma), len(aux_sigma)))]), torch.hstack([torch.zeros((len(aux_sigma), len(sigma))), aux_sigma])])
        
        # STEP 2: computes mean and covariance matrix for the extended component
        new_mu = torch.clone(mu)
        
        new_mu[i] = torch.dot(self.add_coeff, aux_mean) + self.add_const
        new_sigma = torch.clone(sigma)
        new_sigma[i,:] = new_sigma[:,i] = torch.mm(self.add_coeff.reshape((1, len(self.add_coeff))), aux_cov).flatten()[:len(mu)]
        new_sigma[i,i] = torch.mm(torch.mm(self.add_coeff.reshape((1, len(self.add_coeff))), aux_cov), self.add_coeff.reshape((len(self.add_coeff),1)))
        
        # STEP 3: appends weight, new mean and new covariance matrix to the final vectors
        final_pi.append(aux_pi)
        final_mu.append(new_mu)
        final_sigma.append(new_sigma)
    
    return Dist(comp.var_list, GaussianMix(final_pi, final_mu, final_sigma))


class AsgmtRule(ASGMTListener):
    
    def __init__(self, var_list, data):
        self.var_list = var_list
        self.data = data
        self.func = None           # stores the function
        self.target = None         # stores the index of the target variable
        
        self.aux_pis = []          # stores the weights of auxiliary variables
        self.aux_means = []        # stores the means of auxiliary variables
        self.aux_covs = []         # stores the cov matrices of auxiliary variables
        
        self.is_prod = None        # checks whether a term is a product of two vars
        
    def enterAssignment(self, ctx):
        self.target = self.var_list.index(ctx.symvars().getVar(self.data))
   
       
    def enterAdd(self, ctx):
        if len(ctx.add_term())==1 and len(ctx.add_term(0).term()) == 2:
            self.is_prod = 1
            for term in ctx.add_term(0).term():
                self.is_prod = self.is_prod*term.is_var(self.data)
        if self.is_prod:
            self.mul_idx = []
        else:
            self.add_coeff = torch.zeros(len(self.var_list))
            self.add_const = torch.tensor(0.)
    
    def enterAdd_term(self,ctx):
        # product between variables
        if self.is_prod:
            for term in ctx.term():
                if not term.gm() is None:
                    self.aux_pis.append(torch.tensor(eval(term.gm().list_()[0].getText())))
                    self.aux_means.append(torch.tensor(eval(term.gm().list_()[1].getText())))
                    self.aux_covs.append(torch.pow(torch.tensor(eval(term.gm().list_()[2].getText())),2))
                    self.mul_idx.append(int(len(self.var_list)+len(self.aux_pis)-1))
                elif not term.symvars() is None:
                    self.mul_idx.append(self.var_list.index(term.symvars().getVar(self.data)))
            self.func = partial(mul_func,self)
        # linear combination
        else:
            # collects the coefficients of the linear combination
            coeff = torch.tensor(1.)
            var_idx = None
            for term in ctx.term():
                if term.sub() is not None:
                    coeff = -1*coeff
                else:
                    coeff = 1*coeff
                if term.is_const(self.data):
                    coeff = coeff*term.getValue(self.data)
                elif not term.symvars() is None:
                    var_idx = self.var_list.index(term.symvars().getVar(self.data))
                elif not term.gm() is None:
                    self.aux_pis.append(torch.tensor(eval(term.gm().list_()[0].getText())))
                    self.aux_means.append(torch.tensor(eval(term.gm().list_()[1].getText())))
                    self.aux_covs.append(torch.pow(torch.tensor(eval(term.gm().list_()[2].getText())),2))
                    var_idx = len(self.add_coeff) + 1
            if not var_idx is None:
                if var_idx < len(self.add_coeff):
                    self.add_coeff[var_idx] = coeff
                else:
                    self.add_coeff = torch.hstack([self.add_coeff, coeff])
            else:
                self.add_const = self.add_const + coeff
                                
    def exitAdd(self, ctx):
        if not self.is_prod:
            if not torch.all(self.add_coeff == 0):
                self.func = partial(add_func, self)
            # here there was a part implementing constant assignment, but I removed it because everything must be differentiable
        
    
def asgmt_parse(var_list, expr, data):
    """ Parses expr using ANTLR4. Returns a function """
    lexer = ASGMTLexer(InputStream(expr))
    stream = CommonTokenStream(lexer)
    parser = ASGMTParser(stream)
    tree = parser.assignment()
    asgmt_rule = AsgmtRule(var_list, data)
    walker = ParseTreeWalker()
    walker.walk(asgmt_rule, tree) 
    return asgmt_rule.func
        
        
def update_rule(dist, expr, data):
    """ Applies expr to dist. It first parses expr using the function asgmt_parse, implemented as an ANTLR listener. asgmt_parse returns a function rule_func, such that, rule_func(GaussianMix) returns a new GaussianMix object obtained applying expr to the initial distribution. rule_func is applied to each component of dist, and the resulting Gaussian mixtures are stored in a single GaussianMix object."""
    
    if expr == 'skip':
        return dist
    else:
        rule_func = asgmt_parse(dist.var_list, expr, data)    # define function
        new_pi = []
        new_mu = []
        new_sigma = []
        for k in range(dist.gm.n_comp()):
            comp = Dist(dist.var_list, dist.gm.comp(k))
            new_mix = rule_func(comp)
            new_pi += list(dist.gm.pi[k]*np.array(new_mix.gm.pi))
            new_mu += new_mix.gm.mu
            new_sigma += new_mix.gm.sigma
        return Dist(dist.var_list, GaussianMix(new_pi, new_mu, new_sigma))
    
    


    
    
    