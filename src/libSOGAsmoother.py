# The smoother executes a non-differentiable version on SOGA, to check which instructions 
# cause degenerate covariance matrices (i.e. non-differentiable distributions) and corrects those 
# instructions. Loops are checked only once.

from libSOGA import *
from libSOGAtruncate import *
from libSOGAupdate import *
from libSOGAmerge import *

SMOOTH_EPS = 1e-2    # starting noise for smoothing
SMOOTH_DELTA = 1e-2  # addition to gaussian noise for smoothing
TOL_DEG = 1e-15  # matrices with determinants below this threshold are considered degenerate

def check_dist_non_deg(dist):    
    """ Checks whether all cov matrices in dist are non-degenerate."""
    sigma = dist.gm.sigma
    dets = torch.linalg.det(sigma)
    if torch.any(torch.abs(dets) < TOL_DEG):
        return True
    else:
        return False
    
def smooth_asgmt(dist, node, data, params_dict):
    """ dist is a distribution that when updated with node.expre is degenerate.
    This function adds to node.expr a Gaussian noise to avoid degeneracy and computes the new dist.
    The new expression for the update is saved in the attribute node.smooth"""
    current_EPS = SMOOTH_EPS
    new_expr = None
    smooth_flag = True
    while smooth_flag:
        new_expr = node.expr + '+ gm([1.], [0.], [{}])'.format(current_EPS)
        updated_dist = update_rule(dist, new_expr, data, params_dict)
        smooth_flag = check_dist_non_deg(updated_dist)
        if smooth_flag is True:
            current_EPS += SMOOTH_DELTA
    node.smooth = new_expr
    return updated_dist

def smooth_trunc(trunc, node):
    """ trunc is a truncation var ('==' | '!=') val.
    This functions transforms the conditions:
    var == val -> var > val - delta and var < val + delta
    var != val -> var < val - delta or var > val + delta
    where delta is selected by the function select_delta, depending on the current distribution"""
    if '==' in trunc:
        ops = '=='
    elif '!=' in trunc:
        ops = '!='
    target_var, target_val = trunc.split(ops)
    target_val = eval(target_val)
    dist = node.dist 
    delta = select_delta(dist, target_var, target_val)
    if ops == '==':
        new_trunc = '{} > {} and {} < {}'.format(target_var, target_val - delta, target_var, target_val + delta)
    elif ops == '!=':
        new_trunc = '{} < {} or {} > {}'.format(target_var, target_val - delta, target_var, target_val + delta)
    node.smooth = new_trunc
    return new_trunc    

def select_delta(dist, var, val):
    """ Selects the delta for the smooth_trunc function.
    It is based on the standard deviation of the variable var in the distribution dist."""
    var_idx = dist.var_list.index(var)
    # computes the component of the mixturewith mean closest to val
    marg_mean = dist.gm.mu[:,var_idx]
    diff = torch.abs(marg_mean - val)
    min_index = torch.argmin(diff)   # torch.argsort for ranked indices
    # selects the std of the component
    std = torch.sqrt(dist.gm.sigma[min_index,var_idx,var_idx])
    return 5*std

def start_SOGA_smooth(cfg, params_dict={}, pruning=None, Kmax=None, parallel=None):
    """ Invokes SOGA on the root of the CFG object cfg, initializing current_distribution to a Dirac delta centered in zero.
        eps is a dictionary containing 'eps_asgmt' and 'eps_test', the smoothing coefficients for assignments and if branches."""

    # initializes current_dist
    var_list = cfg.ID_list
    data = cfg.data
    n_dim = len(var_list)
    gm = GaussianMix(torch.tensor([[1.]]), torch.zeros((1,n_dim)), EPS*torch.eye(n_dim).reshape(1,n_dim, n_dim))
    init_dist = Dist(var_list, gm)
    cfg.root.set_dist(init_dist)
    
    # initializes visit queue
    exec_queue = [cfg.root]
    
    # executes SOGA on nodes on exec_queue
    while(len(exec_queue)>0):
        SOGAsmooth(exec_queue.pop(0), data, parallel, exec_queue, params_dict)
    
    # returns output distribution
    p, current_dist = merge(cfg.node_list['exit'].list_dist)
    cfg.node_list['exit'].list_dist = []
    return current_dist


def SOGAsmooth(node, data, parallel, exec_queue, params_dict):

    #print('Entering', node)
    #print(node.dist)
    #print('\n')

    if node.type != 'merge' and node.type != 'exit':
        current_dist = node.dist                  # previously copy_dist(node.dist)
        current_p = node.p
        current_trunc = node.trunc
        

    # starts execution
    if node.type == 'entry':
        update_child(node.children[0], node.dist, torch.tensor(1.), None, exec_queue)
            
    
    # if tests saves LBC and calls on children
    if node.type == 'test':
        current_trunc = node.LBC
        if '==' in current_trunc or '!=' in current_trunc:
            current_trunc = smooth_trunc(current_trunc, node)
        for child in node.children:
            update_child(child, node.dist, current_p, current_trunc, exec_queue)
            

    # if loop saves checks the condition and decides which child node must be accessed
    if node.type == 'loop':
        # the first time is accessed set the value of the counter to 0 and converts node.const into a number
        if data[node.idx][0] is None:
            data[node.idx][0] = torch.tensor(0.)
        if type(node.const) is str:
            if '[' in node.const:
                data_name, data_idx = node.const.split('[')
                data_idx = data_idx[:-1]
                # data_idx is a data
                if data_idx in data:
                    data_idx = int(data[data_idx][0])
                # data_idx is a number
                else:
                    data_idx = int(data_idx)
                node.const = torch.tensor(int(data[data_name][data_idx]))
            else:
                node.const = torch.tensor(int(node.const))            
        # successively checks the condition and decides which child node must be accessed
        if data[node.idx][0] < node.const:
            for child in node.children:
                if child.cond == True:
                    update_child(child, node.dist, current_p, current_trunc, exec_queue)
        else:
            data[node.idx][0] = None
            for child in node.children:
                if child.cond == False:
                    update_child(child, node.dist, current_p, current_trunc, exec_queue)
     

    # if state checks wheter cond!=None. If yes, truncates to current_trunc, eventually negating it. In any case applies the rule in expr. Appends the distribution in the next merge node or calls recursively on children. If child is loop node increments its idx.
    if node.type == 'state':
        if node.cond != None and not current_trunc is None:
            if node.cond == False:
                current_trunc = negate(current_trunc) 
            p, current_dist = truncate(current_dist, current_trunc, data, params_dict)     ### see libSOGAtruncate
            current_trunc = None
            current_p = p*current_p
        if current_p > TOL_PROB:
            updated_dist = update_rule(current_dist, node.expr, data, params_dict)         ### see libSOGAupdate
            
            # smoothing
            smooth_flag = check_dist_non_deg(updated_dist)
            if smooth_flag:
                updated_dist = smooth_asgmt(updated_dist, node, data, params_dict)
            current_dist = updated_dist

        # updating child
        child = node.children[0]
        if child.type == 'loop' and not data[child.idx][0] is None:
            data[child.idx][0] += 1
        update_child(child, current_dist, current_p, current_trunc, exec_queue)
        
            
    # if observe truncates to LBC and calls on children
    if node.type == 'observe':
        current_trunc = node.LBC
        #if parallel is not None and parallel >1:
        #    p, current_dist = parallel_truncate(current_dist, current_trunc, data,parallel)
        #else:
        p, current_dist = truncate(current_dist, current_trunc, data, params_dict)                     ### see libSOGAtruncate
        #current_p = current_p*p
        current_trunc = None
        child = node.children[0]
        update_child(child, current_dist, current_p, current_trunc, exec_queue)


    # if merge checks whether all paths have been explored.
    # Either returns or merge distributions and calls on children
    if node.type == 'merge':
        if len(node.list_dist) != len(node.parent):
            return
        else:
            current_p, current_dist = merge(node.list_dist)        ### see libSOGAmerge
            node.list_dist = []
            child = node.children[0]
            update_child(child, current_dist, current_p, None, exec_queue)
                
                
    if node.type == 'exit':
        return
    
    #if node.type == 'prune':
    #    current_dist = prune(current_dist,'classic',node.Kmax)        ### options: 'classic', 'ranking' (see libSOGAmerge)
    #    node.list_dist = []
    #    for child in node.children:
    #        if child.type == 'merge' or child.type == 'exit':
    #            child.list_dist.append((current_p, current_dist))
    #        else:
    #            child.set_dist(copy(current_dist))
    #            child.set_p(current_p)
    #            child.set_trunc(current_trunc)
    #        exec_queue.append(child)

## OLD STUFF

## These functions where written when trying to make a syntactic smoothing (no semantics information used)
## They are not currently used but might be useful in the future

#def add_smooth_gm(gm, eps=1e-2):
#    """ gm is a GmContext to be smoothed """
#    w_list = eval(gm.list_()[0].getText())
#    mean_list = eval(gm.list_()[1].getText())
#    std_list = eval(gm.list_()[2].getText())
#    for i in range(len(std_list)):
#        if std_list[i] == 0.:
#            std_list[i] = eps
#    return 'gm({}, {}, {})'.format(str(w_list), str(mean_list), str(std_list))

#def extract_coeff(add_term):
#    if add_term.NUM():
#        coeff = add_term.NUM().getText()
#    elif add_term.idd():
#        coeff = add_term.idd().getText()
#    elif add_term.par():
#        coeff = add_term.par().getText()
#    else:
#        coeff = None
#    return coeff

#def check_smooth_expr(expr, eps):
#    target_var = expr.symvars().getText()
#    # the assignment is constant
#    if expr.const():
#        new_expr = expr.getText() + '+ gm([1.], [0.], [{}])'.format(eps)
#    # the assignment is a linear expression with possible random terms
#    elif expr.add():
#        var_list = []
#        new_expr = str(target_var) + ' ='
#        ops = extract_operators(expr.add().getText())
#        for i, add_term in enumerate(expr.add().add_term()):
#            # sets the next operation
#            if new_expr[-1] == '=':
#                new_expr = new_expr + ' '
#            else:
#                new_expr = new_expr + ' {} '.format(ops[i-1])
#            if add_term.vars_():
#                # if the term is const*var
#                if add_term.vars_().symvars():
#                    var_list.append(add_term.vars_().symvars().getText())
#                    new_expr = new_expr + add_term.getText()
#                # random term
#                elif add_term.vars_().gm():
#                    gm = add_term.vars_().gm()
#                    coeff = extract_coeff(add_term)
#                    new_gm = add_smooth_gm(gm, eps)
#                    if coeff:
#                        new_expr = new_expr + coeff + '*' + new_gm
#                    else:
#                        new_expr = new_expr + new_gm
#            # constant term
#            elif add_term.const_term():
#                new_expr = new_expr + add_term.const_term().getText()
#        if target_var not in var_list and 'gm' not in new_expr:
#            new_expr = new_expr + ' + gm([1.], [0.], [{}])'.format(eps)       
#    else:        
#        new_expr = expr.getText()
#    return new_expr
#    
#def extract_subprogram(node, subprogram='', stack=[]):
#    """ Extracts the subprogram starting from a state node and returns it as a string"""
#    
#    print(node, subprogram, stack)
#
#    if node.type == 'entry':
#        child = node.children[0]
#        subprogram = extract_subprogram(child, subprogram, stack)
#
#    if node.type == 'state':
#        subprogram += node.expr + ';'
#        child = node.children[0]
#        subprogram = extract_subprogram(child, subprogram, stack)
#    
#    if node.type == 'test':
#        subprogram += 'if ' + node.LBC + ' {'
#        stack.append(node)
#        for child in node.children:
#            if child.cond is True:
#                subprogram = extract_subprogram(child, subprogram, stack)
#        
#    if node.type == 'merge':
#        if len(stack) > 0:
#            subprogram += ' } else {'
#            test = stack.pop()
#            for child in test.children:
#                if child.cond is False:
#                    subprogram = extract_subprogram(child, subprogram, stack)
#                    subprogram += ' } end if;'
#    
#    return subprogram


