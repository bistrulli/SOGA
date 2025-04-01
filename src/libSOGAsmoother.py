# The smoother executes a non-differentiable version on SOGA, to check which instructions 
# cause degenerate covariance matrices (i.e. non-differentiable distributions) and corrects those 
# instructions. Loops are checked only once.

from libSOGA import *
from libSOGAtruncate import *
from libSOGAupdate import *
from libSOGAmerge import *

# FUNCTIONS FOR CHECKING NON DEGENERACY

def check_dist_non_deg(dist):    
    """ Checks whether all cov matrices in dist are non-degenerate. Returns True if some are degenerate"""
    sigma = dist.gm.sigma
    if torch.any(torch.linalg.eigh(sigma)[0] < TOL_EIG):
        #deg_idx, _ = torch.where(torch.linalg.eigh(sigma)[0] < TOL_EIG)
        #print(dist.gm.pi[deg_idx[0]], dist.gm.sigma[deg_idx[0]])
        return True
    else:
        return False

# FUNCTIONS FOR STRING PARSING

def extract_var_and_index(expression):
    """
    Extracts the variable name and index from a string like 'data[i]'.
    """
    # Define the regular expression pattern
    pattern = r'(\w+)\[(\w+)\]'
    
    # Use re.match to find the variable name and index
    match = re.match(pattern, expression)
    
    if match:
        var_name = match.group(1)
        index = match.group(2)
        return var_name, index
    
def extract_variables(expression):
    """
    Estrae le variabili da un'espressione lineare.
    """
    # Pattern for finding variables
    pattern = r'\b[a-zA-Z_]\w*\([^)]*\)|\b[a-zA-Z_]\w*\[[^]]*\]|\b[a-zA-Z_]\w*\b'
    # Finding pattern occurences
    matches = re.findall(pattern, expression)
    # Filter numbers out
    variables = [match for match in matches if not re.match(r'^\d+$', match)]
    return variables


def extract_lists(gm_string):
    """ Estrae le liste di parametri da una stringa gm."""
    # Pattern for lists
    pattern = r'\[[^\]]*\]'
    # Finding pattern occurrences
    matches = re.findall(pattern, gm_string)
    return matches


def format_float_list(float_list):
    """ Formats a list of floats to avoid scientific notation. """
    formatted_list = '['
    for num in float_list:
        formatted_list += f'{float(num):.10f}, '
    formatted_list = formatted_list[:-2] +']'
    return formatted_list

# SMOOTHING FUNCTION

def smooth_asgmt_gm(var, expr, eps):
    """ Smooths var = 'gm([], [], [])' adding gaussian noise to components with 0 variance. """
    weights, mean, variances = extract_lists(var)
    variances = eval(variances)
    for i, variance in enumerate(variances):
        if variance == 0:
            variances[i] = eps
    variances = format_float_list(variances)
    expr = expr.replace(var, 'gm({}, {}, {})'.format(weights, mean, variances))
    return expr


def smooth_asgmt(dist, updated_dist, node, smoothed_vars, data, params_dict):
    """ dist is a distribution that when updated with node.expr gives the degenerate distribution updated_dist.
    This function adds to node.expr a Gaussian noise to avoid degeneracy and computes the new dist.
    The new expression for the update is saved in the attribute node.smooth"""
    
    current_EPS = SMOOTH_EPS
    new_expr = None
    smooth_flag = True
    
    if node.smooth:
        orig_expr = node.smooth
    else:
        orig_expr = node.expr

    lhs, rhs = orig_expr.replace(' ', '').split('=')
    # target variable and corresponding index
    target_var = lhs
    if '[' in target_var:
        var_name, idx = extract_var_and_index(target_var)
        idx = int(data[idx][0].item())
        target_idx = dist.var_list.index('{}[{}]'.format(var_name, idx))
    else:
        target_idx = dist.var_list.index(target_var)
    # variables used in the asgmt
    vars = extract_variables(rhs)

    while smooth_flag and current_EPS < 1.:

        # There are two cases in which the distributions can be degenerate
        # Observe that if CASE 1 is verified, after the smoothing CASE 2 cannot happen

        # CASE 1: for some components the distribution is degenerate
        # In this case the variable needs to be smoothed
        if torch.any(updated_dist.gm.sigma[:,target_idx,target_idx] == 0):
            
            smoothed_vars.append(target_var)
            gm_vars = [var for var in vars if 'gm(' in var]
            # if a random term is present at the rhs we smooth that
            if len(gm_vars) > 0:
                var = gm_vars[0]
                new_expr = smooth_asgmt_gm(var, orig_expr, current_EPS)
            # if no random term is present we add a Gaussian noise
            else:
                new_expr = orig_expr + '+ gm([1.], [0.], [{:.10f}])'.format(current_EPS)
        # CASE 2: one variable is deterministically determined by the others
        # in this case we add Gaussian noise, but we are not smoothing a variable, unless all variables from which it depends are smoothed
        else:
            new_expr = orig_expr + '+ gm([1.], [0.], [{:.10f}])'.format(current_EPS)
            if all(var in smoothed_vars for var in vars):
                smoothed_vars.append(target_var)
        
        # checks if with the new expression the distribution is smooth
        updated_dist = update_rule(dist, new_expr, data, params_dict)
        smooth_flag = check_dist_non_deg(updated_dist)
        if smooth_flag is True:
            current_EPS += SMOOTH_DELTA
    
    # saves the smoothed expression in the node
    node.smooth = new_expr

    # if too much noise needs to be added raises an error 
    if current_EPS >= 1.:
        print('Smoothing failed for node', node, orig_expr)
        print('Cannot smooth', updated_dist)
        raise ValueError

    return updated_dist


def smooth_trunc(trunc, node, smoothed_vars, data, params_dict):
    """ trunc is a truncation var ('==' | '!=') val.
    This functions transforms the conditions:
    var == val -> var > val - delta and var < val + delta
    var != val -> var < val - delta or var > val + delta
    where delta is selected by the function select_delta, depending on the current distribution"""
    
    if '==' in trunc:
        ops = '=='
    elif '!=' in trunc:
        ops = '!='
    elif '<=' in trunc:
        ops = '<='
    elif '<' in trunc:
        ops = '<'
    elif '>=' in trunc:
        ops = '>='
    elif '>' in trunc:
        ops = '>'
        
    target_var, target_val = trunc.split(ops)
    if not target_var.replace(' ', '') in smoothed_vars:
        return trunc 
    
    #if '[' in target_val:
    #    target_val, idx = extract_var_and_index(target_val)
    #    idx = int(data[idx][0].item())
    #    target_val = data[target_val][idx]
    #elif target_val[0] == '_':
    #    target_val = params_dict[target_val[1:]]
    #else:
    #    target_val = eval(target_val)
        
    dist = node.dist 
    delta = select_delta(dist, target_var)
    if ops == '==':
        new_trunc = '{} > {} - {:.10f} and {} < {} + {:.10f}'.format(target_var, target_val, delta, target_var, target_val, delta)
    elif ops == '!=':
        new_trunc = '{} < {} - {:.10f} or {} > {} + {:.10f}'.format(target_var, target_val, delta, target_var, target_val, delta)
    elif ops == '<=':
        new_trunc = '{} {} {} + {:.10f}'.format(target_var, ops, target_val, delta)
    elif ops == '<':
        new_trunc = '{} {} {} - {:.10f}'.format(target_var, ops, target_val, delta)
    elif ops == '>=':
        new_trunc = '{} {} {} - {:.10f}'.format(target_var, ops, target_val, delta)
    elif ops == '>':
        new_trunc = '{} {} {} + {:.10f}'.format(target_var, ops, target_val, delta)
    
    node.smooth = new_trunc
    
    return new_trunc    


def select_delta(dist, var):
    """ Selects the delta for the smooth_trunc function.
    It is based on the standard deviation of the variable var in the distribution dist."""
    var_idx = dist.var_list.index(var)
    # takes the smallest std
    std = torch.max(torch.sqrt(dist.gm.sigma[:,var_idx,var_idx]))
    return 5*std

# SOGA SMOOTH FUNCTIONS

def start_SOGA_smooth(cfg, params_dict={}, pruning=None, Kmax=None, parallel=None):
    """ Invokes SOGA on the root of the CFG object cfg, initializing current_distribution to a Dirac delta centered in zero.
        eps is a dictionary containing 'eps_asgmt' and 'eps_test', the smoothing coefficients for assignments and if branches."""

    # initializes current_dist
    var_list = cfg.ID_list
    data = cfg.data
    smoothed_vars = cfg.smoothed_vars
    n_dim = len(var_list)
    gm = GaussianMix(torch.tensor([[1.]]), torch.zeros((1,n_dim)), EPS*torch.eye(n_dim).reshape(1,n_dim, n_dim))
    init_dist = Dist(var_list, gm)
    cfg.root.set_dist(init_dist)
    
    # initializes visit queue
    exec_queue = [cfg.root]
    
    # executes SOGA on nodes on exec_queue
    while(len(exec_queue)>0):
        SOGAsmooth(exec_queue.pop(0), smoothed_vars, data, parallel, exec_queue, params_dict)
    
    # returns output distribution
    p, current_dist = merge(cfg.node_list['exit'].list_dist)
    cfg.node_list['exit'].list_dist = []
    return current_dist


def SOGAsmooth(node, smoothed_vars, data, parallel, exec_queue, params_dict):

    #print('Entering', node)
    #if node.dist:
    #    print(node.dist.gm.n_comp(), ' components')
    #    print(check_dist_non_deg(node.dist))
    #    print(node.dist)
    #    print('mean', node.dist.gm.mean())
    #    print('cov', node.dist.gm.cov())
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
        current_trunc = smooth_trunc(current_trunc, node, smoothed_vars, data, params_dict)
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
        #print('After truncation:')
        #print('current_p: ', current_p)
        #print('mean: ', current_dist.gm.mean())
        #print('cov: ', current_dist.gm.cov())
        #print('\n')
        if current_p > TOL_PROB:
            if node.smooth:
                updated_dist = update_rule(current_dist, node.smooth, data, params_dict)         ### see libSOGAupdate
            else:
                updated_dist = update_rule(current_dist, node.expr, data, params_dict)
            
            # smoothing
            smooth_flag = check_dist_non_deg(updated_dist)
            if smooth_flag:
                updated_dist = smooth_asgmt(current_dist, updated_dist, node, smoothed_vars, data, params_dict)
            current_dist = updated_dist
        #print('After update:')
        #print('mean: ', current_dist.gm.mean())
        #print('cov: ', current_dist.gm.cov())
        #print('\n')
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
        current_trunc = smooth_trunc(current_trunc, node, smoothed_vars, data, params_dict)
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
            #print('List_dist')
            #for p, dist in node.list_dist:
            #    print('p: ', p, 'mean: ', dist.gm.mean())
            #print('\n')
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
