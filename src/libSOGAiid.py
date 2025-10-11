from libSOGAtruncate import *
from libSOGAupdate import *
from libSOGAmerge import *
from libSOGAshared import *

TOL_IID = 1e-5 # tolerance for checking i.i.d.-ness of loops

def check_iid(dist, trunc, true_child, data, params_dict):
    """
    Check i.i.d.-ness of non-nested while loop with respect to distribution dist.
    The loop has guard trunc and the cfg of its body is rooted in true_child.
    """
    init_dist = dist
    # initializes the subroot
    true_child.dist = init_dist
    true_child.trunc = trunc
    # select the indices for which i.i.d.-ness must be checked
    # first selects the variables in the truncation condition
    trunc_rule = trunc_parse(dist.var_list, trunc, data, params_dict)
    trunc_coeffs = trunc_rule.coeff
    trunc_vars = select_indices(trunc_coeffs, init_dist.gm.sigma)
    # computes the final distribution after the body of the loop, 
    # and collects the variables appearing there
    body_vars = []
    final_dist, body_vars = start_sub_SOGA(true_child, init_dist, data, params_dict, body_vars)
    # puts together trunc_vars and body_vars
    all_vars = list(set(trunc_vars + body_vars))
    # checks whether the final distribution is identical to the initial one
    flag = check_equal_dist(init_dist, final_dist, all_vars)
    return flag 

def check_equal_dist(dist1, dist2, indices):
    """
    Check whether two distributions are identical.
    """
    # extracts marginals to be checked
    marg_vars = [dist1.var_list[idx] for idx in indices]
    marg1 = extract_marginal(dist1, marg_vars)
    marg2 = extract_marginal(dist2, marg_vars)
    marg1 = aggregate_mixture(marg1)
    marg2 = aggregate_mixture(marg2)
    pi1, idx1 = marg1.gm.pi.flatten().sort()
    pi2, idx2 = marg2.gm.pi.flatten().sort()
    mu1 = marg1.gm.mu[idx1]
    mu2 = marg2.gm.mu[idx2]
    sigma1 = marg1.gm.sigma[idx1]
    sigma2 = marg2.gm.sigma[idx2]
    if len(pi1) != len(pi2):
        print('Different number of components:', len(pi1), len(pi2))
        return False
    else:
        # adapting the tolerance to the dimension of the problem 
        tol = (marg1.gm.n_dim()**2)*TOL_IID
        #print('Using tolerance:', tol)
        #print('check on pi:', torch.all(torch.abs(pi1-pi2) < TOL_IID), torch.sum(torch.abs(pi1-pi2)))
        #print('check on mu:', torch.all( torch.abs(mu1-mu2) < TOL_IID ), torch.sum(torch.abs(mu1-mu2)))
        #print('check on sigma:', torch.all( torch.abs(sigma1-sigma2) < TOL_IID ), torch.sum(torch.abs(sigma1-sigma2)))
        return torch.all(torch.abs(pi1-pi2) < tol) and torch.all( torch.abs(mu1-mu2) < tol) and torch.all( torch.abs(sigma1-sigma2) < tol)

def start_sub_SOGA(node, dist, data, params_dict, body_vars):
    """
    Starts a sub-SOGA execution from node with distribution current_dist.
    Returns the distribution at the end of the first occurence of a while node
    """
    node.dist = dist

    # initializes visit queue
    exec_queue = [node]
    
    # executes SOGA on nodes on exec_queue
    while(len(exec_queue)>0):
        sub_SOGA(exec_queue.pop(0), exec_queue, data, params_dict, body_vars)
    # returns output distribution
    while_node = node.parent[0]
    if while_node.type != 'while':
        raise RuntimeError
    p = while_node.p
    final_dist = while_node.dist
    return final_dist, body_vars
    
def sub_SOGA(node, exec_queue, data, params_dict, body_vars):

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
        # collects the variables appeating in the condition
        trunc_rule = trunc_parse(current_dist.var_list, current_trunc, data, params_dict)
        indices = torch.where(trunc_rule.coeff != 0)[0]
        body_vars.append(idx.item() for idx in indices)
        #keeps executing the loop
        if '==' in current_trunc or '!=' in current_trunc:
            print('Degeneracy in if condition detected. Please smooth the program.')
            raise RuntimeError
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
     
    if node.type == 'while':
        return

    # if state checks wheter cond!=None. If yes, truncates to current_trunc, eventually negating it. In any case applies the rule in expr. Appends the distribution in the next merge node or calls recursively on children. If child is loop node increments its idx.
    if node.type == 'state':
        if node.cond != None and not current_trunc is None:
            if node.cond == False:
                current_trunc = negate(current_trunc) 
            # if parallel is not None and parallel >1:
            #     p, current_dist = parallel_truncate(current_dist, current_trunc, data, parallel)   
            # else:
            p, current_dist = truncate(current_dist, current_trunc, data, params_dict)     ### see libSOGAtruncate
            current_trunc = None
            current_p = p*current_p
        if current_p > TOL_PROB:
            # stores the variable appearing in the update in body_vars
            if node.expr != 'skip':
                expr_rule = asgmt_parse(current_dist.var_list, node.expr, data, params_dict)
                body_vars.append(expr_rule.target)
            # applies the update rule
            current_dist = update_rule(current_dist, node.expr, data, params_dict)         ### see libSOGAupdate

        # updating child
        child = node.children[0]
        if child.type == 'loop' and not data[child.idx][0] is None:
            data[child.idx][0] += 1
        update_child(child, current_dist, current_p, current_trunc, exec_queue)
        
            
    # if observe truncates to LBC and calls on children
    if node.type == 'observe':
        current_trunc = node.LBC
        # collects the variables appeating in the condition
        trunc_rule = trunc_parse(current_dist.var_list, current_trunc, data, params_dict)
        indices = torch.where(trunc_rule.coeff != 0)[0]
        for index in indices:
            body_vars.append(index.item())
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