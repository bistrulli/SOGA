# Contains the function start_soga, which is used to invoke SOGA on a CFG object and the recursive function SOGA, which, depending on the type of the visited node, calls the functions needed to update the current distribution. 

#Such functions are contained in the auxiliary libraries:
# - libSOGAtruncate, containing functions for computing the resulting distribution when a truncation occurs (in conditional or observe instructions);
# - libSOGAupdate, containing functions for computing the resulting distribution after applying an assignment instruction;
# - libSOGAmerge, containing functions for computing the resulting distribution when a merge instruction is encountered;

# Additional functions for general purpose are defined in the library libSOGAshared, which is imported by all previous libraries.

# TO DO:
# - improve dependencies on libraries (all auxiliary libraries import libSOGAshared, maybe there is a more efficient way to do this?)
# - libSOGAmerge: add other pruning  

from libSOGAtruncate import *
from libSOGAupdate import *
from libSOGAmerge import *

#def copy_dist(dist):
#    new_dist = Dist(dist.var_list, GaussianMix([], [], []))
#    new_dist.gm.pi = [torch.clone(p) for p in dist.gm.pi]
#    new_dist.gm.mu = [torch.clone(m) for m in dist.gm.mu]
#    new_dist.gm.sigma = [torch.clone(s) for s in dist.gm.sigma]
#    return new_dist

def update_child(child, dist, p, trunc, exec_queue):
    if child.type == 'merge' or child.type == 'exit':
        child.list_dist.append((p, dist))
    else:
        child.set_dist(dist)                    # previously copy_dist(node.dist)
        child.set_p(p)
        child.set_trunc(trunc)
    if child not in exec_queue:
        exec_queue.append(child)

def start_SOGA(cfg, params_dict={}, pruning=None, Kmax=None, parallel=None,useR=False):
    """ Invokes SOGA on the root of the CFG object cfg, initializing current_distribution to a Dirac delta centered in zero.
        If pruning='classic' implements pruning at the merge nodes with maximum number of component Kmax.
        Returns an object Dist (defined in libSOGAshared) with the final computed distribution."""
    if(useR):
        initR()

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
        SOGA(exec_queue.pop(0), data, parallel, exec_queue, params_dict)
    
    # returns output distribution
    p, current_dist = merge(cfg.node_list['exit'].list_dist)
    cfg.node_list['exit'].list_dist = []
    return current_dist


def SOGA(node, data, parallel, exec_queue, params_dict):

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
            current_dist = update_rule(current_dist, node.expr, data, params_dict)         ### see libSOGAupdate

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

