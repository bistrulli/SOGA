# This module contains the functions needed to run the experiments in the notebook FinalExperiment.ipynb

# for SOGA estimations
from gvuln import *  
# for black-box estimations
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import mutual_info_regression 
import numpy as np

## DIFFERENTIAL PRIVACY

# Gaussian Mechanism

def run_gaussianDP_SOGA(eps_list,):

    Delta = 150/10
    delta = 1/100

    gvd_gauss = {}   # g-vulnerability for delta gain
    gvg_gauss = {}   # g-vulnerability for Gaussian gain
    entropy_gauss = {}       # entropy of the secret variable
    cond_entropy_gauss = {}  # conditional entropy of secret given output
    mi_gauss = {}            # mutual information between secret and output
    kl_gauss = {}        # KL divergence between secret given output value and secret 

    for eps in eps_list:

        # current value of epsilon
        #print('Eps: ', eps)
        eps = torch.tensor(eps)

        # associated std value
        pars = {'std': torch.sqrt(2*Delta**2*torch.log(torch.tensor(1.25/delta))/(eps))}
        params_dict = {}
        for key, value in pars.items():
            params_dict[key] = torch.tensor(value.item(), requires_grad=False)    
        #print('std:', 2*Delta**2*torch.log(torch.tensor(1.25/delta))/(eps**2))

        # computes the output for the model with no observations
        compiledFile=compile2SOGA('programs/DP_Gauss.soga')
        cfg = produce_cfg(compiledFile)
        output_dist = start_SOGA(cfg, params_dict=params_dict)
        marg_gauss = extract_marginal(output_dist, ['dataset[0]', 'avg'])
        s_marg_gauss = extract_marginal(output_dist, ['dataset[0]'])

        # computes exact quantities for the model with no observations
        #entropy_gauss['{}'.format(eps)] = entropy_gaussian(s_marg_gauss.gm.mu[0], s_marg_gauss.gm.sigma[0])
        cond_entropy_gauss['{}'.format(eps)] = cond_entropy_gaussian(marg_gauss.gm.mu[0], marg_gauss.gm.sigma[0], idx_o=[1])
        mi_gauss['{}'.format(eps)] = mi_gaussian(marg_gauss.gm.mu[0], marg_gauss.gm.sigma[0], idx_o=[1])
        gvd_gauss['{}'.format(eps)] =  vdelta_bivariate_gaussian(marg_gauss.gm.mu[0], marg_gauss.gm.sigma[0], idx_o=1).item()
        gvg_gauss['{}'.format(eps)] = vgauss_bivariate_gaussian(marg_gauss.gm.mu[0], marg_gauss.gm.sigma[0], eps=1, idx_o=1).item()

        # computes the output for the model with observations
        compiledFile=compile2SOGA('programs/DP_Gauss_obs.soga')
        cfg = produce_cfg(compiledFile)
        output_dist_obs = start_SOGA(cfg, params_dict=params_dict)
        s_marg_gauss_obs = extract_marginal(output_dist_obs, ['dataset[0]'])

        # computes exact quantities for the model with observations
        kl_gauss['{}'.format(eps)] = kl_div_gaussian(s_marg_gauss_obs.gm.mu[0], s_marg_gauss_obs.gm.sigma[0], s_marg_gauss.gm.mu[0], s_marg_gauss.gm.sigma[0])

    return gvd_gauss, gvg_gauss, entropy_gauss, cond_entropy_gauss, mi_gauss, kl_gauss

def print_gaussianDP(gvd_gauss, gvg_gauss, entropy_gauss, cond_entropy_gauss, mi_gauss, kl_gauss):
    
    print('GAUSSIAN MECHANISM \n')

    print('G-Vulnerability (Delta Gain)')
    for key in gvd_gauss.keys():
        print('\t Eps = {:.1f}, exact={:.6f}'.format(float(key), gvd_gauss[key]))

    print('\n G-Vulnerability (Gauss Gain)')
    for key in gvg_gauss.keys():
        print('\t Eps = {:.1f}, exact={:.6f}'.format(float(key), gvg_gauss[key]))

    print('\n Conditional Entropy   ')
    for key in cond_entropy_gauss.keys():
        print('\t Eps = {:.1f}, exact={:.6f}'.format(float(key), cond_entropy_gauss[key]))

    print('\n Mutual Information   ')
    for key in mi_gauss.keys():
        print('\t Eps = {:.1f}, exact={:.6f}'.format(float(key), mi_gauss[key]))

    print('\n KL Divergence   ')
    for key in kl_gauss.keys():
        print('\t Eps = {:.1f}, exact={:.6f}'.format(float(key), kl_gauss[key]))

# Black box estimations

def gaussian_dp_model(batch_size=10, std=5.):
    """ Numpy model of Gaussian DP mechanism to be used to generate samples for black-box methods"""
    dataset = distributions.Normal(40., 20.).sample([batch_size, 10])
    avg = torch.mean(dataset, dim=1, keepdim=True)
    v = distributions.Normal(0., std).sample([batch_size, 1])
    return dataset[:, 0], avg + v

def custom_loss(y_true, y_pred):
    """ Loss function to be used in g-vulnerability estimation """
    return np.mean(np.exp(-0.5 * (y_true - y_pred) ** 2))

def run_gaussianDP_BB(eps_list, n_train=1_000_000, n_test=100_000, n_repeats=5):
    """ 
    Performs black-box estimation using kNN regressor gor gvulnerability and the mutual_info_regression function from sklearn for mutual information.
    - n_samples: number of samples to use for the estimation of MI and for the training set in gvulnerability
    - n_test: number of samples to use for the test set in gvulnerability
    - n_repeats: number of times to repeat the experiment
    """

    Delta = 150/10
    delta = 1/100

    gvuln_bbox = {}
    mi_bbox = {}

    for eps in eps_list:

        # current eps and std
        eps = torch.tensor(eps)
        std = torch.sqrt(2*Delta**2*torch.log(torch.tensor(1.25/delta))/(eps))

        mi_bbox[eps.item()] = []
        gvuln_bbox[eps.item()] = []

        for i in range(n_repeats):

            # samples generation
            s_train, o_train = gaussian_dp_model(batch_size=n_train, std=std)
            s_test, o_test = gaussian_dp_model(batch_size=n_test, std=std)
            # Convert to numpy arrays if needed
            o_train_np = o_train.numpy() if hasattr(o_train, 'numpy') else np.array(o_train)
            s_train_np = s_train.numpy() if hasattr(s_train, 'numpy') else np.array(s_train)
            o_test_np = o_test.numpy() if hasattr(o_test, 'numpy') else np.array(o_test)
            s_test_np = s_test.numpy() if hasattr(s_test, 'numpy') else np.array(s_test)

            # g-vulnerability estimation
            knn = KNeighborsRegressor(n_neighbors=100)
            knn.fit(o_train_np.reshape(-1, 1), s_train_np)  # reshape if o_train is 1D
            s_pred = knn.predict(o_test_np.reshape(-1, 1))
            loss = custom_loss(s_test_np, s_pred)
            gvuln_bbox[eps.item()].append(loss)

            # mutual information estimation
            mi = mutual_info_regression(s_train.reshape(-1, 1), o_train.reshape(o_train.shape[0],))
            mi_bbox[eps.item()].append(mi[0])

    return gvuln_bbox, mi_bbox

def print_gaussianDP_BB(gvuln_bbox, mi_bbox):
    print('GAUSSIAN MECHANISM \n')
    for key in gvuln_bbox.keys():
        print('\t Eps = {:.1f}, G-Vuln = {:.6f} +- {:.6f}, Mutual Information = {:.6f} +- {:.6f}'.format(float(key), np.mean(gvuln_bbox[key]), np.std(gvuln_bbox[key]), np.mean(mi_bbox[key]), np.std(mi_bbox[key])))

# Geometric Mechanism 

geom_program = '''array[10] dataset;

                for i in range(10){{
                    dataset[i] = gm([0.5,0.5], [0., 1.], [0.0, 0.0]);
                }} end for;
                count = 0.0;

                for i in range(10){{
                    count = count + dataset[i];
                }} end for;

                geom1 = {};
                geom2 = {}; 
                noise = geom1 - geom2;

                count = count + noise;
                '''



def run_geomDP_SOGA(eps_list):

    gv_geom = {}
    entropy_geom = {}
    cond_entropy_geom = {}
    mi_geom = {}
    kl_geom = {}

    secret_var = 'dataset[0]'
    output_var = 'count'

    for eps in eps_list:

        # current value of epsilon
        #print('Eps: ', eps)
        eps = torch.tensor(eps)

        # associated p value
        Delta = torch.tensor(1.)
        p = 1 - torch.exp(-eps/Delta)
        gm_str = create_gm_from_p(p)
        #print('p:', p)

        # inserts the noise in the program
        formatted_program = geom_program.format(gm_str, gm_str)

        # computes output for program without observations
        compiledFile = compile2SOGA_text(formatted_program)
        cfg = produce_cfg_text(compiledFile)
        output_dist = start_SOGA(cfg)
        #output_dist = aggregate_mixture(output_dist)
        marg = extract_marginal(output_dist, ['dataset[0]'])
        marg = aggregate_mixture(marg)

        # computes output for program with observations
        program_obs = formatted_program + '\n observe(count == 2.);'
        compiledFile = compile2SOGA_text(program_obs)
        cfg = produce_cfg_text(compiledFile)
        output_dist_obs = start_SOGA(cfg)
        #output_dist_obs = aggregate_mixture(output_dist_obs)
        marg_obs = extract_marginal(output_dist_obs, ['dataset[0]'])
        marg_obs = aggregate_mixture(marg_obs)

        # Computes the exact metrics
        gv_geom['{}'.format(eps)] = discrete_V1(output_dist, secret_var, output_var)    
        cond_entropy_geom['{}'.format(eps)] = discrete_cond_entropy(output_dist, secret_var, output_var)
        mi_geom['{}'.format(eps)] = discrete_mi(output_dist, secret_var, output_var)
        kl_geom['{}'.format(eps)] = discrete_kl(marg_obs, marg, secret_var)
    
    return gv_geom, entropy_geom, cond_entropy_geom, mi_geom, kl_geom


def print_geomDP(gv_geom, entropy_geom, cond_entropy_geom, mi_geom, kl_geom):
    
    print('GEOMETRIC MECHANISM \n')

    print('G-Vulnerability  ')
    for key in gv_geom.keys():
        print('\t Eps = {:.1f},exact={:.6f}'.format(float(key), gv_geom[key]))


    print('\n Conditional Entropy   ')
    for key in cond_entropy_geom.keys():
        print('\t Eps = {:.1f}, exact = {:.6f}'.format(float(key), cond_entropy_geom[key]))

    print('\n Mutual Information   ')
    for key in mi_geom.keys():
        print('\t Eps = {:.1f}, exact = {:.6f}'.format(float(key), mi_geom[key]))

    print('\n KL Divergence   ')
    for key in kl_geom.keys():
        print('\t Eps = {:.1f}, exact = {:.6f}'.format(float(key), kl_geom[key]))

# LAPLACIAN MECHANISM (DIRECT)


def run_laplace_direct_DP_SOGA(eps_list):

    Delta = 150./10.

    gvd_lap_lb = {}
    gvd_lap_ub = {}
    gvg_lap_lb = {}
    gvg_lap_ub = {}

    gvd_lap = {}
    gvg_lap = {}

    cond_entropy_lap_lb = {}
    cond_entropy_lap_ub = {}
    cond_entropy_lap = {}

    mi_lap_lb = {}
    mi_lap_ub = {}
    mi_lap = {}

    kl_lap_lb = {}
    kl_lap_ub = {}
    kl_lap = {}

    n_samples = 100_000
    #sampling_time = 0.
    #bounds_time = 0.

    for eps in eps_list:

        # current value of epsilon and scale
        eps = torch.tensor(eps)
        #print('n Eps: ', eps)
        b = Delta/eps
        #print('b:', b)

        # find the parameters of 2 components mixture Gaussian to approximate Laplace(0,b)
        pi, mu1, mu2, sigma1, sigma2, _ = fit_laplace(torch.tensor(0.), b)
        pars = {'p1': pi,
                'p2': 1-pi,
                'mean1': mu1,
                'mean2': mu2,
                'sigma1': sigma1,
                'sigma2': sigma2}
        params_dict = {}
        for key, value in pars.items():
            params_dict[key] = torch.tensor(value, requires_grad=True)    

        # Computes SOGA output without observation
        compiledFile=compile2SOGA('programs/DP_Laplace.soga')
        cfg = produce_cfg(compiledFile)
        output_dist = start_SOGA(cfg, params_dict=params_dict)
        marg_lap = extract_marginal(output_dist, ['dataset[0]', 'avg'])
        marg_lap = aggregate_mixture(marg_lap)
        s_marg = extract_marginal(output_dist, ['dataset[0]'])
        s_marg = aggregate_mixture(s_marg)

        # Computes SOGA output with observation
        compiledFile=compile2SOGA('programs/DP_Laplace_obs.soga')
        cfg = produce_cfg(compiledFile)
        output_dist_obs = start_SOGA(cfg, params_dict=params_dict)
        s_marg_obs = extract_marginal(output_dist_obs, ['dataset[0]'])
        s_marg_obs = aggregate_mixture(s_marg_obs)

        # Computes bounds
        #start = time()
        # g vulnerability
        gvd_lap_lb['{}'.format(eps)] = vdelta_lower_bound(marg_lap, idx_o=1)
        gvd_lap_ub['{}'.format(eps)] = vdelta_upper_bound(marg_lap, idx_o=1)
        gvg_lap_lb['{}'.format(eps)] = vgauss_lower_bound(marg_lap, 1., idx_o=1)
        gvg_lap_ub['{}'.format(eps)] = vgauss_upper_bound(marg_lap, 1., idx_o=1)
        # conditional entropy
        cond_entropy_lap_lb['{}'.format(eps)] = cond_entropy_lb(marg_lap, idx_o=[1])
        cond_entropy_lap_ub['{}'.format(eps)] = cond_entropy_ub(marg_lap, idx_o=[1])
        # mutual information
        mi_lap_lb['{}'.format(eps)] = mi_lb(marg_lap, idx_o=[1])
        mi_lap_ub['{}'.format(eps)] = mi_ub(marg_lap, idx_o=[1])
        # KL divergence
        kl_lap_lb['{}'.format(eps)] = kl_div_lb(s_marg_obs, s_marg)
        kl_lap_ub['{}'.format(eps)] = kl_div_ub(s_marg_obs, s_marg)
        #end = time()
        #bounds_time += end - start

        # Computes exact values via MCMC
        #start = time()
        # g vulnerability
        gvd_lap['{}'.format(eps)] = sample_gvuln(output_dist, n_samples, gain_type='delta', secret_var='dataset[0]', output_var='avg')
        gvg_lap['{}'.format(eps)] = sample_gvuln(output_dist, n_samples, gain_type='gauss', secret_var='dataset[0]', output_var='avg')
        # conditional entropy
        cond_entropy_lap['{}'.format(eps)] = sample_cond_entropy(marg_lap, idx_o=[1], n_samples=n_samples)
        # mutual information
        mi_lap['{}'.format(eps)] = sample_mi(marg_lap, idx_o=[1], n_samples=n_samples)
        # KL divergence
        kl_lap['{}'.format(eps)] = sample_kl(s_marg_obs, s_marg, n_samples)
        #end = time()
        #sampling_time += end - start

    # organizes results in lists
    gvd = [gvd_lap_lb, gvd_lap, gvd_lap_ub]
    gvg = [gvg_lap_lb, gvg_lap, gvg_lap_ub]
    cond_entropy = [cond_entropy_lap_lb, cond_entropy_lap, cond_entropy_lap_ub]
    mi = [mi_lap_lb, mi_lap, mi_lap_ub]
    kl = [kl_lap_lb, kl_lap, kl_lap_ub]

    return gvd, gvg, cond_entropy, mi, kl

# LAPLACIAN MECHANISM (IID LOOP)

laplacian_noise_program = '''
    /* First exponential (approximated via i.i.d. loop semantics) */
    w1 = uniform([0, {}], 5);
    u1 = uniform([0, 1], 2);
    expw1 = 0 - w1;
    expw1 = _par*expw1;
    expw1 = exp(expw1);
    observe(u1 - expw1 <= 0);
    /* Second exponential (approximated via i.i.d. loop semantics) */
    w2 = uniform([0, {}], 5);
    u2 = uniform([0, 1], 2);
    expw2 = 0 - w2;
    expw2 = _par*expw2;
    expw2 = exp(expw2);
    observe(u2 - expw2 <= 0);
    /* Transforms into Laplace */
    r = w1 - w2;
    '''

laplace_program = '''array[10] dataset;
    for i in range(10){{
        dataset[i] = gauss(40, 20);
    }} end for;
    v = {};
    avg = 0.0;
    for i in range(10){{
        avg = avg + 0.1*dataset[i];
    }} end for;
    avg = avg + v;
    '''

def run_laplace_iid_DP_SOGA(eps_list, M_list):
    """ M_list is the list of the support of the uniform in which I need to sample for the rejection sampling. """
    
    Delta = 150./10.

    gvd_lap_loop_lb = {}
    gvd_lap_loop_ub = {}
    gvg_lap_loop_lb = {}
    gvg_lap_loop_ub = {}
    gvd_lap_loop = {}
    gvg_lap_loop = {}

    cond_entropy_lap_loop_lb = {}
    cond_entropy_lap_loop_ub = {}
    cond_entropy_lap_loop = {}

    mi_lap_loop_lb = {}
    mi_lap_loop_ub = {}
    mi_lap_loop = {}

    kl_lap_loop_lb = {}
    kl_lap_loop_ub = {}
    kl_lap_loop = {}

    #sampling_time = 0.
    #bounds_time = 0.
    n_samples = 100_000

    for i in range(len(eps_list)):

        # current epsilon and scale
        eps = torch.tensor(eps_list[i])
        #print('\n Eps: ', eps)
        b = Delta/eps
        #print('b:', b)

        # Approximating the Laplace
        current_noise = laplacian_noise_program.format(M_list[i], M_list[i])
        compiledFile=compile2SOGA_text(current_noise)
        cfg = produce_cfg_text(compiledFile)
        params_dict = {'par': torch.tensor(1/b.detach().numpy(), requires_grad=False)}
        output_dist = start_SOGA(cfg, params_dict=params_dict)
        noise_dist = extract_marginal(output_dist, ['r'])
        noise_dist = aggregate_mixture(noise_dist)
        pruned_noise = ranking_prune(noise_dist, 10)  # pruning noise to avoid too many components
        #print('Noise mean:', pruned_noise.gm.mean(), 'Noise scale:', torch.sqrt(pruned_noise.gm.cov()))
        gm_str = from_gm_to_string(pruned_noise.gm)

        # adds the noise distribution to the program without observations
        formatted_program = laplace_program.format(gm_str)
        compiledFile=compile2SOGA_text(formatted_program)
        cfg = produce_cfg_text(compiledFile)

        # creates a second program with observations
        program_obs = formatted_program + '\n observe(avg == 38);'
        compiledFile=compile2SOGA_text(program_obs)
        cfg_obs = produce_cfg_text(compiledFile)

        #  Computes outputs without obs
        output_dist = start_SOGA(cfg)
        marg_lap = extract_marginal(output_dist, ['dataset[0]', 'avg'])
        marg_lap = aggregate_mixture(marg_lap)
        #print('marg_lap mean:', marg_lap.gm.mean(), 'marg_lap scale:', torch.sqrt(marg_lap.gm.cov()))
        s_marg = extract_marginal(output_dist, ['dataset[0]'])
        s_marg = aggregate_mixture(s_marg)

        # Computes outputs with obs
        output_dist_obs = start_SOGA(cfg_obs)
        s_marg_obs = extract_marginal(output_dist_obs, ['dataset[0]'])
        s_marg_obs = aggregate_mixture(s_marg_obs)

        # Computes bounds
        #start = time()
        # g vulnerability
        gvd_lap_loop_lb['{}'.format(eps)] = vdelta_lower_bound(marg_lap, idx_o=1)
        gvd_lap_loop_ub['{}'.format(eps)] = vdelta_upper_bound(marg_lap, idx_o=1)
        gvg_lap_loop_lb['{}'.format(eps)] = vgauss_lower_bound(marg_lap, 1., idx_o=1)
        gvg_lap_loop_ub['{}'.format(eps)] = vgauss_upper_bound(marg_lap, 1., idx_o=1)
        # conditional entropy   
        cond_entropy_lap_loop_lb['{}'.format(eps)] = cond_entropy_lb(marg_lap, idx_o=[1])
        cond_entropy_lap_loop_ub['{}'.format(eps)] = cond_entropy_ub(marg_lap, idx_o=[1])
        # mutual information
        mi_lap_loop_lb['{}'.format(eps)] = mi_lb(marg_lap, idx_o=[1])
        mi_lap_loop_ub['{}'.format(eps)] = mi_ub(marg_lap, idx_o=[1])
        # KL divergence
        kl_lap_loop_lb['{}'.format(eps)] = kl_div_lb(s_marg_obs, s_marg)
        kl_lap_loop_ub['{}'.format(eps)] = kl_div_ub(s_marg_obs, s_marg)
        #end = time()
        #bounds_time += end - start

        # Computes exact values via MCMC
        #start = time()
        # g vulnerability
        gvd_lap_loop['{}'.format(eps)] = sample_gvuln(output_dist, n_samples, gain_type='delta', secret_var='dataset[0]', output_var='avg')
        gvg_lap_loop['{}'.format(eps)] = sample_gvuln(output_dist, n_samples, gain_type='gauss', secret_var='dataset[0]', output_var='avg')
        # conditional entropy
        cond_entropy_lap_loop['{}'.format(eps)] = sample_cond_entropy(marg_lap, idx_o=[1], n_samples=n_samples)
        # mutual information
        mi_lap_loop['{}'.format(eps)] = sample_mi(marg_lap, idx_o=[1], n_samples=n_samples)
        # KL divergence
        kl_lap_loop['{}'.format(eps)] = sample_kl(s_marg_obs, s_marg, n_samples)
        #end = time()
        #sampling_time += end - start

    # organizes results in lists
    gvd = [gvd_lap_loop_lb, gvd_lap_loop, gvd_lap_loop_ub]
    gvg = [gvg_lap_loop_lb, gvg_lap_loop, gvg_lap_loop_ub]
    cond_entropy = [cond_entropy_lap_loop_lb, cond_entropy_lap_loop, cond_entropy_lap_loop_ub]
    mi = [mi_lap_loop_lb, mi_lap_loop, mi_lap_loop]
    kl = [kl_lap_loop_lb, kl_lap_loop, kl_lap_loop_ub]

    return gvd, gvg, cond_entropy, mi, kl

def print_laplaceDP(gvd, gvg, cond_entropy, mi, kl):
    
    print('LAPLACIAN MECHANISM (DIRECT ENCODING) \n')
    print('G-Vulnerability (Delta Gain)')
    for key in gvd[0].keys():
        print('\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), gvd[0][key], gvd[1][key], gvd[2][key]))

    print('\nG-Vulnerability (Gaussian Gain)')
    for key in gvg[0].keys():
        print('\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), gvg[0][key], gvg[1][key], gvg[2][key]))

    print('\n Conditional Entropy   ')
    for key in cond_entropy[0].keys():
        print('\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), cond_entropy[0][key], cond_entropy[1][key], cond_entropy[2][key]))

    print('\n Mutual Information   ')
    for key in mi[0].keys():
        print('\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), mi[0][key], mi[1][key], mi[2][key]))

    print('\n KL Divergence   ')
    for key in kl[0].keys():
        print('\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), kl[0][key], kl[1][key], kl[2][key]))

## Black-box estimation

def laplacian_dp_model(batch_size=10, b=1.):
    """ Numpy model of Laplacian DP mechanism to be used to generate samples for black-box methods"""
    dataset = distributions.Normal(40., 20.).sample([batch_size, 10])
    avg = torch.mean(dataset, dim=1, keepdim=True)
    v = distributions.Laplace(0., b).sample([batch_size, 1])
    return dataset[:, 0], avg + v

def run_laplacianDP_BB(eps_list, n_train=1_000_000, n_test=100_000, n_repeats=5):
    """ 
    Performs black-box estimation using kNN regressor gor gvulnerability and the mutual_info_regression function from sklearn for mutual information.
    - n_samples: number of samples to use for the estimation of MI and for the training set in gvulnerability
    - n_test: number of samples to use for the test set in gvulnerability
    - n_repeats: number of times to repeat the experiment
    """

    Delta = 150/10

    gvuln_bbox = {}
    mi_bbox = {}

    for eps in eps_list:

        # current eps and scale
        eps = torch.tensor(eps)
        b = Delta/eps

        mi_bbox[eps.item()] = []
        gvuln_bbox[eps.item()] = []

        for i in range(n_repeats):

            # samples generation
            s_train, o_train = laplacian_dp_model(batch_size=n_train, b=b)
            s_test, o_test = laplacian_dp_model(batch_size=n_test, b=b)
            # Convert to numpy arrays if needed
            o_train_np = o_train.numpy() if hasattr(o_train, 'numpy') else np.array(o_train)
            s_train_np = s_train.numpy() if hasattr(s_train, 'numpy') else np.array(s_train)
            o_test_np = o_test.numpy() if hasattr(o_test, 'numpy') else np.array(o_test)
            s_test_np = s_test.numpy() if hasattr(s_test, 'numpy') else np.array(s_test)

            # g-vulnerability estimation
            knn = KNeighborsRegressor(n_neighbors=100)
            knn.fit(o_train_np.reshape(-1, 1), s_train_np)  # reshape if o_train is 1D
            s_pred = knn.predict(o_test_np.reshape(-1, 1))
            loss = custom_loss(s_test_np, s_pred)
            gvuln_bbox[eps.item()].append(loss)

            # mutual information estimation
            mi = mutual_info_regression(s_train.reshape(-1, 1), o_train.reshape(o_train.shape[0],))
            mi_bbox[eps.item()].append(mi[0])

    return gvuln_bbox, mi_bbox

def print_laplacianDP_BB(lap_gvuln_bbox, lap_mi_bbox):
    print('LAPLACIAN MECHANISM \n')
    for key in lap_gvuln_bbox.keys():
        print('\t Eps = {:.1f}, G-Vuln = {:.6f} +- {:.6f}, Mutual Information = {:.6f} +- {:.6f}'.format(float(key), np.mean(lap_gvuln_bbox[key]), np.std(lap_gvuln_bbox[key]), np.mean(lap_mi_bbox[key]), np.std(lap_mi_bbox[key])))

# GEO-INDISTINGUISHABILITY (DIRECT)

def run_geo_direct_SOGA(eps_list):

    vdelta_noloop_lb = {}
    vdelta_noloop_ub = {}
    vgauss_noloop_lb = {}
    vgauss_noloop_ub = {}
    vdelta_noloop = {}
    vgauss_noloop = {}

    cond_ent_noloop_lb = {}
    cond_ent_noloop_ub = {}
    cond_ent_noloop = {}
    cond_entx_noloop_lb = {}
    cond_entx_noloop_ub = {}
    cond_entx_noloop = {}
    cond_enty_noloop_lb = {}
    cond_enty_noloop_ub = {}
    cond_enty_noloop = {}   

    mi_noloop_lb = {}
    mi_noloop_ub = {}
    mi_noloop = {}
    mix_noloop_lb = {}
    mix_noloop_ub = {}
    mix_noloop = {}
    miy_noloop_lb = {}
    miy_noloop_ub = {}
    miy_noloop = {}

    kl_noloop_lb = {}
    kl_noloop_ub = {}
    kl_noloop = {}
    klx_noloop_lb = {}
    klx_noloop_ub = {}
    klx_noloop = {}
    kly_noloop_lb = {}
    kly_noloop_ub = {}
    kly_noloop = {}

    n_samples = 100_000
    #sampling_time = 0.
    #bounds_time = 0.

    for eps in eps_list:

        # current eps
        eps = torch.tensor(eps)
        #print('\n Eps: ', eps)

        # Computing parameters of a 2-components Gaussian mixture to approximate Gamma(2, 1/eps)    
        mean = torch.tensor(2.)
        scale = 1/eps
        _, pi1, pi2, mu1, mu2, sigma1, sigma2 = fit_gamma(mean, scale)
        pars = {'p1': pi1,
                'p2': pi2,
                'mu1': mu1,
                'mu2': mu2,
                'sigma1': sigma1,
                'sigma2': sigma2
                }
        params_dict = {}
        for key, value in pars.items():
            params_dict[key] = torch.tensor(value.item(), requires_grad=True)    

        # Computing SOGA output without observations
        compiledFile=compile2SOGA('programs/Geo_ind.soga')
        cfg = produce_cfg(compiledFile)
        output_dist = start_SOGA(cfg, params_dict=params_dict)
        marg_dist = extract_marginal(output_dist, ['x', 'y', 'xAnon', 'yAnon'])
        marg_dist = aggregate_mixture(marg_dist)
        s_marg = extract_marginal(output_dist, ['x', 'y'])
        s_marg = aggregate_mixture(s_marg)
        marg_x_dist = extract_marginal(output_dist, ['x', 'xAnon'])
        marg_y_dist = extract_marginal(output_dist, ['y', 'yAnon'])
        marg_x_dist = aggregate_mixture(marg_x_dist)
        marg_y_dist = aggregate_mixture(marg_y_dist)
        x_marg = extract_marginal(output_dist, ['x'])
        y_marg = extract_marginal(output_dist, ['y'])
        x_marg = aggregate_mixture(x_marg)
        y_marg = aggregate_mixture(y_marg)

        # Computing SOGA output with observations
        compiledFile=compile2SOGA('programs/Geo_ind_obs.soga')
        cfg = produce_cfg(compiledFile)
        output_dist = start_SOGA(cfg, params_dict=params_dict)
        s_marg_obs = extract_marginal(output_dist, ['x', 'y'])
        s_marg_obs = aggregate_mixture(s_marg_obs)
        x_marg_obs = extract_marginal(output_dist, ['x'])
        y_marg_obs = extract_marginal(output_dist, ['y'])
        x_marg_obs = aggregate_mixture(x_marg_obs)
        y_marg_obs = aggregate_mixture(y_marg_obs)

        # Computing bounds
        #start = time()
        # g vulnerability
        vdelta_noloop_lb['{}'.format(eps)] = vdelta_lower_bound(marg_dist, idx_o=[2,3], mvariate=True)
        vdelta_noloop_ub['{}'.format(eps)] = vdelta_upper_bound(marg_dist, idx_o=[2,3], mvariate=True)
        vgauss_noloop_lb['{}'.format(eps)] = vgauss_lower_bound(marg_dist, idx_o=[2,3], eps=1., mvariate=True)
        vgauss_noloop_ub['{}'.format(eps)] = vgauss_upper_bound(marg_dist, idx_o=[2,3], eps=1., mvariate=True)
        # conditional entropy
        cond_ent_noloop_lb['{}'.format(eps)] = cond_entropy_lb(marg_dist, idx_o=[2,3])
        cond_ent_noloop_ub['{}'.format(eps)] = cond_entropy_ub(marg_dist, idx_o=[2,3])
        cond_entx_noloop_lb['{}'.format(eps)] = cond_entropy_lb(marg_x_dist, idx_o=[1])
        cond_entx_noloop_ub['{}'.format(eps)] = cond_entropy_ub(marg_x_dist, idx_o=[1])
        cond_enty_noloop_lb['{}'.format(eps)] = cond_entropy_lb(marg_y_dist, idx_o=[1])
        cond_enty_noloop_ub['{}'.format(eps)] = cond_entropy_ub(marg_y_dist, idx_o=[1])
        # mutual information
        mi_noloop_lb['{}'.format(eps)] = mi_lb(marg_dist, idx_o=[2,3])
        mi_noloop_ub['{}'.format(eps)] = mi_ub(marg_dist, idx_o=[2,3])
        mix_noloop_lb['{}'.format(eps)] = mi_lb(marg_x_dist, idx_o=[1])
        mix_noloop_ub['{}'.format(eps)] = mi_ub(marg_x_dist, idx_o=[1])
        miy_noloop_lb['{}'.format(eps)] = mi_lb(marg_y_dist, idx_o=[1])
        miy_noloop_ub['{}'.format(eps)] = mi_ub(marg_y_dist, idx_o=[1])
        # KL divergence
        kl_noloop_lb['{}'.format(eps)] = kl_div_lb(s_marg_obs, s_marg)
        kl_noloop_ub['{}'.format(eps)] = kl_div_ub(s_marg_obs, s_marg)
        klx_noloop_lb['{}'.format(eps)] = kl_div_lb(x_marg_obs, x_marg)
        klx_noloop_ub['{}'.format(eps)] = kl_div_ub(x_marg_obs, x_marg)
        kly_noloop_lb['{}'.format(eps)] = kl_div_lb(y_marg_obs, y_marg)
        kly_noloop_ub['{}'.format(eps)] = kl_div_ub(y_marg_obs, y_marg)
        #end = time()
        #print('Eps {} took {:.2f} seconds to compute bounds'.format(eps, end-start))
        #bounds_time += end - start

        # Computing exact values with MCMC
        #start = time()
        # g vulnerability
        vdelta_noloop['{}'.format(eps)] = sample_gvuln_multivariate(marg_dist, n_samples=1000, secret_var=['x', 'y'], output_var=['xAnon', 'yAnon'], gain_type='delta', lr=0.001)
        vgauss_noloop['{}'.format(eps)] = sample_gvuln_multivariate(marg_dist, n_samples=1000, secret_var=['x', 'y'], output_var=['xAnon', 'yAnon'], gain_type='gauss', lr=0.001)
        # conditional entropy
        cond_ent_noloop['{}'.format(eps)] = sample_cond_entropy(marg_dist, idx_o=[2,3], n_samples=n_samples)
        cond_entx_noloop['{}'.format(eps)] = sample_cond_entropy(marg_x_dist, idx_o=[1], n_samples=n_samples)
        cond_enty_noloop['{}'.format(eps)] = sample_cond_entropy(marg_y_dist, idx_o=[1], n_samples=n_samples)
        # mutual information
        mi_noloop['{}'.format(eps)] = sample_mi(marg_dist, idx_o=[2,3], n_samples=n_samples)
        mix_noloop['{}'.format(eps)] = sample_mi(marg_x_dist, idx_o=[1], n_samples=n_samples)
        miy_noloop['{}'.format(eps)] = sample_mi(marg_y_dist, idx_o=[1], n_samples=n_samples)
        # KL divergence
        kl_noloop['{}'.format(eps)] = sample_kl(s_marg_obs, s_marg, n_samples=n_samples)
        klx_noloop['{}'.format(eps)] = sample_kl(x_marg_obs, x_marg, n_samples=n_samples)
        kly_noloop['{}'.format(eps)] = sample_kl(y_marg_obs, y_marg, n_samples=n_samples)
        #end = time()
        #print('Eps {} took {:.2f} seconds to sample'.format(eps, end-start))
        #sampling_time += end - start

    # organizes results in dictionaries
    vdelta = {'(x,y)': [vdelta_noloop_lb, vdelta_noloop, vdelta_noloop_ub]}
    vgauss = {'(x,y)': [vgauss_noloop_lb, vgauss_noloop, vgauss_noloop_ub]}
    cond_entropy = {'(x,y)': [cond_ent_noloop_lb, cond_ent_noloop, cond_ent_noloop_ub],
                    'x': [cond_entx_noloop_lb, cond_entx_noloop, cond_entx_noloop_ub],
                    'y': [cond_enty_noloop_lb, cond_enty_noloop, cond_enty_noloop_ub]}
    mi = {'(x,y)': [mi_noloop_lb, mi_noloop, mi_noloop_ub],
            'x': [mix_noloop_lb, mix_noloop, mix_noloop_ub],
            'y': [miy_noloop_lb, miy_noloop, miy_noloop_ub]}
    kl = {'(x,y)': [kl_noloop_lb, kl_noloop, kl_noloop_ub],
            'x': [klx_noloop_lb, klx_noloop, klx_noloop_ub],
            'y': [kly_noloop_lb, kly_noloop, kly_noloop_ub]}    
    
    return vdelta, vgauss, cond_entropy, mi, kl


# GEO-INDISTINGUISHABILITY (IID LOOP)

noise_program = '''
    /* First exponential (approximated via i.i.d. loop semantics) */
    w1 = uniform([0,10], 5);
    u1 = uniform([0, 1], 2);
    expw1 = 0 - w1;
    expw1 = exp(expw1);
    observe(u1 - expw1 <= 0);   
    /* Second exponential (approximated via i.i.d. loop semantics) */
    w2 = uniform([0,10], 5);
    u2 = uniform([0, 1], 2);
    expw2 = 0 - w2;
    expw2 = exp(expw2);
    observe(u2 - expw2 <= 0);
    /* Transforms into Gamma(2,_rate) */  
    r = w1 + w2;
    r = _rate*r;
    '''

geo_program = '''
    x=gauss(0., 10.);
    y=gauss(0., 20.);

    r = {};

    theta = uniform([0, 6.28], 2);

    cosTheta = cos(theta);
    xAnon = r*cosTheta;
    xAnon = x + xAnon;

    sinTheta = sin(theta);
    yAnon = r*sinTheta;
    yAnon = y + yAnon;
    '''

def run_geo_iid_SOGA(eps_list):
    
    # g vulnerability
    vdelta_rej_lb = {}
    vdelta_rej_ub = {}
    vgauss_rej_lb = {}
    vgauss_rej_ub = {}
    vdelta_rej = {}
    vgauss_rej = {}
    # conditional entropy
    cond_ent_rej_lb = {}
    cond_ent_rej_ub = {}
    cond_ent_rej = {}
    cond_entx_rej_lb = {}
    cond_entx_rej_ub = {}
    cond_entx_rej = {}
    cond_enty_rej_lb = {}
    cond_enty_rej_ub = {}
    cond_enty_rej = {}
    # mutual information
    mi_rej_lb = {}
    mi_rej_ub = {}
    mi_rej = {}
    mix_rej_lb = {}
    mix_rej_ub = {}
    mix_rej = {}
    miy_rej_lb = {}
    miy_rej_ub = {}
    miy_rej = {}
    # KL divergence
    kl_rej_lb = {}
    kl_rej_ub = {}
    kl_rej = {}
    klx_rej_lb = {}
    klx_rej_ub = {}
    klx_rej = {}
    kly_rej_lb = {}
    kly_rej_ub = {}
    kly_rej = {}

    n_samples = 10_000
    #bounds_time = 0.
    #sampling_time = 0.

    for eps in eps_list:

        # current value of epsilon
        eps = torch.tensor(eps)
        #print('\n Eps: ', eps)

        # creates the noise distribution
        compiledFile = compile2SOGA_text(noise_program)
        cfg = produce_cfg_text(compiledFile)
        noise_dist = start_SOGA(cfg, params_dict={'rate': 1/eps})
        noise_marg = extract_marginal(noise_dist, ['r'])
        noise_marg = aggregate_mixture(noise_marg)
        # prunes the noise distribution to avoid too many components
        pruned_noise_marg = Dist(noise_marg.var_list, GaussianMix(torch.clone(noise_marg.gm.pi), torch.clone(noise_marg.gm.mu), torch.clone(noise_marg.gm.sigma)))
        pruned_noise_marg = ranking_prune(pruned_noise_marg, 10)
        gm_str = from_gm_to_string(pruned_noise_marg.gm)

        # adds the noise distribution to the program
        formatted_program = geo_program.format(gm_str)
        compiledFile=compile2SOGA_text(formatted_program)
        cfg = produce_cfg_text(compiledFile)

        # creates a second program with observations
        program_obs = formatted_program + '\n observe(xAnon==10.);\n observe(yAnon==10.);'
        compiledFile=compile2SOGA_text(program_obs)
        cfg_obs = produce_cfg_text(compiledFile)

        # computes the output of the program without observations
        output_dist = start_SOGA(cfg, params_dict={})
        # marginal over (x, y)
        marg_dist = extract_marginal(output_dist, ['x', 'y', 'xAnon', 'yAnon'])
        marg_dist = aggregate_mixture(marg_dist)    
        s_marg = extract_marginal(marg_dist, ['x', 'y'])
        s_marg = aggregate_mixture(s_marg)
        # marginals over x and y
        x_marg_dist = extract_marginal(marg_dist, ['x', 'xAnon'])
        y_marg_dist = extract_marginal(marg_dist, ['y', 'yAnon'])
        x_marg_dist = aggregate_mixture(x_marg_dist)   
        y_marg_dist = aggregate_mixture(y_marg_dist)
        x_dist = extract_marginal(x_marg_dist, ['x'])
        y_dist = extract_marginal(y_marg_dist, ['y'])
        x_dist = aggregate_mixture(x_dist)
        y_dist = aggregate_mixture(y_dist)

        # output of the program with observations
        output_dist_obs = start_SOGA(cfg_obs, params_dict={})
        # marginal over (x, y)
        s_marg_obs = extract_marginal(output_dist_obs, ['x', 'y'])
        s_marg_obs = aggregate_mixture(s_marg_obs)
        # marginal over x and y
        x_marg_obs = extract_marginal(output_dist_obs, ['x'])
        y_marg_obs = extract_marginal(output_dist_obs, ['y'])
        x_marg_obs = aggregate_mixture(x_marg_obs)
        y_marg_obs = aggregate_mixture(y_marg_obs)

        # Computes bounds
        #start = time()
        # bounds for g-vulnerability
        vdelta_rej_lb['{}'.format(eps)] = vdelta_lower_bound(marg_dist, idx_o=[2,3], mvariate=True)
        vdelta_rej_ub['{}'.format(eps)] = vdelta_upper_bound(marg_dist, idx_o=[2,3], mvariate=True)
        vgauss_rej_lb['{}'.format(eps)] = vgauss_lower_bound(marg_dist, idx_o=[2,3], eps=1., mvariate=True)
        vgauss_rej_ub['{}'.format(eps)] = vgauss_upper_bound(marg_dist, idx_o=[2,3], eps=1., mvariate=True)
        # bounds for conditional entropy
        cond_ent_rej_lb['{}'.format(eps)] = cond_entropy_lb(marg_dist, idx_o=[2,3])
        cond_ent_rej_ub['{}'.format(eps)] = cond_entropy_ub(marg_dist, idx_o=[2,3])
        cond_entx_rej_lb['{}'.format(eps)] = cond_entropy_lb(x_marg_dist, idx_o=[1])
        cond_entx_rej_ub['{}'.format(eps)] = cond_entropy_ub(x_marg_dist, idx_o=[1])
        cond_enty_rej_lb['{}'.format(eps)] = cond_entropy_lb(y_marg_dist, idx_o=[1])
        cond_enty_rej_ub['{}'.format(eps)] = cond_entropy_ub(y_marg_dist, idx_o=[1])
        # bounds for mutual information
        mi_rej_lb['{}'.format(eps)] = mi_lb(marg_dist, idx_o=[2,3])
        mi_rej_ub['{}'.format(eps)] = mi_ub(marg_dist, idx_o=[2,3])
        mix_rej_lb['{}'.format(eps)] = mi_lb(x_marg_dist, idx_o=[1])
        mix_rej_ub['{}'.format(eps)] = mi_ub(x_marg_dist, idx_o=[1])
        miy_rej_lb['{}'.format(eps)] = mi_lb(y_marg_dist, idx_o=[1])
        miy_rej_ub['{}'.format(eps)] = mi_ub(y_marg_dist, idx_o=[1])
        # bounds for KL divergence
        kl_rej_lb['{}'.format(eps)] = kl_div_lb(s_marg_obs, s_marg)
        kl_rej_ub['{}'.format(eps)] = kl_div_ub(s_marg_obs, s_marg)
        klx_rej_lb['{}'.format(eps)] = kl_div_lb(x_marg_obs, x_dist)
        klx_rej_ub['{}'.format(eps)] = kl_div_ub(x_marg_obs, x_dist)
        kly_rej_lb['{}'.format(eps)] = kl_div_lb(y_marg_obs, y_dist)
        kly_rej_ub['{}'.format(eps)] = kl_div_ub(y_marg_obs, y_dist)
        #    end = time()
        #print('Eps = {} took {} seconds to compute bounds'.format(eps, end-start))
        #bounds_time += end - start

        # Computes values with MCMC
        #start = time()
        # g-vulnerability
        vdelta_rej['{}'.format(eps)] = sample_gvuln_multivariate(marg_dist, n_samples=100, secret_var=['x', 'y'], output_var=['xAnon', 'yAnon'], gain_type='delta', num_steps=10, min_samples=5, lr=0.001)
        vgauss_rej['{}'.format(eps)] = sample_gvuln_multivariate(marg_dist, n_samples=100, secret_var=['x', 'y'], output_var=['xAnon', 'yAnon'], gain_type='gauss', num_steps=10, min_samples=5, lr=0.001)
        # conditional entropy
        cond_ent_rej['{}'.format(eps)] = sample_cond_entropy(marg_dist, idx_o=[2,3], n_samples=n_samples)
        cond_entx_rej['{}'.format(eps)] = sample_cond_entropy(x_marg_dist, idx_o=[1], n_samples=n_samples)
        cond_enty_rej['{}'.format(eps)] = sample_cond_entropy(y_marg_dist, idx_o=[1], n_samples=n_samples)
        # mutual information
        mi_rej['{}'.format(eps)] = sample_mi(marg_dist, idx_o=[2,3], n_samples=n_samples)
        mix_rej['{}'.format(eps)] = sample_mi(x_marg_dist, idx_o=[1], n_samples=n_samples)
        miy_rej['{}'.format(eps)] = sample_mi(y_marg_dist, idx_o=[1], n_samples=n_samples)
        # KL divergence
        kl_rej['{}'.format(eps)] = sample_kl(s_marg_obs, s_marg, n_samples=n_samples)
        klx_rej['{}'.format(eps)] = sample_kl(x_marg_obs, x_dist, n_samples=n_samples)
        kly_rej['{}'.format(eps)] = sample_kl(y_marg_obs, y_dist, n_samples=n_samples)
        #end = time()
        #print('Eps = {} took {} seconds to compute sampling'.format(eps, end-start))
        #sampling_time += end - start

    # organizes results in dictionaries
    vdelta = {'(x,y)': [vdelta_rej_lb, vdelta_rej, vdelta_rej_ub]}
    vgauss = {'(x,y)': [vgauss_rej_lb, vgauss_rej, vgauss_rej_ub]}
    cond_entropy = {'(x,y)': [cond_ent_rej_lb, cond_ent_rej, cond_ent_rej_ub], 
                    'x': [cond_entx_rej_lb, cond_entx_rej, cond_entx_rej_ub], 
                    'y': [cond_enty_rej_lb, cond_enty_rej, cond_enty_rej_ub]}
    mi = {'(x,y)': [mi_rej_lb, mi_rej, mi_rej_ub],
            'x': [mix_rej_lb, mix_rej, mix_rej_ub],
            'y': [miy_rej_lb, miy_rej, miy_rej_ub]}
    kl = {'(x,y)': [kl_rej_lb, kl_rej, kl_rej_ub],
            'x': [klx_rej_lb, klx_rej, klx_rej_ub],
            'y': [kly_rej_lb, kly_rej, kly_rej_ub]}

    return vdelta, vgauss, cond_entropy, mi, kl

def print_geo(vdelta, vgauss, cond_entropy, mi, kl):

    print('GEO-INDISTINGUISHABILITY \n')

    print('G-Vulnerability (Delta Gain)')
    vdelta_joint = vdelta['(x,y)']
    for key in vdelta_joint[0].keys():
        print('\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), vdelta_joint[0][key], vdelta_joint[1][key], vdelta_joint[2][key]))

    print('\nG-Vulnerability (Gaussian Gain)')
    vgauss_joint = vgauss['(x,y)']
    for key in vgauss_joint[0].keys():
        print('\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), vgauss_joint[0][key], vgauss_joint[1][key], vgauss_joint[2][key]))

    print('\n Conditional Entropy   ')
    print('\t Joint Conditional Entropy')
    cond_entropy_joint = cond_entropy['(x,y)']
    for key in cond_entropy_joint[0].keys():
        print('\t\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), cond_entropy_joint[0][key], cond_entropy_joint[1][key], cond_entropy_joint[2][key]))
    print('\tMarginal Conditional Entropy X')
    cond_entropy_x = cond_entropy['x']
    for key in cond_entropy_x[0].keys():
        print('\t\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), cond_entropy_x[0][key], cond_entropy_x[1][key], cond_entropy_x[2][key]))
    print('\tMarginal Conditional Entropy Y')
    cond_entropy_y = cond_entropy['y']
    for key in cond_entropy_y[0].keys():
        print('\t\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), cond_entropy_y[0][key], cond_entropy_y[1][key], cond_entropy_y[2][key]))

    print('\n Mutual Information   ')
    print('\t Joint Mutual Information')
    mi_joint = mi['(x,y)']
    for key in mi_joint[0].keys():
        print('\t\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), mi_joint[0][key], mi_joint[1][key], mi_joint[2][key]))
    print('\tMarginal Mutual Information X')
    mi_x = mi['x']
    for key in mi_x[0].keys():
        print('\t\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), mi_x[0][key], mi_x[1][key], mi_x[2][key]))
    print('\tMarginal Mutual Information Y')
    mi_y = mi['y']
    for key in mi_y[0].keys():
        print('\t\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), mi_y[0][key], mi_y[1][key], mi_y[2][key]))

    print('\n KL Divergence   ')
    print('\t Joint KL Divergence')
    kl_joint = kl['(x,y)']
    for key in kl_joint[0].keys():
        print('\t\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), kl_joint[0][key], kl_joint[1][key], kl_joint[2][key]))
    print('\tMarginal KL Divergence X')
    kl_x = kl['x']
    for key in kl_x[0].keys():
        print('\t\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), kl_x[0][key], kl_x[1][key], kl_x[2][key]))
    print('\tMarginal KL Divergence Y')
    kl_y = kl['y']
    for key in kl_y[0].keys():
        print('\t\t Eps = {:.1f}, lb={:.6f},exact={:.6f}, ub={:.6f}'.format(float(key), kl_y[0][key], kl_y[1][key], kl_y[2][key]))

# Black-box estimation

def geo_model(batch_size=10, eps=0.1):
    """ Numpy model of Geo-indistinguishability mechanism to be used to generate samples for black-box methods"""
    x = distributions.Normal(0., 10.).sample([batch_size, 1])
    y = distributions.Normal(0., 20.).sample([batch_size, 1])
    r = distributions.Gamma(2., eps).sample([batch_size, 1])
    theta = distributions.Uniform(0., 2*torch.pi).sample([batch_size, 1])
    x_anon = r*torch.cos(theta) + x
    y_anon = r*torch.sin(theta) + y
    return x, y, x_anon, y_anon

def geo_loss(y_true, y_pred):
    """ Generalized loss for multivariate outputs"""
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    return torch.mean(torch.exp(-0.5 * torch.sum((y_true - y_pred) ** 2, dim=1)))

def run_geo_BB(eps_list, n_train=1_000_000, n_test=100_000, n_repeats=5):

    gvuln_bbox = {}
    mix_bbox = {}
    miy_bbox = {}

    for eps in eps_list:

        eps = torch.tensor(eps)

        gvuln_bbox[eps.item()] = []
        mix_bbox[eps.item()] = []
        miy_bbox[eps.item()] = []

        for i in range(n_repeats):

            # Generates samples
            x, y, x_anon, y_anon = geo_model(batch_size=n_train, eps=eps)
            x_train = x.numpy().flatten()
            y_train = y.numpy().flatten()
            x_anon_train = x_anon.numpy().flatten()
            y_anon_train = y_anon.numpy().flatten()
            # stacks data for gvuln estimation
            output_train = torch.hstack([x, y])
            input_train = torch.hstack([x_anon, y_anon])
            x, y, x_anon, y_anon = geo_model(batch_size=n_test, eps=eps)
            output_test = torch.hstack([x, y])
            input_test = torch.hstack([x_anon, y_anon])

            # g-vulnerability estimation
            knn = KNeighborsRegressor(n_neighbors=100)
            knn.fit(output_train, input_train)  # reshape if o_train is 1D
            input_pred = knn.predict(output_test)
            loss = geo_loss(input_test, input_pred)
            gvuln_bbox[eps.item()].append(loss.item())

            # mutual information estimation
            mi_x = mutual_info_regression(x_train.reshape(-1, 1), x_anon_train)
            mi_y = mutual_info_regression(y_train.reshape(-1, 1), y_anon_train)
            mix_bbox[eps.item()].append(mi_x[0])
            miy_bbox[eps.item()].append(mi_y[0])


    return gvuln_bbox, mix_bbox, miy_bbox

def print_geo_BB(geo_gvuln_bbox, geo_mix_bbox, geo_miy_bbo):
    print('GEO-INDISTINGUISHABILITY (BLACK-BOX ESTIMATION) \n')
    for key in geo_gvuln_bbox.keys():
        print('\t Eps = {:.1f}, G-Vuln = {:.6f} +- {:.6f}, Mutual Information X = {:.6f} +- {:.6f}, Mutual Information Y = {:.6f} +- {:.6f}'.format(float(key), np.mean(geo_gvuln_bbox[key]), np.std(geo_gvuln_bbox[key]), np.mean(geo_mix_bbox[key]), np.std(geo_mix_bbox[key]), np.mean(geo_miy_bbo[key]), np.std(geo_miy_bbo[key])))