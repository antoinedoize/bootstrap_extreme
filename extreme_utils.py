import numpy as np
from scipy.stats import t, invweibull, genextreme, pareto
import matplotlib.pyplot as plt
import json
import random as random
from random import choices
from tqdm import tqdm
from scipy.special import gamma as gamma_func
from numpy.random import normal as normal_gauss


### Simulation variables de Student

# df = 10
# r = t.rvs(df, size=10000)
# plt.hist(r, density=True, bins=100, log=False)

# fig, ax = plt.subplots(1, 1)
# mean, var, skew, kurt = t.stats(df, moments='mvsk')
# x = np.linspace(t.ppf(0.01, df),
#                 t.ppf(0.99, df), 100)
# ax.plot(x, t.pdf(x, df),
#        'r-', lw=5, alpha=0.6, label='t pdf')

# rv = t(df)
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

# vals = t.ppf([0.001, 0.5, 0.999], df)
# np.allclose([0.001, 0.5, 0.999], t.cdf(vals, df))

# ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
# ax.legend(loc='best', frameon=False)
# plt.show()

def get_student_sample(degrees_freedom, size_sample, get_plot=False, log_plot=False, nb_bins=100):
    # Le nombre de degrés de liberté correspond au nombre de paramètres
    # Si degrees_freedom < 1 : pas d'espérance
    r = t.rvs(degrees_freedom, size=size_sample)
    if get_plot:
        plt.hist(r, density=True, bins=nb_bins, log=log_plot, label="student_samples", alpha = 0.8)
        plt.legend()
    return r


### Simulation type II EVD (Frechet distribution)


def get_frechet_sample(inv_gamma, size_sample, get_plot=False, log_plot=False, nb_bins=100):
    # Warning : takes in input inverse of gamma (gamma being the tail index)
    # The smaller inv_gamma, the heavier the tail
    sample = invweibull.rvs(inv_gamma, size=size_sample)
    if get_plot:
        plt.hist(sample, density=True, bins=nb_bins, log=log_plot, label="frechet_samples", alpha = 0.6)
        plt.legend()
    return sample


def get_ma_sample(inv_gamma, size_sample, order, distrib="frechet"):
    if distrib == "frechet":
        hidden_sample = get_frechet_sample(inv_gamma, size_sample+order)
    if distrib == "student":
        hidden_sample = get_student_sample(inv_gamma, size_sample+order)
    # TODO: adapt to other types of distributions such as weibull ?
    ma_sample = list()
    for i in range(size_sample):
        ma_sample.append(np.mean(hidden_sample[i:i+order]))
    return ma_sample


def get_dependant_frechet_max(inv_gamma, size_sample, order, distrib="frechet"):
    if distrib == "frechet":
        hidden_sample = get_frechet_sample(inv_gamma, size_sample+order)
    if distrib == "student":
        hidden_sample = get_student_sample(inv_gamma, size_sample+order)
    if distrib == "pareto":
        hidden_sample = pareto.rvs(inv_gamma, size=size_sample+order)
    # TODO: adapt to other types of distributions such as weibull ?
    max_sample = list()
    for i in range(size_sample):
        max_sample.append(np.max(hidden_sample[i:i+order]/order))
    return max_sample


def get_block_max(inv_gamma, size_sample, block_size, order_ma,
                  slide="disjoint",distrib="frechet",dependency="ma"):
    if slide == "disjoint":
        slide = block_size
    nb_hidden_sample = size_sample*slide + block_size
    if dependency == "ma":
        hidden_sample = get_ma_sample(inv_gamma, nb_hidden_sample, 
                                    order_ma, distrib="frechet")
    if dependency == "max":
        hidden_sample = get_dependant_frechet_max(inv_gamma, nb_hidden_sample, 
                                                  order_ma, distrib="frechet")
    block_max_sample = list()
    for i in range(0,nb_hidden_sample-block_size,slide):
        block = hidden_sample[i:i+block_size]
        max_block = np.max(block)
        block_max_sample.append(max_block)
    return block_max_sample


def get_armax_bucher_segers_2020(inv_gamma,size_sample,beta=0.5,burn_in=200):
    hidden_sample = get_frechet_sample(inv_gamma, size_sample+burn_in)
    current_value = hidden_sample[burn_in]
    armax_sample = [current_value]
    for next_val in hidden_sample[burn_in+1:]:
        if beta*current_value > (1-beta)*next_val:
            current_value = beta*current_value # Current value
        else:
            current_value = (1-beta)*next_val
        armax_sample.append(current_value)
    return armax_sample


def arch_coeff(inv_gamma):
    gamma_fact = gamma_func(inv_gamma+1/2)
    square_pi = (np.pi)**(1/2)
    coeff = (gamma_fact/square_pi)**(-1/inv_gamma)
    coeff = 1/2 * coeff
    return coeff


def get_arch1_sample(inv_gamma, size_sample,beta=1):
    arch_lambda_coeff = arch_coeff(inv_gamma)
    current_sample = normal_gauss(0,1)**2
    arch_sample = [current_sample]
    for i in range(size_sample-1):
        square_hidden_gaussian = normal_gauss(0,1)**2
        factor = beta+arch_lambda_coeff*current_sample
        next_sample = square_hidden_gaussian*factor
        arch_sample.append(next_sample)
        current_sample = next_sample
    return arch_sample


def get_random_block(sample,b_size):
    max_idx = len(sample)-b_size
    random_idx = random.randint(0,max_idx)
    return sample[random_idx:random_idx+b_size]


def get_random_block_size(method_block_size, sample_size):
    """
    Takes a dictionnary methode_block_size that can be :
    - method deterministic : returns a given block size
    - method geometric : returns a geometric_law block size
    """
    if method_block_size["alea"] == "deterministic":
        b_size = method_block_size["size_expectancy"]
        if b_size > sample_size:
            raise ValueError(f"The block size imputed is greater than sample size: {b_size} > {sample_size}")
        return b_size
    if method_block_size["alea"] == "geometric":
        p = 1/method_block_size["size_expectancy"]
        geom_size = np.random.geometric(p)
        geom_size = np.min([geom_size,sample_size]) # size must not be greater than sample size
        geom_size = np.max([geom_size,3]) # size must be greater than 3 to get relevant hill estimate
        return geom_size


# # Trouver la relation entre X suit frechet et Y suit GEV
# def get_frechet_sample_with_genextreme(inv_gamma, size_sample, get_plot=False, log_plot=False, nb_bins=100):
#     # inv_gamma correspond au shape param
#     # Si inv_gamma < 1 pas d'esp
#     c = -1/inv_gamma
#     sigma = 1
#     mu = 0
#     r = genextreme.rvs(c, size=size_sample)
# #     r = (1+1/c*r)**(-c)  # TODO : Trouver la relation entre X suit frechet et Y suit GEV
#     if get_plot:
#         plt.hist(r, density=True, bins=nb_bins, log=log_plot, label="frechet_samples_extreme", alpha = 0.6)
#         plt.legend()
#     return r
    



### On définit les estimateurs de gamma de de Haan 1998
### On définit l'estimateur de la variance asymptotique
### on minimise cet estimateur par rapport à k pour une taille de bootstrap donnée


def gamma_moment_1_estimator(k, sample):
    # Gives moment 1 estimator of gamma (depends on the order statistic k)
    sorted_sample = np.sort(sample)
    k_largest_samples = sorted_sample[-k:]
    kest_largest = sorted_sample[-k]
    moment_1_estimator = np.mean(np.log(k_largest_samples)-np.log(kest_largest))
    return moment_1_estimator

# # Test du moment ordre 1
# inv_gamma = 5
# size_sample = 10000
# sample = get_frechet_sample(inv_gamma, size_sample)
# print(f"estimated value is {gamma_moment_1_estimator(1000,sample)}; true value is {1/inv_gamma}")

def gamma_moment_2_estimator(k, sample):
    # Gives moment 2 estimator of gamma (depends on the order statistic k)
    if k==1:
        print("There was a k order stat =1 : reimplaces by 2")
        k = 2
    sorted_sample = np.sort(sample)
    k_largest_samples = sorted_sample[-k:]
    kest_largest = sorted_sample[-k]
    moment_2_estimator = np.mean((np.log(k_largest_samples)-np.log(kest_largest))**2)
    return moment_2_estimator/(2*gamma_moment_1_estimator(k, sample))

# # Test du moment ordre 2
# inv_gamma = 3
# size_sample = 10000
# sample = get_frechet_sample(inv_gamma, size_sample)
# print(f"estimated value is {gamma_moment_2_estimator(1000,sample)}; true value is {1/inv_gamma}")


def gamma_moment_alpha(k, sample, alpha):
    sorted_sample = np.sort(sample)
    k_largest_samples = sorted_sample[-k:]
    kest_largest = sorted_sample[-k]
    moment_alpha_estimator = np.mean((np.log(k_largest_samples)-np.log(kest_largest))**alpha)
    return moment_alpha_estimator


def S_2_de_haan_mercadier_zhou(k, sample):
    m4 = gamma_moment_alpha(k, sample, 4)
    m3 = gamma_moment_alpha(k, sample, 3)
    m2 = gamma_moment_alpha(k, sample, 2)
    m1 = gamma_moment_alpha(k, sample, 1)
    num = (m4 - 24*(m1)**4) * (m2 - 2*(m1)**2)
    denom = m3 - 6*(m1)**3
    s2 = 3/4 * num/denom
    # print(s2)
    # if s2 < 2/3 or s2 > 3/4:
    #     raise ValueError("s2 value is not in the right range between 2/3 and 3/4: {s2}")
    return s2


def rho_2_de_haan_mercadier_zhou(k, sample):
    s2 = S_2_de_haan_mercadier_zhou(k, sample)
    num = -4 +6*s2 + np.sqrt(3*s2 -2)
    denom = 4*s2 -3
    rho = num / denom
    return rho


def rho_2_gomes_de_haan_rodrigues(k, sample):
    m3 = gamma_moment_alpha(k, sample, 3)
    m2 = gamma_moment_alpha(k, sample, 2)
    m1 = gamma_moment_alpha(k, sample, 1)
    num_t2 = m1**2 - (m2/2)
    denom_t2 = (m2/2) - (m3/6)**2/3
    t2 = num_t2 / denom_t2
    num = - np.abs(3*t2-1)
    denom = np.abs(t2-3)
    rho = - num / denom
    return rho


def unbiaised_gamma_mercadier_zhou(sample, k):
    gamma_biaised = gamma_moment_1_estimator(k, sample)
    m2 = gamma_moment_alpha(k, sample, 2)
    rho2 = rho_2_de_haan_mercadier_zhou(k, sample)
    num_b = m2 - 2*gamma_biaised**2
    denom_2 = 2*gamma_biaised*rho2/(1-rho2)
    biais = num_b / denom_2
    return gamma_biaised - biais



def unbiaised_gamma_gomes_dehaan_rodrigues(sample, k):
    gamma_biaised = gamma_moment_1_estimator(k, sample)
    m2 = gamma_moment_alpha(k, sample, 2)
    rho2 = rho_2_gomes_de_haan_rodrigues(k, sample)
    num_b = m2 - 2*gamma_biaised**2
    denom_2 = 2*gamma_biaised*rho2/(1-rho2)
    biais = num_b / denom_2
    return gamma_biaised - biais



def get_bootstrap_variance_est_de_haan_zhou(method, sample,
                                      nb_bootstrap,
                                      bootstrap_size,
                                      k0_opti):
    """
    When sample is fixed, it returns :
    - the de_hann & zhou std estimation (with substraction) times the root of k0_opti, 
    with bootstraps of a fixed sample
    - the hill estimator of the sample as estimation
    """
    method_bootstrap = method["method_bootstrap"]
    substracted_estimation_list = list()
    btstrap_est_list = list()
    gamma_moment_1_mc = gamma_moment_1_estimator(k0_opti, sample)
    if method_bootstrap["method_name"] == "iid":
        for i in range(nb_bootstrap):
            bootstrapped_sample = choices(sample, k=bootstrap_size)
            gamma_moment_1_btstrap = gamma_moment_1_estimator(k0_opti, bootstrapped_sample)
            btstrap_est_list.append(gamma_moment_1_btstrap)
            substracted_estimation = gamma_moment_1_btstrap - gamma_moment_1_mc
            substracted_estimation_list.append(substracted_estimation*k0_opti**(1/2)) # Normalization by block size root
    elif method_bootstrap["method_name"] == "stationary":
        for i in range(nb_bootstrap):
            sample_size = len(sample)
            method_block_size = {
                "alea": method_bootstrap["alea"],
                "size_expectancy": int(method_bootstrap["size_sample_bootstrap_ratio"]*sample_size)
            }
            b_size = get_random_block_size(method_block_size, sample_size) #Now we know the block size we adapt bootstrap size and k0_opti
            bootstrap_size = b_size
            k0_opti_adapted = int(k0_opti*b_size/sample_size)
            bootstrapped_sample = get_random_block(sample,bootstrap_size)

            gamma_moment_1_btstrap = gamma_moment_1_estimator(k0_opti_adapted, bootstrapped_sample)
            btstrap_est_list.append(gamma_moment_1_btstrap)
            substracted_estimation = gamma_moment_1_btstrap - gamma_moment_1_mc
            substracted_estimation_list.append(substracted_estimation*k0_opti_adapted**(1/2)) # Normalization by block size root
    else:
        btstrap_name = method["method_bootstrap"]["method_name"]
        raise ValueError(f"Method bootstrap unknown: {btstrap_name}")
    
    std_estimator = np.std(substracted_estimation_list)
    # mom1 = gamma_moment_1_mc # As estimator I use the hill estimate on the sample
    mom1 = np.mean(btstrap_est_list) # As estimator I use the mean hill estimator on the bostraps samples (just as de_haan_98 est)
    return std_estimator, mom1 # Normalisation *k0_opti**(1/2) happens when added to std_estimation_list


def de_hann_zhou_utils(config_path,
                        config,
                        path_run):
    """
    Specific function for a test experiment of de_hann_&_zhou experience (to check the 
    variance of the substraction of bootstrap and monte carlo)
    """
    size_sample = config["run_de_hann_zhou_experience_config"]["size_sample"]
    nb_samples = config["run_de_hann_zhou_experience_config"]["nb_samples"]
    nb_bootstrap_samples = config["run_de_hann_zhou_experience_config"]["nb_bootstrap_samples"]
    inv_gamma = config["run_de_hann_zhou_experience_config"]["inv_gamma"]
    variance_raw_est_for_each_k_list = list()
    variance_substract_est_for_each_k_list = list()
    k_list = list()
    for k_stat_ratio in tqdm(range(5,95,4)):
        k_order_statistic = int(k_stat_ratio/100*size_sample)
        k_list.append(k_order_statistic)
        raw_est_list = list()
        substract_est_list = list()
        for i in range(nb_samples):
            new_fixed_sample = get_frechet_sample(inv_gamma, size_sample)
            fixed_sample_gamma_estimation = gamma_moment_1_estimator(k_order_statistic, 
                                                                     new_fixed_sample)
            for j in range(nb_bootstrap_samples):
                bootstrapped_sample = choices(new_fixed_sample, 
                                              k=size_sample) # On rééchantillonne
                btstrap_sample_gamma_est = gamma_moment_1_estimator(k_order_statistic, 
                                                                    bootstrapped_sample)
                denormalized_btstrped_smple_gam_est = btstrap_sample_gamma_est*k_order_statistic**(1/2)
                sbstract_est = btstrap_sample_gamma_est - fixed_sample_gamma_estimation
                denormalized_substract_est = sbstract_est*k_order_statistic**(1/2)
                raw_est_list.append(denormalized_btstrped_smple_gam_est)
                substract_est_list.append(denormalized_substract_est)
        variance_raw_est_for_each_k_list.append(np.std(raw_est_list))
        variance_substract_est_for_each_k_list.append(np.std(substract_est_list))
    return k_list, variance_raw_est_for_each_k_list, variance_substract_est_for_each_k_list


def de_hann_98_de_hann_zhou_experience(config_path,
                                            config,
                                            path_run):
    size_sample = config["run_de_hann_zhou_experience_config"]["size_sample"]
    nb_samples = config["run_de_hann_zhou_experience_config"]["nb_samples"]
    nb_bootstrap_samples = config["run_de_hann_zhou_experience_config"]["nb_bootstrap_samples"]
    inv_gamma = config["run_de_hann_zhou_experience_config"]["inv_gamma"]
    var_mom1_est_for_each_k_list = list()
    var_mom2_est_for_each_k_list = list()
    var_substract_est_for_each_k_list = list()
    k_list = list()
    for k_stat_ratio in tqdm(range(5,95,10)):
        k_order_statistic = int(k_stat_ratio/100*size_sample)
        k_list.append(k_order_statistic)
        mom1_est_list = list()
        mom2_est_list = list()
        substract_est_list = list()
        for i in range(nb_samples):
            new_fixed_sample = get_frechet_sample(inv_gamma, size_sample)
            for j in range(nb_bootstrap_samples):
                bootstrapped_sample = choices(new_fixed_sample, 
                                              k=size_sample) # On rééchantillonne
                btstrap_sample_mom1_est = gamma_moment_1_estimator(k_order_statistic, 
                                                                    bootstrapped_sample)
                btstrap_sample_mom2_est = gamma_moment_2_estimator(k_order_statistic, 
                                                                    bootstrapped_sample)
                denormalized_btstrped_smple_mom1_est = btstrap_sample_mom1_est*k_order_statistic**(1/2)
                denormalized_btstrped_smple_mom2_est = btstrap_sample_mom2_est*k_order_statistic**(1/2)
                denormalized_substract_est = denormalized_btstrped_smple_mom2_est - denormalized_btstrped_smple_mom1_est
                mom1_est_list.append(denormalized_btstrped_smple_mom1_est)
                mom2_est_list.append(denormalized_btstrped_smple_mom2_est)
                substract_est_list.append(denormalized_substract_est)
        var_mom1_est_for_each_k_list.append(np.std(mom1_est_list))
        var_mom2_est_for_each_k_list.append(np.std(mom2_est_list))
        var_substract_est_for_each_k_list.append(np.std(substract_est_list))
    return k_list, var_mom1_est_for_each_k_list, var_mom2_est_for_each_k_list, var_substract_est_for_each_k_list


def gamma_plots_in_front_of_k_order_stats(dict_results, inv_gamma,
                                            config_path, config, path_run):
    # gamma_est and gamma_std same plot
    fig, ax = plt.subplots()
    for name, d in dict_results.items():
        name_key = name + "_est"
        ax.errorbar(d["k_list"], d["est_list"], yerr=d["std_list"], 
                    capsize=5, 
                    alpha=0.7, label=name_key)
    ax.set_title("Gamma_estimation_in_front_of_k_order_statistics")
    ax.legend(framealpha=0.1)
    plt.savefig(path_run+"\\Gamma_estimation_in_front_of_k_order_statistics.jpg")
    plt.clf()
    # plot gamma_biais (only for monte carlo estimations)
    for name, d in dict_results.items():
        if name in ["monte_carlo", "monte_carlo_unbiaised_gomes"]:
            # We only plot for monte carlo estimations
            name_key = name + "_biais"
            true_gamma = 1/inv_gamma
            biais_list = [np.abs(g - true_gamma) for g in d["est_list"]]
            plt.plot(d["k_list"], biais_list, 
                        alpha=0.7, label=name_key)
    plt.title("Gamma_biais_in_front_of_k_order_statistics")
    plt.legend(framealpha=0.1)
    plt.savefig(path_run+"\\Gamma_biais_in_front_of_k_order_statistics.jpg")
    plt.clf()
    # plot gamma_std
    for name, d in dict_results.items():
        name_key = name + "_std"
        plt.plot(d["k_list"], d["std_list"], 
                    alpha=0.7, label=name_key)
    plt.title("Gamma_std_in_front_of_k_order_statistics")
    plt.legend(framealpha=0.1)
    plt.savefig(path_run+"\\Gamma_std_in_front_of_k_order_statistics.jpg")
    plt.clf()
    # plot gamma_biais/gamma_std
    for name, d in dict_results.items():
        name_key = name + "_biais"
        true_gamma = 1/inv_gamma
        biais_list = [np.abs(g - true_gamma) for g in d["est_list"]]
        biais_divided_by_std = [biais / sd 
                                for biais, sd in zip(biais_list,d["std_list"])]
        plt.plot(d["k_list"], biais_divided_by_std, 
                    alpha=0.7, label=name_key)
    plt.title("Gamma_biais_on_std_in_front_of_k_order_statistics")
    plt.legend(framealpha=0.1)
    plt.savefig(path_run+"\\Gamma_biais_on_std_in_front_of_k_order_statistics.jpg")
    plt.clf()
    # plot gamma_std_std
    # TODO: plot variance of variance


# def kulik_modified_de_hann_98_de_hann_zhou_experience(config_path,
#                                             config,
#                                             path_run):
#     size_sample = config["run_kulik_modified_de_hann_zhou_experience_config"]["size_sample"]
#     nb_samples = config["run_kulik_modified_de_hann_zhou_experience_config"]["nb_samples"]
#     nb_bootstrap_samples = config["run_kulik_modified_de_hann_zhou_experience_config"]["nb_bootstrap_samples"]
#     inv_gamma = config["run_kulik_modified_de_hann_zhou_experience_config"]["inv_gamma"]
#     variance_kulik_est_for_each_k_list = list()
#     variance_kulik_mod1_est_for_each_k_list = list()
#     variance_kulik_mod2_est_for_each_k_list = list()
#     k_list = list()
#     for k_stat_ratio in tqdm(range(5,95,4)):
#         k_order_statistic = int(k_stat_ratio/100*size_sample)
#         k_list.append(k_order_statistic)
#         kulik_est_list = list()
#         kulik_mod1_est_for_each_k_list = list()
#         kulik_mod2_est_for_each_k_list = list()
#         for i in range(nb_samples):
#             new_fixed_sample = get_frechet_sample(inv_gamma, size_sample)
#             # fixed_sample_gamma_estimation = gamma_moment_1_estimator(k_order_statistic, 
#             #                                                          new_fixed_sample)
#             for j in range(nb_bootstrap_samples):
#                 # bootstrapped_sample = choices(new_fixed_sample, 
#                 #                               k=size_sample) # On rééchantillonne
#                 btstrap_sample_gamma_est = gamma_moment_1_estimator(k_order_statistic, 
#                                                                     bootstrapped_sample)
#                 denormalized_btstrped_smple_gam_est = btstrap_sample_gamma_est*k_order_statistic**(1/2)
#                 sbstract_est = btstrap_sample_gamma_est - fixed_sample_gamma_estimation
#                 denormalized_substract_est = sbstract_est*k_order_statistic**(1/2)
#                 raw_est_list.append(denormalized_btstrped_smple_gam_est)
#                 substract_est_list.append(denormalized_substract_est)
#         variance_raw_est_for_each_k_list.append(np.std(raw_est_list))
#         variance_substract_est_for_each_k_list.append(np.std(substract_est_list))
#     return k_list, variance_raw_est_for_each_k_list, variance_substract_est_for_each_k_list
