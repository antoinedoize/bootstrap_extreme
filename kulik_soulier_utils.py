from extreme_utils import *
import numpy as np
import scipy.stats


def r(n):
    #La seule condition : r(n) = o(n)
    return 4*np.log(n) # CHOIX : On fixe r(n) = 4*ln(n)


def get_blocks(sample, func_block_size):
    n = len(sample)
    nb_blocks = int(n/func_block_size(n))
    block_list = list()
    for i in range(nb_blocks):
        ind_left = int(i*func_block_size(n))
        ind_right = int((i+1)*func_block_size(n))
        block_list.append(sample[ind_left:ind_right])
    return block_list


def get_poisson_weights(nb_weights):
    # CHOIX : On a choisi la loi de Poisson(1)
    return np.random.poisson(1, nb_weights)


def get_gaussian_weights(nb_weights):
    # Choix : we choose gaussian distribution with mean 1
    return np.random.normal(1,1,nb_weights)


def get_exponential_weights(nb_weights):
    # Choix : we choose exponential distribution with mean 1
    return np.random.exponential(1,nb_weights)


def get_multiplier_version_hill(sample, k0_opti, block_weights,
                                downsample_per_block,bootstrap_ratio,
                                func_block_size):
    num, denom = 0, 0
    block_list = get_blocks(sample, func_block_size)
    if len(block_weights) != len(block_list):
        raise ValueError(f"Not same number of weights {len(block_weights)} and blocks : {len(block_list)}")
    bootstrapped_block_list = list()
    bootstrapped_sample = list()
    for block in block_list:
        if downsample_per_block:
            # Here we downsample per block
            block_btstrp_size = max(int(bootstrap_ratio*len(block)),1) # We take at least 1 per block
            bootstrapped_block = choices(block, k=block_btstrp_size)
        else:
            bootstrapped_block = block
        bootstrapped_block_list.append(bootstrapped_block)
        bootstrapped_sample.extend(bootstrapped_block[:])
    
    sorted_bootstrapped_sample = sorted(bootstrapped_sample)
    k0_opti = int(k0_opti*bootstrap_ratio) if downsample_per_block else k0_opti
    x_threshold = sorted_bootstrapped_sample[-k0_opti]
    for bootstrapped_block, weight in zip(bootstrapped_block_list, block_weights):
        block_num, block_denom = 0, 0
        for x_i in bootstrapped_block:
            if x_i > x_threshold:
                block_num += np.log(x_i/x_threshold)
                block_denom += 1
        num += block_num*weight
        denom += block_denom*weight # TODO: Check : ça devrait être *weight non ?
    return num/denom



def multiplier_estimations(sample, k0_opti, nb_boostraps,
                             weight_distribution, downsample_per_block,
                             bootstrap_ratio, block_size):
    if block_size == "auto":
        func_block_size = r
    elif type(block_size) == int:
        func_block_size = lambda nb: block_size
    else:
        raise ValueError(f"Missmatched argument for block size function kulik expected int of auto, and got: {block_size}")
    distribution_name_to_function = {
        "poisson": get_poisson_weights,
        "gaussian": get_gaussian_weights,
        "exponential": get_exponential_weights
    }
    get_weights = distribution_name_to_function[weight_distribution]
    n = len(sample)
    nb_weights = int(n/func_block_size(n))
    mult_hill_list = list()
    for i in range(nb_boostraps):
        block_weights = get_weights(nb_weights)
        mult_hill = get_multiplier_version_hill(sample, k0_opti, block_weights,
                                                downsample_per_block,bootstrap_ratio,
                                                func_block_size)
        mult_hill_list.append(mult_hill)
    return mult_hill_list


def get_bootstrap_variance_kulik(method, sample,
                                 nb_bootstrap,
                                 bootstrap_size,
                                 k0_opti):
    weight_distribution = method["weight_distribution"]
    downsampling_method = method["kulik_downsampling_procedure"]
    bootstrap_ratio = method["size_sample_bootstrap_ratio"]

    if downsampling_method == "global":
        bootstrapped_sample = choices(sample, k=bootstrap_size) # On rééchantillonne
        downsample_per_block = False
    if downsampling_method == "per_block":
        bootstrapped_sample = sample
        downsample_per_block = True
    mult_hill_list = multiplier_estimations(bootstrapped_sample, k0_opti, nb_bootstrap,
                                            weight_distribution,downsample_per_block,
                                            bootstrap_ratio)
    mult_hill_list = sorted(mult_hill_list)
    q25_ind, q75_ind = int(25/100*len(mult_hill_list)), int(75/100*len(mult_hill_list))
    q75, q25 = mult_hill_list[q75_ind], mult_hill_list[q25_ind]
    qn_75, qn_25 = scipy.stats.norm.ppf(0.75), scipy.stats.norm.ppf(0.25)
    std_estimator = (q75 - q25)/(qn_75 - qn_25)
    return std_estimator*k0_opti**(1/2)


def get_bootstrap_variance_est_kulik(method, sample,
                                 nb_bootstrap,
                                 bootstrap_size,
                                 k0_opti):
    weight_distribution = method["weight_distribution"]
    downsampling_method = method["kulik_downsampling_procedure"]
    bootstrap_ratio = method["size_sample_bootstrap_ratio"]
    block_size = method["block_size"]

    if downsampling_method == "global":
        bootstrapped_sample = choices(sample, k=bootstrap_size) # On rééchantillonne
        downsample_per_block = False
    elif downsampling_method == "per_block":
        bootstrapped_sample = sample # we keep the sample
        downsample_per_block = True # but we will downsample among blocks
    elif downsampling_method == "none":
        bootstrapped_sample = sample # We keep the sample
        downsample_per_block = False # and we won't downsample among vlocks
    else:
        raise ValueError("Give kulik downsampling method among global, per_block, none")
    mult_hill_list = multiplier_estimations(bootstrapped_sample, k0_opti, nb_bootstrap,
                                            weight_distribution,downsample_per_block,
                                            bootstrap_ratio, block_size)
    mult_hill_list = sorted(mult_hill_list)
    q25_ind, q75_ind = int(25/100*len(mult_hill_list)), int(75/100*len(mult_hill_list))
    q75, q25 = mult_hill_list[q75_ind], mult_hill_list[q25_ind]
    qn_75, qn_25 = scipy.stats.norm.ppf(0.75), scipy.stats.norm.ppf(0.25)
    std_estimator = (q75 - q25)/(qn_75 - qn_25)
    estimator = np.mean(mult_hill_list)
    return std_estimator*k0_opti**(1/2), estimator