from extreme_utils import *
import numpy as np
import scipy.stats


def r(n):
    #La seule condition : r(n) = o(n)
    return 4*np.log(n) # CHOIX : On fixe r(n) = 4*ln(n)


def get_blocks(sample):
    n = len(sample)
    nb_blocks = int(n/r(n))
    block_list = list()
    for i in range(nb_blocks):
        ind_left = int(i*r(n))
        ind_right = int((i+1)*r(n))
        block_list.append(sample[ind_left:ind_right])
    return block_list


def get_multiplier_version_hill(sample, k0_opti, block_weights):
    num, denom = 0, 0
    block_list = get_blocks(sample)
    if len(block_weights) != len(block_list):
        raise ValueError(f"Not same number of weights {len(block_weights)} and blocks : {len(block_list)}")
    sorted_sample = sorted(sample)
    x_threshold = sorted_sample[-k0_opti]
    for block, weight in zip(block_list, block_weights):
        block_num, block_denom = 0, 0
        for x_i in block:
            if x_i > x_threshold:
                block_num += np.log(x_i/x_threshold)
                block_denom += 1
        num += block_num*weight
        denom += block_denom*1
    return num/denom


def multiplier_estimations(sample, k0_opti, nb_boostraps):
    n = len(sample)
    nb_weights = int(n/r(n))
    mult_hill_list = list()
    for i in range(nb_boostraps):
        block_weights = np.random.poisson(1, nb_weights) # CHOIX : On a choisi la loi de Poisson(1)
        mult_hill = get_multiplier_version_hill(sample, k0_opti, block_weights)
        mult_hill_list.append(mult_hill)
    return mult_hill_list


def get_bootstrap_variance_kulik(sample,
                                 nb_bootstrap,
                                 bootstrap_size,
                                 k0_opti):
    mult_hill_list = multiplier_estimations(sample, k0_opti, nb_bootstrap)
    mult_hill_list = sorted(mult_hill_list)
    q25_ind, q75_ind = int(25/100*len(mult_hill_list)), int(75/100*len(mult_hill_list))
    q75, q25 = mult_hill_list[q75_ind], mult_hill_list[q25_ind]
    qn_75, qn_25 = scipy.stats.norm.ppf(0.75), scipy.stats.norm.ppf(0.25)
    std_estimator = (q75 - q25)/(qn_75 - qn_25)
    return std_estimator*k0_opti**(1/2)