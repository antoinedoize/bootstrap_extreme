import numpy as np
from scipy.stats import t, invweibull, genextreme
import matplotlib.pyplot as plt
import json


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

from random import choices
from tqdm.notebook import tqdm


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
