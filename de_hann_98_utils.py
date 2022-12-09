from extreme_utils import *



def Q_bootstrapped_quadratic_asymp_error_estimator(k1, n1_bootstrap, sample, bootstrap_steps):
    # Q est l'espérance bootstrapée (pas monte-carlo) de l'écart quadratique entre estimateur d'ordre 1 et 2
    # A- On repète bootstrap_steps fois les étapes suivantes
    #    1- rééchantillonnage de taille n1_bootstrap
    #    2- calcul de l'écart quadratique entre estimateur d'ordre 1 et d'ordre 2
    # B- Puis on moyenne ces écarts quadratiques
    # ### TODO : Checker s'il faudrait diviser Q par estimateur d'ordre 1 pour avoir qqchose de normalisé
    q_values_list = list()
    for i in range(bootstrap_steps):
        bootstrapped_sample = choices(sample, k=n1_bootstrap) # On rééchantillonne
        gamma_moment_1 = gamma_moment_1_estimator(k1, bootstrapped_sample)
        gamma_moment_2 = gamma_moment_2_estimator(k1, bootstrapped_sample)
        q_value = (gamma_moment_2*(2*gamma_moment_1) - 2*gamma_moment_1**2)**2  # Mn = gamma2_estim*2gamma_1_est
        q_values_list.append(q_value)
    return np.mean(q_values_list)


def get_bootstrap_variance_de_hann_98(sample,
                                      nb_bootstrap,
                                      bootstrap_size,
                                      k0_opti):
    std_estimator_list = list()
    for i in range(nb_bootstrap):
        bootstrapped_sample = choices(sample, k=bootstrap_size) # On rééchantillonne
        gamma_moment_1 = gamma_moment_1_estimator(k0_opti, bootstrapped_sample)
        gamma_moment_2 = gamma_moment_2_estimator(k0_opti, bootstrapped_sample)
        control_variate = gamma_moment_2 - gamma_moment_1
        std_estimator_list.append(control_variate)
    std_estimator = np.std(std_estimator_list)
    return std_estimator*k0_opti**(1/2)


def find_argmin_Q(n1_bootstrap, sample,
                  k_min=5, k_max="default", step="default",
                  bootstrap_steps=200,
                  plot_q_currents = False):
    # On minimise Q à une taille n1_bootstrap donnée par rapport à la statistique d'ordre k
    # Pour une grid de k, on calcule Q(n1,k)
    # On choisit argmin de Q
    if k_max == "default":
        k_max = int(n1_bootstrap*3/4)
    if step == "default":
        step = max(int(n1_bootstrap//10),1)
    list_of_k = range(k_min, k_max, step)
    argmin_k = k_min
    min_q = Q_bootstrapped_quadratic_asymp_error_estimator(k_min, n1_bootstrap, sample, bootstrap_steps)
    list_q_current = list()
    for k_current in list_of_k:
        q_current = Q_bootstrapped_quadratic_asymp_error_estimator(k_current, n1_bootstrap, sample, bootstrap_steps)
        list_q_current.append(q_current)
        if q_current < min_q:
            min_q = q_current
            argmin_k = k_current
    if plot_q_currents:
        opti_gamma = gamma_moment_1_estimator(argmin_k, sample)
        # TODO add description k_min..etc...
        plt.plot(list_of_k, list_q_current, label="q values")
        plt.title(f"minimization of Q, k_min={argmin_k}; gamma value is {opti_gamma}")
        plt.legend()
        plt.show()
    return argmin_k

# # Test find_argmin_Q
# inv_gamma = 3
# size_sample = 10000
# sample = get_frechet_sample(inv_gamma, size_sample)
# find_argmin_Q(500, sample, plot_q_currents=True)


# # Test plot_q_minimization
# inv_gamma = 3
# size_sample = 10000
# sample = get_frechet_sample(inv_gamma, size_sample)
# test_plot_q_minimization(500, sample, 10)


### On applique l'algo de de Haan_1998

def de_haan_1998(sample):
    n1_bootstrap = int(len(sample)**(2/3))  # CHOIX n1_bootstrap
    argmin_k1 = find_argmin_Q(n1_bootstrap, sample,
                              k_min=5)
    n2_bootstrap = int((n1_bootstrap/len(sample))**2*len(sample))
    argmin_k2 = find_argmin_Q(n2_bootstrap, sample,
                              k_min=5)
    exposant = (np.log(n1_bootstrap) - np.log(argmin_k1))/np.log(n1_bootstrap)
    factor_2 = (np.log(argmin_k1)**2/(2*np.log(n1_bootstrap) - np.log(argmin_k1))**2) ** exposant
    k0_opti = argmin_k1**2/argmin_k2 * factor_2
    return k0_opti
