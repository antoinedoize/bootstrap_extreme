from extreme_utils import *
from de_hann_98_utils import *
from kulik_soulier_utils import *


def get_bootstrap_vs_monte_carlo(nb_monte_carlo_steps,
                                inv_gamma,
                                size_sample,
                                list_methods):
    output_dict = dict()
    output_dict["hill_estimator_list"] = list()
    for method in list_methods:
        output_dict[method["method_name"] + "_std"] = list()
    for i in tqdm(range(nb_monte_carlo_steps), desc = "monte_carlo steps"):
        sample = get_frechet_sample(inv_gamma, size_sample)
        # On lance une optimisation et on calcul le hill estimator
        k0_opti = int(de_haan_1998(sample))
        hill_estimator = gamma_moment_1_estimator(k0_opti, sample)
        output_dict["hill_estimator_list"].append(hill_estimator)
        # Pour chaque méthode on va calculer le bootstrap std
        for method in list_methods:
            std_method = method["method"](sample, nb_monte_carlo_steps,
                                          len(sample), k0_opti)
            output_dict[method["method_name"] + "_std"].append(std_method)
    output_dict["hill_estimator_mean"] = np.mean(output_dict["hill_estimator_list"])
    output_dict["hill_estimator_std"] = np.std(output_dict["hill_estimator_list"])
    return output_dict


# nb_monte_carlo_steps = 5
# inv_gamma = 3
# size_sample = 20000
# list_methods = [
#     {"method_name": "de_hann_98",
#      "method": get_bootstrap_variance_de_hann_98,
# #      "argument_list": [sample, nb_bootstrap, bootstrap_size, k0_opti]
#     },
#     {"method_name": "kulik",
#      "method": get_bootstrap_variance_kulik,
# #      "argument_list": [sample, nb_bootstrap, bootstrap_size, k0_opti] 
#     }
# ]
# output_dict = get_bootstrap_vs_monte_carlo(nb_monte_carlo_steps,
#                                 inv_gamma,
#                                 size_sample,
#                                 list_methods)

# with open(f"mc_{nb_monte_carlo_steps}_invg_{inv_gamma}_size_{size_sample}.json", "w") as write_file:
#     json.dump(output_dict, write_file)
    
# with open(f"mc_{nb_monte_carlo_steps}_invg_{inv_gamma}_size_{size_sample}.json", 'r') as fp:
#     output_dict = json.load(fp)

# plt.plot(output_dict["de_hann_98_std"], label="de_hann_98")
# plt.plot(output_dict["kulik_std"], label="kulik")
# plt.plot([output_dict["hill_estimator_std"] for i in range(len(output_dict["kulik_std"]))], label="")
# plt.title("Standard deviation for each monte carlo step")
# plt.legend()


def get_monte_carlo_methods_std_for_a_step(nb_monte_carlo_steps,
                                inv_gamma,
                                size_sample,
                                list_methods):
    ratio = 0
    output_dict = dict()
    hill_estimator_list = list()
    for i in range(nb_monte_carlo_steps):
        sample = get_frechet_sample(inv_gamma, size_sample)
        # On lance une optimisation et on calcul le hill estimator
        k0_opti = int(de_haan_1998(sample))
        try:
            hill_estimator = gamma_moment_1_estimator(k0_opti, sample)
            hill_estimator_list.append(hill_estimator)
        except:
            ratio += 1
            hill_estimator_list.append(hill_estimator)
            continue
    output_dict["monte_carlo_std"] = np.std(hill_estimator_list)
    # Pour chaque méthode on va calculer le bootstrap std
    for method in list_methods:
        std_method = method["method"](sample, nb_monte_carlo_steps,
                                      len(sample), k0_opti)
        output_dict[method["method_name"] + "_std"] = std_method
    ratio = ratio / nb_monte_carlo_steps
    if ratio != 0:
        print(f"Ratio of missed estimations is {ratio} for size_sample = {size_sample}")
    return output_dict


# list_methods = [
#     {"method_name": "de_hann_98",
#      "method": get_bootstrap_variance_de_hann_98,
# #      "argument_list": [sample, nb_bootstrap, bootstrap_size, k0_opti]
#     },
#     {"method_name": "kulik",
#      "method": get_bootstrap_variance_kulik,
# #      "argument_list": [sample, nb_bootstrap, bootstrap_size, k0_opti] 
#     }
# ] 

# std_vs_size_dict = {
#     "monte_carlo_std": list(),
#     "de_hann_98_std": list(),
#     "kulik_std": list(),
# }
# inv_gamma = 3
# nb_monte_carlo_steps = 50
# size_samples = list(range(500,10000,500))
# for size_sample in tqdm(size_samples, desc = f"size_sample_loop"):
#     output_dict = get_monte_carlo_methods_std_for_a_step(nb_monte_carlo_steps,
#                                 inv_gamma,
#                                 size_sample,
#                                 list_methods)
#     for key in output_dict.keys():
#         std_vs_size_dict[key].append(output_dict[key])

# with open(f"std_vs_sample_size_nb_monte_carlo_{nb_monte_carlo_steps}.json", "w") as write_file:
#     json.dump(std_vs_size_dict, write_file)
    
# with open(f"std_vs_sample_size_nb_monte_carlo_{nb_monte_carlo_steps}.json", 'r') as fp:
#     std_vs_size_dict = json.load(fp)
        

# plt.plot(size_samples, std_vs_size_dict["monte_carlo_std"], label="monte_carlo_std")
# plt.plot(size_samples, std_vs_size_dict["de_hann_98_std"], label="de_hann_98_std")
# plt.plot(size_samples, std_vs_size_dict["kulik_std"], label="kulik_std")
# plt.title("Standard deviation for each method vs size_sample")
# plt.legend()