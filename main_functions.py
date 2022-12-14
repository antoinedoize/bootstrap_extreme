from extreme_utils import *
from de_hann_98_utils import *
from kulik_soulier_utils import *
import os
import datetime
import json
import numpy as np


def get_monte_carlo_variance(method, nb_monte_carlo_steps,
                                      inv_gamma, size_sample):
    k_order_statistic_mc_list = list()
    hill_estimator_denormalized_list = list()
    ratio = 0
    k_order_statistics_ratio = method["k_order_statistics_ratio"]
    if k_order_statistics_ratio == "version2":
        get_auto_k_order = True
    elif 0<= k_order_statistics_ratio <= 1:
        get_auto_k_order = False
        k_order_statistic = int(k_order_statistics_ratio*size_sample)
    # start mc
    for i in range(nb_monte_carlo_steps):
        sample = get_frechet_sample(inv_gamma, size_sample)
        if get_auto_k_order: # we optimize k
            k_order_statistic = int(de_haan_1998(sample))
        k_order_statistic_mc_list.append(k_order_statistic)
        try:
            hill_estimator = gamma_moment_1_estimator(k_order_statistic, sample)
            hill_estimator_denormalized = hill_estimator*k_order_statistic**(1/2)
            hill_estimator_denormalized_list.append(hill_estimator_denormalized)
        except:
            ratio += 1
            hill_estimator_denormalized_list.append(hill_estimator_denormalized)
            continue
    monte_carlo_std = np.std(hill_estimator_denormalized_list)
    mc_mean_k_order_statistic = np.mean(k_order_statistic_mc_list)
    ratio = ratio / nb_monte_carlo_steps
    if ratio >=0.1:
        raise ValueError(f"Ratio of missed estimations is {ratio} for size_sample = {size_sample}")
    return monte_carlo_std, mc_mean_k_order_statistic


# TODO : Check if we will use this function : if not : delete, if we do : adapt to config
# def get_bootstrap_vs_monte_carlo(nb_monte_carlo_steps,
#                                 nb_bootstrap_steps,
#                                 inv_gamma,
#                                 size_sample,
#                                 list_methods,
#                                 k_order_statistic="version2"):
#     output_dict = dict()
#     output_dict["m_c_hill_estimator_denormalized_list"] = list()
#     output_dict["m_c_hill_estimator_normalized_list"] = list()
#     output_dict["k_order_statistic"] = list()
#     ratio = 0
#     method_name_to_function = {
#         "get_bootstrap_variance_de_hann_98": get_bootstrap_variance_de_hann_98,
#         "get_bootstrap_variance_kulik": get_bootstrap_variance_kulik
#     }
#     if k_order_statistic == "version2":
#         get_auto_k_order = True
#     else :
#         get_auto_k_order = False

#     for method in list_methods:
#         output_dict[method["method_name"] + "_std"] = list()
#     for i in tqdm(range(nb_monte_carlo_steps), desc = "monte_carlo steps"):
#         sample = get_frechet_sample(inv_gamma, size_sample)
#         # On lance une optimisation et on calcul le hill estimator
#         try:
#             if get_auto_k_order:
#                 k_order_statistic = int(de_haan_1998(sample))
#             m_c_hill_estimator = gamma_moment_1_estimator(k_order_statistic, sample)
#             m_c_hill_estimator_denomalized = m_c_hill_estimator*k_order_statistic**(1/2)
#         except:
#             ratio += 1
#         output_dict["m_c_hill_estimator_denormalized_list"].append(m_c_hill_estimator_denomalized)
#         output_dict["m_c_hill_estimator_normalized_list"].append(m_c_hill_estimator)
#         output_dict["k_order_statistic"].append(k_order_statistic)
#         # Pour chaque m??thode on va calculer le bootstrap std
#         for method in list_methods:
#             method_function = method_name_to_function[method["method"]]
#             size_sample_bootstrap_ratio = method["size_sample_bootstrap_ratio"]
#             std_method = method_function(sample, nb_bootstrap_steps,
#                                         int(size_sample*size_sample_bootstrap_ratio), 
#                                         int(k_order_statistic*size_sample_bootstrap_ratio))
#             output_dict[method["method_name"] + "_std"].append(std_method)
#     output_dict["m_c_hill_estimator_mean"] = np.mean(output_dict["m_c_hill_estimator_normalized_list"])
#     output_dict["monte_carlo_std"] = np.std(output_dict["m_c_hill_estimator_denormalized_list"])
#     ratio = ratio / nb_monte_carlo_steps
#     if ratio > 0.1:
#         raise ValueError(f"Ratio of missed estimations is {ratio} for size_sample = {size_sample}")
#     return output_dict


def get_methods_std_for_a_step(nb_monte_carlo_steps,
                                nb_bootstrap_steps,
                                inv_gamma,
                                size_sample,
                                list_methods):
    ratio = 0
    method_name_to_function = {
        "get_bootstrap_variance_de_hann_98": get_bootstrap_variance_de_hann_98,
        "get_bootstrap_variance_kulik": get_bootstrap_variance_kulik,
        "get_monte_carlo_variance": get_monte_carlo_variance
    }
    # if k_order_statistic == "version2":
    #     get_auto_k_order = True
    # else :
    #     get_auto_k_order = False
    output_dict = dict()
    output_dict["hill_estimator_denormalized_list"] = list()
    output_dict["k_order_statistic_mc_list"] = list()
    # for i in range(nb_monte_carlo_steps):
    #     sample = get_frechet_sample(inv_gamma, size_sample)
    #     # On lance une optimisation et on calcul le hill estimator
    #     if get_auto_k_order:
    #         k_order_statistic = int(de_haan_1998(sample))
    #     output_dict["k_order_statistic_mc_list"].append(k_order_statistic)
    #     try:
    #         hill_estimator = gamma_moment_1_estimator(k_order_statistic, sample)
    #         hill_estimator_denormalized = hill_estimator*k_order_statistic**(1/2)
    #         output_dict["hill_estimator_denormalized_list"].append(hill_estimator_denormalized)
    #     except:
    #         ratio += 1
    #         output_dict["hill_estimator_denormalized_list"].append(hill_estimator_denormalized)
    #         continue
    # output_dict["monte_carlo_std"] = np.std(output_dict["hill_estimator_denormalized_list"])
    # first_time_auto_k_order_for_given_sample = True
    # bootstrap_sample_undefined = True
    bootstrap_sample = get_frechet_sample(inv_gamma, size_sample)
    try: # try to get auto k_order twice with two different bootstrap_sample
        auto_k_order_for_this_sample = int(de_haan_1998(bootstrap_sample))
        gamma_moment_1_estimator(auto_k_order_for_this_sample,bootstrap_sample)
    except:
        bootstrap_sample = get_frechet_sample(inv_gamma, size_sample)
        auto_k_order_for_this_sample = int(de_haan_1998(bootstrap_sample))
        gamma_moment_1_estimator(auto_k_order_for_this_sample,bootstrap_sample)
    # Pour chaque m??thode on va calculer le bootstrap std
    for method in list_methods:
        method_function_name = method["method"]
        method_name = method["method_name"]
        method_function = method_name_to_function[method_function_name]
        method_k_order_statistics_ratio = method["k_order_statistics_ratio"]
        # if method_function_name== "get_monte_carlo_variance":
        #     sample = get_frechet_sample(inv_gamma, size_sample)
        #     if bootstrap_sample_undefined:
        #         bootstrap_sample = sample # we set the bootstrap sample
        #         bootstrap_sample_undefined = False
        if method_k_order_statistics_ratio == "version2": # We use auto k order statistics
            # if method_function_name== "get_monte_carlo_variance": # if mc : we compute auto_k_order for sample
            #     method_k_order_statistics = int(de_haan_1998(bootstrap_sample))
            # elif first_time_auto_k_order_for_given_sample:
            #     # If order statistics never computed for this sample: compute it
            #     auto_k_order_for_this_sample = int(de_haan_1998(bootstrap_sample))
            #     first_time_auto_k_order_for_given_sample = False
            #     method_k_order_statistics = auto_k_order_for_this_sample
            # else: # We already computed auto order for this sample
            method_k_order_statistics = auto_k_order_for_this_sample
        else: # We use defined order statistics
            method_k_order_statistics = int(method_k_order_statistics_ratio*size_sample)
        if method_function_name== "get_monte_carlo_variance": # if mc : call the right function
            std_method, k_order_stat = method_function(method, nb_monte_carlo_steps,
                                      inv_gamma, size_sample)
            output_dict[method_name+"_k_order_statistic"] = k_order_stat
        else: # if bootstrap : compute on boostrap_sample
            size_sample_bootstrap_ratio = method["size_sample_bootstrap_ratio"]
            ratio_method_btstrp_order = int(method_k_order_statistics*size_sample_bootstrap_ratio)
            ratio_sample_size = int(size_sample*size_sample_bootstrap_ratio)
            std_method = method_function(method, bootstrap_sample, nb_bootstrap_steps,
                                        ratio_sample_size, 
                                        ratio_method_btstrp_order)
            output_dict[method_name+"_k_order_statistic"] = ratio_method_btstrp_order
        output_dict[method_name+"_std"] = std_method
        

    # output_dict["mc_mean_k_order_statistic"] = np.mean(output_dict["k_order_statistic_mc_list"])
    # ratio = ratio / nb_monte_carlo_steps
    # if ratio >=0.1:
    #     raise ValueError(f"Ratio of missed estimations is {ratio} for size_sample = {size_sample}")
    return output_dict


def save_results_initialization(config_path):
    # Open config_file
    with open(config_path) as config_file:
        config = json.load(config_file)
        name_config = config["name"]
    # Preparation to save the results
    cwd = os.getcwd()
    now = datetime.datetime.now()
    path = f"{cwd}\\output_runs\\{now.year}_{now.month}_{now.day}"
    if not os.path.isdir(path):
        os.mkdir(path)
    time_hash = f"{now.hour}_{now.minute}_{now.second}"
    path_run = path+"\\"+time_hash+f"_{name_config}"
    os.mkdir(path_run)
    print(f"Results will be stored at {path_run}")
    with open(path_run+"\\config_file_"+time_hash+".json", "w") as outfile:
        json.dump(config, outfile)
    return path_run, time_hash, config


def run_variance_vs_size_sample_from_config(config_path,
                                            config,
                                            path_run):

    list_methods = config["run_variance_vs_size_sample_from_config"]["list_methods"]
    inv_gamma = config["run_variance_vs_size_sample_from_config"]["inv_gamma"]
    nb_monte_carlo_steps = config["run_variance_vs_size_sample_from_config"]["nb_monte_carlo_steps"]
    nb_bootstrap_steps = config["run_variance_vs_size_sample_from_config"]["nb_bootstrap_steps"]
    std_vs_size_dict = dict()
    for d in list_methods:
        std_vs_size_dict[d["method_name"]+"_std"] = list() 
        std_vs_size_dict[d["method_name"]+"_k_order_statistic"] = list()
    size_sample_ranges = config["run_variance_vs_size_sample_from_config"]["size_sample_ranges"]
    size_samples = np.sum([list(range(*l)) for l in size_sample_ranges])
    
    for size_sample in tqdm(size_samples, desc = f"size_sample_loop"):
        output_dict = get_methods_std_for_a_step(nb_monte_carlo_steps,
                                    nb_bootstrap_steps,
                                    inv_gamma,
                                    size_sample,
                                    list_methods)
        for key in output_dict.keys():
            if key in std_vs_size_dict.keys():
                std_vs_size_dict[key].append(output_dict[key])

    # plt.plot(size_samples, std_vs_size_dict["monte_carlo_std"], label="monte_carlo_std")
    for d in list_methods:
        name_key = d["method_name"] + "_std"
        plt.plot(size_samples, std_vs_size_dict[name_key], label=name_key)
    plt.title("Standard deviation for each method vs size_sample")
    plt.legend()
    plt.savefig(path_run+"\\Standard_deviation_vs_size_sample.jpg")
    plt.clf()
    for d in list_methods:
        method_k_order_name = d["method_name"] + "_k_order_statistic"
        plt.plot(size_samples, std_vs_size_dict[method_k_order_name], label=method_k_order_name)
    plt.title("k_order_statistic for each step")
    plt.legend()
    plt.savefig(path_run+"\\k_order_statistic_for_each_step.jpg")
    plt.clf()


def run_config_file(config_path):
    path_run, time_hash, config = save_results_initialization(config_path)
    function_to_run = config["function_to_run"]
    if function_to_run == "run_variance_vs_size_sample_from_config":
        run_variance_vs_size_sample_from_config(config_path, config, path_run)
