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


def get_monte_carlo_variance_est(nb_monte_carlo_steps,
                                 size_sample,
                                 get_distrib,
                                 method_k_order_statistics):
    list_denormalized_estimations = list()
    list_hill_estimation = list()
    for i in range(nb_monte_carlo_steps):
        mc_sample = get_distrib(size_sample)
        hill_estimator = gamma_moment_1_estimator(method_k_order_statistics, mc_sample)
        hill_estimator_denormalized = hill_estimator*method_k_order_statistics**(1/2)
        list_denormalized_estimations.append(hill_estimator_denormalized)
        list_hill_estimation.append(hill_estimator)
    std, est = np.std(list_denormalized_estimations), np.mean(list_hill_estimation)
    return std, est


def get_unbiaised_monte_carlo_variance_est_gomes(nb_monte_carlo_steps,
                                 size_sample,
                                 get_distrib,
                                 method_k_order_statistics):
    list_denormalized_estimations = list()
    list_hill_estimation = list()
    for i in range(nb_monte_carlo_steps):
        mc_sample = get_distrib(size_sample)
        hill_estimator = unbiaised_gamma_gomes_dehaan_rodrigues(mc_sample, method_k_order_statistics)
        hill_estimator_denormalized = hill_estimator*method_k_order_statistics**(1/2)
        list_denormalized_estimations.append(hill_estimator_denormalized)
        list_hill_estimation.append(hill_estimator)
    std, est = np.std(list_denormalized_estimations), np.mean(list_hill_estimation)
    return std, est

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
#         # Pour chaque méthode on va calculer le bootstrap std
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
    # Pour chaque méthode on va calculer le bootstrap std
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
    plt.legend(framealpha=0.1)
    plt.savefig(path_run+"\\Standard_deviation_vs_size_sample.jpg")
    plt.clf()
    for d in list_methods:
        method_k_order_name = d["method_name"] + "_k_order_statistic"
        plt.plot(size_samples, std_vs_size_dict[method_k_order_name], label=method_k_order_name)
    plt.title("k_order_statistic for each step")
    plt.legend()
    plt.savefig(path_run+"\\k_order_statistic_for_each_step.jpg")
    plt.clf()


def run_variance_vs_downsampling_from_config(config_path,
                                            config,
                                            path_run):
    method_name_to_function = {
        "get_bootstrap_variance_de_hann_98": get_bootstrap_variance_de_hann_98,
        "get_bootstrap_variance_kulik": get_bootstrap_variance_kulik,
        "get_monte_carlo_variance": get_monte_carlo_variance
    }
    list_methods = config["run_variance_vs_size_sample_from_config"]["list_methods"]
    inv_gamma = config["run_variance_vs_size_sample_from_config"]["inv_gamma"]
    nb_samples_for_variance_estimation = config["nb_samples_for_variance_estimation"]
    nb_bootstrap_steps = config["run_variance_vs_size_sample_from_config"]["nb_bootstrap_steps"]
    subsampling_rate = config["subsampling_rate"]
    subsampling_ratio_stop = config["subsampling_ratio_stop"]
    for method in list_methods:
        method_name = method["method_name"]
        method_k_order_statistics_ratio = method["k_order_statistics_ratio"]
        method_function = method_name_to_function[method["method"]]
        initial_sample_size = method["initial_sample_size"]
        std_estimations_per_ratio_list = list()
        ratio_list = list()
        for sbsmpling_ratio in tqdm(range(100,int(100*subsampling_ratio_stop),-int(100*subsampling_rate))):
            # Convert ratio in percent to ratio
            ratio_list.append(sbsmpling_ratio)
            ratio = sbsmpling_ratio/100
            std_estimations_per_sample_list = list()
            for i in range(nb_samples_for_variance_estimation):
                sample = get_frechet_sample(inv_gamma, initial_sample_size)
                if method["k_order_statistics_ratio"] == "version2":
                    method_k_order_statistics = int(de_haan_1998(sample))
                    try: # try to get auto k_order twice with two different bootstrap_sample
                        gamma_moment_1_estimator(auto_k_order_for_this_sample,sample)
                    except:
                        sample = get_frechet_sample(inv_gamma, initial_sample_size)
                        auto_k_order_for_this_sample = int(de_haan_1998(sample))
                        gamma_moment_1_estimator(auto_k_order_for_this_sample,sample)
                else:
                    method_k_order_statistics = int(method_k_order_statistics_ratio*initial_sample_size)
                ratio_method_btstrp_order = int(method_k_order_statistics*ratio)
                ratio_sample_size = int(initial_sample_size*ratio)
                
                std_method = method_function(method, sample, nb_bootstrap_steps,
                                            ratio_sample_size, 
                                            ratio_method_btstrp_order)
                std_estimations_per_sample_list.append(std_method)
            std_estimations_per_ratio_list.append(np.std(std_estimations_per_sample_list))
        plt.plot(ratio_list, std_estimations_per_ratio_list)
        title = "Std of std for several downsampling ratio for " + method_name
        plt.title(title)
        plt.savefig(path_run+"\\"+title+".jpg")
        plt.clf()


def run_de_hann_zhou_experience_from_config(config_path,
                                            config,
                                            path_run):
    k_list, variance_raw_est_for_each_k_list, variance_substract_est_for_each_k_list = (
        de_hann_zhou_utils(config_path,
                                            config,
                                            path_run)
    )
   
    plt.plot(k_list, variance_raw_est_for_each_k_list, label="variance raw estimator")
    plt.plot(k_list, variance_substract_est_for_each_k_list, label="variance substracted estimator")
    plt.title("Standard deviation for each variance estimator vs k_order_statistic")
    plt.legend(framealpha=0.1)
    plt.savefig(path_run+"\\standart_deviation_for_each_order_statistics.jpg")
    plt.clf()


def run_test_de_hann_98_accuracy_with_de_hann_zhou_experience(config_path, config, path_run):
    k_list, var_mom1_est_for_each_k_list, var_mom2_est_for_each_k_list, var_substract_est_for_each_k_list = (
        de_hann_98_de_hann_zhou_experience(config_path,
                                            config,
                                            path_run)
    )
   
    plt.plot(k_list, var_mom1_est_for_each_k_list, label="variance mom1 estimator")
    plt.plot(k_list, var_mom2_est_for_each_k_list, label="variance mom2 estimator")
    plt.plot(k_list, var_substract_est_for_each_k_list, label="variance substracted estimator")
    plt.title("Standard deviation for each variance estimator vs k_order_statistic")
    plt.legend(framealpha=0.1)
    plt.savefig(path_run+"\\standart_deviation_for_each_order_statistics.jpg")
    plt.clf()


def run_est_and_var_vs_k_dep_data_from_config(config_path, config, path_run):
    list_methods = config["run_est_and_var_vs_k_dep_data_from_config"]["list_methods"]
    inv_gamma = config["run_est_and_var_vs_k_dep_data_from_config"]["inv_gamma"]
    nb_monte_carlo_steps = config["run_est_and_var_vs_k_dep_data_from_config"]["nb_monte_carlo_steps"]
    nb_bootstrap_steps = config["run_est_and_var_vs_k_dep_data_from_config"]["nb_bootstrap_steps"]
    nb_averaging_btstrp = config["run_est_and_var_vs_k_dep_data_from_config"]["nb_averaging_btstrp"]
    bootstrap_type = config["run_est_and_var_vs_k_dep_data_from_config"]["bootstrap_type"]
    distribution_type = config["run_est_and_var_vs_k_dep_data_from_config"]["distribution_type"]
    ma_order = config["run_est_and_var_vs_k_dep_data_from_config"]["ma_order"]
    size_sample = config["run_est_and_var_vs_k_dep_data_from_config"]["size_sample"]
    method_name_to_function = {
            "get_bootstrap_variance_est_de_hann_98": get_bootstrap_variance_est_de_hann_98,
            "get_bootstrap_variance_est_kulik": get_bootstrap_variance_est_kulik,
            "get_monte_carlo_variance_est": get_monte_carlo_variance_est,
            "get_unbiaised_monte_carlo_variance_est_gomes": get_unbiaised_monte_carlo_variance_est_gomes,
            "get_bootstrap_variance_est_de_haan_zhou": get_bootstrap_variance_est_de_haan_zhou
        }
    distribution_type_to_function = {
        "frechet_iid": lambda size_smple: get_frechet_sample(inv_gamma, size_smple),
        "ma": lambda size_smple: get_ma_sample(inv_gamma, size_smple, ma_order),
        "arch1": lambda size_smple: get_arch1_sample(inv_gamma, size_smple,),
    }
    dict_results = dict()

    get_distrib = distribution_type_to_function[distribution_type]
    sample = get_distrib(size_sample)
    # Maybe decomment if we use auto_k_order
    # try: # try to get auto k_order twice with two different bootstrap_sample
    #     auto_k_order_for_this_sample = int(de_haan_1998(sample))
    #     gamma_moment_1_estimator(auto_k_order_for_this_sample,sample)
    # except:
    #     sample = get_distrib(size_sample)
    #     auto_k_order_for_this_sample = int(de_haan_1998(sample))
    #     gamma_moment_1_estimator(auto_k_order_for_this_sample,sample)
    for method in list_methods:
        method_function_name = method["method"]
        method_name = method["method_name"]
        method_function = method_name_to_function[method_function_name]
        method_bootstrap = method["method_bootstrap"]
        dict_results[method_name] = {
            "k_list": list(),
            "est_list": list(),
            "std_list": list()
        }

        # Maybe decomment if we use auto_k_order
        # if method_k_order_statistics_ratio == "version2": # We use auto k order statistics
        #     method_k_order_statistics = auto_k_order_for_this_sample
        # else: # We use defined order statistics
        for k_order_statistics_percentage in tqdm(range(10,75,5)):
            ratio =  k_order_statistics_percentage/100
            method_k_order_statistics = int(size_sample*ratio)
            dict_results[method_name]["k_list"].append(method_k_order_statistics)

            if method_function_name== "get_monte_carlo_variance_est": # if mc: draw new samples and compute mc
                aver_std, aver_est = get_monte_carlo_variance_est(nb_monte_carlo_steps,
                                size_sample,
                                get_distrib,
                                method_k_order_statistics)
            elif method_function_name=="get_unbiaised_monte_carlo_variance_est_gomes":
                aver_std, aver_est = get_unbiaised_monte_carlo_variance_est_gomes(nb_monte_carlo_steps,
                                size_sample,
                                get_distrib,
                                method_k_order_statistics)
            else: # if bootstrap : compute boostrap_sample
                list_std = list()
                list_est = list()
                for i in range(nb_averaging_btstrp):
                    # We smooth the btstrap on several samples
                    if method_bootstrap["method_name"] == "iid":
                        size_sample_bootstrap_ratio = method_bootstrap["size_sample_bootstrap_ratio"]
                        order_stat_ratio_btstrp = int(method_k_order_statistics*size_sample_bootstrap_ratio)
                        sample_size_btstrp = int(size_sample*size_sample_bootstrap_ratio)
                        std, est = method_function(method, sample, nb_bootstrap_steps,
                                                    sample_size_btstrp, 
                                                    order_stat_ratio_btstrp)
                        list_est.append(est)
                        list_std.append(std)
                    if method_bootstrap["method_name"] == "iid":
                        size_sample_bootstrap_ratio = method_bootstrap["size_sample_bootstrap_ratio"]
                        order_stat_ratio_btstrp = int(method_k_order_statistics*size_sample_bootstrap_ratio)
                        sample_size_btstrp = int(size_sample*size_sample_bootstrap_ratio)
                        std, est = method_function(method, sample, nb_bootstrap_steps,
                                                    sample_size_btstrp, 
                                                    order_stat_ratio_btstrp)
                    elif method_bootstrap["method_name"] == "stationary":
                        size_sample_bootstrap_ratio = method_bootstrap["size_sample_bootstrap_ratio"]
                        order_stat_ratio_btstrp = method_k_order_statistics # Warning : will be adapted when block size is known (*block_size)
                        sample_size_btstrp = "tbd" # Warning : will be adapted when block size is known (=block_size)
                        std, est = method_function(method, sample, nb_bootstrap_steps,
                                                    sample_size_btstrp, 
                                                    order_stat_ratio_btstrp)
                    else:
                        btstrp_name = method_bootstrap["method_name"]
                        raise ValueError(f"Method bootstrap unknown: {btstrp_name}")
                    list_est.append(est)
                    list_std.append(std)
                aver_est = np.mean(list_est)
                list_var= [sd**2 for sd in list_std]
                aver_var = np.mean(list_var)
                aver_std = aver_var**(1/2)
                # TODO: Maybe adapt for de haan and zhou : here we average the variances
                # TODO: Should we instead compute the "total variance" among all substracted estimates
                # TODO: (instead)

            dict_results[method_name]["est_list"].append(aver_est)
            dict_results[method_name]["std_list"].append(aver_std)
    ## PLOTS
    gamma_plots_in_front_of_k_order_stats(dict_results, inv_gamma,
                                        config_path, config, path_run)

def run_bm_est_and_var_vs_block_size_dep_data_from_config(config_path, config, path_run):
    list_methods = config["run_est_and_var_vs_k_dep_data_from_config"]["list_methods"]
    inv_gamma = config["run_est_and_var_vs_k_dep_data_from_config"]["inv_gamma"]
    extra_block_sizes = config["run_est_and_var_vs_k_dep_data_from_config"]["extra_block_sizes"]
    nb_monte_carlo_steps = config["run_est_and_var_vs_k_dep_data_from_config"]["nb_monte_carlo_steps"]
    nb_bootstrap_steps = config["run_est_and_var_vs_k_dep_data_from_config"]["nb_bootstrap_steps"]
    nb_averaging_btstrp = config["run_est_and_var_vs_k_dep_data_from_config"]["nb_averaging_btstrp"]
    distribution_type = config["run_est_and_var_vs_k_dep_data_from_config"]["distribution_type"]
    ma_order = config["run_est_and_var_vs_k_dep_data_from_config"]["ma_order"]
    size_sample = config["run_est_and_var_vs_k_dep_data_from_config"]["size_sample"]
    method_name_to_function = {
            "get_bootstrap_variance_est_de_hann_98": get_bootstrap_variance_est_de_hann_98,
            "get_bootstrap_variance_est_kulik": get_bootstrap_variance_est_kulik,
            "get_monte_carlo_variance_est": get_monte_carlo_variance_est,
            "get_unbiaised_monte_carlo_variance_est_gomes": get_unbiaised_monte_carlo_variance_est_gomes,
            "get_bootstrap_variance_est_de_haan_zhou": get_bootstrap_variance_est_de_haan_zhou
        }
    # distribution_type_to_function = {
    #     "frechet_iid": lambda size_smple: get_frechet_sample(inv_gamma, size_smple),
    #     "ma": lambda size_smple: get_ma_sample(inv_gamma, size_smple, ma_order),
    #     "arch1": lambda size_smple: get_arch1_sample(inv_gamma, size_smple,),
    # }
    dict_results = dict()


    # sample = get_distrib(size_sample)
    # Maybe decomment if we use auto_k_order
    # try: # try to get auto k_order twice with two different bootstrap_sample
    #     auto_k_order_for_this_sample = int(de_haan_1998(sample))
    #     gamma_moment_1_estimator(auto_k_order_for_this_sample,sample)
    # except:
    #     sample = get_distrib(size_sample)
    #     auto_k_order_for_this_sample = int(de_haan_1998(sample))
    #     gamma_moment_1_estimator(auto_k_order_for_this_sample,sample)
    for method in list_methods:
        method_function_name = method["method"]
        method_name = method["method_name"]
        method_function = method_name_to_function[method_function_name]
        method_bootstrap = method["method_bootstrap"]
        dict_results[method_name] = {
            "block_size": list(),
            "est_list": list(),
            "std_list": list()
        }

        # Maybe decomment if we use auto_k_order
        # if method_k_order_statistics_ratio == "version2": # We use auto k order statistics
        #     method_k_order_statistics = auto_k_order_for_this_sample
        # else: # We use defined order statistics
        for bm_block_size in tqdm(range(1, ma_order+extra_block_sizes,2)):
            # TODO remplacer par boucle sur block size
            if distribution_type == "block_maxima_ma":
                get_distrib = lambda size_smple: get_block_max(inv_gamma, size_smple, bm_block_size, 
                                                               ma_order, dependency="ma")
            if distribution_type == "block_maxima_max":
                # import pdb
                # pdb.set_trace()
                get_distrib = lambda size_smple: get_block_max(inv_gamma, size_smple, bm_block_size, 
                                                               ma_order, dependency="max")
            k_order_statistics_percentage = 10
            ratio =  k_order_statistics_percentage/100
            method_k_order_statistics = int(size_sample*ratio)
            dict_results[method_name]["block_size"].append(bm_block_size)

            if method_function_name== "get_monte_carlo_variance_est": # if mc: draw new samples and compute mc
                aver_std, aver_est = get_monte_carlo_variance_est(nb_monte_carlo_steps,
                                size_sample,
                                get_distrib,
                                method_k_order_statistics)
            elif method_function_name=="get_unbiaised_monte_carlo_variance_est_gomes":
                aver_std, aver_est = get_unbiaised_monte_carlo_variance_est_gomes(nb_monte_carlo_steps,
                                size_sample,
                                get_distrib,
                                method_k_order_statistics)
            else: # if bootstrap : compute boostrap_sample
                list_std = list()
                list_est = list()
                for i in range(nb_averaging_btstrp):
                    sample = get_distrib(size_sample)
                    # We smooth the btstrap on several samples
                    if method_bootstrap["method_name"] == "iid":
                        size_sample_bootstrap_ratio = method_bootstrap["size_sample_bootstrap_ratio"]
                        order_stat_ratio_btstrp = int(method_k_order_statistics*size_sample_bootstrap_ratio)
                        sample_size_btstrp = int(size_sample*size_sample_bootstrap_ratio)
                        std, est = method_function(method, sample, nb_bootstrap_steps,
                                                    sample_size_btstrp, 
                                                    order_stat_ratio_btstrp)
                        # list_est.append(est)
                        # list_std.append(std)
                    # if method_bootstrap["method_name"] == "iid":
                    #     size_sample_bootstrap_ratio = method_bootstrap["size_sample_bootstrap_ratio"]
                    #     order_stat_ratio_btstrp = int(method_k_order_statistics*size_sample_bootstrap_ratio)
                    #     sample_size_btstrp = int(size_sample*size_sample_bootstrap_ratio)
                    #     std, est = method_function(method, sample, nb_bootstrap_steps,
                    #                                 sample_size_btstrp, 
                    #                                 order_stat_ratio_btstrp)
                    elif method_bootstrap["method_name"] == "stationary":
                        size_sample_bootstrap_ratio = method_bootstrap["size_sample_bootstrap_ratio"]
                        order_stat_ratio_btstrp = method_k_order_statistics # Warning : will be adapted when block size is known (*block_size)
                        sample_size_btstrp = "tbd" # Warning : will be adapted when block size is known (=block_size)
                        std, est = method_function(method, sample, nb_bootstrap_steps,
                                                    sample_size_btstrp, 
                                                    order_stat_ratio_btstrp)
                    else:
                        btstrp_name = method_bootstrap["method_name"]
                        raise ValueError(f"Method bootstrap unknown: {btstrp_name}")
                    list_est.append(est)
                    list_std.append(std)
                aver_est = np.mean(list_est)
                list_var= [sd**2 for sd in list_std]
                aver_var = np.mean(list_var)
                aver_std = aver_var**(1/2)
                # TODO: Maybe adapt for de haan and zhou : here we average the variances
                # TODO: Should we instead compute the "total variance" among all substracted estimates
                # TODO: (instead)

            dict_results[method_name]["est_list"].append(aver_est)
            dict_results[method_name]["std_list"].append(aver_std)
    ## PLOTS

    gamma_plots_in_front_of_block_size(dict_results, inv_gamma,
                                        config_path, config, path_run)


def gamma_plots_in_front_of_block_size(dict_results, inv_gamma,
                                    config_path, config, path_run):
    # gamma_est and gamma_std same plot
    fig, ax = plt.subplots()
    for name, d in dict_results.items():
        name_key = name + "_est"
        ax.errorbar(d["block_size"], d["est_list"], yerr=d["std_list"], 
                    capsize=5, 
                    alpha=0.7, label=name_key)
    ax.set_title("Gamma_estimation_in_front_of_block_size")
    ax.legend(framealpha=0.1)
    plt.savefig(path_run+"\\Gamma_estimation_in_front_of_block_size.jpg")
    plt.clf()
    # plot gamma_biais (only for monte carlo estimations)
    for name, d in dict_results.items():
        name_key = name + "_biais"
        true_gamma = 1/inv_gamma
        biais_list = [np.abs(g - true_gamma) for g in d["est_list"]]
        plt.plot(d["block_size"], biais_list, 
                    alpha=0.7, label=name_key)
    plt.title("Gamma_biais_in_front_of_block_size")
    plt.legend(framealpha=0.1)
    plt.savefig(path_run+"\\Gamma_biais_in_front_of_block_size.jpg")
    plt.clf()
    # plot gamma_std
    for name, d in dict_results.items():
        name_key = name + "_std"
        plt.plot(d["block_size"], d["std_list"], 
                    alpha=0.7, label=name_key)
    plt.title("Gamma_std_in_front_of_block_size")
    plt.legend(framealpha=0.1)
    plt.savefig(path_run+"\\Gamma_std_in_front_of_block_size.jpg")
    plt.clf()
    # plot gamma_biais/gamma_std
    for name, d in dict_results.items():
        name_key = name + "_biais"
        true_gamma = 1/inv_gamma
        biais_list = [np.abs(g - true_gamma) for g in d["est_list"]]
        biais_divided_by_std = [biais / sd 
                                for biais, sd in zip(biais_list,d["std_list"])]
        plt.plot(d["block_size"], biais_divided_by_std, 
                    alpha=0.7, label=name_key)
    plt.title("Gamma_biais_on_std_in_front_of_block_size")
    plt.legend(framealpha=0.1)
    plt.savefig(path_run+"\\Gamma_biais_on_std_in_front_of_block_size.jpg")
    plt.clf()


def run_config_file(config_path):
    path_run, time_hash, config = save_results_initialization(config_path)
    function_to_run = config["function_to_run"]
    if function_to_run == "run_variance_vs_size_sample_from_config":
        run_variance_vs_size_sample_from_config(config_path, config, path_run)
    if function_to_run == "run_variance_vs_downsampling_from_config":
        run_variance_vs_downsampling_from_config(config_path, config, path_run)
    if function_to_run == "run_de_hann_zhou_experience_from_config":
        run_de_hann_zhou_experience_from_config(config_path, config, path_run)
    if function_to_run == "run_test_de_hann_98_accuracy_with_de_hann_zhou_experience":
        run_test_de_hann_98_accuracy_with_de_hann_zhou_experience(config_path, config, path_run)
    if function_to_run == "run_est_and_var_vs_k_dep_data_from_config":
        run_est_and_var_vs_k_dep_data_from_config(config_path, config, path_run)
    if function_to_run == "run_bm_est_and_var_vs_block_size_dep_data_from_config":
        run_bm_est_and_var_vs_block_size_dep_data_from_config(config_path, config, path_run)