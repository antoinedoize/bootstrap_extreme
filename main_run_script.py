from main_functions import *


def main():
    # Test de_hann_98_with_extreme_subsampling_sizes (10% 50% 90%)
    path_to_config = 'c:\\Users\\Antoine\\Desktop\\JupyterNotebooks\\bootstrap_extreme\\config_files\\'
    # run_config_file(path_to_config + "de_hann_98_extreme_subsample_size.json")
    # run_config_file(path_to_config + "2023_01_31_chekcpoint_all_methods.json")
    # run_config_file(path_to_config + "2023_01_31_de_hann_variance_of_variance.json")
    # run_config_file(path_to_config + "2023_01_31_de_hann_variance_of_variance_auto_k_order.json")



    # Test de_hann_&_zhou
    # run_config_file(path_to_config + "2023_02_13_de_hann_&_zhou_check_factor_2.json")
    # run_config_file(path_to_config + "2023_02_14_de_hann_&_zhou_check_de_hann_98_performance.json")

    # Correction kulik
    # run_config_file(path_to_config + "2023_01_31_chekcpoint_all_methods.json")

    # 14/03 Run toutes méthodes avec données dépendantes
    # run_config_file(path_to_config + "2023_03_14_all_methods_dependant_data_iid_case.json")
    # run_config_file(path_to_config + "2023_04_03_all_methods_dependant_data_ma.json")
    # run_config_file(path_to_config + "2023_04_03_all_methods_dependant_data_arch1.json")

    # 24/04 Run avec integration de de_hann _and_zhou estimation
    # run_config_file(path_to_config + "2023_04_24_all_methods_dependant_data_iid_case.json")
    # run_config_file(path_to_config + "2023_04_24_all_methods_dependant_data_ma2.json")
    # run_config_file(path_to_config + "2023_04_24_all_methods_dependant_data_ma5.json")
    # run_config_file(path_to_config + "2023_04_24_all_methods_dependant_data_ma10.json")
    # run_config_file(path_to_config + "2023_04_24_all_methods_dependant_data_arch1.json")

    # 25/04 Run test kulik en fonction de block size custom, nouveaux plots, variance de variance, stationary bootstrap
    # Custom block size
    # run_config_file(path_to_config + "2023_04_25_kulik_block_size_ma2.json")
    # run_config_file(path_to_config + "2023_04_25_kulik_block_size_ma5.json")
    # run_config_file(path_to_config + "2023_04_25_kulik_block_size_ma10.json")
    # run_config_file(path_to_config + "2023_04_25_kulik_block_size_ma50.json")

    # 26/04 Run stationary bootstrap
    # run_config_file(path_to_config + "2023_04_26_all_methods_stationary_btstrap_ma2_v8.json")
    # run_config_file(path_to_config + "2023_04_26_all_methods_stationary_btstrap_ma30_block_smaller.json")
    # run_config_file(path_to_config + "2023_04_26_all_methods_stationary_btstrap_ma30_block_equal.json")
    # run_config_file(path_to_config + "2023_04_26_all_methods_stationary_btstrap_ma30_block_greater.json")
    # run_config_file(path_to_config + "2023_04_26_all_methods_stationary_btstrap_ma30_block_very_greater.json")
    # 26/04 Run stationary bootstrap focus on de_hann_98
    # run_config_file(path_to_config + "2023_04_26_de_haan_98_focus_iid.json")
    # run_config_file(path_to_config + "2023_04_26_de_haan_98_focus_deterministic_blocks.json")
    # run_config_file(path_to_config + "2023_04_26_de_haan_98_focus_geom_blocks.json")
    # 26/04 Run on arch1 data
    # run_config_file(path_to_config + "2023_04_26_all_methods_stationary_btstrap_arch1_block_smaller.json")
    # run_config_file(path_to_config + "2023_04_26_all_methods_stationary_btstrap_arch1_block_equal.json")
    # run_config_file(path_to_config + "2023_04_26_all_methods_stationary_btstrap_arch1_block_greater.json")
    # run_config_file(path_to_config + "2023_04_26_all_methods_stationary_btstrap_arch1_block_very_greater.json")

    # 08/05 Run unbiaised estimator gomes + biais mc + var
    # run_config_file(path_to_config + "2023_05_08_all_methods_stationary_btstrap_ma5_block_30.json")
    # run_config_file(path_to_config + "2023_05_08_all_methods_stationary_btstrap_ma10_block_30.json")
    # run_config_file(path_to_config + "2023_05_08_all_biased_methods_stationary_btstrap_ma5_block_30.json")
    # run_config_file(path_to_config + "2023_05_08_all_biased_methods_stationary_btstrap_ma10_block_30.json")
    # 09/05 Run unbiaised estimator gomes on several gamma
    # run_config_file(path_to_config + "2023_05_09_unbiaised_est_inva_gamma_0_3_ma10.json")
    # run_config_file(path_to_config + "2023_05_09_unbiaised_est_inva_gamma_0_5_ma10.json")
    # run_config_file(path_to_config + "2023_05_09_unbiaised_est_inva_gamma_1_ma10.json")
    # run_config_file(path_to_config + "2023_05_09_unbiaised_est_inva_gamma_5_ma10.json")
    # run_config_file(path_to_config + "2023_05_09_unbiaised_est_inva_gamma_10_ma10.json")
    # run_config_file(path_to_config + "2023_05_09_unbiaised_est_inva_gamma_3_ma30.json")

    # 30/05 Run estimators on block maxima data
    # run_config_file(path_to_config + "2023_05_30_all_methods_bm_ma_inv_gamma_3_ma20.json")
    # run_config_file(path_to_config + "2023_05_30_all_methods_bm_max_inv_gamma_3_ma20.json")

    #26/08 Redaction of final beamer
    # run_config_file(path_to_config + "2023_08_26_config_file_checkepoint_iid_data.json")
    # run_config_file(path_to_config + "2023_08_26_config_file_checkepoint_ma2.json")
    # run_config_file(path_to_config + "2023_08_26_config_file_checkepoint_ma5.json")
    # run_config_file(path_to_config + "2023_08_26_config_file_checkepoint_ma10.json")


if __name__ == "__main__":
    main()