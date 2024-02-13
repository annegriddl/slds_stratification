import json
import numpy as np
import pandas as pd
import os
#import argparse

from utils_evaluation import flatten_data, generate_hyperparameter_combinations_dict, descreptives, save_results_to_csv, csv_to_list, flatten_nested_lists, error_estimator, grouped_bar_plot_hyperparameters

# remove waringns
import warnings
warnings.filterwarnings("ignore")

###########################################################################################################
###### Load json files ######
# set file path to json files (change model) -> get names of json files
model_name = 'rf' # @param: needs to be changed to the model name
json_path = "/Users/nadja/Documents/University/Master_Statistik/WS2023_24/SLDS/slds_stratification/Simulations/Final_Setup/results/" + model_name + "/"
json_files = os.listdir(json_path)

# load data from json file
data = []
for file in json_files:
    with open(json_path + file) as f:
        data_new = json.load(f)
        data = data + data_new

# flatten dictionary and convert to dataframe
data_all_flatten, keys_dic = flatten_data(data)
data = pd.DataFrame(data_all_flatten)

print('\nData Loading from json file:')
print('Loaded data from jason file: shape', data.shape)
###########################################################################################################



###########################################################################################################
###### Filter for one experimental parameter combination ######
parameter_grid = {
    "model_info_model": [model_name],
    "model_info_transformation": ['identity', 'log', 'sqrt'],
    "model_info_n_train": [200, 1000],
    "model_info_group_size": [5, 10],
    "model_info_noise": [0, 3]
}

parameter_combinatons = generate_hyperparameter_combinations_dict(parameter_grid)
print('In total', len(parameter_combinatons), 'parameter combinations')

# iterate over each parameter combination
for parameter_combination in parameter_combinatons:
    # First create unique identifier of parameter combination for saving results in csv
    parameter_combination_string = ''
    for key, value in parameter_combination.items():
        parameter_combination_string = parameter_combination_string + '_' + str(value)
    parameter_combination_string = parameter_combination_string[1:]
    parameter_combination_string

    # filter data for experimental parameter combination
    filtered_data = data[
        (data['model_info_model'] == parameter_combination['model_info_model']) &
        (data['model_info_transformation'] == parameter_combination['model_info_transformation']) &
        (data['model_info_n_train'] == parameter_combination['model_info_n_train']) &
        (data['model_info_group_size'] == parameter_combination['model_info_group_size']) &
        (data['model_info_noise'] == parameter_combination['model_info_noise'])
    ]

    print("\n--------------------------------------------------------")
    print('experimental parameter combination:', parameter_combination, 'with shape', filtered_data.shape)
    print("--------------------------------------------------------")

    # create folders for saving results per unique parameter_combination
    path_evaluation_tables = "/Users/nadja/Documents/University/Master_Statistik/WS2023_24/SLDS/slds_stratification/Simulations/Final_Setup/Evaluation/tables/"  + model_name + "/" 
    path_evaluation_plots = "/Users/nadja/Documents/University/Master_Statistik/WS2023_24/SLDS/slds_stratification/Simulations/Final_Setup/Evaluation/plots/"  + model_name + "/" + parameter_combination_string + "/" 

    if not os.path.exists(path_evaluation_tables):
        os.makedirs(path_evaluation_tables)
        print('Folder for tables created:', parameter_combination_string)

    if not os.path.exists(path_evaluation_plots):
        os.makedirs(path_evaluation_plots)
        print('Folder for plots created:', parameter_combination_string)
    ###########################################################################################################
        


    ###########################################################################################################
    ##### Evaluation per parameter combinaiton (see fiterling above) #####
    ### Number of experiments
    num_exp = filtered_data.shape[0]
    print('Number of experiments:', num_exp)


    ###### 1. Analysis: Selected Best Hyperparameters ######
    ###  Investigate 'hyperparameters_same'
    hyp_same = filtered_data['hyperparameters_same'].value_counts()
    hyp_different_rel = hyp_same[0]/(hyp_same[0]+hyp_same[1])
    print('hyp_different:', hyp_same[0], '\nhyp_same:', hyp_same[1], '\nrelative difference:', hyp_different_rel)

    ### Investigate hyperparameters of RandomSearch: hyperparameters_RS
    if model_name == 'rf':
        hyperparameters_RS = ['min_samples_split', 'min_samples_leaf', 'max_features']
    elif model_name == 'xgb':
        hyperparameters_RS = ['learning_rate', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'gamma']
    else:
        print('Model not implemented')

    for hypRS in hyperparameters_RS:
        num_unique_stratified, counts_stratified, num_unique_unstratified, counts_unstratifed = grouped_bar_plot_hyperparameters(filtered_data['stratified_best_params_' + hypRS], 
                                                                                                                                 filtered_data['unstratified_best_params_' + hypRS],
                                                                                                                                   hypRS, path_evaluation_plots )
        
    ###### 2. Final Performance ######
    ''' variables: 'unstratified_results_train r2', 'unstratified_results_test r2',
       'unstratified_results_train mse', 'unstratified_results_test mse',
       'unstratified_results_train mae', 'unstratified_results_test mae',
       'stratified_results_train r2', 'stratified_results_test r2',
       'stratified_results_train mse', 'stratified_results_test mse',
       'stratified_results_train mae', 'stratified_results_test mae','''


    ###### 3. Performance within cross-validation ######
    ### Mean MSE from RandomSearch
    RandomSearch_Mean_Val_MSE_unstratified = abs(np.mean(filtered_data['cv_unstratified_iterations_mean_test_score'].explode().tolist()))
    RandomSearch_Mean_Val_SD_unstratified = np.sqrt(np.var(filtered_data['cv_unstratified_iterations_std_test_score'].explode().tolist()))

    RandomSearch_Mean_Val_MSE_stratified = abs(np.mean(filtered_data['cv_stratified_iterations_mean_test_score'].explode().tolist()))
    RandomSearch_Mean_Val_SD_stratified = np.sqrt(np.var(filtered_data['cv_stratified_iterations_std_test_score'].explode().tolist()))

    RandomSearch_Mean_Val_MSE_diff = RandomSearch_Mean_Val_MSE_stratified - RandomSearch_Mean_Val_MSE_unstratified

    print(f"Unstratified: Mean MSE RandomSearch {RandomSearch_Mean_Val_MSE_unstratified} with sd of  {RandomSearch_Mean_Val_SD_unstratified}")
    print(f"Stratified: Mean MSE RandomSearch {RandomSearch_Mean_Val_MSE_stratified} with sd of {RandomSearch_Mean_Val_SD_stratified}")
    print(f"Difference: {RandomSearch_Mean_Val_MSE_diff}")



    ###### 4. Generalisation error ######
    ### Error Estimation
    error_estimator_result = error_estimator(filtered_data, path_evaluation_plots)
    error_estimator_list_name, error_estimator_values = csv_to_list(error_estimator_result, title = '')




    ##### 5.Descreptives in cross-validation ######
    # Descreptives: 'ks_statistic', 'p_value', 'intersection_area'
    val_train_descriptives = descreptives(filtered_data, path_evaluation_plots)
    val_train_descriptives_list_name, val_train_descriptives_list_values = csv_to_list(val_train_descriptives, title = 'val_train_descriptives_')
    ###########################################################################################################


    ###########################################################################################################
    ### save results to one csv for each parameter_combination
    header = ['parameter_combination_string', 'num_exp',
            'hyperparameter_different', 'hyperparameter_same', 'hyperparameter_different_rel', # best hyperparameter same ?
            val_train_descriptives_list_name, # val_train_descriptives_
            'RandomSearch_Mean_Val_MSE_unstratified', 'RandomSearch_Mean_Val_MSE_stratified', 'RandomSearch_Mean_Val_SD_unstratified', 'RandomSearch_Mean_Val_SD_stratified', 'RandomSearch_Mean_Val_MSE_diff', #RandomSearch_Mean_Val
            error_estimator_list_name, # error_estimator
            'num_unique_stratified', 'counts_stratified', 'num_unique_unstratified', 'counts_unstratifed',
            ] 
    values = [parameter_combination_string, num_exp, hyp_same[0], hyp_same[1], hyp_different_rel, 
            val_train_descriptives_list_values, # @Nadja: einmal 1.0, 0.0,
            RandomSearch_Mean_Val_MSE_unstratified, RandomSearch_Mean_Val_MSE_stratified, RandomSearch_Mean_Val_SD_unstratified, RandomSearch_Mean_Val_SD_stratified, RandomSearch_Mean_Val_MSE_diff,
            error_estimator_values,
            np.array(num_unique_stratified), np.array(counts_stratified), np.array(num_unique_unstratified), np.array(counts_unstratifed)] # convert to numpy to not flatten in flatten_nested_lists

    #flatten lists val_train_descriptives_list_name and error_estimator_list_name
    header = flatten_nested_lists(header)
    values = flatten_nested_lists(values)

    save_results_to_csv(file_path = path_evaluation_tables + '/results_per_parameter.csv', 
                        header = header,
                        values = values)
    print("\n")




#if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--model_name", type=str, default='rf', help="rf or xgb")    
    #model_name = parser.parse_args()          