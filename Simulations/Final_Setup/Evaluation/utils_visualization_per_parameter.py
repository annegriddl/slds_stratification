import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os
import csv


def flatten_data(data_all):
    # Flatten the nested dictionaries
    flat_data = []
    key_dic = []
    for entry in data_all:
        flat_entry = {}
        for key, value in entry.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_entry[key + "_" + sub_key] = sub_value
                    key_dic.append(key)

            else:
                flat_entry[key] = value
        flat_data.append(flat_entry)

    df = pd.DataFrame(flat_data)
    return df, list(set(key_dic))


def generate_hyperparameter_combinations_dict(hyperparameter_options):
    """
    Generate all possible combinations of hyperparameters.

    Parameters:
    hyperparameter_options (dict): Dictionary where keys are hyperparameter names and values are lists of options.

    Returns:
    List of dictionaries, where each dictionary represents a combination of hyperparameters.
    """
    hyperparameter_names = hyperparameter_options.keys()
    all_hyperparameter_combinations = list(itertools.product(*hyperparameter_options.values()))

    all_hyperparameter_dicts = []
    for combination in all_hyperparameter_combinations:
        hyperparameter_dict = dict(zip(hyperparameter_names, combination))
        all_hyperparameter_dicts.append(hyperparameter_dict)

    return all_hyperparameter_dicts


def descreptives(data, path_plots):
    statistics = ['ks_statistic', 'p_value', 'intersection_area']
    mean_stratified_list = []
    sd_stratified_list = []
    mean_unstratified_list = []
    sd_unstratified_list = []
    mean_diff = []
    sd_diff = []

    for s in statistics:
        # filter stratified and unstratified descreptives columns
        s_stratified = '_stratified_' + s
        s_unstratified = '_unstratified_' + s
        s_stratified = [col for col in data.columns if s_stratified in col]
        s_unstratified = [col for col in data.columns if s_unstratified in col]

        plot_combined_boxplots(data_x= data[s_stratified], data_y = data[s_unstratified], title= s, path=path_plots + s)
        
        # values for stratified
        mean_stratified = data[s_stratified].mean()[0]
        sd_stratified = data[s_stratified].std()[0]
        mean_stratified_list.append(mean_stratified)
        sd_stratified_list.append(sd_stratified)

        # values for unstratified
        mean_unstratified = data[s_unstratified].mean()[0]
        sd_unstratified = data[s_unstratified].std()[0]
        mean_unstratified_list.append(mean_unstratified)
        sd_unstratified_list.append(sd_unstratified)

        # diference
        mean_diff.append(mean_stratified - mean_unstratified)
    df_result = pd.DataFrame({'mean_stratified': mean_stratified_list, 'sd_stratified': sd_stratified_list, 
                              'mean_unstratified': mean_unstratified_list, 'sd_unstratified': sd_unstratified_list, 
                              'mean_diff' : mean_diff}, 
                              index= statistics)
    
    return df_result



def plot_combined_boxplots(data_x, data_y, title, path):
    # check data format
    data_x = np.array(data_x).flatten()
    data_y = np.array(data_y).flatten()
    # plot boxplots
    _, ax = plt.subplots(figsize=(7, 3))
    ax.boxplot([data_x, data_y], vert=False, labels=['stratified', 'unstratified'])
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    #plt.show()

def csv_to_list(val_train_descriptives, title):
    # save results from def descreptives for val-train comparison
    val_train_descriptives_list_name = []
    val_train_descriptives_list_value = []
    # Iterate over rows
    for row_name in val_train_descriptives.index:
        # Iterate over columns
        for col_name in val_train_descriptives.keys():
            # Combine column and row names to get the value
            value = val_train_descriptives[col_name][row_name]
            val_train_descriptives_string = f"{title}{row_name}_{col_name}"
            val_train_descriptives_list_name.append(val_train_descriptives_string)
            val_train_descriptives_list_value.append(value)
    return val_train_descriptives_list_name, val_train_descriptives_list_value


def save_results_to_csv(file_path, header, values):
    if not os.path.exists(file_path): # file does not exist
        with open(file_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
        
    with open(file_path, 'a', newline='') as f: # file exists, append row
        csv_writer = csv.writer(f)
        csv_writer.writerow(values)


def flatten_nested_lists(data):
    #val_train_descriptives_list_name, error_estimator_list_name
    flattened_data = []
    for sublist in data:
        if isinstance(sublist, list):
            flattened_data.extend(sublist)
        else:
            flattened_data.append(sublist)
    return flattened_data




def error_estimator(data, path):
    error_estimate_stratified_list = []
    error_estimate_unstratified_list = []

    for i in range(len(data)):
        # generalisation error: stratified
        error_estimate_stratified = np.array(data['cv_iteration_refit_test_mse'])[i] + np.array(data['cv_stratified_iterations_mean_test_score'])[i]
        error_estimate_stratified_list.append(error_estimate_stratified)
        # generalisation error: unstratified
        error_estimate_unstratified = np.array(data['cv_iteration_refit_test_mse'])[i] + np.array(data['cv_unstratified_iterations_mean_test_score'])[i]
        error_estimate_unstratified_list.append(error_estimate_unstratified)

    mean_stratified = np.mean(error_estimate_stratified_list)
    sd_stratified = np.sqrt(np.var(error_estimate_stratified_list))
    mean_unstratified = np.mean(error_estimate_unstratified_list)
    sd_unstratified = np.sqrt(np.var(error_estimate_unstratified_list))
    mean_diff = mean_stratified - mean_unstratified

    #data frame
    df_result = pd.DataFrame({'mean_stratified': mean_stratified, 'sd_stratified': sd_stratified, 
                                'mean_unstratified': mean_unstratified, 'sd_unstratified': sd_unstratified,
                                'mean_diff': mean_diff},
                                index= ['error_estimate'])
    print(df_result)


    # plot error_estimate_stratified and error_estimate_unstratified in two boxplots next to each other
    plot_combined_boxplots(error_estimate_stratified_list, error_estimate_unstratified_list, title = 'Error of Estimator', path= path + 'error_estimator.png')
    return df_result



def grouped_bar_plot_hyperparameters(data_stratified, data_unstratified, hyperparameter, path_evaluation_plots):
    data_stratified = list(data_stratified)
    data_unstratified = list(data_unstratified)

    unique_values1 = np.unique(data_stratified) #sorted unique elements of an array.
    unique_values2 = np.unique(data_unstratified)

    num_unique1 = len(unique_values1)
    num_unique2 = len(unique_values2)

    counts_stratified = [data_stratified.count(val) for val in unique_values1]
    counts_unstratifed = [data_unstratified.count(val) for val in unique_values2]

    width = 0.35  # Width of each bar

    fig, ax = plt.subplots()
    bars1 = ax.bar(np.arange(num_unique1), counts_stratified , width, label= 'stratified') 
    bars2 = ax.bar(np.arange(num_unique2) + width, counts_unstratifed, width, label= 'unstratified')

    ax.set_xlabel(hyperparameter +' values')
    ax.set_ylabel('Count')
    ax.set_xticks(np.arange(num_unique1) + width / 2)
    ax.set_xticklabels(unique_values1)
    ax.legend()
    plt.savefig(path_evaluation_plots + hyperparameter + '_grouped_bar_plot')
    return np.arange(num_unique1), counts_stratified, np.arange(num_unique2), counts_unstratifed

