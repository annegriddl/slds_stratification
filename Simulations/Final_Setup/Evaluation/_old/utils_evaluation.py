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
    _, ax = plt.subplots(figsize=(8, 3))
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
        # Note: cv_stratified_iterations_mean_test_score is negagtive MSE, wherase cv_iteration_refit_test_mse is absolute MSE
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
                                index= ['error_estimator'])
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

    fig, ax = plt.subplots(figsize=(6, 3))  # Set the figure size

    bars1 = ax.bar(np.arange(num_unique1), counts_stratified , width, label= 'stratified') 
    bars2 = ax.bar(np.arange(num_unique2) + width, counts_unstratifed, width, label= 'unstratified')

    ax.set_xlabel(hyperparameter +' values')
    ax.set_ylabel('Count')
    ax.set_xticks(np.arange(num_unique1) + width / 2)
    ax.set_xticklabels(unique_values1)
    ax.legend()
    plt.savefig(path_evaluation_plots + hyperparameter + '_grouped_bar_plot')
    return np.arange(num_unique1), counts_stratified, np.arange(num_unique2), counts_unstratifed


# for summary_evualation
def colours_scheme(df, experimental_parameter):
    colours = []
    for value in df[experimental_parameter]:
        if experimental_parameter == 'param_transformation':
            # blue values
            if value == 'log':
                col = '#33E6FF'
            elif value == 'sqrt':
                col = '#0A7685'
            elif value == 'identity':
                col = '#2F52DF'
            else:
                raise ValueError('Error: No valid transformation')
            colours.append(col)
            set_colours = [ '#33E6FF', '#0A7685', '#2F52DF']
            legend_labels = ['log', 'sqrt', 'identity']
        elif experimental_parameter == 'param_n_train':
            # green values
            if value == 200:
                col = '#2FDF54'
            elif value == 1000:    
                col = '#16892E'
            else:
                raise ValueError('Error: No valid n_train')
            colours.append(col)
            set_colours = ['#2FDF54', '#16892E']
            legend_labels = ['200', '1000']
        elif experimental_parameter == 'param_group_size':
            # lila
            if value == 5:
                col = '#6D1689'
            elif value == 10:
                col = '#A21FCC'
            else:
                raise ValueError('Error: No valid group_size')
            colours.append(col)
            set_colours = ['#6D1689', '#A21FCC']
            legend_labels = ['5', '10']
        elif experimental_parameter == 'param_noise':
            # orange
            if value == 0:
                col = '#E77A0D'
            elif value == 3:
                col = '#EAA560'
            else:
                raise ValueError('Error: No valid group_size')
            colours.append(col)
            set_colours = ['#E77A0D', '#EAA560']
            legend_labels = ['0', '3']
        else: #error
            raise ValueError('Error: No valid experimental parameter')
    return colours, set_colours, legend_labels


def barplot_coloured_by_parameter(data, experimental_parameter, variable_y, title, variable_y_title):
   colors , set_colours, legend_labels =  colours_scheme(data.sort_values(variable_y), experimental_parameter)
   
   plt.figure(figsize=(8, 3))
   ax = sns.barplot(x='parameter_combination_string', y=variable_y, data=data, order=data.sort_values(variable_y)["parameter_combination_string"], palette=colors)
   ax.set_ylabel(variable_y_title)
   ax.set_xlabel('Experimental Parameter Combination')
   ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
   ax.set_title(title)
   
   # Add legend
   legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in  set_colours]
   ax.legend(legend_handles, legend_labels, title=experimental_parameter, bbox_to_anchor=(1, 1))
   
   plt.show()


def  barplot_one_var(df, var, title, y_label):
    df_sorted = df.sort_values(by=var)

    if title == 'Random Forest':
        color = 'darkgreen'
    else:
        color = 'green'

    # Barplot
    plt.figure(figsize=(10, 3))  # Adjust figure size as needed
    plt.bar(df_sorted['parameter_combination_string'], df_sorted[var], color= color)
    plt.xlabel('Experimental Parameter Combination')
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(df_sorted['parameter_combination_string'], rotation=90)  #
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Boxplot
    plt.figure(figsize=(10, 3))  # Adjust figure size as needed
    sns.boxplot(x=var, data=df_sorted, color='darkgreen')
    plt.xlabel(y_label)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # stats
    stats = df_sorted[var].describe()
    return stats