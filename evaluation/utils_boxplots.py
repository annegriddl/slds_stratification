import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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







def plot_eval(value_vars , value_name, data, model_vars_title, transformation = 'None', model = 'None', figsize = (6, 4)):
        '''
        value_vars: list of strings, names of columns to be plotted for stratified and unstratified ['error_estimator_mean_stratified', 'error_estimator_mean_unstratified']. 
                        Important: stratified and unstratified must be written in the end after '_' otherwise automatic labeling won't work and you'll get an error
                        Important: stratifed first element, unstratified second element
        value_name: string, name of the value variable to be plotted. Basically name of value_vars that is plotted on the y-axis
        differences_table_all: pd.DataFrame, table of differences in mean and sd of stratified and unstratified , output of function with plots
        '''
        plots_report_path = '/Users/nadja/Documents/University/Master_Statistik/WS2023_24/SLDS/slds_stratification/Simulations/Final_Setup/Evaluation/plots/report_plots'
        differences_table_all = pd.DataFrame()
        # make data long
        data_long = data.melt(id_vars=['param_model'], 
                                                value_vars= value_vars , 
                                                var_name='Stratification', value_name= value_name)

        # definer ordering of boxplots
        filtered_data = data_long[data_long['Stratification'] ==  value_vars[1]]
        mean_intersection = filtered_data.groupby(['param_model'])[value_name].mean().reset_index().sort_values(by= value_name, ascending=False)

        ### Deacreptives
        descriptives = data_long.groupby(['param_model', 'Stratification']).describe()
        keys = descriptives[value_name]['mean'].keys()
        # descriptives['Intesection']['count'].values  # check N per boxplot
        descriptives_table = pd.DataFrame({'Expermintel Hyperparameter Combinaiton': keys.get_level_values('param_model'), 
                                                'Stratification': keys.get_level_values('Stratification').str.rsplit('_', n=1).str[-1], 
                                                'Mean': descriptives[value_name]['mean'].values, 
                                                'SD': descriptives[value_name]['std'].values})
        difference_mean =  descriptives_table[descriptives_table['Stratification']==  'stratified']['Mean'].values - descriptives_table[descriptives_table['Stratification']==  'unstratified']['Mean'].values
        difference_sd =  descriptives_table[descriptives_table['Stratification']==  'stratified']['SD'].values - descriptives_table[descriptives_table['Stratification']==  'unstratified']['SD'].values
        differences = pd.DataFrame({'Expermintel Hyperparameter Combinaiton': keys.get_level_values('param_model').unique(), 
                                                'Difference Mean': difference_mean, 
                                                'Difference SD': difference_sd,
                                                'Stratified': descriptives_table[descriptives_table['Stratification']==  'stratified']['Mean'].values,
                                                'Unstratified': descriptives_table[descriptives_table['Stratification']==  'unstratified']['Mean'].values})
        differences_table_all = pd.concat([differences_table_all, differences])

        ### Plot
        plt.figure(figsize= figsize)  # Set the figure size to 10 inches by 6 inches
        if model == 'None' and transformation != 'None':
                plt.title('Transformation ' + transformation, fontsize=16)
        elif transformation == 'None' and model != 'None':
                plt.title('Model: ' + model, fontsize=16)
        elif transformation != 'None' and model != 'None':
                plt.title('Transformation: ' + transformation, fontsize=16)
        #else:
                #plt.title('Ordered and grouped boxplot', fontsize=15)
        sns.boxplot(x= data_long['param_model'], 
                        y= data_long[value_name], 
                        hue= data_long['Stratification'],  palette={value_vars[0]: 'royalblue', 
                        value_vars[1]: 'orangered'}, 
                        showfliers=False, # hide outliers
                        order = list(mean_intersection['param_model']))  #showmeans=True, meanline=True
        plt.yticks(fontsize=16)  
        plt.xticks(rotation=90, fontsize=16)  # rotate x labels by 90 degrees
        plt.ylabel(value_name, fontsize=16)
        plt.xlabel('Experimental Parameter Combination: '+ model_vars_title,  fontsize=16)
        legend = plt.legend( fontsize=16)
        for i, label in enumerate(legend.get_texts()):
                if label.get_text() == value_vars[0]:
                        label.set_text('Stratified')
                elif label.get_text() == value_vars[1]:
                        label.set_text('Unstratified')#
        if transformation == 'None' and model == 'None':
                plt.savefig(plots_report_path + '/boxplot_' + value_name + "_all" + '.png', bbox_inches='tight')
        elif transformation != 'None' and model == 'None':
                plt.savefig(plots_report_path + '/boxplot_' + value_name + "_" + transformation + '.png', bbox_inches='tight')
        elif transformation == 'None' and model != 'None':
                plt.savefig(plots_report_path + '/boxplot_' + value_name + "_" + model + '.png', bbox_inches='tight')
        elif transformation != 'None' and model != 'None':
               plt.savefig(plots_report_path + '/boxplot_' + value_name + "_" + transformation +  "_"+ model + '.png', bbox_inches='tight')
        plt.show()
        differences.sort_values(by='Difference Mean', ascending=False, inplace=True)
        print(differences)
        return differences_table_all


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