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
                plt.title('Transformation ' + transformation, fontsize=15)
        elif transformation == 'None' and model != 'None':
                plt.title('Model: ' + model, fontsize=15)
        elif transformation != 'None' and model != 'None':
                plt.title('Transformation: ' + transformation, fontsize=15)
        else:
                plt.title('Ordered and grouped boxplot', fontsize=15)
        sns.boxplot(x= data_long['param_model'], 
                        y= data_long[value_name], 
                        hue= data_long['Stratification'],  palette={value_vars[0]: 'royalblue', 
                        value_vars[1]: 'orangered'}, 
                        showfliers=False, # hide outliers
                        order = list(mean_intersection['param_model']))  #showmeans=True, meanline=True
        plt.yticks(fontsize=14)  
        plt.xticks(rotation=90, fontsize=14)  # rotate x labels by 90 degrees
        plt.ylabel(value_name, fontsize=14)
        plt.xlabel('Experimental Parameter Combination: '+ model_vars_title,  fontsize=14)
        legend = plt.legend( fontsize=14)
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