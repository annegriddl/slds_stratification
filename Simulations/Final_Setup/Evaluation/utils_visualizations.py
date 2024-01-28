import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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



def plot_boxplots(data200, data1000, title_left, title_right, metric='r2', order=None):
    if order is None:
        order = [
            f'unstratified_results_test {metric}', 
            f'stratified_results_test {metric}'
        ]

    df_melted200 = pd.melt(data200, value_vars=order, var_name='Metric', value_name='Value')
    df_melted1000 = pd.melt(data1000, value_vars=order, var_name='Metric', value_name='Value')

    fig, axes = plt.subplots(1, 2, figsize=(6, 4), sharey=False)

    for i in range(2):
        if i == 0:
            df_melted = df_melted200
            df=data200
            title = title_left
        else:
            df_melted = df_melted1000    
            df=data1000
            title = title_right
        ax = sns.boxplot(x='Metric', y='Value', data=df_melted, ax=axes[i],
                         order=order,
                         showmeans=True,
                         meanline=True,
                         color="grey",
                         meanprops={"marker": "+",
                                    "markerfacecolor": "white",
                                    "markeredgecolor": "green",
                                    "markersize": "10"},
                         width=.5,
                         #linecolor="#137", linewidth=.75,
                         medianprops={"color": "#137", "linewidth": 1})
        ax.set_ylabel(metric.upper())
        ax.set_xlabel('')
        plt.text(0.98, 0.98, f'N = {len(df)}', horizontalalignment='right', verticalalignment='top',
                 transform=ax.transAxes, size=14)
        xtickNames = plt.setp(ax, xticklabels=['Test Unstratified', 'Test Stratified'])
        plt.setp(xtickNames, rotation=0)
        ax.set_title(title)
    plt.show()

'''
def plot_differences_mean(data, title_variable, plot = True):
    evaluation_metrics = data.loc[:,['unstratified_results_train r2', 'unstratified_results_test r2',
       'unstratified_results_train mse', 'unstratified_results_test mse',
       'unstratified_results_train mae', 'unstratified_results_test mae',
       'stratified_results_train r2', 'stratified_results_test r2',
       'stratified_results_train mse', 'stratified_results_test mse',
       'stratified_results_train mae', 'stratified_results_test mae']]
    data = evaluation_metrics.mean()

    data_stats = {
    'mean': evaluation_metrics.mean(),
    'sd': evaluation_metrics.std()}
    data_stats = pd.DataFrame(data_stats)
    print(data_stats)
    print(data_stats.index)

    # order data asc by index
    data_stats = data_stats.sort_index()
    #print(data)
    index = ['mae test', 'mse test', 'r2 test', 'mae train', 'mse train', 'r2 train']
    print(data_stats.shape)
  
    means = pd.DataFrame({
        'Stratified': data_stats.values[0:  6],
        'Unstratified': data_stats.values[6: data.shape[0]]
    }, index=index)
    
    # add column with difference
    means['difference'] = means['Stratified'] - means['Unstratified']
    means['sd'] =  means['difference'].std()

    # plot barplot difference
    if plot == True:
        means['difference'].plot.barh()
        plt.xlabel('Difference')
        plt.ylabel('Evaluation metric')
        plt.title('Difference between stratified and unstratified sampling: ' + title_variable)
        plt.tight_layout()
        plt.savefig('./plots/difference_stratified_unstratified' + title_variable+ '.png', dpi=300, bbox_inches='tight')
        plt.show()
    return means
''' 

def differences_eval(data):
    data_diff = pd.DataFrame()
    data_diff['diff_train_r2'] = data['stratified_results_train r2'] - data['unstratified_results_train r2']
    data_diff['diff_test_r2'] = data['stratified_results_test r2'] - data['unstratified_results_test r2']
    data_diff['diff_train_mse'] = data['stratified_results_train mse'] - data['unstratified_results_train mse']
    data_diff['diff_test_mse'] = data['stratified_results_test mse'] - data['unstratified_results_test mse']
    data_diff['diff_train_mae'] = data['stratified_results_train mae'] - data['unstratified_results_train mae']
    data_diff['diff_test_mae'] = data['stratified_results_test mae'] - data['unstratified_results_test mae']

    #print(data_diff)
    #print(data_diff.shape)

    data_stats = {
        'mean_diff': data_diff.mean(),
        'sd_diff': data_diff.std()
        } 
    data_stats = pd.DataFrame(data_stats)
    #print(data_stats)
    #print(data_stats.index)
    ''' 
    if plot == True:
        data_stats['mean_diff'].plot.barh()
        plt.xlabel('Difference')
        plt.ylabel('Evaluation metric')
        plt.title('Difference between stratified and unstratified sampling: ' + title_variable)
        plt.tight_layout()
        plt.savefig('./plots/difference_stratified_unstratified' + title_variable+ '.png', dpi=300, bbox_inches='tight')
        plt.show()
    '''
    return data_stats

def filter_data(data, conditions, value1, value2):
    # write query for whole data: filter over all hyperparameters
    query_string = ' and '.join([f"{key} == {repr(value)}" for key, value in conditions.items() if value is not None])
    data_filtered = data.query(query_string)

    # comparison boxplots by one variable
    filtered_parameter = [key for key, value in conditions.items() if value is None][0]
    data_filtered_1 = data_filtered[(data_filtered[filtered_parameter] == value1)]
    data_filtered_2 = data_filtered[(data_filtered[filtered_parameter] == value2)]
    print('data_filtered_1 shape:', data_filtered_1.shape)
    print('data_filtered_2 shape:', data_filtered_2.shape)
    return data_filtered_1, data_filtered_2, value1, value2, filtered_parameter

def plots_per_condition(data, conditions, value1, value2):
    # filter data
    data_filtered_1, data_filtered_2, value1, value2, filtered_parameter = filter_data(data, conditions, value1, value2)

    # plot boxplots
    plot_boxplots(data_filtered_1, data_filtered_2, title_left= filtered_parameter +':'+ str(value1), title_right = filtered_parameter +':'+ str(value2), metric='mse')

    # plot difference seperatly
    means1 = differences_eval(data_filtered_1)
    means2 = differences_eval(data_filtered_2)

    # plot difference together
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_width = 0.35
    y_positions = np.arange(len(means1))     # Create positions for bars on the y-axis
    ax.barh(y_positions - bar_width/2, means1['mean_diff'], bar_width, label=str(value1), color='b') # Plotting bars for the 'difference' column in the first data frame
    ax.barh(y_positions + bar_width/2, means2['mean_diff'], bar_width, label=str(value2), color='g') # Plotting bars for the 'difference' column in the second data frame #@Nadja: xerr=means2['sd_diff']

    # Adding labels, title, and legend
    ax.set_xlabel('Difference Values', fontsize=15)
    ax.set_ylabel('Metrics', fontsize=15)
    ax.set_title('Grouped Horizontal Bar Plot for Difference', fontsize=18)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(means1.index)
    ax.legend(title=filtered_parameter)

    # Display the plot
    plt.tight_layout()
    plt.show()

    return means1, means2


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
    plt.show()


def descreptives(data, path_plots):
    statistics = ['ks_statistic', 'p_value', 'intersection_area']
    mean_stratified_list = []
    sd_stratified_list = []
    mean_unstratified_list = []
    sd_unstratified_list = []

    for s in statistics:
        # filter stratified and unstratified descreptives columns
        s_stratified = '_stratified_' + s
        s_unstratified = '_unstratified_' + s
        s_stratified = [col for col in data.columns if s_stratified in col]
        s_unstratified = [col for col in data.columns if s_unstratified in col]

        plot_combined_boxplots(data_x= data[s_stratified], data_y = data[s_unstratified], title= s, path=path_plots + '_' + s)
        
        # values for stratified
        mean_stratified = data[s_stratified].mean()[0]
        sd_stratified = data[s_stratified].std()[0]
        mean_stratified_list.append(mean_stratified)
        mean_stratified_list.append(sd_stratified)

        # values for unstratified
        mean_unstrtified = data[s_unstratified].mean()[0]
        sd_unstratified = data[s_unstratified].std()[0]
        mean_unstratified_list.append(mean_unstrtified)
        sd_unstratified_list.append(sd_unstratified)
    df_result = pd.DataFrame({'mean_strtified': mean_stratified_list, 'sd_stratified': mean_stratified_list, 
                              'mean_unstratified': mean_unstratified_list, 'sd_unstratified_list': sd_unstratified_list}, 
                              index= statistics)
    
    return df_result
