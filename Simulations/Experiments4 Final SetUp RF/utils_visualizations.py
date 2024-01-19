import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def flatten_data(data_all):
    # Flatten the nested dictionaries
    flat_data = []
    for entry in data_all:
        flat_entry = {}
        for key, value in entry.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_entry[key + "_" + sub_key] = sub_value
            else:
                flat_entry[key] = value
        flat_data.append(flat_entry)

    df = pd.DataFrame(flat_data)
    return df



def plot_boxplots(data200, data1000, title_left, title_right, metric='r2', order=None):
    if order is None:
        order = [
            f'unstratified_results_test {metric}', 
            f'stratified_results_test {metric}'
        ]

    df_melted200 = pd.melt(data200, value_vars=order, var_name='Metric', value_name='Value')
    df_melted1000 = pd.melt(data1000, value_vars=order, var_name='Metric', value_name='Value')

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=False)

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

def plot_differences_mean(data, title_variable):
    data = data.iloc[:,-14:].mean()
    print(data)

    # order data asc by index
    data = data.sort_index()
    index = ['mae test', 'mse test', 'r2 test', 'mae train', 'mse train', 'r2 train', 'running time']

    means = pd.DataFrame({
        'Unstratified': data.values[0:  7],
        'Stratified': data.values[7: data.shape[0]]
    }, index=index)
    
    # add column with difference
    means['difference'] = means['Stratified'] - means['Unstratified']
    
    # plot barplot difference
    means['difference'].plot.barh()
    plt.xlabel('Difference')
    plt.ylabel('Evaluation metric')
    plt.title('Difference between stratified and unstratified sampling: ' + title_variable)
    plt.tight_layout()
    plt.savefig('./plots/difference_stratified_unstratified' + title_variable+ '.png', dpi=300, bbox_inches='tight')
    plt.show()
    return means



def filter_and_boxplot(data, conditions, value1, value2):
    # write query for whole data
    query_string = ' and '.join([f"{key} == {repr(value)}" for key, value in conditions.items() if value is not None])
    data_filtered = data.query(query_string)

    # comparison boxplots by one variable
    keys_with_none_value = [key for key, value in conditions.items() if value is None][0]
    data_filtered_1 = data_filtered[(data_filtered[keys_with_none_value] == value1)]
    data_filtered_2 = data_filtered[(data_filtered[keys_with_none_value] == value2)]


    # plot boxplots
    plot_boxplots(data_filtered_1, data_filtered_2, title_left= keys_with_none_value +':'+ str(value1), title_right = keys_with_none_value +':'+ str(value2), metric='r2')

    # plot difference seperatly
    means1 = plot_differences_mean(data_filtered_1, title_variable=keys_with_none_value +':'+ str(value1))
    means2 = plot_differences_mean(data_filtered_2, title_variable=keys_with_none_value +':'+ str(value2))

    # plot difference together
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_width = 0.35
    y_positions = np.arange(len(means1))     # Create positions for bars on the y-axis
    ax.barh(y_positions - bar_width/2, means1['difference'], bar_width, label=str(value1), color='b') # Plotting bars for the 'difference' column in the first data frame
    ax.barh(y_positions + bar_width/2, means2['difference'], bar_width, label=str(value2), color='g') # Plotting bars for the 'difference' column in the second data frame

    # Adding labels, title, and legend
    ax.set_xlabel('Difference Values', fontsize=15)
    ax.set_ylabel('Metrics', fontsize=15)
    ax.set_title('Grouped Horizontal Bar Plot for Difference', fontsize=18)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(means1.index)
    ax.legend(title=keys_with_none_value)

    # Display the plot
    plt.tight_layout()
    plt.show()

    return means1, means2