import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
            f'unstratified_results_train {metric}', 
            f'stratified_results_train {metric}'
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