# File for helper functions and classes

import sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import make_friedman1
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error

def generate_friedman1(n_samples=10000, n_features=5, noise=0.0, random_state=42):
    '''
    Function to generate dataset according to Friedman1.
    Inputs:
        n_samples: number of data points
        n_features: number of features (have to be at least 5)
        noise: The standard deviation of the gaussian noise applied to the output.
        random_state: to repreoduce dataset
    Outputs:
        features: array
        y: array

    '''
    features, y = sklearn.datasets.make_friedman1(n_samples=n_samples, 
                                                  n_features=n_features, 
                                                  noise=noise, 
                                                  random_state=random_state)
    return (features, y)


def to_dataframe(features, y):
    ''' 
    Function to convert arrays to combined dataframe of X and y. (could also add normalization?)
    Inputs: 
        features: first output from generate_friedman1
        y: second output from generate_friedman1
    Output:
        combined dataframe
    '''
    features=pd.DataFrame(features, 
                            columns=[f'X{i}' for i in range(1, features.shape[1] + 1)])
    y=pd.DataFrame(y, 
                     columns=['y'])
    
    df=pd.concat([features, y], axis=1)
    return df


def plot_data_3D(df, axes=['X1', 'X2', 'X3']):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['X1'], df['X2'], df['X3'], c=df['y'], cmap='viridis', marker='o')

    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    ax.set_zlabel(axes[2])
    ax.set_title(f'3D Scatter Plot of {axes[0]}, {axes[1]} and {axes[2]}')

    plt.show()


def train_test_stratified(df, n_quantiles=20, train_size=0.8, seed=42):
    # Step 1: Sort the DataFrame based on the target variable
    df_sorted = df.sort_values(by='y')
    # Step 2: Divide the sorted DataFrame into deciles
    quantiles = pd.qcut(df_sorted['y'], q=n_quantiles, labels=False)
    # Step 3: Randomly sample 80% of the data from each decile for training
    train_data = pd.concat([group.sample(frac=train_size, random_state=seed) for _, group in df_sorted.groupby(quantiles)])
    # The remaining 20% will be used for testing
    test_data = df_sorted.drop(train_data.index)
    # Step 4: Already divide by features and target variables
    X_train, y_train = train_data.drop(columns=['y']), train_data['y']
    X_test, y_test = test_data.drop(columns=['y']), test_data['y']
    return X_train, X_test, y_train, y_test


def fold_visualizer(data, fold_idxs, seed_num):
    fig, axs = plt.subplots(len(fold_idxs)//2, 2, figsize=(10,(len(fold_idxs)//2)*2))
    fig.suptitle("Seed: " + str(seed_num), fontsize=10)
    for fold_id, (train_ids, val_ids) in enumerate(fold_idxs):
        sns.histplot(data=data[train_ids],
                     kde=True,
                     stat="density",
                     alpha=0.15,
                     label="Train Set",
                     bins=30,
                     line_kws={"linewidth":1},
                     ax=axs[fold_id%(len(fold_idxs)//2), fold_id//(len(fold_idxs)//2)])
        sns.histplot(data=data[val_ids],
                     kde=True,
                     stat="density", 
                     color="darkorange",
                     alpha=0.15,
                     label="Validation Set",
                     bins=30,
                     line_kws={"linewidth":1},
                     ax=axs[fold_id%(len(fold_idxs)//2), fold_id//(len(fold_idxs)//2)])
        axs[fold_id%(len(fold_idxs)//2), fold_id//(len(fold_idxs)//2)].legend()
        axs[fold_id%(len(fold_idxs)//2), fold_id//(len(fold_idxs)//2)].set_title("Split " + str(fold_id+1))
    plt.show()
    

    def evaluate_rf(model, X_train, X_test, y_train, y_test, cv_rs=True):
        if cv_rs:
            model=model.best_estimator_
        train_r2, test_r2=model.score(X_train, y_train), model.score(X_test, y_test)
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
        train_mse, test_mse=mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)
        train_mae, test_mae=mean_absolute_error(y_train, y_train_pred), mean_absolute_error(y_test, y_test_pred)
        return {'train r2': train_r2, 
                'test r2': test_r2, 
                'train mse': train_mse,
                'test mse': test_mse,
                'train mae': train_mae,
                'test mae': test_mae}