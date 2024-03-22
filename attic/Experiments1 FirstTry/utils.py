# File for helper functions and classes

import sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import make_friedman1
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

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
    # Step 1: Divide the DataFrame into quantiles
    quantiles = pd.qcut(df['y'], q=n_quantiles, labels=False)
    # Step 2: Randomly sample 80% of the data from each quantile for training
    train_data = pd.concat([group.sample(frac=train_size, random_state=seed) for _, group in df.groupby(quantiles)])
    # The remaining 20% will be used for testing
    test_data = df.drop(train_data.index)
    train_data = shuffle(train_data)
    test_data = shuffle(test_data)
    # Step 3: Already divide by features and target variables
    X_train, y_train = train_data.drop(columns=['y']), train_data['y']
    X_test, y_test = test_data.drop(columns=['y']), test_data['y']
    return X_train, X_test, y_train, y_test


def evaluate_rf(model, X_train, X_test, y_train, y_test, cv_rs=True):
    if cv_rs:
        model=model.best_estimator_
    train_r2, test_r2=model.score(X_train, y_train), model.score(X_test, y_test)
    y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
    train_mse, test_mse=mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)
    return {'train r2': train_r2, 
            'test r2': test_r2, 
            'train mse': train_mse,
            'test mse': test_mse}

def create_cont_folds(y, n_folds=5, n_groups=5, seed=1):
    # create StratifiedKFold like for classification
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # create groups in y with pd.qcut: quantile-based discretization 
    y_grouped = pd.qcut(y, n_groups, labels=False)

    # create fold numbers    
    fold_nums = np.zeros(len(y))
    #split(X, y[, groups]): Generate indices to split data into training and test set
    for fold_no, (t, v) in enumerate(skf.split(y_grouped, y_grouped)): #@Nadja: unabhängig von n_folds? n_folds = fol_no, test_data_size = N/n_folds
        fold_nums[v] = fold_no
  

    cv_splits = []

    # iterate over folds and creat train and test indices for each fold
    for i in range(n_folds):
        test_indices = np.argwhere(fold_nums==i).flatten()
        train_indices = list(set(range(len(y_grouped))) - set(test_indices))
        cv_splits.append((train_indices, test_indices))

    return cv_splits