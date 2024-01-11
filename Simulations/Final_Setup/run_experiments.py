import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import make_friedman1
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import json
from sklearn.dummy import DummyRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
from utils_final import FriedmanDataset, ModelOptimizerFinal, generate_hyperparameter_combinations_dict
import itertools

####### 1. Choose model ################

model_name = 'rf'

####### 2. Initialize parameters #######
json_file = "C:/Users/anneg/Documents/Documents/StatistikMaster/slds_stratification/Simulations/Final_Setup/test.json" # set path to save json-file 
path_to_seeds = "C:/Users/anneg/Documents/Documents/StatistikMaster/slds_stratification/Simulations/Final_Setup/seeds_available.json" # set path to all available seeds 
n_features = 5
n_folds = 5
n_iter= 5
n_jobs= -1
n_repetitions = 2
n_test= 100000
scoring= 'neg_mean_squared_error'

# Define hyperparameter options
train_list = [200, 1000]
#noise_list = [0, 5]
noise_list = [0]
transformation_list = ['identity', 'sqrt']
#group_size_list = [5, 10]
group_size_list = [10]

# Set param grid for RF
rf_param_grid = {
    'min_samples_split': np.arange(2, 21),
    'min_samples_leaf': np.arange(1, 21),
    'max_features': np.arange(1, n_features + 1) #@nadja is that right?
}

# Set param grid for XGBoost
xgb_param_grid = {}

all_combinations = generate_hyperparameter_combinations_dict(n_train=train_list, 
                                                             noise=noise_list,
                                                             transformation=transformation_list, 
                                                             group_size=group_size_list)


if __name__ == '__main__':
    for hyperparameter_dict in all_combinations:

        # Save Parameters in a dictionary
        params = {'model': model_name,
          'n_train': hyperparameter_dict['n_train'],
          'n_test': n_test,
          'n_features': n_features,
          'FD_noise': hyperparameter_dict['noise'],
          'transformation': hyperparameter_dict['transformation'],
          'group_size': hyperparameter_dict['group_size'],
          'n_folds': n_folds,
          'n_iter': n_iter,
          'n_repetitions': n_repetitions,
          'scoring': scoring, 
          'n_jobs': n_jobs,
          'json_file': json_file}
        print(params)

        if model_name == 'rf': param_grid = rf_param_grid
        elif model_name == 'xgb': param_grid = xgb_param_grid
        else: raise ValueError('Model name not found')

        # Initalize Model
        modelOptimizer = ModelOptimizerFinal(param_grid=param_grid,
                                            model_name=model_name,
                                            path_to_seeds=path_to_seeds)
        modelOptimizer.optimize(params=params)


