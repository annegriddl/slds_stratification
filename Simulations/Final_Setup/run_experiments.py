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
import re
import warnings
warnings.filterwarnings('ignore')
from utils_final import FriedmanDataset, ModelOptimizerFinal, generate_hyperparameter_combinations_dict
import itertools
import os

####### 1. Choose model ################

#### TODO: Choose 'rf' for Random Forest Regressor
####       Or 'xgb' for XGBoost Regressor
model_name = 'rf'
#### Ende #######################


####### 2. Initialize parameters #######

#### TODO: Set parameters for the experiment
#json_file = os.path.join(os.path.dirname(os.path.realpath("test.ipynb")), "test.json") 
#path_to_seeds = os.path.join(os.path.dirname(os.path.realpath("run_experiments.py")), "seeds_available.json") 
#json_file = re.sub(r"\\", "/", json_file)
#path_to_seeds = re.sub(r"\\", "/", path_to_seeds)
json_file = "C:/Users/anneg/Documents/Documents/StatistikMaster/slds_stratification/Simulations/Final_Setup/test.json"
path_to_seeds = "C:/Users/anneg/Documents/Documents/StatistikMaster/slds_stratification/Simulations/Final_Setup/seeds_available.json"
n_features = 5
n_folds = 5
n_iter= 5
n_jobs= -1
n_repetitions = 2
n_test= 100000
scoring= 'neg_mean_squared_error'

# Define hyperparameter options
hyperparameter_options = {'n_train': [200, 1000],
                          'transformation': ['identity', 'sqrt'],
                          'noise': [0],
                          'group_size': [10]}
#### Ende #######################

# Generate hyperparameter combinations
all_combinations = generate_hyperparameter_combinations_dict(hyperparameter_options)

# Set param grid for RF
rf_param_grid = {
    'min_samples_split': np.arange(2, 21),
    'min_samples_leaf': np.arange(1, 21),
    'max_features': np.arange(1, n_features + 1) #@nadja is that right?
}

# Set param grid for XGBoost
xgb_param_grid = {}


if __name__ == '__main__':
    if not os.path.exists(path_to_seeds):   
        print("cant find path")
        with open(path_to_seeds, 'w') as file:
            json.dump([x for x in range(100000)], file, indent=4)
            print("File created: ", path_to_seeds)
            #random_states = [x for x in range(n_repetitions)]
            #seeds_available = [x for x in range(100000)][n_repetitions:]

    for hyperparameter_dict in all_combinations:

        # Save Parameters in a dictionary
        params = {'model': model_name,
          'n_train': hyperparameter_dict['n_train'],
          'n_test': n_test,
          'n_features': n_features,
          'noise': hyperparameter_dict['noise'],
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


