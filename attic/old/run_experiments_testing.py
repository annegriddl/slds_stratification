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
import re
import warnings
warnings.filterwarnings('ignore')
#from utils_final_nadja import FriedmanDataset, ModelOptimizer, generate_hyperparameter_combinations_dict
from utils_final_nadja import ModelOptimizer, generate_hyperparameter_combinations_dict
import itertools
import os




####### 1. Choose model ################################
#### TODO: Choose 'rf' for Random Forest Regressor
####       Or 'xgb' for XGBoost Regressor
model_name = 'rf'
#### END ##############################################



####### 2. Initialize experimental parameters #######################
#### TODO: Set experimental parameters 
# Here: fixed and not varied over experiments
json_file = "./Simulations/Final_Setup/test_nadja.json"
path_to_seeds = "./Simulations/Final_Setup/seeds_available_nadja.json"
n_features = 5
n_folds = 4
n_iter= 5
n_jobs= -1
n_repetitions = 2
n_test= 100000
scoring= 'neg_mean_squared_error'

# Here: varied over experiments #@Anne: eigenlich muss ich hier nur die parameter mit mehr als einem wert angeben, oder?
hyperparameter_options = {'n_train': [200],
                          'transformation': ['log', 'sqrt'],
                          'noise': [0],
                          'group_size': [10]}
#### END ##############################################







######### 3. Run experiemnts (nothing to change here) ##############################
# Generate grid for experimental parameter combinations over that we iterate later
all_combinations = generate_hyperparameter_combinations_dict(hyperparameter_options)
print('\n-----------------------------------')
print('Number of hyperparameter combinations:', len(all_combinations))
print('-----------------------------------\n')

# Set model hyperparameter grid for Random Search for RF
rf_param_grid = {
    'min_samples_split': np.arange(2, 21),
    'min_samples_leaf': np.arange(1, 21),
    'max_features': np.arange(1, n_features + 1) #@nadja is that right?
}

# Set model hyperparameter grid for Random Search for XGBoost
xgb_param_grid = {}






if __name__ == '__main__':
    if not os.path.exists(path_to_seeds):   
        print("cant find path")
        #with open(path_to_seeds, 'w') as file: #@Anne: würd auskommentieren, d.h. error und dann manuell wieder einkommentieren, um evlt. file Fehelr zu vermeiden
            #json.dump([x for x in range(100000)], file, indent=4)
            #print("File created: ", path_to_seeds)

    tracker = 1
    for hyperparameter_dict in all_combinations:
        # Save Parameters in a dictionary
        params_experiment = {'model': model_name,
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
        print('EXPERIMENTAL PARPAMETER COMBINATION ', tracker, ' (out of ', len(all_combinations), '): \n', params_experiment)

        if model_name == 'rf': hyp_param_grid = rf_param_grid
        elif model_name == 'xgb': hyp_param_grid = xgb_param_grid
        else: raise ValueError('Model name not found')

        # Initalize Model
        modelOptimizer = ModelOptimizer(hyp_param_grid=hyp_param_grid, #@Anne: unbenannt von ModelOptimizerFinal 
                                            model_name=model_name,
                                            path_to_seeds=path_to_seeds, checks=True)
        modelOptimizer.optimize(params_experiment=params_experiment)
        print('End of hyperparameter combinaiton', tracker)
        tracker += 1
        print('\n-----------------------------------')
        print()
        


