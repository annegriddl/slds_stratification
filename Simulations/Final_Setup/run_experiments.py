"""
This script runs the experiments for the final setup. 
Important to either have files seeds_final_rf.json and seeds_final_xgb.json in folder seeds/. Otherwise run create_seeds.py first.
Important to specify in bash command line: 
- Specify whether you want to perform parallel repetitions or parallel random search by setting the variable parallel_repetitions to True or False.
  We recommend to set parallel_repetitions to True, as it is usually faster.
- Choose the model by setting the variable model_name to 'rf' for Random Forest Regressor or 'xgb' for XGBoost Regressor.
- Choose the number of repetitions by setting the variable n_repetitions.
"""

import sys
try:
    parallel_repetitions = sys.argv[1].lower() == 'true'
except:
    parallel_repetitions = True
    print("No argument given for parallel_repetitions. Using default value True")

try: 
    model_name = sys.argv[2]
except:
    model_name = 'xgb'
    print("No argument given for model_name. Using default value 'xgb'")

try:
    n_repetitions = int(sys.argv[3])
except:
    n_repetitions = 10
    print("No argument given for n_repetitions. Using default value 10")
    
print("Value of parallel_repetitions:", parallel_repetitions)
print("Model:", model_name)
print("Number of repetitions:", n_repetitions)

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import time

# Import utils for parallel repetitions or parallel random search
if parallel_repetitions == True:
    from utils_parallel import ModelOptimizer, generate_hyperparameter_combinations_dict
    print("Performing parallel repetitions")
else:
    from utils_final import ModelOptimizer, generate_hyperparameter_combinations_dict
    print("Performing parallel random search")

# Check if results folder already exists, if not create it
if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/results/"):
    os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/results/")
    print("Folder for results created.")

if not os.path.exists(os.path.dirname(os.path.abspath(__file__)) + "/results/" + model_name + "/"):
    os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/results/" + model_name + "/")
    print(f"Folder for results of {model_name} created.")

# Specify name of output files
run = len(os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/results/" + model_name + "/")) + 1
print(run)

results_file = f"results_{model_name}{run}.json"
seed_file = f"seeds_final_{model_name}.json"


####### 2. Initialize experimental parameters #######################
####  Set experimental parameters 
# Here: fixed and not varied over experiments

script_dir = os.path.dirname(os.path.abspath(__file__))
json_file =  f"{script_dir}/results/{model_name}/{results_file}" 
path_to_seeds = f"{script_dir}/seeds/{seed_file}"

n_features = 8
n_folds = 5
n_iter= 200
if parallel_repetitions: n_jobs= 1
else: n_jobs= -1
n_test= 100000
scoring= 'neg_mean_squared_error'

# Here: varied over experiments.
hyperparameter_options = {'n_train': [200, 1000],
                          'transformation': ['identity', 'log', 'sqrt'],
                          'noise': [0, 3],
                          'group_size': [5, 10]} # Number of datapoints per group
#### END ##############################################


######### 3. Run experiemnts  ##############################
# Generate grid for experimental parameter combinations over that we iterate later
all_combinations = generate_hyperparameter_combinations_dict(hyperparameter_options)
print('\n-----------------------------------')
print('Number of hyperparameter combinations:', len(all_combinations))
print('-----------------------------------\n')

# Set model hyperparameter grid for Random Search for RF
rf_param_grid = {
    'min_samples_split': np.arange(2, 11), 
    'min_samples_leaf': np.arange(1, 11),
    'max_features': np.arange(1, n_features + 1) 
}

# Set model hyperparameter grid for Random Search for XGBoost
xgb_param_grid = {
    'learning_rate': np.linspace(0.001, 0.4, num =10), #default 0.3
    'max_depth':  [int(i) for i in np.logspace(np.log10(2), np.log10(20), num =20)], # defualt 6
    'subsample': [np.random.uniform(0.5,1) for i in range(10)], # default 1
    'colsample_bytree': [np.random.uniform(0.5,1) for i in range(10)],# default 1
    'gamma': np.logspace(0, np.log10(20), num =10), #default 0
    'min_child_weight': np.arange(1, 10) #default 1
}

if __name__ == '__main__':
    tracker = 1
    start_time = time.time()
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
        print('Model name: ', model_name)
        # Initalize Model
        modelOptimizer = ModelOptimizer(hyp_param_grid=hyp_param_grid, 
                                        model_name=model_name,
                                        path_to_seeds=path_to_seeds, 
                                        checks=False)
        # Train model (utils_final.py for parallel random search; utils_parallel.py for parallel repetitions)
        modelOptimizer.optimize(params_experiment=params_experiment)
        print('End of hyperparameter combination', tracker)
        tracker += 1
        print('\n-----------------------------------')
        print()
    if parallel_repetitions: run = "parallel repetitions"    
    else: run = "parallel Random Search"
    print(f"Total execution time of {run}: {round((time.time() - start_time)/60, 4)} min")
        


