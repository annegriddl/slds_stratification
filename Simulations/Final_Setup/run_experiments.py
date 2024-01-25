import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import time

parallel_repetitions = False
if parallel_repetitions:
    from utils_parallel import ModelOptimizer, generate_hyperparameter_combinations_dict
else:
    from utils_final import ModelOptimizer, generate_hyperparameter_combinations_dict


####### 1. Choose model ################################
#### TODO: Choose 'rf' for Random Forest Regressor
####       Or 'xgb' for XGBoost Regressor
model_name = 'rf'
#### END ##############################################



####### 2. Initialize experimental parameters #######################
#### TODO: Set experimental parameters 
# Here: fixed and not varied over experiments
script_dir = os.path.dirname(os.path.abspath(__file__))
json_file =  script_dir + "/results/test-parallel.json" 
path_to_seeds = script_dir + "/seeds/json-test.json"  

n_features = 8
n_folds = 5
n_iter= 5
n_jobs= -1
n_repetitions = 2
n_test= 100000
scoring= 'neg_mean_squared_error'

# Here: varied over experiments.
hyperparameter_options = {'n_train': [200],
                          'transformation': ['log'],
                          'noise': [0],
                          'group_size': [5, 10]} # Number of datapoints per group
#### END ##############################################


######### 3. Run experiemnts (nothing to change here) ##############################
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
    'max_depth':  [int(i) for i in np.logspace(np.log10(2), np.log10(20), num =20) ], # defualt 6
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

        # Initalize Model
        modelOptimizer = ModelOptimizer(hyp_param_grid=hyp_param_grid, 
                                            model_name=model_name,
                                            path_to_seeds=path_to_seeds, checks=False)
        modelOptimizer.optimize(params_experiment=params_experiment)
        print('End of hyperparameter combination', tracker)
        tracker += 1
        print('\n-----------------------------------')
        print()
    if parallel_repetitions: run = "parallel repetitions"    
    else: run = "parallel Random Search"
    print(f"Total execution time of {run}: {round((time.time() - start_time)/60, 4)} min")
        


