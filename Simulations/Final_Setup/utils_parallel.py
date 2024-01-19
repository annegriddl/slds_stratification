
import sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import make_friedman1
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
import json
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV 
import random
from sklearn.ensemble import RandomForestRegressor
import time
import xgboost as xgb
import math
import os
from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp
from joblib import Parallel, delayed

import itertools

def generate_hyperparameter_combinations_dict(hyperparameter_options):
    """
    Generate all possible combinations of hyperparameters.

    Parameters:
    hyperparameter_options (dict): Dictionary where keys are hyperparameter names and values are lists of options.

    Returns:
    List of dictionaries, where each dictionary represents a combination of hyperparameters.
    """
    hyperparameter_names = hyperparameter_options.keys()
    all_hyperparameter_combinations = list(itertools.product(*hyperparameter_options.values()))

    all_hyperparameter_dicts = []
    for combination in all_hyperparameter_combinations:
        hyperparameter_dict = dict(zip(hyperparameter_names, combination))
        all_hyperparameter_dicts.append(hyperparameter_dict)

    return all_hyperparameter_dicts




class ModelOptimizer:
    '''
    Class to optimize the model.
    Inputs:
        model: the model to be optimized
        hyp_param_grid: the parameter grid to be used for the optimization
        random_state: the random state to be used
    '''
    def __init__(self, hyp_param_grid, model_name, path_to_seeds, checks):
        self.hyp_param_grid = hyp_param_grid
        self.model_name = model_name
        self.path_to_seeds = path_to_seeds
        self.checks = checks
        #Create testing data with seed 1718 (use same seed in all experiments)
        self.global_seed_testing_data = 1718 # static, don't change this over experiments!!!
        self.n_estimators = 500 # static, don't change this over experiments !!!
        


    def optimize(self, 
                 params_experiment,
                 data='friedman',
                 random_states=None): 
        '''
        Function to optimize the model.
        Inputs:
            X_train, X_test, y_train, y_test: the train and test data
            cv: the number of folds
            n_groups: the number of groups (based on quantiles)
            scoring: the scoring to be used
            n_jobs: the number of jobs to be used
            n_iter: the number of iterations
            ROOT_PATH: the root path to be used (results are stored in JSON file)
            transformation: the transformation of target to be applied
        Outputs:
            unstratified_results: the results of the unstratified cross-validation
            stratified_results: the results of the stratified cross-validation
        Important: The results are stored in a JSON file. Initialize a new file with an empty list as content.
        '''
        start_time = time.time()                                             
        
        ### get experimental parameters from entered params_experiment dictionary
        n_train = params_experiment['n_train']
        n_test = params_experiment['n_test']
        n_features = params_experiment['n_features']
        noise = params_experiment['noise']
        transformation = params_experiment['transformation']
        n_folds= params_experiment['n_folds']
        group_size = params_experiment['group_size']
        scoring = params_experiment['scoring'] 
        n_jobs = params_experiment['n_jobs']
        n_iter = params_experiment['n_iter']
        n_repetitions = params_experiment['n_repetitions']
        json_file = params_experiment['json_file']

        # Calculate number of groups for stratified cross-validation
        # Goal: same number of observations in each group independent of n_train    
        n_groups = int(n_train/group_size)
        if self.checks:
            print("Number of groups: ", n_groups)

        ### set seeds for all repetitions
        #######################################################################################
        if not isinstance(random_states, list):   # if random_states is None: load seeds forom json_file
            if self.checks: print("\nLoad seeds from json: ", self.path_to_seeds)   
            try:
                with open(self.path_to_seeds, 'r') as file:
                    seeds_available = json.load(file)
            except json.JSONDecodeError:
                print("Error decoding JSON. The file might be empty or not properly formatted.")
            except FileNotFoundError:
                print("Seed file not found. The file might not exist or the path is wrong.")
            if self.checks:
                print(f"available seeds in {self.path_to_seeds}: {len(seeds_available)}")
            random_states = seeds_available[:n_repetitions]
            seeds_available = seeds_available[n_repetitions:]
            with open(self.path_to_seeds, 'w') as file:
                json.dump(seeds_available, file, indent=4)
            if self.checks:
                print(f"available seeds in {self.path_to_seeds}: {len(seeds_available)}") 
            print(f"Successfully loaded and deleted picked seeds from json file {self.path_to_seeds}!:\n {random_states}")
        else: # if random_states is list: use list as seeds 
            random_states = random_states[:n_repetitions]
            print("Set seeds to: ", random_states, "for all iterations.\n")
        #######################################################################################


   
        # initalize dictionary with final_results, which is saved to json_file for evaluation
        final_results = {
            'model_info': params_experiment
        }

        #### Run Experiments for each repetition independently 
        results_list=Parallel(n_jobs=-1)(
            delayed(self.run_experiments)(random_states, repetition, data, n_train, n_test, noise, n_features, transformation,
                   n_folds, n_groups, scoring, n_jobs, n_iter)
            for repetition in range(n_repetitions))
        
        for result in results_list:
            final_results.update(result)
            self.save_results(final_results, json_file)

        parallel_execution_time = round((time.time() - start_time)/60, 4)                                    
        print(f"\nParallel execution time: {parallel_execution_time} min")    
            
        
    def run_experiments(self, random_states, repetition, data, n_train, n_test, noise, n_features, transformation,
                   n_folds, n_groups, scoring, n_jobs, n_iter):        
   
        if data == 'friedman':
            X_train, y_train, X_test, y_test = self.generate_data(n_samples_train=n_train, 
                                                                      n_samples_test= n_test, 
                                                                      noise = noise, 
                                                                      n_features = n_features, 
                                                                      random_state_training = random_states[repetition], 
                                                                      transformation = transformation)    


        # Perform optimization with unstratified cross-validation
        unstratified_results, unstratified_iteration, unstratified_params, unstratified_running_time, unstratified_results_descriptives_folds = self._perform_optimization(X_train, 
                                                            y_train, 
                                                            X_test,
                                                            y_test,
                                                            n_folds, 
                                                            n_groups,
                                                            scoring, 
                                                            n_jobs, 
                                                            n_iter, 
                                                            random_states[repetition],
                                                            stratified=False)
            

        # Perform optimization with stratified cross-validation
        # note: here iteration_refit_test is additionally calculated as it's the same for stratified and unstratified (fit of same hyperparameters on the same data)
        stratified_results, stratified_iteration, stratified_params, stratified_running_time, stratified_results_descriptives_folds, iteration_refit_test = self._perform_optimization(X_train, 
                                                            y_train, 
                                                            X_test,
                                                            y_test,
                                                            n_folds, 
                                                            n_groups, 
                                                            scoring, 
                                                            n_jobs, 
                                                            n_iter, 
                                                            random_states[repetition],
                                                            stratified=True)

        if unstratified_params == stratified_params:
            hyperparameters_same = True
        else:
            hyperparameters_same = False 
                
        # Save results and parameters to a file
        results = {
                'repetition': repetition,
                'random_state': random_states[repetition],
                'hyperparameters_same': hyperparameters_same,
                'unstratified_results': unstratified_results,
                'stratified_results': stratified_results,
                'cv_unstratified_iterations': unstratified_iteration,
                'cv_stratified_iterations': stratified_iteration, 
                'cv_iteration_refit_test': iteration_refit_test,
                'cv_folds_descriptives_unstratified': unstratified_results_descriptives_folds,
                'cv_folds_descriptives_stratified': stratified_results_descriptives_folds,
                'unstratified_best_params': unstratified_params,
                'stratified_best_params': stratified_params,
        }
        if self.checks:
            print('seed for training data: ', random_states[repetition])
        if hyperparameters_same: hype_same = 'the same'
        else: hype_same = 'different'
        print(f"Repetition {repetition+1}: Hyperparameters are {hype_same}")

        return results
        
        
    
    def save_results(self, results, path):
        '''
        Function to save the results to a JSON file.
        Inputs:
            results: the results to be saved
            json_file: the JSON file to be used
        Outputs:
            None (it saves the results to the JSON file)
        '''

        if not os.path.exists(path):  
            with open(path, 'w') as file:
                json.dump([], file, indent=4)
            print("File not found, created new file: ", path)
    
        # Load existing data or create an empty list
        with open(path, 'r') as file:
            existing_data = json.load(file)

        # Append the new results dictionary to the existing data
        existing_data.append(results)

        # Write the updated data back to the JSON file
        with open(path, 'w') as file:
            json.dump(existing_data, file, indent=4, default=self._convert_numpy_types)


    def _perform_optimization(self, 
                              X_train, 
                              y_train, 
                              X_test, 
                              y_test, 
                              cv, 
                              n_groups, 
                              scoring, 
                              n_jobs, 
                              n_iter, 
                              random_state,
                              stratified):
        '''
        Function to perform the optimization.
        Inputs:
            the same as in optimize function
            stratified: whether to use stratified k-fold or not
        Outputs:
            evaluation_results: the evaluation results in a dictionary
            best_params: the best parameters in a dictionary
        '''
        # Create cross-validation splits
        if stratified:
            cv_splits = self.create_cont_folds(y=y_train, 
                                               n_folds=cv, 
                                               n_groups=n_groups, 
                                               seed=random_state) 
            output_text = 'Stratified Split Cross-validation'
            results_descriptives_folds = self.analysis_folds(data = y_train, 
                                                             fold_idxs = cv_splits, 
                                                             seed_num= random_state, 
                                                             stratified=True, 
                                                             plot=False)
            if self.checks:
                print(f"{output_text} with seed {random_state}: {results_descriptives_folds}")
        else:
            #cv_splits = cv
            cv_splits = list(KFold(cv, shuffle=True, random_state=random_state).split(y_train))
            output_text = 'Random Split Cross-validation'
            results_descriptives_folds = self.analysis_folds(data = y_train, fold_idxs = cv_splits, seed_num= random_state, stratified=True, plot=False)
            if self.checks:
                print(f"\n{output_text} with seed {random_state}: {results_descriptives_folds}")
        
        # Initialize the model
        try:
            if self.model_name == "rf":
                model = RandomForestRegressor(n_estimators= self.n_estimators,
                                              random_state=random_state)
            elif self.model_name == "xgb":
                model = xgb.XGBRegressor(random_state=random_state)
        except:
            raise ValueError("Model not implemented. Only 'rf' and 'xgb' are implemented.")
        
        # Perform the optimization
        start_time = time.time()
        random_search = RandomizedSearchCV(estimator=model,
                                           param_distributions=self.hyp_param_grid,
                                           n_iter=n_iter,
                                           cv=cv_splits,
                                           scoring=scoring,
                                           #n_jobs=n_jobs,
                                           random_state=random_state)
        
        random_search.fit(X_train, y_train)
        end_time = time.time()
        running_time = round((end_time - start_time)/60, 4)
        cv_results = random_search.cv_results_

        # Evaluate the model
        evaluation_results = self.evaluate_rf(random_search, X_train, X_test, y_train, y_test)

        # Evaluation on test set with all hyperparameter combinations of Random Search; only do it once, here for stratified 
        if stratified: 
            iteration_refit_test = self.iteration_results(random_search, X_train, y_train, X_test, y_test, cv_results)   
            return evaluation_results, cv_results, random_search.best_params_, running_time, results_descriptives_folds, iteration_refit_test
        else:
            return evaluation_results, cv_results, random_search.best_params_, running_time, results_descriptives_folds
        




    def iteration_results(self, RandomSearchObject, X_train, y_train, X_test, y_test, results_cv):
        mse_list = []
        mae_list = []
        r2_list = []
        for index, params_experiment in enumerate(results_cv["params"]):
            model = RandomSearchObject.best_estimator_ 

            model.set_params(**params_experiment)

            model.fit(X_train, y_train)
            test_r2 = round(model.score(X_test, y_test), 8)
            y_pred = model.predict(X_test)
            test_mse = round(mean_squared_error(y_test, y_pred), 8)
            test_mae = round(mean_absolute_error(y_test, y_pred), 8)
            r2_list.append(test_r2)
            mse_list.append(test_mse)
            mae_list.append(test_mae)
        return {'r2': r2_list, 'mse': mse_list, 'mae': mae_list}


    def create_cont_folds(self, 
                          y, 
                          n_folds, 
                          n_groups, 
                          seed):
        '''
        Function to create continuous folds.
        Inputs:
            y: the target variable
            n_folds: the number of folds
            n_groups: the number of groups (based on quantiles)
            seed: the seed to be used
        Outputs:
            cv_splits: the indices for the folds
        '''
        # create StratifiedKFold like for classification
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        # create groups in y with pd.qcut: quantile-based discretization 
        y_grouped = pd.qcut(y, math.ceil(n_groups), labels=False)

        # create fold numbers    
        fold_nums = np.zeros(len(y))
        #split(X, y[, groups]): Generate indices to split data into training and test set
        for fold_no, (t, v) in enumerate(skf.split(y_grouped, y_grouped)): 
            fold_nums[v] = fold_no

        cv_splits = []

        # iterate over folds and creat train and test indices for each fold
        for i in range(n_folds):
            test_indices = np.argwhere(fold_nums==i).flatten()
            train_indices = list(set(range(len(y_grouped))) - set(test_indices))
            cv_splits.append((train_indices, test_indices))

        return cv_splits


    def evaluate_rf(self, model, X_train, X_test, y_train, y_test):
        '''
        Function to evaluate the model.
        Inputs:
            model: the model to be evaluated
            X_train, X_test, y_train, y_test: the train and test data
        Outputs:
            dictionary with the evaluation results (R2, MSE, MAE)
        '''
        model=model.best_estimator_

        train_r2, test_r2 = round(model.score(X_train, y_train), 8), round(model.score(X_test, y_test), 8)
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
        train_mse, test_mse = round(mean_squared_error(y_train, y_train_pred), 8), round(mean_squared_error(y_test, y_test_pred), 8)
        train_mae, test_mae = round(mean_absolute_error(y_train, y_train_pred), 8), round(mean_absolute_error(y_test, y_test_pred), 8)
        return {'train r2': train_r2, 
                'test r2': test_r2, 
                'train mse': train_mse,
                'test mse': test_mse,
                'train mae': train_mae,
                'test mae': test_mae}
        
     
    def _convert_numpy_types(self, obj):
        '''
        Function to convert numpy types.
        Inputs:
            obj: the object to be converted
        Outputs:
            the converted object
        '''
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj

    
    def generate_data(self, n_samples_train, n_samples_test, noise, n_features, random_state_training, transformation):
        X_test, y_test = make_friedman1(n_samples=n_samples_test, n_features=n_features, noise=noise, random_state= self.global_seed_testing_data)
        X_train, y_train = make_friedman1(n_samples=n_samples_train, n_features=n_features, noise=noise, random_state=random_state_training)
        min_y_test = min(y_test)
        min_y_train = min(y_train)
        min_data = min(min_y_train, min_y_test)

        if min_data < 0:
            shifting = abs(min_data)
        else:
            shifting = 0
        
        y_train = self.transform(y_train, transformation, shifting=shifting)
        y_test = self.transform(y_test, transformation, shifting=shifting)
        
        return X_train, y_train, X_test, y_test


    def transform(self, y, transformation='identity', shifting=0):
        '''
        Function to transform the target variable.
        Inputs:
            transformation: the transformation to be applied
        Outputs:
            None (it transforms the target variable of the dataframe and of y itself)
        '''

        if transformation == 'identity':
            return y
        elif transformation == 'log':
            if shifting > 0:
                shifting = shifting + 1.00000001
            y = np.log(y + shifting)
        elif transformation == 'sqrt':
            y = np.sqrt(y + shifting)
        else:
            raise ValueError('Transformation not implemented.')
        return y
    

    def analysis_folds(self, data, fold_idxs, seed_num, stratified=False, plot = False):
        '''
        Function to visualize the folds.
        Inputs:
            data: the target variable
            fold_idxs: the number of folds
            seed_num: the seed numbers as a list
        Outputs:
            results: a dictionary containing the outputs of perform_ks_test() and plot_intersection()
        '''
        ks_statistic_list = []
        p_value_list = []
        intersection_area_list = []
    

        # for plotting title
        if stratified:
            stratified_title = "Stratified Split"
        else:
            stratified_title = "Random Split"

        if plot:
            fig, axs = plt.subplots(len(fold_idxs)//2, 2, figsize=(10,(len(fold_idxs)//2)*2))
            fig.suptitle(stratified_title + " with seed: " + str(seed_num), fontsize=10)
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

        for fold_id, (train_ids, val_ids) in enumerate(fold_idxs):
            # Perform the Kolmogorov-Smirnov test and store the results in the dictionary
            ks_statistic, p_value = self.perform_ks_test(data[train_ids], data[val_ids])

            # Calculate the intersection and store the result in the dictionary
            intersection_area = self.plot_intersection(data[train_ids], data[val_ids])
            ks_statistic_list.append(ks_statistic)
            p_value_list.append(p_value)
            intersection_area_list.append(intersection_area)
        results = {'ks_statistic': ks_statistic_list, 'p_value': p_value_list, 'intersection_area': intersection_area_list}
        
        mean_results = {}
        for key, values in results.items():
            mean_results[key] = np.mean(values)
        return mean_results
    

    def perform_ks_test(slef, data1, data2):
        # Perform the Kolmogorov-Smirnov test
        statistic, p_value = ks_2samp(data1, data2)
        return statistic, p_value
    

    def plot_intersection(self, data1, data2):
        data_dict = {'data1': data1, 'data2': data2}
        min_data = min(min(data) for data in data_dict.values())
        max_data = max(max(data) for data in data_dict.values())
        min_len = min([len(data) for data in data_dict.values()])
        x = np.linspace(min_data, max_data, min_len)
        ys = []
        for label, data in data_dict.items():
            kde_func = gaussian_kde(data)
            y = kde_func(x)
            ys.append(y)
        y_intersection = np.amin(ys, axis=0)
        area = np.trapz(y_intersection, x)
        return area

        