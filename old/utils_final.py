# File for helper functions and classes

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


# Now some classes
class FriedmanDataset:
    # nur friedman ? oder auch anwendbar für andere datasets?
    def __init__(self, n_samples=1000, n_features=5, noise=0.0, random_state=42):
        self.features, self.y = self.generate_friedman1(n_samples, n_features, noise, random_state)
        self.df = self.to_dataframe(self.features, self.y)

    def generate_friedman1(self, n_samples, n_features, noise, random_state):
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
        features, y = make_friedman1(n_samples=n_samples, 
                                    n_features=n_features, 
                                    noise=noise, 
                                    random_state=random_state)
        return features, y

    def to_dataframe(self, features, y):
        ''' 
        Function to convert arrays to combined dataframe of X and y. (could also add normalization?)
        Inputs: 
            features: first output from generate_friedman1
            y: second output from generate_friedman1
        Output:
            combined dataframe
        '''
        features_df = pd.DataFrame(features, columns=[f'X{i}' for i in range(1, features.shape[1] + 1)])
        y_df = pd.DataFrame(y, columns=['y'])
        return pd.concat([features_df, y_df], axis=1)
    
    def transform(self, transformation='log'):
        '''
        Function to transform the target variable.
        Inputs:
            transformation: the transformation to be applied
        Outputs:
            None (it transforms the target variable of the dataframe and of y itself)
        '''
        if transformation == 'identity':
            pass
        elif transformation == 'log':
            self.df['y'] = np.log(self.df['y'])
            self.y = self.df['y'].values  
        elif transformation == 'sqrt':
            self.df['y'] = np.sqrt(self.df['y'])
            self.y = self.df['y'].values  
        else:
            raise ValueError('Transformation not implemented.')



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
        # Goal: same number of observations in each group independent of n_train #@Anne: richtig?    
        n_groups = int(n_train/group_size)
        if self.checks:
            print("Number of groups: ", n_groups)
        #print(f"RandomizesdSearchCV with parameters of experiment n_folds = {n_folds}, group_size = {group_size}, n_groups = {n_groups}, scoring = {scoring}, n_jobs = {n_jobs}, n_iter = {n_iter} and save to {json_file}")

        
        ### Create testing data with seed 1718 (use same seed in all experiments)
        if data == 'friedman':
            # maybe implement accessing and generating the data nicer
            X_test, y_test = self.generate_friedman1(n_samples=n_test,
                                                     n_features=n_features,
                                                     noise=noise,
                                                     random_state= self.global_seed_testing_data,
                                                     transformation=transformation)
            self.y_test_min = y_test.min()
            #print("Created testing data with seed: ", self.global_seed_testing_data)

        
        ### set seeds for all repetitions
        #######################################################################################
        # TODO: abchecken ob es richtig funktioniert. @Anne: verseth ich nicht 100%, habs mal auskommentiert, weil seeds wurden ja eignetlich schon erzeugt oder?
        if not isinstance(random_states, list):   # if random_states is None: load sedds form json_file
            print("\nLoad seeds from json: ", self.path_to_seeds)
            #if path does not exist, create file with empty list -> @Anen: glaub besser error und dann manuell Liste erzeugen
            if not os.path.exists(self.path_to_seeds):   
                print("cant find path")
                with open(self.path_to_seeds, 'w') as file:
                    json.dump([], file, indent=4)
                print("File created: ", self.path_to_seeds)
                random_states = [x for x in range(n_repetitions)]
                seeds_available = [x for x in range(100000)][n_repetitions:]
                print("Can't find path to seeds! Current paht: ", self.path_to_seeds) #@anne: would need to include in try and except
            # Else read the content of the JSON file
            else:
                try:
                    with open(self.path_to_seeds, 'r') as file:
                        seeds_available = json.load(file)
                except json.JSONDecodeError:
                    print("Error decoding JSON. The file might be empty or not properly formatted.")
                random_states = seeds_available[:n_repetitions]
                seeds_available = seeds_available[n_repetitions:]
            with open(self.path_to_seeds, 'w') as file:
                json.dump(seeds_available, file, indent=4)
                print(f"Successfully loaded and deleted picked seeds from json file!:\n {random_states}")
        else: # if random_states is list: use list as seeds #@Anne: hier auch nochmal checken, ob list
            random_states = random_states[:n_repetitions]
            print("Set seeds to: ", random_states, "for all iterations.\n")
        #######################################################################################


   
        # initalize dictionary with final_results, which is saved to json_file for evaluation
        final_results = {
            'model_info': params_experiment
        }

        #### Run Experiments for each repetition independently 
        for repetition in range(n_repetitions):
            start_time_repetition = time.time()
            # create training data with seed from random_states
            # check if generated data is negative -> then shift it to positive
            if data == 'friedman':
                X_train, y_train = self.generate_friedman1(n_samples=n_train,
                                                        n_features=n_features,
                                                        noise=noise,
                                                        random_state=random_states[repetition],
                                                        transformation=transformation)
              
            ##########################################################
            # TODO: weniger rechenintensiv (Funktion oben umschreiben) ->  check if generated data is negative -> then shift it to positive with min
            # Check for NaN values in the data
            if np.isnan(y_train).any() or np.isnan(y_test).any(): 
                X_train, y_train = make_friedman1(n_samples=n_train,
                                    n_features=n_features, 
                                    noise=noise, 
                                    random_state=random_states[repetition])
                X_test, y_test = make_friedman1(n_samples=n_test,
                                    n_features=n_features,
                                    noise=noise,
                                    random_state= self.global_seed_testing_data)
                min_val = min(y_train.min(), y_test.min())
                
                # Logarithmus +1, für Square Root nicht
                y_train = y_train + abs(min_val) + 1
                y_test = y_test + abs(min_val) + 1
                if transformation=='identity':
                    pass
                elif transformation == 'log':
                    y_train = np.log(y_train)
                    y_test = np.log(y_test)
                elif transformation == 'sqrt':
                    y_train = np.sqrt(y_train)
                    y_test = np.sqrt(y_test)
            ##########################################################  


            # Perform optimization with unstratified cross-validation
            unstratified_results, unstratified_iteration, unstratified_params, unstratified_running_time = self._perform_optimization(X_train, 
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
            stratified_results, stratified_iteration, stratified_params, stratified_running_time = self._perform_optimization(X_train, 
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
                'unstratified_running_time': round(unstratified_running_time,4), 
                'stratified_running_time': round(stratified_running_time, 4),
                'unstratified_iteration': unstratified_iteration,
                'stratified_iteration': stratified_iteration
            }

            final_results.update(results)

            self.save_results(final_results, json_file)

            # for printing
            if self.checks:
                print('seed for training data: ', random_states[repetition])

            if hyperparameters_same:
                hype_same = 'the same'
            else:
                hype_same = 'different'
            ent_time_repetition = time.time()

            print(f"Repetition {repetition+1} out of {n_repetitions} hyperparameter are {hype_same} and took {round((ent_time_repetition - start_time_repetition)/60, 4)} min")


        
    
    def save_results(self, results, path):
        '''
        Function to save the results to a JSON file.
        Inputs:
            results: the results to be saved
            json_file: the JSON file to be used
        Outputs:
            None (it saves the results to the JSON file)
        '''

        if not os.path.exists(path):   #@Anne: hier vlt auch besser Fehlermedung anstatt neue Datei
            with open(path, 'w') as file:
                json.dump([], file, indent=4)
            print("File not found, current path: ", path)
    
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
            cv_splits = self.create_cont_folds(y=y_train, n_folds=cv, n_groups=n_groups, seed=random_state)
            output_text = 'Stratified Split Cross-validation'
        else:
            cv_splits = cv
            output_text = 'Random Split Cross-validation'
        
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
                                           n_jobs=n_jobs,
                                           random_state=random_state)
        
        random_search.fit(X_train, y_train)
        end_time = time.time()
        running_time = end_time - start_time
        cv_results = random_search.cv_results_
        #print(f'Runing time Random Search of {output_text}: {round(running_time/60, 4)} min')
        #print("Best Parameters:", random_search.best_params_)

        # Evaluate the model
        evaluation_results = self.evaluate_rf(random_search, X_train, X_test, y_train, y_test)

        # Evaluation on test set with all hyperparameter combinations of Random Search #@Anne: Interpretation richtig?
        if stratified: 
            iteration_res = self.iteration_results(random_search, X_train, y_train, X_test, y_test, cv_results)  
            #print(iteration_res)
            cv_results.update(iteration_res)
        #print("Evaluation Results of", output_text, ': ', evaluation_results)
       
        
        return evaluation_results, cv_results, random_search.best_params_, running_time




    def iteration_results(self, RandomSearchObject, X_train, y_train, X_test, y_test, results_cv):
        mse_list = []
        mae_list = []
        r2_list = []
        for index, params_experiment in enumerate(results_cv["params"]):
            model = RandomSearchObject.best_estimator_ #@Anne: renamed model (input) to  RandomSearchObject, as two times model raised error for me; maybe double check
            #print(params_experiment)
            #print(model)
            model.set_params(**params_experiment)
            #print(model.get_params())
            test_r2 = round(model.score(X_test, y_test), 8)
            model.fit(X_train, y_train)
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
        for fold_no, (t, v) in enumerate(skf.split(y_grouped, y_grouped)): #@Nadja: unabhängig von n_folds? n_folds = fol_no, test_data_size = N/n_folds
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
        
    def generate_friedman1(self, n_samples, n_features, noise, random_state, transformation='identity'):
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
        features, y = make_friedman1(n_samples=n_samples, 
                                    n_features=n_features, 
                                    noise=noise, 
                                    random_state=random_state)
        if transformation=='identity':
            pass
        elif transformation == 'log':
            y = np.log(y)
             
        elif transformation == 'sqrt':
            y = np.sqrt(y)
             
        else:
            raise ValueError('Transformation not implemented.')

        return features, y
    

    