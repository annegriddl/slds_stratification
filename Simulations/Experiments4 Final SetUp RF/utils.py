# File for helper functions and classes

import sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import make_friedman1
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
import json
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV 
import random
import time


def plot_data_3D(df, axes=['X1', 'X2', 'X3']):
    '''
    Function to plot the data (here for Friedman dataset) in 3D. 
    Works also for other datasets with at least 3 features.
    Inputs:
        df: the dataframe with the data
        axes: the axes to be plotted
    Outputs:
        None (it plots the data)
    '''
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['X1'], df['X2'], df['X3'], c=df['y'], cmap='viridis', marker='o')

    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    ax.set_zlabel(axes[2])
    ax.set_title(f'3D Scatter Plot of {axes[0]}, {axes[1]} and {axes[2]}')

    plt.show()


def train_test_stratified(df, n_quantiles=20, train_size=0.8, seed=42):
    '''
    Function to split the data into train and test set.
    Not needed anymore since we use a fixed test set.
    '''
    # 1. Divide the DataFrame into quantiles
    quantiles = pd.qcut(df['y'], q=n_quantiles, labels=False)
    # 2. Randomly sample 80% of the data from each quantile for training
    train_data = pd.concat([group.sample(frac=train_size, random_state=seed) for _, group in df.groupby(quantiles)])
    # The remaining 20% will be used for testing
    test_data = df.drop(train_data.index)
    train_data = shuffle(train_data)
    test_data = shuffle(test_data)
    # 3. Already divide by features and target variables
    X_train, y_train = train_data.drop(columns=['y']), train_data['y']
    X_test, y_test = test_data.drop(columns=['y']), test_data['y']
    return X_train, X_test, y_train, y_test


def create_cont_folds(y,
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


def fold_visualizer(data, fold_idxs, seed_num, stratified= False):
    '''
    Function to visualize the folds.
    Inputs:
        data: the target variable
        fold_idxs: the number of folds
        seed_num: the seed numbers as a list
    Outputs:
        None (it plots the folds)
    '''
    # for plotting title
    if stratified:
        stratified_title = "Stratified Split"
    else:
        stratified_title = "Random Split"

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
    

def run_kfold_visualizer(labels, k, seeds, n_groups=10, stratified=False):
    ''' 
    Run the fold_visualizer function for a given dataset and number of folds.
    It visualizes the folds for each seed.
    Input:
        labels: the target variable
        k: number of folds
        seeds: list of seeds
        n_groups: number of groups for the stratification
        stratified: whether to use stratified k-fold or not
    Output:
        None (it plots the folds)
    '''
    for seed in seeds:
        if stratified:
            cv_splits = create_cont_folds(labels, 
                                          n_folds=k, 
                                          n_groups=n_groups, 
                                          seed=seed)
        else:
            cv_splits = list(KFold(k, 
                                   shuffle=True, 
                                   random_state=seed
                                   ).split(labels))
        
        fold_visualizer(data=labels, 
                        fold_idxs=cv_splits, 
                        seed_num=seed, stratified=stratified)

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



class ModelOptimizerFinal:
    '''
    Class to optimize the model.
    Inputs:
        model: the model to be optimized
        param_grid: the parameter grid to be used for the optimization
        random_state: the random state to be used
    '''
    def __init__(self, model, param_grid, random_state):
        self.model = model
        self.param_grid = param_grid
        self.random_state = random_state


    def optimize(self, 
                 params,
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
                                                     
        
        # get parameters from params dictionary
        n_train = params['n_train']
        n_test = params['n_test']
        n_features = params['n_features']
        noise = params['FD_noise']
        transformation = params['transformation']
        n_folds= params['n_folds']
        n_groups = params['n_groups']
        scoring = params['scoring']
        n_jobs = params['n_jobs']
        n_iter = params['n_iter']
        n_repetitions = params['n_repetitions']
        json_file = params['json_file']

        if data == 'friedman':
            # maybe implement accessing and generating the data nicer
            X_test, y_test = self.generate_friedman1(n_samples=n_test,
                                                     n_features=n_features,
                                                     noise=noise,
                                                     random_state=1718,
                                                     transformation=transformation)
        # @Anne und auch @Nadja: macht das Sinn?
        if not isinstance(random_states, list):
            random.seed(self.random_state)
            random_states = random.sample(range(1, 10000), n_repetitions)

        else:
            random_states = random_states[:n_repetitions]

        print("RandomizesdSearchCV with params n_folds =",
            n_folds, ", ngroups: ", n_groups, ", scoring: ",scoring, ", n_jobs: ",n_jobs,
            ", n_iter: ", n_iter, " and save to  ", json_file, "\n")
        
        all_results = {}
        all_results_stratified = {}

        initialization = {
            'model_info': params,
            'seed': self.random_state
        }
        for repetition in range(n_repetitions):
            if data == 'friedman':
                X_train, y_train = self.generate_friedman1(n_samples=n_train,
                                                        n_features=n_features,
                                                        noise=noise,
                                                        random_state=random_states[repetition],
                                                        transformation=transformation)
            # Check for NaN values in the data
            if np.isnan(y_train).any() or np.isnan(y_test).any(): 
                X_train, y_train = make_friedman1(n_samples=n_train,
                                    n_features=n_features, 
                                    noise=noise, 
                                    random_state=random_states[repetition])
                X_test, y_test = make_friedman1(n_samples=n_test,
                                    n_features=n_features,
                                    noise=noise,
                                    random_state=1718)
                min_val = min(y_train.min(), y_test.min())
                # @Anne: Noch absprechen... +1 weil es sonst Skala durch Werte [0, 1] hauptsächlich bei y_test größer wird.           
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

            # Perform optimization with unstratified cross-validation
            unstratified_results, unstratified_params, unstratified_running_time = self._perform_optimization(X_train, 
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
            all_results.update({f'Repetition {repetition}': unstratified_results})

            # Perform optimization with stratified cross-validation
            stratified_results, stratified_params, stratified_running_time = self._perform_optimization(X_train, 
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
            all_results_stratified.update({f'Repetition {repetition}': stratified_results}) # @Anne: check ich nicht ganz, was das macht

            if unstratified_params == stratified_params:
                hyperparameters_same = True
            else:
                hyperparameters_same = False
            
            # Save results and parameters to a file
            results = {
                'repetition': repetition,
                'random_state': random_states[repetition],
                'hyperparameters_same': hyperparameters_same,
                'unstratified_params':unstratified_params,
                'stratified_params': stratified_params,
                'unstratified_results': unstratified_results,
                'stratified_results': stratified_results,
                'unstratified_running_time': round(unstratified_running_time,2), 
                'stratified_running_time': round(stratified_running_time, 2)
            }

            initialization.update(results)
            # Load existing data or create an empty list
            with open(json_file, 'r') as file:
                existing_data = json.load(file)

            # Append the new results dictionary to the existing data
            existing_data.append(initialization)

            # Write the updated data back to the JSON file
            with open(json_file, 'w') as file:
                json.dump(existing_data, file, indent=4, default=self._convert_numpy_types)

        return all_results, all_results_stratified
    
    def save_results(self, results, path):
        '''
        Function to save the results to a JSON file.
        Inputs:
            results: the results to be saved
            json_file: the JSON file to be used
        Outputs:
            None (it saves the results to the JSON file)
        '''
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
        if stratified:
            cv_splits = self.create_cont_folds(y=y_train, n_folds=cv, n_groups=n_groups, seed=random_state)
            output_text = 'Stratified Split Cross-validation'
        else:
            cv_splits = cv
            output_text = 'Random Split Cross-validation'
        
        start_time = time.time()
        random_search = RandomizedSearchCV(estimator=self.model,
                                           param_distributions=self.param_grid,
                                           n_iter=n_iter,
                                           cv=cv_splits,
                                           scoring=scoring,
                                           n_jobs=n_jobs,
                                           random_state=random_state)
        random_search.fit(X_train, y_train)
        end_time = time.time()
        running_time = end_time - start_time
        print("Best Parameters:", random_search.best_params_)

        # Evaluate the model
        evaluation_results = self.evaluate_rf(random_search, X_train, X_test, y_train, y_test)
        print("Evaluation Results of", output_text, ': ', evaluation_results)
        print('running_time: ', round(running_time/60, 2), ' min')
        
        return evaluation_results, random_search.best_params_, running_time

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
        # @Anne: This somehow also does not work, do not know why.
        #best_score = model.best_score_

        train_r2, test_r2 = round(model.score(X_train, y_train), 4), round(model.score(X_test, y_test), 4)
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
        train_mse, test_mse = round(mean_squared_error(y_train, y_train_pred), 4), round(mean_squared_error(y_test, y_test_pred), 4)
        train_mae, test_mae = round(mean_absolute_error(y_train, y_train_pred), 4), round(mean_absolute_error(y_test, y_test_pred), 4)
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

        
