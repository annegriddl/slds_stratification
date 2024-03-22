# Stratified Sampling for Regression
## Seminar Statistical Learning and Data Science. 

Author: Nadja Sauter, Anne Gritto      
Supervisor: Prof. Dr. Matthias Feurer and Dr. Giuseppe Casalicchio   
Date: 22.03.2024   

## Abstract
Stratification is typically applied in classification to sample proportionally to the respective
class sizes in order to preserve the original class proportion in resulting subsets. In this way,
stratification can address issues arising from imbalanced and small datasets, overall leading
to a better generalization performance of machine learning models. In our experiemnts we
investigate the application of stratification in regression, focusing on its potential to improve
hyperparameter optimization and the final performance of regression models. We apply a
group-sized discretization of the target variable to create bins from which one can sample
proportionally during the cross-validation of Random Search. The results indicate that
our stratification approache compared to naive random sampling increases the overlap of
training and validation data within cross-validation, but does not significantly improve the
final performance of the models. However, the error of the Mean Squared Error (MSE)
estimator and its variance are reduced through stratification, indicating a better estimation.
The findings suggest that stratification could be particularly beneficial for smaller datasets.
We think that stratification should be considered as an optional sampling approach within
cross-validation for hyperparameter tuning and model selection.


## Orientation in our Repository
Hello and welcome to our research seminar. The following is supposed to guide you through our repository and explain how to reproduce the results of our experiments. The python script ``Simulations/Final_Setup/run_experiments.py`` can be run either in the Terminal with *python run_experiments.py* or in VS Code by clicking on the Play button after the environment had been set up (Python version 3.12) and the packages in ``requirements.txt`` had been installed. Two new JSON files will then be created in ``Simulations/Final_Setup``, one contains a list with all seeds that are still available, the other one stores the results. Once created, everything will be saved in those files. 

**Important:** Set the path in the beginning of *json_file* and *path_to_seeds* to the directory + filename you want (for example *path_to_seeds = path_of_your_directory + "seeds_available.json"*). Also make sure to set the hyperparameters you want. If you want to try different settings, e.g. *n_train=200* and *n_train=1000*, specify that in the dictionary **hyperparameter_options**. All possible combinations of hyperparameters are then built.

## THAT'S ALL, THANKS!
