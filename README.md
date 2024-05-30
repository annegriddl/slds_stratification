# Stratified Sampling for Regression

Stratification is typically applied in classification to sample proportionally to the respective
class sizes in order to preserve the original class proportion in resulting subsets. In this way,
stratification can address issues arising from imbalanced and small datasets, overall leading
to a better generalization performance of machine learning models. In our experiemnts we
investigate the application of stratification in regression, focusing on its potential to improve
hyperparameter optimization and the final performance of regression models. We apply a
group-sized discretization of the target variable to create bins from which one can sample
proportionally during the cross-validation of Random Search. The results indicate that
our stratification approache compared to naive random sampling increases the overlap of
the training and validation data within cross-validation, but does not improve the final
performance of the models in our setting. However, the error of the Mean Squared Error
(MSE) estimator and its variance are reduced through stratification, indicating a better
estimation. The findings suggest that stratification could be particularly beneficial for
smaller datasets. We think that stratification should be considered as an optional sampling
approach within cross-validation for hyperparameter tuning and model selection.


## Project Organization

Hello and welcome to our research seminar!  
  
The following is supposed to guide you through our repository and explain how to reproduce the results of our experiments. Make sure to install all packages required in an environment with a Python version 3.12. To run the experiments, seed lists first need to be generated, if they do not already exists using the file ``create_seeds.py``. 

Then, the python script ``run_experiments.py`` can be executed which trains the defined models using both, unstratified and stratified cross-validation for 10 repetitions (default) and all 24 experimental parameter combinations. Make sure to specify the necessary parameters in the command line:

1. ``sys.argv[1]``: Whether to use parallelization      - True or False (default: True).
2. ``sys.argv[2]``: Model - Random Forest or XGBoost    - rf / xgb     (default: xgb)
3. ``sys.argv[3]``: Number of repetitions (integer, default: 10)

For example:
```bash
python run_experiments.py True xgb 20
```

A detailed explanation of our experiments and results can be found in our report.
The results of the conducted simulations are stored on the [LRZ](https://syncandshare.lrz.de/getlink/fi9NpAtAbwiJzAtvJKUv3T/) platform.


## Contents of Repository
```
    ├── attic              <- Old experiments that are not relevant
    │
    ├── evaluation         <- Evaluation of final results
    │
    ├── seeds              <- Seed lists for Random Forest and XGBoost to replicate results and 
    │                         to make sure to not use same seeds during experiments
    │  
    ├── .gitattributes     <- To track results saved in JSON format with Git LFS
    │
    ├── .gitignore     
    │
    ├── create_seeds.py    <- File to create two JSON files (for RF and XGB) with 100,000 seeds
    │                         Make sure to run this file if the seed lists in seeds/ do not exist      
    │
    ├── environment.yml    <- YAML file for setting up environment
    │
    ├── README.md  
    │
    ├── report_slds.pdf    <- Report with explanation of experiments and results 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment,
    │                         generated with `pip freeze > requirements.txt    
    │
    ├── run_experiments.py <- Main file to run experiments. 
    │                         Make sure to specify the parameters (see text above) when running the 
    │                         script in the command line.
    │                         
    ├── utils_final.py     <- Contains class to optimize the models with parallel Random Search
    │
    └── utils_parallel.py  <- Contains class to optimize the models with parallel repetitions
```
---


