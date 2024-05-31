# Towards understanding the usefulness of stratified resampling in regression task

Stratification is commonly used in classification to sample proportionally to the respective
class sizes, preserving the original class distribution in subsets. This approach addresses
issues with imbalanced and small datasets, improving the estimation of the generalization
performance of machine learning algorithms. This paper explores the use of stratification
in regression, examining its impact on the estimation of the generalization performance
and investigating its potential to enhance the hyperparameter optimization. We show that,
compared to naive random sampling in cross-validation, our stratification approach better
maintains the target distribution between training and validation data. Our stratification
approach provides more reliable estimates of the generalization performance, as measured
by the Mean Squared Error (MSE) of the performance estimate. Although this did not help
in finding better hyperparameter configurations, it also did not cause any harm.


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


