# Stratified Sampling for Regression
Author: Nadja Sauter and Anne Gritto         
Supervisor: Prof. Dr. Matthias Feurer and Dr. Giuseppe Casalicchio     
Seminar: Statistical Learning and Data Science    
Date: 22.03.2024     


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

Hello and welcome to our research seminar. The following is supposed to guide you through our repository and explain how to reproduce the results of our experiments. The python script ``Simulations/Final_Setup/run_experiments.py`` can be run either in the Terminal with *python run_experiments.py* or in VS Code by clicking on the Play button after the environment had been set up (Python version 3.12) and the packages in ``requirements.txt`` had been installed. Two new JSON files will then be created in ``Simulations/Final_Setup``, one contains a list with all seeds that are still available, the other one stores the results. Once created, everything will be saved in those files. 


    ├── attic              <- Old experiments that are not relevant
    │
    ├── evaluation         <- Evaluation of final results
    │
    ├── results            <- Results of runs for Random Forest and XGBoost used for evaluation
    │   └── rf
    │   └── xgb
    │
    ├── seeds              <- Seed lists for Random Forest and XGBoost to replicate results and 
    │                         to make sure to not use same seeds during experiments
    │  
    ├── .gitattributes     <- To track results saved in JSON format with Git LFS
    │
    ├── .gitignore          
    │
    ├── README.md   
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt    
    │
    ├── create_seeds.py    <- File to create two JSON files (for RF and XGB) with 100,000 seeds
    │                         Make sure to run this file if the seed lists in ``seeds/`` do not exist
    │
    ├── run_experiments.py <- Main file to run experiments. 
    │                         Make sure to specify the following parameters when running the scipt in the command line:
    │                            - sys.argv[1]: Whether to use parallelization - True / False (default: True)
    │                            - sys.argv[2]: Model name                     - rf / xgb     (default: xgb)
    │                            - sys.argv[3]: Number of repetitions (int)
    │
    ├── utils_final.py     <- Contains class to optimize the models with **parallel Random Search**
    │
    └── utils_parallel.py  <- Contains class to optimize the models with **parallel repetitions**

---


