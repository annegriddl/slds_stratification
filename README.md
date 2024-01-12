# Seminar Statistical Learning and Data Science. 

Research question: To stratify in regression or not?

## Orientation in our Repository
Hello and welcome to our research seminar. The following is supposed to guide you through our repository and explain how to reproduce the results of our experiments. The python script ``Simulations/Final_Setup/run_experiments.py`` can be run either in the Terminal with *python run_experiments.py* or in VS Code by clicking on the Play button after the environment had been set up (Python version 3.12) and the packages in ``requirements.txt`` had been installed. Two new JSON files will then be created in ``Simulations/Final_Setup``, one contains a list with all seeds that are still available, the other one stores the results. Once created, everything will be saved in those files. 

**Important:** Set the path in the beginning of *json_file* and *path_to_seeds* to the directory + filename you want (for example *path_to_seeds = path_of_your_directory + "seeds_available.json"*). Also make sure to set the hyperparameters you want. If you want to try different settings, e.g. *n_train=200* and *n_train=1000*, specify that in the dictionary **hyperparameter_options**. All possible combinations of hyperparameters are then built.

## THAT'S ALL, THANKS!
