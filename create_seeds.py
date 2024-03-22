"""
File to create a json file with 100000 random seeds.
"""

import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
filename_xgb = "seeds/seeds_final_xgb.json"
filename_rf = "seeds/seeds_final_rf.json"
path_to_seeds_xgb = script_dir + "/" + filename_xgb
path_to_seeds_rf = script_dir + "/" + filename_rf
overwrite = False

if __name__ == '__main__':
    for path_to_seeds, filename in zip([path_to_seeds_xgb, path_to_seeds_rf], [filename_xgb, filename_rf]):
        if not os.path.exists(path_to_seeds):
            random_states = [x for x in range(100000)]
            with open(path_to_seeds, 'w') as file:
                json.dump(random_states, file, indent=4)
            print("File created: ", filename)

        elif os.path.exists(path_to_seeds) and not overwrite:
            print("File for seeds already exists. If you want to overwrite it, set overwrite to True.")
        
        elif os.path.exists(path_to_seeds) and overwrite:
            random_states = [x for x in range(100000)]
            with open(path_to_seeds, 'w') as file:
                json.dump(random_states, file, indent=4)
                print("File overwritten: ", filename)