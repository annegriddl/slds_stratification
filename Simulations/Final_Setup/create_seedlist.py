import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
filename = "jsontest.json"
path_to_seeds = script_dir + "/" + filename
overwrite = False

if __name__ == '__main__':
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