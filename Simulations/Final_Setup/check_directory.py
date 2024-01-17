import os
import json

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
# Construct the path to the test.json file
#path2 = os.path.join(script_dir, "Simulations", "Final_Setup", "test.json")
with open(script_dir + "/results/results-run-20240114-1.json", "r") as f:
    data = json.load(f)

