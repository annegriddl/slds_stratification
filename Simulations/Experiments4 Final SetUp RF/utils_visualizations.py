import pandas as pd

def flatten_data(data_all):
    # Flatten the nested dictionaries
    flat_data = []
    for entry in data_all:
        flat_entry = {}
        for key, value in entry.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_entry[key + "_" + sub_key] = sub_value
            else:
                flat_entry[key] = value
        flat_data.append(flat_entry)

    df = pd.DataFrame(flat_data)
    return df