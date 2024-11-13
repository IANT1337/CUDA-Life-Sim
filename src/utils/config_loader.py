import json

def load_config(file_path):
    with open(file_path) as config_file:
        return json.load(config_file)