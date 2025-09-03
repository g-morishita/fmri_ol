import json
import pandas as pd
from pathlib import Path

STAN_DATA_PATH = Path("../../preprocess/stan_data.json")
OUTPUT_DATA_PATH = Path("results/")

def load_json_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

