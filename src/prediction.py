import pickle
import pandas as pd
from pathlib import Path
import os
import json


def prediction(input_json):
    df = pd.DataFrame.from_records(json.loads(input_json))
    current_directory = Path(__file__).parent.parent # Get current directory
    file = open(os.path.join(current_directory, 'model', 'fullkmeans1474+15.sav'), 'rb')
    model = pickle.load(file)
    return model.predict(df)

