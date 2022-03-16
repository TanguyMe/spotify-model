import pickle
import pandas as pd
from sklearn import metrics
from pathlib import Path
import os
import json


def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    cluster_labels = estimator['model'].labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(X.index)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return metrics.silhouette_score(X, cluster_labels)


def prediction(input_json):
    df = pd.DataFrame.from_records(json.loads(input_json))
    current_directory = Path(__file__).parent.parent # Get current directory
    file = open(os.path.join(current_directory, 'model', 'birch5705+74.sav'), 'rb')
    model = pickle.load(file)
    return model.predict(df)

