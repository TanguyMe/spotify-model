from src import prediction
import pandas as pd
import numpy

def test_predict():
    input_dict = {
        'key': [5], 'mode': [1], 'popularity': [44], 'duration_ms': [219707], 'danceability': [0.57], 'time_signature': [4],
        'loudness': [-8.968], 'speechiness': [0.043], 'instrumentalness': [0.000106], 'liveness': [0.13], 'valence': [0.569],
        'tempo': [119.865], 'acousticness*energy': [0.104606]
    }
    input_df = pd.DataFrame.from_dict(input_dict)
    result_predict = prediction.prediction(input_df.to_json())
    assert isinstance(result_predict, numpy.ndarray)
    assert len(result_predict) == 1
    assert isinstance(result_predict[0], numpy.int32)
