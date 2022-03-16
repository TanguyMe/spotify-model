# from typing import Optional

from src import app
# from pydantic import BaseModel

from src.prediction import prediction
from flask import Flask, request, jsonify, Response
import json

import numpy as np

# class Item(BaseModel):
#     key: int
#     mode: int
#     popularity: int
#     duration_ms: int
#     danceability: float
#     time_signature: int
#     loudness: float
#     speechiness: float
#     instrumentalness: float
#     liveness: float
#     valence: float
#     tempo: float
#     acousticness_energy: float

# #  uvicorn api:api --host 0.0.0.0 --port 80
# app = FastAPI()


@app.route("/")
async def root():
    return {"message": "Hello World"}


@app.route("/prediction")
async def prediction_request():
    """Request that gives the predicted classes for a json containing the songs information"""
    audio_features = request.get_json()
    predictions = prediction(audio_features).tolist()
    print(predictions)
    print(json.dumps(predictions))
    return Response(json.dumps(predictions))
