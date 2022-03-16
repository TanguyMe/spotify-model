#! /usr/bin/env python
from src import app
from prediction import cv_silhouette_scorer

if __name__ == "__main__":
    app.run(port=5000, debug=True)
