# Deployment Guide

1. Install requirements

       `pip install -r requirements.txt`

2. Enter News-Market-Predictions directory

3. Check if embeddings exist on your device

       `ls -lh data/embeddings/ 2>/dev/null || echo "Run embeddings.py :)"`

  3.a Create embeddings if they don't exist for some reason (may take 10-20 minutes depending on your device)

     `python3 ML.embedding.py`

4. Run train.py

       `python3 -m ML.train --config config.yaml`

Should be good! Please reach out to haalexander@ucdavis.edu if you have any issues. 

## Backend API

To start the FastAPI backend server (for the frontend to connect to):

       `uvicorn ML.api:app --reload --port 8000`

The API will be running on `http://localhost:8000`. The backend loads the trained model and data on startup, and serves historical performance analysis endpoints. The React frontend connects to this API - make sure the backend is running before starting the frontend.

Note: The backend loads everything into memory at startup, so it might take a moment to be ready. Normalization stats are computed from the training set only to avoid lookahead bias. The dashboard shows historical performance of the next-day prediction model across the entire dataset (2018-2023), not live forecasts.

