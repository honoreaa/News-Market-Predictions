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

The API will be running on `http://localhost:8000`. The backend loads the trained model and data on startup, and serves:
- **Historical performance analysis endpoints** — Dashboard with ticker summaries, search, etc.
- **Live prediction endpoints** — Real-time predictions using yfinance and Finnhub news

Note: The backend loads everything into memory at startup, so it might take a moment to be ready. Normalization stats are computed from the training set only to avoid lookahead bias.

## Live Prediction Feature

The `/predict-tomorrow` endpoint uses live news data from Finnhub. To enable this feature:
   Make a .env with:
      FINNHUB_API_KEY=d4j1ca1r01queual28cgd4j1ca1r01queual28d0

3. Start the backend:
   uvicorn ML.api:app --reload

The free tier allows 60 API calls per minute, which is sufficient for testing and moderate usage.

