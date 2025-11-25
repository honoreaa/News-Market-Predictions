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

# Cloud Run Deployment

Backend (FastAPI + Torch)

1. Build and push the container (run from repo root):

   ```bash
   gcloud config set project newsmarketpredictions
   gcloud builds submit \
     --tag gcr.io/newsmarketpredictions/news-api \
     .
   ```

2. Deploy to Cloud Run (remember to allow unauthenticated access):

   ```bash
   gcloud run deploy news-api \
     --image gcr.io/newsmarketpredictions/news-api \
     --region=us-west1 \
     --allow-unauthenticated \
     --port=8080
   ```

   Copy the resulting URL; it becomes the `VITE_API_URL` for the frontend build.
   If your frontend is hosted elsewhere, set `ALLOWED_ORIGINS` (comma-separated) on
   the Cloud Run service so CORS allows that origin, e.g.
   `--set-env-vars ALLOWED_ORIGINS=https://newsmarketpredictions-frontend-xyz.a.run.app`.

Frontend (Vite + React)

1. Build + push, injecting the backend URL at build time:

   ```bash
   gcloud builds submit \
     --tag gcr.io/newsmarketpredictions/news-frontend \
     --build-arg VITE_API_URL="https://news-api-xxxxxx-uc.a.run.app" \
     .
   ```

2. Deploy:

   ```bash
   gcloud run deploy news-frontend \
     --image gcr.io/newsmarketpredictions/news-frontend \
     --region=us-west1 \
     --allow-unauthenticated \
     --port=8080
   ```

The published frontend URL is what you share with users on other computers. For local dev, copy `frontend/.env.example` to `.env` and run `npm run dev` after pointing `VITE_API_URL` at the locally running backend (`uvicorn ML.api:app --reload`).

## Live Prediction Feature

The `/predict-tomorrow` endpoint uses live news data from Finnhub. To enable this feature:

1. Get a free API key from [Finnhub](https://finnhub.io/register)
2. Set the `FINNHUB_API_KEY` environment variable when running the backend:
   ```bash
   export FINNHUB_API_KEY=your_api_key_here
   uvicorn ML.api:app --reload
   ```
3. For Cloud Run deployment, add it as an environment variable:
   ```bash
   gcloud run services update news-api \
     --set-env-vars FINNHUB_API_KEY=your_api_key_here \
     --region=us-west1
   ```

The free tier allows 60 API calls per minute, which is sufficient for testing and moderate usage.

