import os
import logging
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import torch
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv

from ML.model import NewsClassifier
from ML import live_news

# Load environment variables from .env file
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/news_classifier.pt")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "yiyanghkust/finbert-tone")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
PRICE_MEAN = np.array(
    [0.0007642, 15.93788, 0.00225, 0.00525, 0.01761, 0.5160, 0.005935],
    dtype=np.float32,
)
PRICE_STD = np.array(
    [0.02126, 1.3451, 0.035676, 0.053319, 0.01252, 0.3053, 0.322114],
    dtype=np.float32,
)

logger = logging.getLogger("uvicorn")


class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=8)
    headline: str = Field(..., min_length=5, max_length=512)


class PredictResponse(BaseModel):
    ticker: str
    probability: float
    direction: str


class PredictTomorrowRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=8)


class PredictTomorrowResponse(BaseModel):
    ticker: str
    date: str  # prediction date (tomorrow)
    probability: float
    direction: str
    headlines_used: int


@lru_cache(maxsize=1)
def load_classifier():
    logger.info("Loading NewsClassifier weights from %s", MODEL_PATH)
    model = NewsClassifier()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


@lru_cache(maxsize=1)
def load_text_models():
    cache_dir = os.getenv("HF_HOME")
    logger.info("Loading transformer backbone %s", TEXT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, cache_dir=cache_dir)
    bert_model = AutoModel.from_pretrained(TEXT_MODEL_NAME, cache_dir=cache_dir)
    bert_model.eval()
    return tokenizer, bert_model


def get_price_features(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1mo")
        if len(df) < 8:
            return None

        df["close_return_1d"] = df["Close"].pct_change().fillna(0)
        df["vol_log"] = np.log1p(df["Volume"]).fillna(0)
        df["close_return_3d"] = df["Close"].pct_change(periods=3).fillna(0)
        df["close_return_7d"] = df["Close"].pct_change(periods=7).fillna(0)
        df["volatility_7d"] = df["close_return_1d"].rolling(window=7).std().fillna(0)
        df["price_position"] = ((df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-8)).fillna(0.5)
        vol_avg_7d = df["Volume"].rolling(window=7).mean().fillna(df["Volume"])
        df["volume_trend"] = (df["Volume"] / (vol_avg_7d + 1e-8) - 1).fillna(0)

        latest = df.iloc[-1]
        return np.array(
            [
                latest["close_return_1d"],
                latest["vol_log"],
                latest["close_return_3d"],
                latest["close_return_7d"],
                latest["volatility_7d"],
                latest["price_position"],
                latest["volume_trend"],
            ],
            dtype=np.float32,
        )
    except Exception as exc:  # pragma: no cover - downstream API errors
        logger.error("Failed to fetch YFinance data for %s: %s", ticker, exc)
        return None


def get_text_embedding(text: str):
    tokenizer, bert_model = load_text_models()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * attention_mask).sum(1)
        counts = attention_mask.sum(1).clamp(min=1)
    embedding = summed / counts
    return embedding


def run_inference(ticker: str, headline: str):
    raw_prices = get_price_features(ticker)
    if raw_prices is None:
        raise HTTPException(
            status_code=400,
            detail="Could not fetch price history for ticker.",
        )

    norm_prices = (raw_prices - PRICE_MEAN) / (PRICE_STD + 1e-8)
    price_tensor = torch.tensor(norm_prices, dtype=torch.float32).unsqueeze(0)
    emb_tensor = get_text_embedding(headline)

    model = load_classifier()
    with torch.no_grad():
        prob = model(emb_tensor, price_tensor).item()

    direction = "UP" if prob > 0.5 else "DOWN"
    return prob, direction


def _parse_allowed_origins(raw_origins: str):
    if raw_origins == "*":
        return ["*"]
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


app = FastAPI(title="News Market Predictions API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    ticker = payload.ticker.upper()
    probability, direction = run_inference(ticker, payload.headline)
    return PredictResponse(
        ticker=ticker,
        probability=probability,
        direction=direction,
    )


@app.post("/predict-tomorrow", response_model=PredictTomorrowResponse)
async def predict_tomorrow(payload: PredictTomorrowRequest):
    """
    Predict tomorrow's stock movement for a given ticker using today's data.
    Fetches live news from Finnhub and current price data from yfinance.
    """
    ticker = payload.ticker.upper()
    
    # Get today's price features
    raw_prices = get_price_features(ticker)
    if raw_prices is None:
        raise HTTPException(
            status_code=400,
            detail="Could not fetch price history for ticker.",
        )
    
    # Normalize price features using training statistics
    norm_prices = (raw_prices - PRICE_MEAN) / (PRICE_STD + 1e-8)
    price_tensor = torch.tensor(norm_prices, dtype=torch.float32).unsqueeze(0)
    
    # Get today's news embedding
    tokenizer, bert_model = load_text_models()
    emb_tensor, headlines_count = live_news.get_today_news_embedding(
        ticker, tokenizer, bert_model
    )
    
    # Run inference
    model = load_classifier()
    with torch.no_grad():
        prob = model(emb_tensor, price_tensor).item()
    
    direction = "UP" if prob > 0.5 else "DOWN"
    
    # Calculate tomorrow's date
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    return PredictTomorrowResponse(
        ticker=ticker,
        date=tomorrow,
        probability=prob,
        direction=direction,
        headlines_used=headlines_count,
    )



