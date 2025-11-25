"""
FastAPI backend for News Market Predictor API.
Serves date-based batch predictions from pre-computed embeddings and trained model,
plus live prediction using Finnhub news API.
"""
import os
import logging
from datetime import datetime, date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv

from .model import NewsClassifier
from . import live_news

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="News Market Predictor API", version="1.0.0")

# Configure CORS for frontend access
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

def _parse_allowed_origins(raw_origins: str):
    if raw_origins == "*":
        return ["*"]
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded data and model (historical analysis)
META_DF: Optional[pd.DataFrame] = None
EMBS: Optional[np.ndarray] = None
PROCESSED_DF: Optional[pd.DataFrame] = None
MODEL: Optional[NewsClassifier] = None
PRICE_MEAN: Optional[torch.Tensor] = None
PRICE_STD: Optional[torch.Tensor] = None
PRICE_FEATURE_COLS = [
    "close_return_1d",
    "vol_log",
    "close_return_3d",
    "close_return_7d",
    "volatility_7d",
    "price_position",
    "volume_trend",
]

# Live prediction constants (pre-computed from training)
LIVE_PRICE_MEAN = np.array(
    [0.0007642, 15.93788, 0.00225, 0.00525, 0.01761, 0.5160, 0.005935],
    dtype=np.float32,
)
LIVE_PRICE_STD = np.array(
    [0.02126, 1.3451, 0.035676, 0.053319, 0.01252, 0.3053, 0.322114],
    dtype=np.float32,
)

TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "yiyanghkust/finbert-tone")


def _build_price_feature_vector(df: pd.DataFrame, idx: int) -> np.ndarray:
    """
    Build the 7-dimensional price feature vector for a given row index.
    Missing columns or NaN values are treated as 0.0.
    """
    price_feat = []
    for feat_name in PRICE_FEATURE_COLS:
        if feat_name in df.columns:
            val = df.loc[idx, feat_name]
            price_feat.append(float(val) if pd.notna(val) else 0.0)
        else:
            price_feat.append(0.0)

    if len(price_feat) < len(PRICE_FEATURE_COLS):
        price_feat.extend([0.0] * (len(PRICE_FEATURE_COLS) - len(price_feat)))

    return np.array(price_feat[: len(PRICE_FEATURE_COLS)], dtype=float)


def load_data_and_model():
    """Load all required data and model on startup."""
    global META_DF, EMBS, PROCESSED_DF, MODEL, PRICE_MEAN, PRICE_STD
    
    logger.info("Loading data and model...")
    
    # Paths relative to project root
    project_root = Path(__file__).parent.parent
    meta_path = project_root / "data" / "embeddings" / "meta.csv"
    embeddings_path = project_root / "data" / "embeddings" / "embeddings.npz"
    processed_path = project_root / "data" / "processed" / "processed.csv"
    model_path = project_root / "models" / "news_classifier.pt"
    config_path = project_root / "config.yaml"
    
    # Load meta.csv
    logger.info(f"Loading meta.csv from {meta_path}")
    META_DF = pd.read_csv(meta_path, parse_dates=['Date'])
    logger.info(f"Loaded {len(META_DF)} rows from meta.csv")
    
    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings_data = np.load(embeddings_path)
    EMBS = embeddings_data['embeddings']
    logger.info(f"Loaded embeddings with shape {EMBS.shape}")
    
    # Load processed.csv for headlines
    logger.info(f"Loading processed.csv from {processed_path}")
    PROCESSED_DF = pd.read_csv(processed_path, parse_dates=['Date'])
    logger.info(f"Loaded {len(PROCESSED_DF)} rows from processed.csv")
    
    # Ensure all three sources align
    min_len = min(len(EMBS), len(META_DF), len(PROCESSED_DF))
    if len(EMBS) != min_len:
        logger.warning("Truncating embeddings to match min_len")
        EMBS = EMBS[:min_len]
    if len(META_DF) != min_len:
        logger.warning("Truncating meta dataframe to match min_len")
        META_DF = META_DF.iloc[:min_len].reset_index(drop=True)
    if len(PROCESSED_DF) != min_len:
        logger.warning("Truncating processed dataframe to match min_len")
        PROCESSED_DF = PROCESSED_DF.iloc[:min_len].reset_index(drop=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    MODEL = NewsClassifier(
        emb_dim=768,
        price_dim=7,
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout']
    )
    
    MODEL.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    MODEL.eval()
    MODEL.to('cpu')
    logger.info("Model loaded successfully")
    
    # Compute normalization statistics from training set only
    logger.info("Computing normalization statistics from training set...")
    train_split_date = pd.to_datetime(config["training"]["train_split_date"])
    train_mask = META_DF["Date"] < train_split_date
    train_indices = META_DF[train_mask].index.tolist()
    
    if not train_indices:
        raise RuntimeError(f"No training rows found before train_split_date={train_split_date!r}")

    sample_size = min(1000, len(train_indices))
    sampled_indices = np.random.choice(train_indices, size=sample_size, replace=False)

    train_price_features = []
    for idx in sampled_indices:
        price_vec = _build_price_feature_vector(META_DF, idx)
        train_price_features.append(price_vec)

    train_price_features = np.array(train_price_features, dtype=float)
    PRICE_MEAN = torch.tensor(train_price_features.mean(axis=0), dtype=torch.float32)
    PRICE_STD = torch.tensor(train_price_features.std(axis=0) + 1e-8, dtype=torch.float32)

    logger.info("Data and model loading complete!")


# Load data on startup
@app.on_event("startup")
async def startup_event():
    """Load data and model when the API starts."""
    try:
        load_data_and_model()
    except Exception as e:
        logger.error(f"Failed to load data/model on startup: {e}", exc_info=True)
        raise


# ============== Pydantic models ==============

# Live prediction models
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
    date: str
    probability: float
    direction: str
    headlines_used: int


# Historical analysis models
class PredictionItem(BaseModel):
    """Single prediction result for one ticker."""
    ticker: str
    date: str
    headline: str
    prob_up: float
    prediction: str
    true_label: Optional[int] = None


class DatePredictionResponse(BaseModel):
    """Batch prediction response for a specific date."""
    date: str
    count: int
    items: list[PredictionItem]
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    rows: int
    available_dates_range: Optional[dict] = None


class AvailableDatesResponse(BaseModel):
    """Available dates response."""
    dates: list[str]
    count: int


class OverallSummaryResponse(BaseModel):
    """Overall summary statistics."""
    all_predictions: dict
    with_headlines: dict


class TickerSummaryItem(BaseModel):
    """Summary for a single ticker."""
    ticker: str
    n_total: int
    n_with_headline: int
    accuracy_total: Optional[float] = None
    accuracy_with_headline: Optional[float] = None
    avg_prob_up: float
    avg_prob_up_with_headline: Optional[float] = None


class TickerSummaryResponse(BaseModel):
    """Ticker summary response."""
    tickers: list[TickerSummaryItem]
    count: int


class TickerPredictionsResponse(BaseModel):
    """All predictions for a specific ticker."""
    ticker: str
    count: int
    items: list[PredictionItem]


class HeadlineSearchResponse(BaseModel):
    """Headline search results."""
    query: str
    count: int
    items: list[PredictionItem]


# ============== Live prediction helpers ==============

@lru_cache(maxsize=1)
def load_text_models():
    cache_dir = os.getenv("HF_HOME")
    logger.info("Loading transformer backbone %s", TEXT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, cache_dir=cache_dir)
    bert_model = AutoModel.from_pretrained(TEXT_MODEL_NAME, cache_dir=cache_dir)
    bert_model.eval()
    bert_model.to('cpu')  # Ensure consistent device with main model
    return tokenizer, bert_model


def get_live_price_features(ticker: str):
    """Fetch live price features from yfinance."""
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
    except Exception as exc:
        logger.error("Failed to fetch YFinance data for %s: %s", ticker, exc)
        return None


def get_text_embedding(text: str):
    """Generate FinBERT embedding for a single text."""
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


# ============== Historical analysis helpers ==============

def _compute_predictions_for_indices(indices: list[int]) -> list[PredictionItem]:
    """Compute predictions for a list of row indices."""
    if not indices:
        return []
    
    try:
        emb_batch = EMBS[indices]
        price_batch = [_build_price_feature_vector(META_DF, idx) for idx in indices]
        price_batch = np.array(price_batch, dtype=np.float32)
        
        emb_tensor = torch.FloatTensor(emb_batch)
        price_tensor = torch.FloatTensor(price_batch)
        
        price_tensor_normalized = (price_tensor - PRICE_MEAN) / PRICE_STD
        
        with torch.no_grad():
            probs = MODEL(emb_tensor, price_tensor_normalized)
        
        items = []
        for i, idx in enumerate(indices):
            prob = probs[i].item()
            prediction = "UP" if prob > 0.5 else "DOWN"
            
            headline = ""
            if idx < len(PROCESSED_DF) and 'headlines' in PROCESSED_DF.columns:
                headline_val = PROCESSED_DF.loc[idx, 'headlines']
                headline = str(headline_val) if pd.notna(headline_val) else ""
            
            ticker = str(META_DF.loc[idx, 'ticker'])
            date_str = META_DF.loc[idx, 'Date'].strftime('%Y-%m-%d')
            true_label = None
            if 'label' in META_DF.columns:
                label_val = META_DF.loc[idx, 'label']
                if pd.notna(label_val):
                    true_label = int(label_val)
            
            items.append(PredictionItem(
                ticker=ticker,
                date=date_str,
                headline=headline,
                prob_up=prob,
                prediction=prediction,
                true_label=true_label
            ))
        
        return items
    except Exception as e:
        logger.error(f"Error computing predictions for indices: {e}", exc_info=True)
        raise


# ============== Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if META_DF is None or MODEL is None:
        return HealthResponse(status="error", model_loaded=False, rows=0)
    
    date_range = None
    if len(META_DF) > 0:
        date_range = {
            "min": META_DF['Date'].min().isoformat(),
            "max": META_DF['Date'].max().isoformat()
        }
    
    return HealthResponse(
        status="ok",
        model_loaded=True,
        rows=len(META_DF),
        available_dates_range=date_range
    )


@app.get("/healthz")
async def health_check_simple():
    """Simple health check for load balancers."""
    return {"status": "ok"}


# ============== Live Prediction Endpoints ==============

@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    """Predict using a manually provided headline."""
    ticker = payload.ticker.upper()
    
    raw_prices = get_live_price_features(ticker)
    if raw_prices is None:
        raise HTTPException(status_code=400, detail="Could not fetch price history for ticker.")

    norm_prices = (raw_prices - LIVE_PRICE_MEAN) / (LIVE_PRICE_STD + 1e-8)
    price_tensor = torch.tensor(norm_prices, dtype=torch.float32).unsqueeze(0)
    emb_tensor = get_text_embedding(payload.headline)

    with torch.no_grad():
        prob = MODEL(emb_tensor, price_tensor).item()

    direction = "UP" if prob > 0.5 else "DOWN"
    return PredictResponse(ticker=ticker, probability=prob, direction=direction)


@app.post("/predict-tomorrow", response_model=PredictTomorrowResponse)
async def predict_tomorrow(payload: PredictTomorrowRequest):
    """
    Predict tomorrow's stock movement using today's live news from Finnhub.
    """
    ticker = payload.ticker.upper()
    
    raw_prices = get_live_price_features(ticker)
    if raw_prices is None:
        raise HTTPException(status_code=400, detail="Could not fetch price history for ticker.")
    
    norm_prices = (raw_prices - LIVE_PRICE_MEAN) / (LIVE_PRICE_STD + 1e-8)
    price_tensor = torch.tensor(norm_prices, dtype=torch.float32).unsqueeze(0)
    
    tokenizer, bert_model = load_text_models()
    emb_tensor, headlines_count = live_news.get_today_news_embedding(
        ticker, tokenizer, bert_model
    )
    
    with torch.no_grad():
        prob = MODEL(emb_tensor, price_tensor).item()
    
    direction = "UP" if prob > 0.5 else "DOWN"
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    return PredictTomorrowResponse(
        ticker=ticker,
        date=tomorrow,
        probability=prob,
        direction=direction,
        headlines_used=headlines_count,
    )


# ============== Historical Analysis Endpoints ==============

@app.get("/available-dates", response_model=AvailableDatesResponse)
async def get_available_dates():
    """Get list of all available dates in the dataset."""
    if META_DF is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    unique_dates = sorted(META_DF['Date'].dt.date.unique())
    date_strings = [d.isoformat() for d in unique_dates]
    
    return AvailableDatesResponse(dates=date_strings, count=len(date_strings))


@app.get("/summary/overall", response_model=OverallSummaryResponse)
async def get_overall_summary():
    """Get overall summary statistics across all predictions."""
    if META_DF is None or EMBS is None or PROCESSED_DF is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    if PRICE_MEAN is None or PRICE_STD is None:
        raise HTTPException(status_code=503, detail="Normalization statistics not computed")
    
    try:
        all_indices = list(range(len(META_DF)))
        all_items = _compute_predictions_for_indices(all_indices)
        
        items_with_headlines = [
            item for item in all_items 
            if item.headline and item.headline.strip() and item.headline.lower() != 'nan'
        ]
        
        def calculate_metrics(items):
            if not items:
                return {"total": 0, "accuracy": None, "avg_prob_up": 0.0, "up_count": 0, "down_count": 0}
            
            total = len(items)
            up_count = sum(1 for item in items if item.prediction == "UP")
            down_count = total - up_count
            avg_prob_up = sum(item.prob_up for item in items) / total
            
            items_with_labels = [item for item in items if item.true_label is not None]
            accuracy = None
            if items_with_labels:
                correct = sum(1 for item in items_with_labels if (item.prob_up > 0.5) == (item.true_label == 1))
                accuracy = (correct / len(items_with_labels)) * 100
            
            return {"total": total, "accuracy": accuracy, "avg_prob_up": avg_prob_up, "up_count": up_count, "down_count": down_count}
        
        return OverallSummaryResponse(
            all_predictions=calculate_metrics(all_items),
            with_headlines=calculate_metrics(items_with_headlines)
        )
    
    except Exception as e:
        logger.error(f"Error computing overall summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/summary/tickers", response_model=TickerSummaryResponse)
async def get_ticker_summary():
    """Get summary statistics for each ticker."""
    if META_DF is None or EMBS is None or PROCESSED_DF is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    if PRICE_MEAN is None or PRICE_STD is None:
        raise HTTPException(status_code=503, detail="Normalization statistics not computed")
    
    try:
        all_indices = list(range(len(META_DF)))
        all_items = _compute_predictions_for_indices(all_indices)
        
        ticker_groups = {}
        for item in all_items:
            if item.ticker not in ticker_groups:
                ticker_groups[item.ticker] = []
            ticker_groups[item.ticker].append(item)
        
        ticker_summaries = []
        for ticker, items in ticker_groups.items():
            items_with_headlines = [
                item for item in items
                if item.headline and item.headline.strip() and item.headline.lower() != 'nan'
            ]
            
            n_total = len(items)
            avg_prob_up = sum(item.prob_up for item in items) / n_total if items else 0.0
            
            items_with_labels = [item for item in items if item.true_label is not None]
            accuracy_total = None
            if items_with_labels:
                correct = sum(1 for item in items_with_labels if (item.prob_up > 0.5) == (item.true_label == 1))
                accuracy_total = (correct / len(items_with_labels)) * 100
            
            n_with_headline = len(items_with_headlines)
            avg_prob_up_with_headline = None
            accuracy_with_headline = None
            
            if items_with_headlines:
                avg_prob_up_with_headline = sum(item.prob_up for item in items_with_headlines) / n_with_headline
                
                items_with_headlines_and_labels = [item for item in items_with_headlines if item.true_label is not None]
                if items_with_headlines_and_labels:
                    correct = sum(1 for item in items_with_headlines_and_labels if (item.prob_up > 0.5) == (item.true_label == 1))
                    accuracy_with_headline = (correct / len(items_with_headlines_and_labels)) * 100
            
            ticker_summaries.append(TickerSummaryItem(
                ticker=ticker,
                n_total=n_total,
                n_with_headline=n_with_headline,
                accuracy_total=accuracy_total,
                accuracy_with_headline=accuracy_with_headline,
                avg_prob_up=avg_prob_up,
                avg_prob_up_with_headline=avg_prob_up_with_headline
            ))
        
        ticker_summaries.sort(key=lambda x: x.ticker)
        
        return TickerSummaryResponse(tickers=ticker_summaries, count=len(ticker_summaries))
    
    except Exception as e:
        logger.error(f"Error computing ticker summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/ticker/{ticker}/predictions", response_model=TickerPredictionsResponse)
async def get_ticker_predictions(ticker: str):
    """Get all historical predictions for a specific ticker."""
    if META_DF is None or EMBS is None or PROCESSED_DF is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    if PRICE_MEAN is None or PRICE_STD is None:
        raise HTTPException(status_code=503, detail="Normalization statistics not computed")
    
    try:
        ticker_mask = META_DF['ticker'] == ticker
        matching_indices = META_DF[ticker_mask].index.tolist()
        
        if len(matching_indices) == 0:
            return TickerPredictionsResponse(ticker=ticker, count=0, items=[])
        
        items = _compute_predictions_for_indices(matching_indices)
        
        return TickerPredictionsResponse(ticker=ticker, count=len(items), items=items)
    
    except Exception as e:
        logger.error(f"Error getting ticker predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/search/headlines", response_model=HeadlineSearchResponse)
async def search_headlines(q: str):
    """Search predictions by headline text (case-insensitive)."""
    if META_DF is None or EMBS is None or PROCESSED_DF is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    if PRICE_MEAN is None or PRICE_STD is None:
        raise HTTPException(status_code=503, detail="Normalization statistics not computed")
    
    if not q or not q.strip():
        return HeadlineSearchResponse(query=q, count=0, items=[])
    
    try:
        query_lower = q.lower().strip()
        matching_indices = []
        
        for idx in range(len(PROCESSED_DF)):
            if 'headlines' in PROCESSED_DF.columns:
                headline_val = PROCESSED_DF.loc[idx, 'headlines']
                if pd.notna(headline_val):
                    headline_str = str(headline_val).lower()
                    if query_lower in headline_str:
                        matching_indices.append(idx)
        
        if len(matching_indices) == 0:
            return HeadlineSearchResponse(query=q, count=0, items=[])
        
        items = _compute_predictions_for_indices(matching_indices)
        
        return HeadlineSearchResponse(query=q, count=len(items), items=items)
    
    except Exception as e:
        logger.error(f"Error searching headlines: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/predict/date/{date_string}", response_model=DatePredictionResponse)
async def predict_by_date(date_string: str):
    """Get predictions for all tickers on a specific date."""
    if META_DF is None or EMBS is None or PROCESSED_DF is None or MODEL is None:
        raise HTTPException(status_code=503, detail="Model or data not loaded")
    
    if PRICE_MEAN is None or PRICE_STD is None:
        raise HTTPException(status_code=503, detail="Normalization statistics not computed")
    
    try:
        requested_date = datetime.strptime(date_string, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date_string}. Expected YYYY-MM-DD.")
    
    date_mask = META_DF['Date'].dt.date == requested_date
    matching_indices = META_DF[date_mask].index.tolist()
    
    if len(matching_indices) == 0:
        return DatePredictionResponse(date=date_string, count=0, items=[], message=f"No records found for date {date_string}")
    
    try:
        items = _compute_predictions_for_indices(matching_indices)
        return DatePredictionResponse(date=date_string, count=len(items), items=items, message=None)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
