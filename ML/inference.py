import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModel

class NewsClassifier(nn.Module):
    def __init__(self, emb_dim=768, price_dim=7, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + price_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, emb, price):
        x = torch.cat([emb, price], dim=1)
        return self.net(x).squeeze(1)

MODEL_PATH = '../models/news_classifier.pt'
TEXT_MODEL_NAME = 'yiyanghkust/finbert-tone' # embedding model

# from meta.csv file
PRICE_MEAN = np.array([0.0007642, 15.93788, 0.002250, 0.005250, 0.01761, 0.5160, 0.005935])
PRICE_STD  = np.array([0.0212600, 1.345100, 0.035676, 0.053319, 0.01252, 0.3053, 0.322114])

def get_price_features(ticker):
    # last 30 days to ensure we have enough data for rolling windows (7 days)
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1mo")
        if len(df) < 8:
            raise ValueError("Not enough history data for this ticker")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

    # features exactly as defined in data_processing.py
    df['close_return_1d'] = df['Close'].pct_change().fillna(0)
    df['vol_log'] = np.log1p(df['Volume']).fillna(0)
    df['close_return_3d'] = df['Close'].pct_change(periods=3).fillna(0)
    df['close_return_7d'] = df['Close'].pct_change(periods=7).fillna(0)
    
    # rolling volatility
    df['volatility_7d'] = df['close_return_1d'].rolling(window=7).std().fillna(0)
    
    # price position (close relative to high/low)
    df['price_position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)).fillna(0.5)
    
    # volume trend (Current vol relative to 7d avg)
    vol_avg_7d = df['Volume'].rolling(window=7).mean().fillna(df['Volume'])
    df['volume_trend'] = (df['Volume'] / (vol_avg_7d + 1e-8) - 1).fillna(0)

    # we only need the very last row (most recent data)
    latest = df.iloc[-1]
    
    features = np.array([
        latest['close_return_1d'],
        latest['vol_log'],
        latest['close_return_3d'],
        latest['close_return_7d'],
        latest['volatility_7d'],
        latest['price_position'],
        latest['volume_trend']
    ])
    
    return features

def get_text_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        # mean pooling calculation
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        summed = (last_hidden * attention_mask).sum(1)
        counts = attention_mask.sum(1).clamp(min=1)
        embedding = summed / counts
    return embedding

# main function for inference
def predict(ticker, headline):
    print("Loading models...")
    model = NewsClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # FinBERT
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(TEXT_MODEL_NAME)

    print(f"Fetching financial data for {ticker}...")
    raw_price_features = get_price_features(ticker)
    if raw_price_features is None:
        return "Could not fetch price data."

    # normiilazation
    norm_price_features = (raw_price_features - PRICE_MEAN) / (PRICE_STD + 1e-8)
    price_tensor = torch.tensor(norm_price_features, dtype=torch.float32).unsqueeze(0) # Batch size 1

    print("Processing text...")
    emb_tensor = get_text_embedding(headline, tokenizer, bert_model)

    # inference
    print("Running prediction...")
    with torch.no_grad():
        probability = model(emb_tensor, price_tensor).item()

    return probability

if __name__ == "__main__":
    # testing
    ticker_symbol = "AAPL"
    news_headline = "Apple reports record breaking quarterly earnings beating all estimates."
    
    pred = predict(ticker_symbol, news_headline)
    print(f"\nPrediction for {ticker_symbol}: {pred:.4f}")
    print(f"(0 = Price Down, 1 = Price Up)")