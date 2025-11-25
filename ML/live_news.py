import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import requests
import torch
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv

# Load .env before reading environment variables
load_dotenv()

logger = logging.getLogger("uvicorn")

# Finnhub API configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Ticker keyword mapping for filtering news
# Based on the preprocessing pipeline used in training
TICKER_KEYWORDS = {
    "AAPL": ["AAPL", "Apple", "Apple Inc", "iPhone", "MacBook", "iPad", "Apple Watch", "AirPods", "Tim Cook", "App Store"],
    "MSFT": ["MSFT", "Microsoft", "Windows", "Azure", "Office 365", "Xbox", "Surface", "Bill Gates", "Satya Nadella", "LinkedIn"],
    "GOOGL": ["GOOGL", "Google", "Alphabet", "YouTube", "Android", "Google Cloud", "Gemini AI", "Sundar Pichai", "Waymo"],
    "AMZN": ["AMZN", "Amazon", "AWS", "Amazon Prime", "Kindle", "Whole Foods", "Alexa", "Jeff Bezos", "Andy Jassy"],
    "NVDA": ["NVDA", "NVIDIA", "GeForce", "RTX", "CUDA", "DGX", "H100", "Jensen Huang", "NVIDIA GPU"],
    "META": ["META", "Meta", "Facebook", "Instagram", "WhatsApp", "Threads", "Oculus", "Quest", "Mark Zuckerberg"],
    "TSLA": ["TSLA", "Tesla", "Elon Musk", "Model 3", "Model Y", "Cybertruck", "Gigafactory", "Autopilot", "Supercharger"],
    "LLY": ["LLY", "Eli Lilly", "Eli Lilly and Company", "Mounjaro", "Zepbound", "Trulicity", "Lilly drug", "David Ricks"],
    "V": ["V", "Visa", "Visa Inc", "Visa card", "Al Kelly", "Ryan McInerney"],
    "TSM": ["TSM", "TSMC", "Taiwan Semiconductor", "TSMC chips", "Morris Chang", "TSMC wafer"],
    "UNH": ["UNH", "UnitedHealth", "United Health Group", "Optum", "Andrew Witty"],
    "AVGO": ["AVGO", "Broadcom", "VMware", "Broadcom Inc", "Hock Tan", "Broadcom chip"],
    "NVO": ["NVO", "Novo Nordisk", "Ozempic", "Wegovy", "Lars Fruergaard Jørgensen"],
    "JPM": ["JPM", "JPMorgan", "JPMorgan Chase", "Jamie Dimon", "Chase Bank"],
    "WMT": ["WMT", "Walmart", "Walmart Inc", "Sam's Club"],
    "MA": ["MA", "Mastercard", "Mastercard Inc", "Michael Miebach"],
    "XOM": ["XOM", "Exxon Mobil", "ExxonMobil", "Darren Woods"],
    "JNJ": ["JNJ", "Johnson & Johnson", "J&J", "Joaquin Duato"],
    "PG": ["PG", "Procter & Gamble", "P&G", "Jon Moeller"],
    "CVX": ["CVX", "Chevron", "Chevron Corporation", "Mike Wirth"],
}

TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "yiyanghkust/finbert-tone")


def fetch_headlines(ticker: str, from_date: str, to_date: str) -> List[str]:
    """
    Fetch company news headlines from Finnhub API for a given ticker and date range.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format
        
    Returns:
        List of headline strings
    """
    if not FINNHUB_API_KEY:
        logger.warning("FINNHUB_API_KEY not set, cannot fetch live news")
        return []
    
    try:
        url = f"{FINNHUB_BASE_URL}/company-news"
        params = {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
            "token": FINNHUB_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        news_data = response.json()
        
        if not isinstance(news_data, list):
            logger.warning(f"Unexpected response format from Finnhub for {ticker}")
            return []
        
        # Extract headlines and filter by keywords
        keywords = TICKER_KEYWORDS.get(ticker.upper(), [ticker.upper()])
        headlines = []
        
        for article in news_data:
            headline = article.get("headline", "")
            summary = article.get("summary", "")
            
            # Check if headline or summary contains any ticker keywords
            text_to_check = f"{headline} {summary}".lower()
            if any(keyword.lower() in text_to_check for keyword in keywords):
                # Prefer headline, fallback to summary if headline is empty
                if headline:
                    headlines.append(headline)
                elif summary:
                    headlines.append(summary)
        
        logger.info(f"Fetched {len(headlines)} relevant headlines for {ticker}")
        return headlines
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch news from Finnhub for {ticker}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching news for {ticker}: {e}")
        return []


@torch.no_grad()
def get_text_embedding(text: str, tokenizer, bert_model) -> torch.Tensor:
    """
    Generate FinBERT embedding for a single text string.
    Uses mean pooling over tokens with attention mask.
    
    Args:
        text: Input text string
        tokenizer: FinBERT tokenizer
        bert_model: FinBERT model
        
    Returns:
        768-dimensional embedding tensor
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    
    outputs = bert_model(**inputs)
    last_hidden = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)
    summed = (last_hidden * attention_mask).sum(1)
    counts = attention_mask.sum(1).clamp(min=1)
    embedding = summed / counts
    
    return embedding


def get_aggregated_embedding(headlines: List[str], tokenizer, bert_model) -> torch.Tensor:
    """
    Generate aggregated embedding from multiple headlines by averaging.
    
    Args:
        headlines: List of headline strings
        tokenizer: FinBERT tokenizer
        bert_model: FinBERT model
        
    Returns:
        768-dimensional embedding tensor (averaged across headlines)
    """
    if not headlines:
        logger.warning("No headlines provided, returning zero vector")
        # Return zero vector matching FinBERT embedding dimension
        return torch.zeros(1, 768, dtype=torch.float32)
    
    embeddings = []
    for headline in headlines:
        emb = get_text_embedding(headline, tokenizer, bert_model)
        embeddings.append(emb)
    
    # Stack and average
    stacked = torch.stack(embeddings, dim=0)
    aggregated = stacked.mean(dim=0)
    
    return aggregated


def get_today_news_embedding(ticker: str, tokenizer, bert_model) -> Tuple[torch.Tensor, int]:
    """
    Fetch today's news for a ticker and return aggregated embedding.
    
    Args:
        ticker: Stock ticker symbol
        tokenizer: FinBERT tokenizer
        bert_model: FinBERT model
        
    Returns:
        Tuple of (embedding tensor, number of headlines used)
    """
    today = datetime.now().date()
    today_str = today.strftime("%Y-%m-%d")
    
    headlines = fetch_headlines(ticker, today_str, today_str)
    
    if not headlines:
        logger.warning(f"No headlines found for {ticker} on {today_str}")
        # Try last 7 days as fallback
        week_ago = today - timedelta(days=7)
        week_ago_str = week_ago.strftime("%Y-%m-%d")
        headlines = fetch_headlines(ticker, week_ago_str, today_str)
        
        if not headlines:
            logger.warning(f"No headlines found for {ticker} in the last 7 days")
    
    embedding = get_aggregated_embedding(headlines, tokenizer, bert_model)
    return embedding, len(headlines)
