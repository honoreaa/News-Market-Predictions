import streamlit as st
import torch
import numpy as np
import yfinance as yf
from transformers import AutoTokenizer, AutoModel
# architecture from your existing model.py file
from model import NewsClassifier 

MODEL_PATH = '../models/news_classifier.pt'
TEXT_MODEL_NAME = 'yiyanghkust/finbert-tone'

# stats from earlier
PRICE_MEAN = np.array([0.0007642, 15.93788, 0.002250, 0.005250, 0.01761, 0.5160, 0.005935])
PRICE_STD  = np.array([0.0212600, 1.345100, 0.035676, 0.053319, 0.01252, 0.3053, 0.322114])

#cache so we only have to load model and other things once 
@st.cache_resource
def load_models():
    model = NewsClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    # Load BERT
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(TEXT_MODEL_NAME)
    
    return model, tokenizer, bert_model

# helper Functions (actually same as inference.py) ---
def get_price_features(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1mo")
        if len(df) < 8:
            return None
            
        df['close_return_1d'] = df['Close'].pct_change().fillna(0)
        df['vol_log'] = np.log1p(df['Volume']).fillna(0)
        df['close_return_3d'] = df['Close'].pct_change(periods=3).fillna(0)
        df['close_return_7d'] = df['Close'].pct_change(periods=7).fillna(0)
        df['volatility_7d'] = df['close_return_1d'].rolling(window=7).std().fillna(0)
        df['price_position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)).fillna(0.5)
        vol_avg_7d = df['Volume'].rolling(window=7).mean().fillna(df['Volume'])
        df['volume_trend'] = (df['Volume'] / (vol_avg_7d + 1e-8) - 1).fillna(0)
        
        latest = df.iloc[-1]
        return np.array([
            latest['close_return_1d'], latest['vol_log'], latest['close_return_3d'],
            latest['close_return_7d'], latest['volatility_7d'], latest['price_position'],
            latest['volume_trend']
        ])
    except Exception as e:
        return None

def get_text_embedding(text, tokenizer, bert_model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        summed = (last_hidden * attention_mask).sum(1)
        counts = attention_mask.sum(1).clamp(min=1)
    return summed / counts

# website
st.set_page_config(page_title="Market News AI", page_icon="📈")

st.title("AI Market Predictor")
st.write("Enter a stock ticker and a news headline to see if the AI predicts a price increase.")

# load models instantly
model, tokenizer, bert_model = load_models()

# inputs
col1, col2 = st.columns([1, 3])
with col1:
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
with col2:
    headline = st.text_input("News Headline", value="Apple reports record earnings for Q4.")

if st.button("Predict Movement", type="primary"):
    with st.spinner(f"Fetching live data for {ticker}..."):
        raw_prices = get_price_features(ticker)
    
    if raw_prices is None:
        st.error(f"Could not fetch data for {ticker}. Check the symbol or internet connection.")
    else:
        # normalize
        norm_prices = (raw_prices - PRICE_MEAN) / (PRICE_STD + 1e-8)
        price_tensor = torch.tensor(norm_prices, dtype=torch.float32).unsqueeze(0)
        
        # embed Text
        emb_tensor = get_text_embedding(headline, tokenizer, bert_model)
        
        # actual predict
        with torch.no_grad():
            prob = model(emb_tensor, price_tensor).item()
        
        # result
        st.metric(label="Prediction Confidence", value=f"{prob:.2%}")
        
        if prob > 0.5:
            st.success(f"The model predicts **{ticker}** price will go **UP**")
        else:
            st.error(f"The model predicts **{ticker}** price will go **DOWN**")