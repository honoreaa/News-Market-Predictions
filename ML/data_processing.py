import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def basic_clean(text):
    # handle missing values
    if pd.isna(text):
        return ""
    s = str(text)
    s = s.replace('\n', ' ').strip()  # clean up newlines
    return s

def compute_labels(df):
    # need to sort by date first or this breaks
    #source: https://pandas.pydata.org/docs/reference/groupby.html
    df = df.sort_values(['ticker', 'Date'])
    df['Close_next'] = df.groupby('ticker')['Close'].shift(-1)  # get next day price
    df['future_return_1d'] = df['Close_next'] / df['Close'] - 1
    df['label'] = (df['future_return_1d'] > 0).astype(int)  # 1 if up, 0 if down
    return df

def add_price_features(df):
    #source: https://pandas.pydata.org/docs/reference/api/pandas.Series.pct_change.html
    df['close_return_1d'] = df.groupby('ticker')['Close'].pct_change().fillna(0)
    df['vol_log'] = np.log1p(df['Volume']).fillna(0)  # log transform helps with scale
    
    #adding momentum stuff
    #source: https://pandas.pydata.org/docs/reference/api/pandas.Series.pct_change.html
    df['close_return_3d'] = df.groupby('ticker')['Close'].pct_change(periods=3).fillna(0)
    df['close_return_7d'] = df.groupby('ticker')['Close'].pct_change(periods=7).fillna(0)
    
    # rolling std for volatility
    df['volatility_7d'] = df.groupby('ticker')['close_return_1d'].rolling(window=7, min_periods=1).std().fillna(0).reset_index(0, drop=True)
    
    # where price is in the days range, 0 is low 1 is high
    df['price_position'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)).fillna(0.5)
    
    # compare current volume to recent avg
    vol_avg_7d = df.groupby('ticker')['Volume'].rolling(window=7, min_periods=1).mean().fillna(df['Volume']).reset_index(0, drop=True)
    df['volume_trend'] = (df['Volume'] / (vol_avg_7d + 1e-8) - 1).fillna(0)
    
    return df

def main(raw_csv, out_dir):
    #source: https://docs.python.org/3/library/pathlib.html
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_csv, parse_dates=['Date'])
    #oh my goodness this is such a mess <-- review later tonight
    df['headlines'] = df['headlines'].apply(basic_clean)
    df = compute_labels(df)
    df = add_price_features(df)
    # drop last row per ticker bc we dont have a future return for that day to train with
    df = df[~df['Close_next'].isna()].copy()
    
    df = df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    processed_csv = out_dir / 'processed.csv'
    df.to_csv(processed_csv, index=False)
    print('Saved processed CSV to', processed_csv)

if __name__ == '__main__':
    #source: https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='data/processed')
    args = parser.parse_args()
    main(args.raw_csv, args.out_dir)