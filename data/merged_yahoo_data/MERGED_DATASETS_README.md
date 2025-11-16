# Merged Yahoo News Datasets

## Overview
Two merged datasets combining Yahoo Finance stock price data (2018-2023) with news articles.
---

## Files

### 1. `merged_yahoo_news.csv`
**Complete dataset with all the trading days**
- **Rows:** 602,962
- **Contains:** All ticker-date combinations from Yahoo Finance
- **News coverage:** 36,815 rows have news (6.1%), 566,147 rows have no news (93.9%)

### 2. `merged_yahoo_news__news_only.csv`
**Subset with only days that have news**
- **Rows:** 36,815
- **Contains:** Only ticker-date combinations where at least one news article exists
- **News coverage:** 100% (every row has news)

---

## Data Schema

Both files have identical column structure:

| Column | Type | Description |
|--------|------|-------------|
| `Date` | string | Trading date (YYYY-MM-DD) |
| `Open` | float | Opening price |
| `High` | float | Highest price |
| `Low` | float | Lowest price |
| `Close` | float | Closing price |
| `Volume` | int | Trading volume |
| `Dividends` | float | Dividends paid |
| `Stock Splits` | float | Stock split ratio |
| `ticker` | string | Stock ticker symbol (e.g., AAPL, MSFT) |
| `headlines` | string | Combined news headlines, separated by " \| " |
| `news_count` | int | Number of news articles for this ticker+date |

---

## Data Statistics

- **Unique tickers:** 491 stocks
- **Tickers with news:** 362 (73.7% coverage)
- **Date range:** 2018-11-29 to 2023-11-29
- **News concentration:** Most news in 2019-2020

---

## Example Rows

**Row with single news article:**
```
Date: 2018-11-29
Open: 54.176498
High: 55.007500
Low: 54.099998
Close: 54.729000
Volume: 31004000
Dividends: 0.0
Stock Splits: 0.0
ticker: GOOGL
headlines: 'White House to Hold Roundtable With Tech Executives' -WSJ
news_count: 1
```

**Row with multiple news articles:**
```
Date: 2018-11-29
Open: 39.692784
High: 40.064904
Low: 38.735195
Close: 39.037853
Volume: 54917200
Dividends: 0.04
Stock Splits: 0.0
ticker: NVDA
headlines: Nvidia And Epic Games Bartering On Fortnite Bundle, Variety Reports | Stocks Trading Ex Dividend For Thurs., Nov. 29, 2018
news_count: 2
```

**Row without news (only in `merged_yahoo_news.csv`):**
```
Date: 2018-11-29
Open: 43.829761
High: 43.863354
Low: 42.639594
Close: 43.083508
Volume: 167080000
Dividends: 0.0
Stock Splits: 0.0
ticker: AAPL
headlines: 
news_count: 0
```

