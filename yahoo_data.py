# yahoo_data.py

import yfinance as yf
import pandas as pd

def fetch_yahoo_intraday(symbol="SPY", interval="1m", period="5d"):
    data = yf.download(tickers=symbol, interval=interval, period=period)
    data = data.reset_index()
    data = data.rename(columns={"Datetime": "datetime", "Close": "Close", "High": "High", "Low": "Low"})

    data["RSI"] = compute_rsi(data["Close"], 14)
    data["Momentum"] = data["Close"] - data["Close"].shift(5)
    data["H-L"] = data["High"] - data["Low"]
    data["H-PC"] = abs(data["High"] - data["Close"].shift(1))
    data["L-PC"] = abs(data["Low"] - data["Close"].shift(1))
    data["TR"] = data[["H-L", "H-PC", "L-PC"]].max(axis=1)
    data["ATR"] = data["TR"].rolling(14).mean()
    data["Volume"] = 1000000  # Placeholder volume

    return data.dropna()

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
