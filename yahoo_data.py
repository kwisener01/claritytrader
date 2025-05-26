import yfinance as yf
import pandas as pd

def fetch_yahoo_intraday(symbol="SPY", interval="1m", period="7d"):
    try:
        data = yf.download(tickers=symbol, interval=interval, period=period, progress=False)
        if data.empty:
            return pd.DataFrame()
        data = data.reset_index()
        data = data.rename(columns={"Datetime": "datetime", "Close": "Close", "High": "High", "Low": "Low"})

        # Indicator calculations
        data["Momentum"] = data["Close"] - data["Close"].shift(5)
        data["H-L"] = data["High"] - data["Low"]
        data["H-PC"] = abs(data["High"] - data["Close"].shift(1))
        data["L-PC"] = abs(data["Low"] - data["Close"].shift(1))
        data["TR"] = data[["H-L", "H-PC", "L-PC"]].max(axis=1)
        data["ATR"] = data["TR"].rolling(14).mean()

        # RSI
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        data["Volume"] = 1000000  # Placeholder
        return data.dropna()

    except Exception as e:
        print(f"Yahoo Finance fetch error: {e}")
        return pd.DataFrame()
