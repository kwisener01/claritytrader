import yfinance as yf
import pandas as pd

def fetch_yahoo_intraday(symbol="SPY", interval="1m", period="7d"):
    try:
        df = yf.download(tickers=symbol, interval=interval, period=period, progress=False)
        if df.empty:
            return pd.DataFrame()
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df = df.tz_convert('US/Eastern')
        df.reset_index(inplace=True)

        # Rename and calculate features
        df.rename(columns={"Datetime": "datetime"}, inplace=True)
        df["Momentum"] = df["Close"] - df["Close"].shift(5)
        df["H-L"] = df["High"] - df["Low"]
        df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
        df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
        df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        df["ATR"] = df["TR"].rolling(14).mean()

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df["Volume"] = 1000000  # Placeholder
        return df.dropna()

    except Exception as e:
        print(f"[Yahoo Fetch Error] {e}")
        return pd.DataFrame()
