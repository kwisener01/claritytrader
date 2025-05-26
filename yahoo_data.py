import yfinance as yf
import pandas as pd

def fetch_yahoo_intraday(symbol="SPY", interval="1m", period="7d"):
    try:
        df = yf.download(tickers=symbol, interval=interval, period=period, progress=False)
        if df.empty:
            print(f"[Yahoo] No data returned for {symbol} with interval={interval} and period={period}")
            return pd.DataFrame()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.tz_convert("US/Eastern")
        df = df.reset_index()
        df.rename(columns={"Datetime": "datetime"}, inplace=True)
        return df
    except Exception as e:
        print(f"[Yahoo Fetch Error] {e}")
        return pd.DataFrame()
