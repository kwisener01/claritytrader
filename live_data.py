#api_key="7c53601780c14ef5a6893e0d522e2388


import requests
import pandas as pd

def fetch_latest_data(symbol="SPY", interval="1min", api_key="7c53601780c14ef5a6893e0d522e2388"):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=30&apikey={api_key}&format=JSON"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        return {"error": data.get("message", "Unknown error")}

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "Datetime", "close": "Close", "high": "High", "low": "Low"})
    df["Close"] = df["Close"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df = df.sort_values("Datetime")

    # Calculate RSI (14)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Momentum (Close - Close[n])
    df["Momentum"] = df["Close"] - df["Close"].shift(5)

    # ATR (Average True Range over 14 periods)
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()

    latest = df.dropna().iloc[-1]

    return {
        "datetime": latest["Datetime"],
        "close": latest["Close"],
        "RSI": latest["RSI"],
        "Momentum": latest["Momentum"],
        "ATR": latest["ATR"],
        "Volume": 1000000  # Placeholder since Twelve Data free tier doesn't include volume
    }
