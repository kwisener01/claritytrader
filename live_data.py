import requests
import pandas as pd

def fetch_latest_data(symbol="SPY", interval="1min", api_key="7c53601780c14ef5a6893e0d522e2388"):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=1&apikey={api_key}&format=JSON"
    response = requests.get(url)
    data = response.json()

    if "values" in data:
        latest = data["values"][0]
        return {
            "datetime": latest["datetime"],
            "close": float(latest["close"]),
            "RSI": 50.0,        # Placeholder until you compute this live
            "Momentum": 1.0,    # Placeholder
            "ATR": 1.0,         # Placeholder
            "Volume": 1000000   # Placeholder
        }
    else:
        return {"error": data.get("message", "Unknown error")}
