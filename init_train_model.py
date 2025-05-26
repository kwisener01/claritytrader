import pandas as pd
import pickle
from strategy_utils import generate_signal, add_custom_features
from train_model import train_model
from yahoo_data import fetch_yahoo_intraday

# Fetch 7 days of Yahoo Finance data
symbol = "SPY"
period = "7d"
df = fetch_yahoo_intraday(symbol=symbol, period=period)

# Rename and calculate base features
df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
df.rename(columns={"Datetime": "datetime", "Close_SPY": "Close", "High_SPY": "High", "Low_SPY": "Low", "Volume_SPY": "Volume"}, inplace=True)
df["Momentum"] = df["Close"] - df["Close"].shift(5)
df["TR"] = df[["High", "Low"]].max(axis=1) - df[["High", "Low"]].min(axis=1)
df["ATR"] = df["TR"].rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + (df["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() / -df["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean())))
df["Volume"] = df["Volume"].fillna(1000000)
df["Label"] = df.apply(generate_signal, axis=1)
df = add_custom_features(df).dropna()

# Train model
model = train_model(df)
pickle.dump(model, open("baseline_model.pkl", "wb"))
df.to_csv("training_data.csv", index=False)
print("✅ Model and data saved.")