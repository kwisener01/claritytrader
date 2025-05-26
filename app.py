import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import os
import io

from live_data import fetch_latest_data
from strategy_utils import add_custom_features
from train_model import train_model

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from strategy_utils import generate_signal, run_backtest, add_custom_features
from train_model import train_model
from live_data import fetch_latest_data
from send_slack_alert import send_slack_alert
from yahoo_data import fetch_yahoo_intraday

st.set_page_config(page_title="ClarityTrader Signal", layout="centered")
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

if 'training_data' not in st.session_state:
    st.session_state.training_data = pd.read_csv("spy_training_data.csv")

if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []

if 'trade_journal' not in st.session_state:
    st.session_state.trade_journal = []

st_autorefresh(interval=300000, key="train_refresh")

source = st.radio("📡 Choose Data Source", ["Twelve Data (Live)", "Yahoo Finance (Historical)"])
ticker = st.selectbox("Choose Ticker", ["SPY", "QQQ", "DIA", "IWM"])
api_key = st.text_input("🔑 Twelve Data API Key", type="password")

# ⏱️ Live Auto-Update Model Using Twelve Data
if source == "Twelve Data (Live)" and api_key:
    try:
        new_row = fetch_latest_data(ticker, api_key=api_key)
        if "error" not in new_row:
            df = pd.concat([st.session_state.training_data, pd.DataFrame([new_row])], ignore_index=True)
            df = add_custom_features(df).dropna()
            if len(df) > 30000:
                df = df[-30000:]
            model = train_model(df)
            st.session_state.training_data = df
            st.session_state.model = model
            pickle.dump(model, open("model.pkl", "wb"))
            df.to_csv("training_data.csv", index=False)
            st.success("✅ Model retrained with live data.")
        else:
            st.warning(f"❌ API Error: {new_row['error']}")
    except Exception as e:
        st.warning(f"⚠️ Could not update model: {e}")



if source == "Yahoo Finance (Historical)":
    period = st.selectbox("📆 Yahoo Period", ["1d", "5d", "7d", "1mo", "3mo"])
    hist_df = fetch_yahoo_intraday(symbol=ticker, period=period)
    hist_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in hist_df.columns]

    hist_df.rename(columns={
        "Datetime": "datetime", "Close_SPY": "Close", "High_SPY": "High",
        "Low_SPY": "Low", "Volume_SPY": "Volume"
    }, inplace=True)

    if hist_df.empty:
        st.warning("⚠️ No data retrieved.")
    else:
        hist_df["Momentum"] = hist_df["Close"] - hist_df["Close"].shift(5)
        hist_df["TR"] = hist_df[["High", "Low"]].max(axis=1) - hist_df[["High", "Low"]].min(axis=1)
        hist_df["ATR"] = hist_df["TR"].rolling(14).mean()
        hist_df["RSI"] = 100 - (100 / (1 + (
            hist_df["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
            -hist_df["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean()
        )))
        if "Volume" not in hist_df.columns or hist_df["Volume"].nunique() <= 1:
            hist_df["Volume"] = 1000000
        hist_df = hist_df.dropna()
        hist_df["Label"] = np.where(hist_df["Close"].shift(-5) > hist_df["Close"], "Buy", "Sell")
        st.session_state.training_data = pd.concat([st.session_state.training_data, hist_df], ignore_index=True)
        st.success(f"✅ Loaded {len(hist_df)} rows")
        timestamp = str(datetime.datetime.now())
        price = hist_df["Close"].iloc[-1] if not hist_df["Close"].empty else 0

# Training
st.write("### 📊 Label Distribution")
st.bar_chart(st.session_state.training_data["Label"].value_counts())

data = st.session_state.training_data.dropna(subset=["RSI", "Momentum", "ATR", "Volume", "Label"])
data = add_custom_features(data)
features = ["RSI", "Momentum", "ATR", "Volume", "Accel", "VolSpike"]

if not data.empty:
    X = data[features]
    y = data["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = train_model(pd.concat([X_train, y_train], axis=1))
    st.success("✅ Model trained")

    st.write("### 🧪 Backtest")
    st.json(run_backtest(data))

    st.write("### 📊 Classification Report")
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))

    st.write("### 📊 Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=["Buy", "Sell"])
    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Buy", "Sell"])
    ax.set_yticklabels(["Buy", "Sell"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="white" if conf_matrix[i, j] > 0 else "black")
    st.pyplot(fig, clear_figure=True)
else:
    st.warning("⚠️ Not enough data to train.")
