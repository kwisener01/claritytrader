import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import os
import io

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from live_data import fetch_latest_data
from strategy_utils import add_custom_features, generate_signal, run_backtest
from train_model import train_model
from yahoo_data import fetch_yahoo_intraday

st.set_page_config(page_title="ClarityTrader Signal", layout="centered")
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

# Session Initialization
if 'training_data' not in st.session_state:
    st.session_state.training_data = pd.DataFrame()

if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []

if 'latest_signal' not in st.session_state:
    st.session_state.latest_signal = None

st_autorefresh(interval=300000, key="auto_refresh")

# Actions UI
source = st.radio("📡 Choose Action", ["🔁 Predict Signal from Twelve Data", "📥 Load & Train from Yahoo Finance"])
ticker = st.selectbox("Choose Ticker", ["SPY", "QQQ", "DIA", "IWM"])
api_key = st.text_input("🔑 Twelve Data API Key", type="password")

# Load model immediately after source selection
if 'model' not in st.session_state or st.session_state.model is None:
    try:
        with open("model.pkl", "rb") as f:
            st.session_state.model = pickle.load(f)
        st.success("✅ Model loaded from disk.")
    except:
        st.session_state.model = None
        st.warning("⚠️ No model found on disk. Please train with Yahoo data first.")

# Display latest signal info if exists
if st.session_state.latest_signal:
    signal, confidence, price = st.session_state.latest_signal
    st.subheader("🔔 Latest Signal")
    st.metric("Signal", signal)
    st.metric("Confidence", f"{confidence}%")
    st.metric("Price", f"${price:.2f}")

# Manual model reload
if st.button("🔁 Reload Model"):
    try:
        with open("model.pkl", "rb") as f:
            st.session_state.model = pickle.load(f)
        st.success("✅ Model reloaded from disk.")
    except:
        st.warning("⚠️ No model file found. Please train first.")

# Prediction from Twelve Data
if source == "🔁 Predict Signal from Twelve Data" and api_key:
    try:
        new_row = fetch_latest_data(ticker, api_key=api_key)
        if "error" in new_row:
            st.warning(f"❌ API Error: {new_row['error']}")
        else:
            df = pd.DataFrame([new_row])
            df = add_custom_features(df).dropna()

            st.write("### 🧠 Live Signal")
            if st.session_state.model is not None and not df.empty:
                features = [col for col in df.columns if col in ["RSI", "Momentum", "ATR", "Volume", "Accel", "VolSpike"]]
                X_live = df[features]
                pred = st.session_state.model.predict(X_live)[0]
                proba = st.session_state.model.predict_proba(X_live)[0]
                confidence = round(100 * max(proba), 2)
                price = new_row.get("close", 0)
                st.metric("Signal", pred)
                st.metric("Confidence", f"{confidence}%")
                st.metric("Price", f"${price:.2f}")
                st.session_state.latest_signal = (pred, confidence, price)

                # Append signal log
                st.session_state.signal_log.append({
                    "timestamp": str(datetime.datetime.now()),
                    "ticker": ticker,
                    "signal": pred,
                    "confidence": confidence,
                    "price": price
                })
            else:
                st.warning("⚠️ No trained model available. Please train with Yahoo data first.")
    except Exception as e:
        st.error(f"Error fetching live data: {e}")

# Training from Yahoo Finance
elif source == "📥 Load & Train from Yahoo Finance":
    period = st.selectbox("📆 Yahoo Period", ["1d", "5d", "7d", "1mo", "3mo"])
    if st.button("🧠 Train Model"):
        try:
            hist_df = fetch_yahoo_intraday(symbol=ticker, period=period)
            hist_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in hist_df.columns]
            hist_df.rename(columns={
                "Datetime": "datetime", "Close_SPY": "Close", "High_SPY": "High",
                "Low_SPY": "Low", "Volume_SPY": "Volume"
            }, inplace=True)

            hist_df["Momentum"] = hist_df["Close"] - hist_df["Close"].shift(5)
            hist_df["TR"] = hist_df[["High", "Low"]].max(axis=1) - hist_df[["High", "Low"]].min(axis=1)
            hist_df["ATR"] = hist_df["TR"].rolling(14).mean()
            hist_df["RSI"] = 100 - (100 / (1 + (
                hist_df["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
                -hist_df["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean()
            )))
            hist_df["Label"] = np.where(hist_df["Close"].shift(-5) > hist_df["Close"], "Buy", "Sell")
            hist_df["Volume"] = hist_df["Volume"].fillna(1000000)
            hist_df = hist_df.dropna()

            df = add_custom_features(hist_df)
            df.to_csv("training_data.csv", index=False)
            st.session_state.training_data = df

            if len(df) < 30:
                raise ValueError("Not enough samples to train a model. Please select a longer period.")

            model = train_model(df)
            st.session_state.model = model
            pickle.dump(model, open("model.pkl", "wb"))
            st.success("✅ Model trained and saved to model.pkl.")

            st.write("### 📊 Label Distribution")
            st.bar_chart(df["Label"].value_counts())

            X = df[["RSI", "Momentum", "ATR", "Volume", "Accel", "VolSpike"]]
            y = df["Label"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            y_pred = model.predict(X_test)

            st.write("### 🧪 Backtest")
            st.json(run_backtest(df))

            st.write("### 📊 Classification Report")
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
                    ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="white")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ Training failed: {e}")
