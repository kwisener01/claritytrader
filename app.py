# Full ClarityTrader app with Yahoo Finance & Twelve Data, live training, journaling, visualization, and safe logic

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import os
import io
from streamlit_autorefresh import st_autorefresh
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from strategy_utils import generate_signal, run_backtest, train_model, bayesian_update_user
from live_data import fetch_latest_data
from send_slack_alert import send_slack_alert
from yahoo_data import fetch_yahoo_intraday

# Set up Streamlit
st.set_page_config(page_title="ClarityTrader Signal", layout="centered")
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

# Initialize state
if 'training_data' not in st.session_state:
    st.session_state.training_data = pd.read_csv("spy_training_data.csv")

if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []

if 'trade_journal' not in st.session_state:
    st.session_state.trade_journal = []

# Refresh every 5 minutes
st_autorefresh(interval=300000, key="train_refresh")

# Source selection
source = st.radio("📡 Choose Data Source", ["Twelve Data (Live)", "Yahoo Finance (Historical)"])
ticker = st.selectbox("Choose Ticker", ["SPY", "QQQ", "DIA", "IWM"])
api_key = st.text_input("🔑 Twelve Data API Key", type="password")

# Live data from Twelve Data
if source == "Twelve Data (Live)" and api_key:
    new_row = fetch_latest_data(symbol=ticker, api_key=api_key)
    if "error" not in new_row:
        df_new = pd.DataFrame([new_row])
        df_new["Label"] = "Hold"
        st.session_state.training_data = pd.concat([st.session_state.training_data, df_new], ignore_index=True)
        timestamp = new_row["datetime"]
        price = new_row["close"]
    else:
        st.error(f"API Error: {new_row['error']}")
        timestamp = str(datetime.datetime.now())
        price = 0

# Historical from Yahoo
elif source == "Yahoo Finance (Historical)":
    period = st.selectbox("📆 Yahoo Period", ["1d", "5d", "7d", "1mo", "3mo"])
    hist_df = fetch_yahoo_intraday(symbol=ticker, period=period)
    hist_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in hist_df.columns]

    hist_df.rename(columns={
        "Datetime": "datetime", "Close_SPY": "Close", "High_SPY": "High",
        "Low_SPY": "Low", "Volume_SPY": "Volume"
    }, inplace=True)

    if hist_df.empty:
        st.warning("⚠️ No data retrieved from Yahoo Finance.")
    else:
        hist_df["Momentum"] = hist_df["Close"] - hist_df["Close"].shift(5)
        hist_df["H-L"] = hist_df["High"] - hist_df["Low"]
        hist_df["H-PC"] = abs(hist_df["High"] - hist_df["Close"].shift(1))
        hist_df["L-PC"] = abs(hist_df["Low"] - hist_df["Close"].shift(1))
        hist_df["TR"] = hist_df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        hist_df["ATR"] = hist_df["TR"].rolling(14).mean()
        hist_df["RSI"] = 100 - (100 / (1 + (
            hist_df["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
            -hist_df["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean()
        )))

        if "Volume" not in hist_df.columns or hist_df["Volume"].nunique() <= 1:
            hist_df["Volume"] = 1000000

        hist_df = hist_df.dropna()

        required = ["datetime", "Open", "High", "Low", "Close", "RSI", "Momentum", "ATR", "Volume"]
        for col in required:
            if col not in hist_df.columns:
                hist_df[col] = None
        hist_df = hist_df[required + [c for c in hist_df.columns if c not in required]]

        hist_df["Label"] = np.where(hist_df["Close"].shift(-5) > hist_df["Close"], "Buy", "Sell")
        st.session_state.training_data = pd.concat([st.session_state.training_data, hist_df], ignore_index=True)
        st.success(f"✅ Pulled {len(hist_df)} rows from Yahoo Finance")
        timestamp = str(datetime.datetime.now())
        price = hist_df["Close"].iloc[-1] if "Close" in hist_df.columns and not hist_df["Close"].empty else 0

# Manual entry
with st.expander("➕ Add Training Row Manually"):
    rsi = st.number_input("RSI", min_value=0.0, max_value=100.0, step=0.1)
    momentum = st.number_input("Momentum", step=0.01)
    atr = st.number_input("ATR", step=0.01)
    volume = st.number_input("Volume", step=1000)
    label = st.selectbox("Label", ["Buy", "Sell", "Hold"])
    if st.button("📌 Add to Training Data"):
        new_row = pd.DataFrame([{"RSI": rsi, "Momentum": momentum, "ATR": atr, "Volume": volume, "Label": label}])
        st.session_state.training_data = pd.concat([st.session_state.training_data, new_row], ignore_index=True)
        st.success("✅ Row added to training data.")

# Display and export
st.write("### 📊 Label Distribution")
st.bar_chart(st.session_state.training_data["Label"].value_counts())
st.write("### 🧾 Current Training Data")
st.dataframe(st.session_state.training_data.tail(10))
st.download_button("📥 Download Data", st.session_state.training_data.to_csv(index=False).encode("utf-8"), file_name="updated_training_data.csv")

threshold = st.slider("🎯 Confidence Threshold (%)", 50, 100, 70, 1)
if st.checkbox("Use Bayesian Forecasting", value=True):
    bayesian_update_user()

# Train model
clean_data = st.session_state.training_data.dropna(subset=["RSI", "Momentum", "ATR", "Volume", "Label"])
if not clean_data.empty:
    model = train_model(clean_data, apply_bayesian=True)
    st.session_state['model'] = pickle.dumps(model)
    st.success("✅ Model trained")

    st.write("### 🧪 Backtest")
    st.json(run_backtest(clean_data))

    y_true = clean_data["Label"]
    y_pred = model.predict(clean_data[["RSI", "Momentum", "ATR", "Volume"]])
    st.write("### 📊 Classification Report")
    st.text(classification_report(y_true, y_pred))
    st.write("### 📊 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Buy", "Sell"], yticklabels=["Buy", "Sell"])
    st.pyplot(fig)

    st.write("### 📈 Feature Importance")
    try:
        fi = model.feature_importances_
        st.bar_chart(pd.DataFrame({"Feature": ["RSI", "Momentum", "ATR", "Volume"], "Importance": fi}).set_index("Feature"))
    except Exception as e:
        st.warning(f"Feature importances unavailable: {e}")

    st.write("### 📉 OHLC Chart")
    try:
        chart_data = clean_data[["datetime", "Open", "High", "Low", "Close"]].copy()
        chart_data["datetime"] = pd.to_datetime(chart_data["datetime"])
        chart_data.set_index("datetime", inplace=True)
        fig = go.Figure(data=[go.Candlestick(x=chart_data.index, open=chart_data["Open"], high=chart_data["High"], low=chart_data["Low"], close=chart_data["Close"])])
        for signal, color in [("Buy", "green"), ("Sell", "red")]:
            s = clean_data[clean_data["Label"] == signal]
            fig.add_trace(go.Scatter(x=s["datetime"], y=s["Close"], mode="markers", name=signal, marker=dict(color=color, size=8)))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart error: {e}")
else:
    model = None
    st.warning("⚠️ Not enough clean data to train.")

# Signal logic
st.write("### 🧠 Generate Signal")
latest = st.session_state.training_data.iloc[-1].drop("Label", errors="ignore")
input_df = pd.DataFrame([latest])[["RSI", "Momentum", "ATR", "Volume"]]
if model is not None:
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    confidence = round(100 * max(proba), 2)
    st.metric("Signal", pred)
    st.metric("Confidence", f"{confidence}%")
    st.session_state.signal_log.append([timestamp, ticker, pred, confidence, price])
else:
    st.warning("⚠️ Model unavailable")

# Journal
with st.form("journal_form"):
    st.subheader("📝 Trade Journal")
    reason = st.text_input("Reason")
    emotion = st.selectbox("Emotion", ["Neutral", "Confident", "Anxious", "Fearful", "Greedy"])
    reflection = st.text_area("Reflection")
    file = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg"])
    if st.form_submit_button("Save Entry"):
        fn = file.name if file else None
        st.session_state.trade_journal.append([timestamp, ticker, pred if model else "N/A", reason, emotion, reflection, fn])
        if file:
            os.makedirs("uploads", exist_ok=True)
            with open(os.path.join("uploads", fn), "wb") as f:
                f.write(file.read())
        st.success("✅ Entry saved")

st.write("### 📚 Recent Entries")
st.dataframe(pd.DataFrame(st.session_state.trade_journal, columns=["Timestamp", "Ticker", "Signal", "Reason", "Emotion", "Reflection", "Attachment"]).tail(5))

st.write("### 🕒 Signal Log")
st.dataframe(pd.DataFrame(st.session_state.signal_log, columns=["Timestamp", "Ticker", "Signal", "Confidence", "Price"]).tail(10))
