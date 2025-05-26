# Updated app.py with Yahoo Finance integration, auto-training, session-safe storage, logging, and export

import streamlit as st
import pandas as pd
import pickle
import datetime
import pytz
import os
import io
from streamlit_autorefresh import st_autorefresh

from strategy_utils import generate_signal, run_backtest, train_model, bayesian_update_user
from live_data import fetch_latest_data
from send_slack_alert import send_slack_alert
from yahoo_data import fetch_yahoo_intraday

# Set up app UI
st.set_page_config(page_title="ClarityTrader Signal", layout="centered")
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

# Initialize session state
if 'training_data' not in st.session_state:
    st.session_state.training_data = pd.read_csv("spy_training_data.csv")

if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []

if 'trade_journal' not in st.session_state:
    st.session_state.trade_journal = []

# Auto-refresh every 300 seconds (5 minutes)
st_autorefresh(interval=300000, key="train_refresh")

# Source selector
source = st.radio("📡 Choose Data Source", ["Twelve Data (Live)", "Yahoo Finance (Historical)"])
ticker = st.selectbox("Choose Ticker (ETF Proxy)", ["SPY", "QQQ", "DIA", "IWM"])
api_key = st.text_input("🔑 Twelve Data API Key", type="password")

if source == "Twelve Data (Live)" and api_key:
    new_row = fetch_latest_data(symbol=ticker, api_key=api_key)
    if "error" not in new_row:
        df_new = pd.DataFrame([new_row])
        df_new["Label"] = "Hold"  # Placeholder label
        st.session_state.training_data = pd.concat([st.session_state.training_data, df_new], ignore_index=True)
        timestamp = new_row["datetime"]
        price = new_row["close"]
    else:
        st.error(f"API Error: {new_row['error']}")
        timestamp = str(datetime.datetime.now())
        price = 0
elif source == "Yahoo Finance (Historical)":
    hist_df = fetch_yahoo_intraday(symbol=ticker)
    hist_df["Label"] = "Hold"
    st.session_state.training_data = pd.concat([st.session_state.training_data, hist_df], ignore_index=True)
    st.success(f"✅ Pulled {len(hist_df)} rows from Yahoo Finance")
    timestamp = str(datetime.datetime.now())
    price = 0

# Manual training row input
with st.expander("➕ Add Training Row Manually"):
    rsi = st.number_input("RSI", min_value=0.0, max_value=100.0, step=0.1)
    momentum = st.number_input("Momentum", step=0.01)
    atr = st.number_input("ATR", step=0.01)
    volume = st.number_input("Volume", step=1000)
    label = st.selectbox("Label", ["Buy", "Sell", "Hold"])
    if st.button("📌 Add to Training Data"):
        new_row = pd.DataFrame([{
            "RSI": rsi,
            "Momentum": momentum,
            "ATR": atr,
            "Volume": volume,
            "Label": label
        }])
        st.session_state.training_data = pd.concat([st.session_state.training_data, new_row], ignore_index=True)
        st.success("✅ Row added to training data.")

# Show current training data
st.write("### 🧾 Current Training Data")
st.dataframe(st.session_state.training_data.tail(10))

csv_export = st.session_state.training_data.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Updated Training Data", csv_export, file_name="updated_training_data.csv")

# Confidence threshold & Bayesian option
threshold = st.slider("🎯 Confidence Threshold (%)", min_value=50, max_value=100, value=70, step=1)
apply_bayes = st.checkbox("Use Bayesian Forecasting", value=True)
if apply_bayes:
    bayesian_update_user()

# Train the model every refresh with cleaned data
clean_data = st.session_state.training_data.dropna(subset=["RSI", "Momentum", "ATR", "Volume", "Label"])
if not clean_data.empty:
    model = train_model(clean_data, apply_bayesian=apply_bayes)
    buf = io.BytesIO()
    pickle.dump(model, buf)
    st.session_state['model'] = buf.getvalue()
    st.success("✅ Model auto-trained")
else:
    model = None
    st.warning("⚠️ No valid rows available for training.")

# Signal from last row
st.write("### 🧠 Generate Signal from Latest Data")
latest_row = st.session_state.training_data.iloc[-1].drop("Label", errors="ignore")
input_df = pd.DataFrame([latest_row])[ [col for col in ["RSI", "Momentum", "ATR", "Volume"] if col in latest_row] ]
if model is not None:
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    confidence = round(100 * max(proba), 2)
    st.metric(label="Signal", value=pred)
    st.metric(label="Confidence", value=f"{confidence}%")

    # Log signal
    st.session_state.signal_log.append([timestamp, ticker, pred, confidence, price])
else:
    st.warning("⚠️ Model not available to generate signal.")

# Journal entry
with st.form(key="journal_form"):
    st.subheader("📝 Log Trade Journal Entry")
    reason = st.text_input("🧠 Why did you take this trade?")
    emotion = st.selectbox("😐 Emotion", ["Neutral", "Confident", "Anxious", "Fearful", "Greedy"])
    reflection = st.text_area("📓 Reflection")
    file = st.file_uploader("📎 Upload Screenshot", type=["png", "jpg", "jpeg"])
    submit = st.form_submit_button("Save Journal Entry")
    if submit:
        filename = file.name if file else None
        st.session_state.trade_journal.append([timestamp, ticker, pred if model else "N/A", reason, emotion, reflection, filename])
        if file:
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            with open(os.path.join("uploads", file.name), "wb") as f:
                f.write(file.read())
        st.success("✅ Journal entry saved!")

# Journal and signal export
st.write("### 📚 Recent Journal Entries")
journal_df = pd.DataFrame(st.session_state.trade_journal, columns=["Timestamp", "Ticker", "Signal", "Reason", "Emotion", "Reflection", "Attachment"])
st.dataframe(journal_df.tail(5))
journal_csv = journal_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Journal CSV", journal_csv, file_name="trade_journal.csv")

st.write("### 🕒 Signal History")
signal_df = pd.DataFrame(st.session_state.signal_log, columns=["Timestamp", "Ticker", "Signal", "Confidence", "Price"])
st.dataframe(signal_df.tail(10))
signal_csv = signal_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Signal Log", signal_csv, file_name="signal_log.csv")
