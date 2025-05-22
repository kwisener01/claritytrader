
import streamlit as st
import pandas as pd
import pickle
from strategy_utils import generate_signal, run_backtest, train_model, bayesian_update_user
from live_data import fetch_latest_data
from signal_log import log_signal
import datetime
import pytz
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="ClarityTrader Signal", layout="centered")
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

uploaded_file = st.file_uploader("📤 Upload CSV or use example data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("spy_training_data.csv")

if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])

st.write("### ⏱ Backtest Time Window")
start_idx = st.number_input("Start Row", min_value=0, max_value=len(df)-2, value=0)
end_idx = st.number_input("End Row", min_value=start_idx+1, max_value=len(df), value=len(df))
df_window = df.iloc[int(start_idx):int(end_idx)]

if 'datetime' in df.columns:
    st.write(f"📅 Backtest Date Range: {df_window.iloc[0]['datetime']} → {df_window.iloc[-1]['datetime']}")

use_slice = st.checkbox("Train model on selected range only", value=False)
training_data = df_window if use_slice else df

st.write("### 📊 Preview Data (From Backtest Window)")
st.dataframe(df_window.head())

threshold = st.slider("🎯 Confidence Threshold (%)", min_value=50, max_value=100, value=70, step=1)

apply_bayes = st.checkbox("Use Bayesian Forecasting", value=True)
if apply_bayes:
    bayesian_update_user()

if st.button("🛠️ Train Model Now"):
    model = train_model(training_data, apply_bayesian=apply_bayes)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("✅ Model trained and saved as model.pkl")

try:
    model = pickle.load(open("model.pkl", "rb"))
    st.success("📦 Model loaded successfully.")
except:
    model = None
    st.warning("⚠️ No trained model found. Please train it first.")

st.write("### 🧠 Generate Signal from Latest Row")
row = df.iloc[-1].drop("Label", errors="ignore").to_dict()
if model:
    input_df = pd.DataFrame([row])[["RSI", "Momentum", "ATR", "Volume"]]
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    confidence = round(100 * max(proba), 2)
    if confidence >= threshold:
        st.metric(label="Predicted Signal", value=pred)
        st.metric(label="Confidence", value=f"{confidence}%")
    else:
        st.warning(f"No signal. Confidence too low ({confidence}%)")
else:
    pred = generate_signal(row)
    st.metric(label="Predicted Signal (Rule-Based)", value=pred)

st.write("### 🔁 Auto Refresh Settings")
enable_refresh = st.checkbox("Enable Auto Refresh During Market Hours", value=True)
refresh_interval = st.slider("Set Refresh Interval (seconds)", min_value=15, max_value=120, value=60, step=15)

eastern = pytz.timezone("US/Eastern")
now_et = datetime.datetime.now(eastern)
market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
in_market = market_open <= now_et <= market_close and now_et.weekday() < 5

if enable_refresh and in_market:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=refresh_interval * 1000, key="auto-refresh")

st.write("### 📡 Live Signal (1-min Data Feed)")
col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Choose Ticker (ETF Proxy)", ["SPY", "QQQ", "DIA", "IWM"])
with col2:
    api_key = st.text_input("🔑 Twelve Data API Key", type="password")

if api_key and model:
    live_row = fetch_latest_data(symbol=ticker, api_key=api_key)
    if "error" in live_row:
        st.error(f"API Error: {live_row['error']}")
    else:
        live_input = pd.DataFrame([{
            "RSI": live_row["RSI"],
            "Momentum": live_row["Momentum"],
            "ATR": live_row["ATR"],
            "Volume": live_row["Volume"]
        }])
        pred = model.predict(live_input)[0]
        proba = model.predict_proba(live_input)[0]
        confidence = round(100 * max(proba), 2)

        timestamp = live_row["datetime"]
        price = round(live_row["close"], 2)
        hour = pd.to_datetime(timestamp).hour

        st.markdown("---")
        st.markdown(f"🧠 **LIVE SIGNAL for {ticker}**  \n🕒 Timestamp: `{timestamp}`")
        st.metric(label="Signal", value=pred)
        st.metric(label="Live Price", value=f"${price}")
        st.metric(label="Confidence", value=f"{confidence}%")

        if confidence >= threshold:
            log_signal(ticker, pred, confidence, price, timestamp)

st.write("### 📈 Backtest Strategy Results")
results = run_backtest(df_window)
st.write(results)

st.write("### 🗂 Signal Log (Label and Stats)")
if os.path.exists("signal_log.csv"):
    logs = pd.read_csv("signal_log.csv")
    logs['Timestamp'] = pd.to_datetime(logs['Timestamp'])
    logs['Hour'] = logs['Timestamp'].dt.hour

    st.write("#### 🧮 Add Labels (Win/Loss)")
    for i in logs.tail(5).index:
        logs.at[i, 'Label'] = st.selectbox(f"Result for {logs.loc[i, 'Timestamp']} – {logs.loc[i, 'Signal']}", ["", "Win", "Loss"], key=i)

    logs.to_csv("signal_log.csv", index=False)

    st.write("#### 📊 Buy/Sell Distribution")
    st.bar_chart(logs['Signal'].value_counts())

    st.write("#### 📊 Win vs Loss Breakdown")
    if 'Label' in logs.columns:
        st.bar_chart(logs['Label'].value_counts())

    st.write("#### ⏱ Signals by Hour")
    st.bar_chart(logs.groupby('Hour').size())

    if 'Confidence' in logs.columns:
        st.write("#### 📊 Avg Confidence by Signal")
        st.bar_chart(logs.groupby('Signal')['Confidence'].mean())

else:
    st.info("No signal log found yet.")
