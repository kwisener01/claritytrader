# -- full corrected Streamlit app.py file --
# Imports
import streamlit as st
import pandas as pd
import pickle
from strategy_utils import generate_signal, run_backtest, train_model, bayesian_update_user
from live_data import fetch_latest_data
from signal_log import log_signal
from trade_journal import log_journal
from send_slack_alert import send_slack_alert
import datetime
import pytz
import os
import matplotlib.pyplot as plt

# Config
st.set_page_config(page_title="ClarityTrader Signal", layout="centered")
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

# File uploader
uploaded_file = st.file_uploader("📤 Upload CSV or use example data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("spy_training_data.csv")

if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()

    st.write("### ⏱ Backtest Date Range Selector")
    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    df_window = df[df['datetime'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))]
    st.write(f"📅 Backtest Date Range: {df_window.iloc[0]['datetime']} → {df_window.iloc[-1]['datetime']}")
else:
    st.warning("No datetime column found. Defaulting to full dataset.")
    df_window = df

# Training set
use_slice = st.checkbox("Train model on selected range only", value=False)
training_data = df_window if use_slice else df

st.write("### 📊 Preview Data (From Backtest Window)")
st.dataframe(df_window.head())

# Threshold
threshold = st.slider("🎯 Confidence Threshold (%)", min_value=50, max_value=100, value=70, step=1)

# Bayesian
apply_bayes = st.checkbox("Use Bayesian Forecasting", value=True)
if apply_bayes:
    bayesian_update_user()

# Train button
if st.button("🛠️ Train Model Now"):
    model = train_model(training_data, apply_bayesian=apply_bayes)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("✅ Model trained and saved as model.pkl")

# Load model
try:
    model = pickle.load(open("model.pkl", "rb"))
    st.success("📦 Model loaded successfully.")
except:
    model = None
    st.warning("⚠️ No trained model found. Please train it first.")

# Predict last row
st.write("### 🧠 Generate Signal from Latest Row")
row = df.iloc[-1].drop("Label", errors="ignore").to_dict()
if model:
    input_df = pd.DataFrame([row])[ [col for col in ["RSI", "Momentum", "ATR", "Volume"] if col in row] ]
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

# Refresh settings
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

# Live Signal
st.write("### 📡 Live Signal (1-min Data Feed)")
col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Choose Ticker (ETF Proxy)", ["SPY", "QQQ", "DIA", "IWM"])
with col2:
    api_key = st.text_input("🔑 Twelve Data API Key", type="password")

# Slack
slack_webhook = st.text_input("🔗 Enter Slack Webhook URL", type="password")
enable_slack = st.checkbox("📣 Send Slack Alerts", value=True)
already_in_trade = True

# Slack test
if st.button("🧪 Send Test Slack Alert"):
    if slack_webhook:
        send_slack_alert(slack_webhook, "SPY", "Buy", 95.0, 522.15, str(datetime.datetime.now()))
        st.success("✅ Test Slack alert sent!")
    else:
        st.warning("⚠️ Please enter a valid Slack Webhook URL.")

# Live signal execution
if api_key and model:
    live_row = fetch_latest_data(symbol=ticker, api_key=api_key)
    if "error" in live_row:
        st.error(f"API Error: {live_row['error']}")
    else:
        live_input = pd.DataFrame([{key: live_row[key] for key in ["RSI", "Momentum", "ATR", "Volume"] if key in live_row}])
        pred = model.predict(live_input)[0]
        proba = model.predict_proba(live_input)[0]
        confidence = round(100 * max(proba), 2)

        timestamp = live_row["datetime"]
        price = round(live_row["close"], 2)

        st.markdown("---")
        st.markdown(f"🧠 **LIVE SIGNAL for {ticker}**  \n🕒 Timestamp: `{timestamp}`")
        st.metric(label="Signal", value=pred)
        st.metric(label="Live Price", value=f"${price}")
        st.metric(label="Confidence", value=f"{confidence}%")

        if confidence >= threshold:
            log_signal(ticker, pred, confidence, price, timestamp)

            if enable_slack and slack_webhook and (pred in ["Buy", "Sell"] or (pred == "Hold" and already_in_trade)):
                send_slack_alert(slack_webhook, ticker, pred, confidence, price, timestamp)

            with st.form(key="journal_form"):
                st.subheader("📝 Log Trade Journal Entry")
                reason = st.text_input("🧠 Why did you take this trade?")
                emotion = st.selectbox("😐 Emotion before/after trade", ["Neutral", "Confident", "Anxious", "Fearful", "Greedy"])
                reflection = st.text_area("📓 Trade outcome / reflection")
                file = st.file_uploader("📎 Upload Screenshot (optional)", type=["png", "jpg", "jpeg"])

                submit = st.form_submit_button("Save Journal Entry")
                if submit:
                    filename = file.name if file else None
                    log_journal(timestamp, ticker, pred, reason, emotion, reflection, filename)
                    if file:
                        with open(os.path.join("uploads", file.name), "wb") as f:
                            f.write(file.read())
                    st.success("✅ Journal entry saved!")

# Backtest Results
st.write("### 📈 Backtest Strategy Results")
results = run_backtest(df_window)
st.write(results)

# Journal View
st.write("### 📚 Recent Trade Journal Entries")
if os.path.exists("trade_journal.csv"):
    journal_df = pd.read_csv("trade_journal.csv")
    st.dataframe(journal_df.tail(5))
else:
    st.info("No journal entries yet.")
