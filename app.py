
import streamlit as st
import pandas as pd
import pickle
from strategy_utils import generate_signal, run_backtest, train_model, bayesian_update_user
from live_data import fetch_latest_data
import datetime
import pytz

# Set page config first
st.set_page_config(page_title="ClarityTrader Signal", layout="centered")

st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

# Upload CSV or load default
uploaded_file = st.file_uploader("📤 Upload CSV or use example data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("spy_training_data.csv")

# Convert datetime if it exists
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])

# Time slice for preview + training
st.write("### ⏱ Backtest Time Window")
start_idx = st.number_input("Start Row", min_value=0, max_value=len(df)-2, value=0)
end_idx = st.number_input("End Row", min_value=start_idx+1, max_value=len(df), value=len(df))
df_window = df.iloc[int(start_idx):int(end_idx)]

# Show datetime range if available
if 'datetime' in df.columns:
    st.write(f"📅 Backtest Date Range: {df_window.iloc[0]['datetime']} → {df_window.iloc[-1]['datetime']}")

# Use slice for training?
use_slice = st.checkbox("Train model on selected range only", value=False)
training_data = df_window if use_slice else df

# Show only relevant preview
st.write("### 📊 Preview Data (From Backtest Window)")
st.dataframe(df_window.head())

# Confidence slider
threshold = st.slider("🎯 Confidence Threshold (%)", min_value=50, max_value=100, value=70, step=1)

# Bayesian option
apply_bayes = st.checkbox("Use Bayesian Forecasting", value=True)
if apply_bayes:
    bayesian_update_user()

# Train model
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

# Signal from latest row
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

# Auto refresh
st.write("### 🔁 Auto Refresh Settings")
enable_refresh = st.checkbox("Enable Auto Refresh During Market Hours", value=True)

# Market check
eastern = pytz.timezone("US/Eastern")
now_et = datetime.datetime.now(eastern)
market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
in_market = market_open <= now_et <= market_close and now_et.weekday() < 5

if enable_refresh and in_market:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60000, key="auto-refresh")  # 60 seconds

# Live signal section
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

        st.markdown("---")
        st.markdown(f"🧠 **LIVE SIGNAL for {ticker}**  
🕒 Timestamp: `{live_row['datetime']}`")
        st.metric(label="Signal", value=pred)
        st.metric(label="Live Price", value=f"${live_row['close']:.2f}")
        st.metric(label="Confidence", value=f"{confidence}%")

# Checklist
with st.expander("🧾 ClarityTrader Market Open Checklist", expanded=False):
    st.markdown("### 1. 🔧 Pre-Market Prep (8:30–9:25 AM ET)")
    st.markdown("- [ ] Open ClarityTrader")
    st.markdown("- [ ] Upload new training data (optional)")
    st.markdown("- [ ] Retrain model")
    st.markdown("- [ ] Set confidence threshold")
    st.markdown("- [ ] Enable Bayesian Forecasting")

    st.markdown("### 2. ⏱ Market Open (9:30 AM ET)")
    st.markdown("- [ ] Check for first 1-min candle")
    st.markdown("- [ ] View live signal with auto-refresh")

    st.markdown("### 3. 🧪 Execution Plan")
    st.markdown("- [ ] Trade only strong signals")
    st.markdown("- [ ] Log trades")
    st.markdown("- [ ] Stay disciplined")

    st.markdown("### 4. 📈 Post-Session")
    st.markdown("- [ ] Run backtest")
    st.markdown("- [ ] Review win rate")
    st.markdown("- [ ] Save model")

# Backtest Results
st.write("### 📈 Backtest Strategy Results")
results = run_backtest(df_window)
st.write(results)
