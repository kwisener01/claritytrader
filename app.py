import streamlit as st
import pandas as pd
import pickle
from strategy_utils import generate_signal, run_backtest, train_model, bayesian_update_user
from live_data import fetch_latest_data

st.set_page_config(page_title="ClarityTrader Signal", layout="centered")
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

# Upload data or load default training set
uploaded_file = st.file_uploader("📤 Upload CSV or use example data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("spy_training_data.csv")

# Confidence threshold control
threshold = st.slider("🎯 Confidence Threshold (%)", min_value=50, max_value=100, value=70, step=1)

st.write("### 📊 Preview Data", df.head())

# Bayesian option
apply_bayes = st.checkbox("Use Bayesian Forecasting", value=True)
if apply_bayes:
    bayesian_update_user()

# Train model
if st.button("🛠️ Train Model Now"):
    model = train_model(df, apply_bayesian=apply_bayes)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("✅ Model trained and saved as model.pkl")

# Try loading model if exists
try:
    model = pickle.load(open("model.pkl", "rb"))
    st.success("📦 Model loaded successfully.")
except:
    model = None
    st.warning("⚠️ No trained model found. Please train it first.")

# Predict signal from latest row
st.write("### 🧠 Generate Signal from Latest Row")
row = df.iloc[-1].drop("Label", errors="ignore").to_dict()

if model:
    input_df = pd.DataFrame([row])[["RSI", "Momentum", "ATR", "Volume"]]
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    confidence = round(100 * max(proba), 2)
    if confidence >= threshold:
        st.metric(label="Predicted Signal", value=pred)
        st.write(f"🧠 Confidence: **{confidence}%**")
    else:
        st.warning(f"No signal. Confidence too low ({confidence}%)")
else:
    pred = generate_signal(row)
    st.metric(label="Predicted Signal (Rule-Based)", value=pred)

# Live Signal from SPY using Twelve Data
st.write("### 📡 Live Signal (SPY via Twelve Data)")
api_key = st.text_input("🔑 Enter your Twelve Data API Key", type="password")

if st.button("Get Live Signal"):
    if model is None:
        st.error("⚠️ Train or load a model before using live signals.")
    elif not api_key:
        st.warning("Please enter your API key.")
    else:
        live_row = fetch_latest_data(symbol="SPY", api_key=api_key)
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
            if confidence >= threshold:
                st.metric(label="LIVE Signal", value=pred)
                st.write(f"📈 Live Price: ${live_row['close']}")
                st.write(f"🧠 Confidence: **{confidence}%**")
            else:
                st.warning(f"No signal. Confidence too low ({confidence}%)")

# Backtest summary
st.write("### 📈 Backtest Strategy Results")
backtest_results = run_backtest(df)
st.write(backtest_results)
