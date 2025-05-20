import streamlit as st
import pandas as pd
import pickle
from strategy_utils import generate_signal, run_backtest, train_model, bayesian_update_user
from live_data import fetch_latest_data

st.set_page_config(page_title="ClarityTrader Signal", layout="centered")
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

uploaded_file = st.file_uploader("Upload CSV or use example data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("spy_training_data.csv")

st.write("### Preview Data", df.head())

apply_bayes = st.checkbox("Use Bayesian Forecasting", value=True)
if apply_bayes:
    bayesian_update_user()


if 'model' not in locals():
    try:
        model = pickle.load(open("model.pkl", "rb"))
    except:
        model = None


if st.button("📡 Get Live Signal"):
    live_row = fetch_latest_data(symbol="SPY", api_key="7c53601780c14ef5a6893e0d522e2388")
    if "error" in live_row:
        st.error(f"API Error: {live_row['error']}")
    else:
        input_df = pd.DataFrame([{
            "RSI": live_row["RSI"],
            "Momentum": live_row["Momentum"],
            "ATR": live_row["ATR"],
            "Volume": live_row["Volume"]
        }])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        confidence = round(100 * max(proba), 2)
        if confidence >= threshold:
            st.metric(label="LIVE Signal", value=pred)
            st.write(f"🧠 Confidence: **{confidence}%**")
        else:
            st.warning(f"No signal. Confidence too low ({confidence}%)")



if st.button("🛠️ Train Model Now"):
    model = train_model(df, apply_bayes)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model trained and saved as model.pkl")

try:
    model = pickle.load(open("model.pkl", "rb"))
    st.success("Model loaded successfully.")
except:
    model = None
    st.warning("No model found. Using rule-based fallback.")

st.write("### Generate Signal")
row = df.iloc[-1].drop("Label", errors="ignore").to_dict()
threshold = 70

if model:
    input_df = pd.DataFrame([row])[["RSI", "Momentum", "ATR", "Volume"]]
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    confidence = round(100 * max(proba), 2)
    if confidence >= threshold:
        st.metric(label="Predicted Signal", value=pred)
        st.write(f"🧠 Confidence: **{confidence}%**")
    else:
        st.warning(f"🧠 No signal. Confidence too low to act ({confidence}%)")
else:
    pred = generate_signal(row)
    st.metric(label="Predicted Signal (Rule-Based)", value=pred)

st.write("### Backtest Strategy")
backtest_results = run_backtest(df)
st.write(backtest_results)
