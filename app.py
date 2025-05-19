import streamlit as st
import pandas as pd
import pickle
from strategy_utils import generate_signal, run_backtest

st.set_page_config(page_title="ClarityTrader Signal", layout="centered")
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

# Upload or sample data
uploaded_file = st.file_uploader("Upload CSV or use example data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("example_data.csv")

st.write("### Preview Data", df.head())

# Load model
try:
    model = pickle.load(open("model.pkl", "rb"))
    st.success("Model loaded successfully.")
except:
    model = None
    st.warning("No model found. Using rule-based fallback.")

# Predict section
st.write("### Generate Signal")
row = df.iloc[-1].to_dict()
if model:
    pred = model.predict(pd.DataFrame([row]))[0]
else:
    pred = generate_signal(row)

st.metric(label="Predicted Signal", value=pred)

# Run backtest
st.write("### Backtest Strategy")
backtest_results = run_backtest(df)
st.write(backtest_results)
