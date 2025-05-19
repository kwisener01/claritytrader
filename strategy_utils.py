import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def generate_signal(row):
    if row["RSI"] < 30 and row["Momentum"] > 0:
        return "Buy"
    elif row["RSI"] > 70 and row["Momentum"] < 0:
        return "Sell"
    else:
        return "Hold"

def run_backtest(df):
    signals = []
    for _, row in df.iterrows():
        signals.append(generate_signal(row))
    df["Signal"] = signals
    buys = df[df["Signal"] == "Buy"]
    sells = df[df["Signal"] == "Sell"]
    return {
        "Total Signals": len(df),
        "Buy Signals": len(buys),
        "Sell Signals": len(sells),
        "Hold Signals": len(df) - len(buys) - len(sells)
    }

def train_model(df, apply_bayesian=False):
    X = df[["RSI", "Momentum", "ATR", "Volume"]]
    y = df["Label"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def bayesian_update_user():
    st.write("### 🎯 Bayesian Adjustment Inputs")
    prior_success = st.slider("📊 Prior Win Rate (P(Success))", 0.1, 0.99, 0.6, 0.01)
    likelihood_success_signal = st.slider("📈 Likelihood of Signal in Winning Trades (P(Signal|Success))", 0.1, 1.0, 0.7, 0.01)
    likelihood_signal = st.slider("📉 Overall Signal Frequency (P(Signal))", 0.1, 1.0, 0.5, 0.01)

    try:
        posterior = (likelihood_success_signal * prior_success) / likelihood_signal
        posterior = min(posterior, 1.0)
    except ZeroDivisionError:
        posterior = 0.0

    st.success(f"🧠 Adjusted Probability of Success: {posterior * 100:.2f}%")
    return posterior
