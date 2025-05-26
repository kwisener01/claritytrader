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
    correct_predictions = 0
    total_trades = 0

    for i, row in df.iterrows():
        signal = generate_signal(row)
        signals.append(signal)
        if "Label" in df.columns:
            if signal == row["Label"] and signal != "Hold":
                correct_predictions += 1
            if signal != "Hold":
                total_trades += 1

    df["Signal"] = signals
    buys = df[df["Signal"] == "Buy"]
    sells = df[df["Signal"] == "Sell"]
    win_rate = (correct_predictions / total_trades * 100) if total_trades > 0 else 0

    return {
        "Total Signals": len(df),
        "Buy Signals": len(buys),
        "Sell Signals": len(sells),
        "Hold Signals": len(df) - len(buys) - len(sells),
        "Backtest Range": f"{df.index.min()} to {df.index.max()}",
        "Win Rate (%)": round(win_rate, 2)
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
    correct_predictions = 0
    total_trades = 0

    for i, row in df.iterrows():
        signal = generate_signal(row)
        signals.append(signal)
        if "Label" in df.columns:
            if signal == row["Label"] and signal != "Hold":
                correct_predictions += 1
            if signal != "Hold":
                total_trades += 1

    df["Signal"] = signals
    buys = df[df["Signal"] == "Buy"]
    sells = df[df["Signal"] == "Sell"]
    win_rate = (correct_predictions / total_trades * 100) if total_trades > 0 else 0

    return {
        "Total Signals": len(df),
        "Buy Signals": len(buys),
        "Sell Signals": len(sells),
        "Hold Signals": len(df) - len(buys) - len(sells),
        "Backtest Range": f"{df.index.min()} to {df.index.max()}",
        "Win Rate (%)": round(win_rate, 2)
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

def add_custom_features(df):
    df["Accel"] = df["Momentum"].diff()
    df["VolSpike"] = df["Volume"] / df["Volume"].rolling(20).mean()
    return df

