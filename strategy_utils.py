import pandas as pd
import streamlit as st

def generate_signal(row):
    # Basic rule-based logic
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


from sklearn.ensemble import RandomForestClassifier

def train_model(df, apply_bayesian=False):
    if apply_bayesian:
        # Estimate signal quality (you can refine this with backtest stats)
        prior_success = 0.6  # example base win rate
        likelihood_signal = len(df) / len(df)  # dummy for now
        likelihood_success_signal = 0.7  # assume this signal shows up in 70% of winners
        adjustment = bayesian_update(prior_success, likelihood_success_signal, likelihood_signal)
        st.info(f"Bayesian Adjusted Win Probability: {adjustment * 100:.2f}%")


    X = df[["RSI", "Momentum", "ATR", "Volume"]]
    y = df["Label"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def bayesian_update(prior_success, likelihood_success_signal, likelihood_signal):
    """
    Apply Bayes' Theorem:
    P(Success | Signal) = (P(Signal | Success) * P(Success)) / P(Signal)
    """
    try:
        posterior = (likelihood_success_signal * prior_success) / likelihood_signal
        return round(posterior, 4)
    except ZeroDivisionError:
        return 0.0
