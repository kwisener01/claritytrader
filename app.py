# Updated app.py with Yahoo Finance integration, auto-training, session-safe storage, logging, and export

import streamlit as st
import pandas as pd
import pickle
import datetime
import pytz
import os
import io
import numpy as np
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
        df_new["Label"] = "Hold"
        st.session_state.training_data = pd.concat([st.session_state.training_data, df_new], ignore_index=True)
        timestamp = new_row["datetime"]
        price = new_row["close"]
    else:
        st.error(f"API Error: {new_row['error']}")
        timestamp = str(datetime.datetime.now())
        price = 0
elif source == "Yahoo Finance (Historical)":
    period = st.selectbox("📆 Yahoo Finance Period", ["1d", "5d", "7d", "1mo", "3mo"])
    hist_df = fetch_yahoo_intraday(symbol=ticker, period=period)

    hist_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in hist_df.columns]
    hist_df.rename(columns={
        "Datetime": "datetime",
        "Close_SPY": "Close",
        "High_SPY": "High",
        "Low_SPY": "Low",
        "Volume_SPY": "Volume"
    }, inplace=True)

    if hist_df.empty:
        st.warning("⚠️ No data retrieved from Yahoo Finance.")
    else:
        hist_df["Momentum"] = hist_df["Close"] - hist_df["Close"].shift(5)
        hist_df["H-L"] = hist_df["High"] - hist_df["Low"]
        hist_df["H-PC"] = abs(hist_df["High"] - hist_df["Close"].shift(1))
        hist_df["L-PC"] = abs(hist_df["Low"] - hist_df["Close"].shift(1))
        hist_df["TR"] = hist_df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        hist_df["ATR"] = hist_df["TR"].rolling(14).mean()
        hist_df["RSI"] = 100 - (100 / (1 + (hist_df["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() / (-hist_df["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean()))))

        if "Volume" not in hist_df.columns or hist_df["Volume"].nunique() <= 1:
            hist_df["Volume"] = 1000000

        hist_df = hist_df.dropna()

        required_columns = ["datetime", "Open", "High", "Low", "Close", "RSI", "Momentum", "ATR", "Volume"]
        missing_columns = [col for col in required_columns if col not in hist_df.columns]
        if missing_columns:
            st.warning(f"⚠️ Missing columns in historical data: {missing_columns}")
            for col in missing_columns:
                hist_df[col] = None

        ordered_cols = required_columns + [col for col in hist_df.columns if col not in required_columns]
        hist_df = hist_df[ordered_cols]

        if "Close" in hist_df.columns:
            hist_df["Label"] = np.where(hist_df["Close"].shift(-5) > hist_df["Close"], "Buy", "Sell")
        else:
            st.warning("⚠️ 'Close' column is missing. Defaulting label to 'Hold'.")
            hist_df["Label"] = "Hold"

        st.session_state.training_data = pd.concat([st.session_state.training_data, hist_df], ignore_index=True)
        st.success(f"✅ Pulled {len(hist_df)} rows from Yahoo Finance")
        timestamp = str(datetime.datetime.now())
        price = hist_df["Close"].iloc[-1] if "Close" in hist_df.columns else 0
