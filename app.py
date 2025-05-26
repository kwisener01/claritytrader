import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import os
import io
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from strategy_utils import generate_signal, run_backtest, train_model, bayesian_update_user, add_custom_features
from live_data import fetch_latest_data
from send_slack_alert import send_slack_alert
from yahoo_data import fetch_yahoo_intraday

st.set_page_config(page_title="ClarityTrader Signal", layout="centered")
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

if 'training_data' not in st.session_state:
    st.session_state.training_data = pd.read_csv("spy_training_data.csv")

if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []

if 'trade_journal' not in st.session_state:
    st.session_state.trade_journal = []

st_autorefresh(interval=300000, key="train_refresh")

source = st.radio("📡 Choose Data Source", ["Twelve Data (Live)", "Yahoo Finance (Historical)"])
ticker = st.selectbox("Choose Ticker", ["SPY", "QQQ", "DIA", "IWM"])
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
    period = st.selectbox("📆 Yahoo Period", ["1d", "5d", "7d", "1mo", "3mo"])
    hist_df = fetch_yahoo_intraday(symbol=ticker, period=period)
    hist_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in hist_df.columns]

    hist_df.rename(columns={
        "Datetime": "datetime", "Close_SPY": "Close", "High_SPY": "High",
        "Low_SPY": "Low", "Volume_SPY": "Volume"
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
        hist_df["RSI"] = 100 - (100 / (1 + (
            hist_df["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
            -hist_df["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean()
        )))

        if "Volume" not in hist_df.columns or hist_df["Volume"].nunique() <= 1:
            hist_df["Volume"] = 1000000

        hist_df = hist_df.dropna()

        required = ["datetime", "Open", "High", "Low", "Close", "RSI", "Momentum", "ATR", "Volume"]
        for col in required:
            if col not in hist_df.columns:
                hist_df[col] = None
        hist_df = hist_df[required + [c for c in hist_df.columns if c not in required]]

        hist_df["Label"] = np.where(hist_df["Close"].shift(-5) > hist_df["Close"], "Buy", "Sell")
        st.session_state.training_data = pd.concat([st.session_state.training_data, hist_df], ignore_index=True)
        st.success(f"✅ Pulled {len(hist_df)} rows from Yahoo Finance")
        timestamp = str(datetime.datetime.now())
        price = hist_df["Close"].iloc[-1] if "Close" in hist_df.columns and not hist_df["Close"].empty else 0

st.write("### 📊 Label Distribution")
st.bar_chart(st.session_state.training_data["Label"].value_counts())
st.dataframe(st.session_state.training_data.tail(10))
st.download_button("📥 Download CSV", st.session_state.training_data.to_csv(index=False).encode("utf-8"), "training_data.csv")

threshold = st.slider("🎯 Confidence Threshold", 50, 100, 70, 1)
if st.checkbox("Use Bayesian Forecasting", value=True):
    bayesian_update_user()

# Feature engineering + training
raw_data = st.session_state.training_data.dropna(subset=["RSI", "Momentum", "ATR", "Volume", "Label"])
clean_data = add_custom_features(raw_data.copy())
features = ["RSI", "Momentum", "ATR", "Volume", "Accel", "VolSpike"]

if not clean_data.empty:
    X = clean_data[features]
    y = clean_data["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = train_model(pd.concat([X_train, y_train], axis=1), apply_bayesian=True)
    st.session_state['model'] = pickle.dumps(model)
    st.success("✅ Model trained")

    st.write("### 🧪 Backtest")
    st.json(run_backtest(clean_data))

    y_pred = model.predict(X_test)
    st.write("### 📊 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.write("### 📊 Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=["Buy", "Sell"])
    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Buy", "Sell"])
    ax.set_yticklabels(["Buy", "Sell"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="white" if conf_matrix[i, j] > 0 else "black")
    st.pyplot(fig)

    st.write("### 📈 Feature Importance")
    try:
        st.bar_chart(pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).set_index("Feature"))
    except:
        st.warning("Could not display feature importance.")

    st.write("### 📉 OHLC Chart")
    try:
        chart_data = clean_data[["datetime", "Open", "High", "Low", "Close"]].copy()
        chart_data["datetime"] = pd.to_datetime(chart_data["datetime"])
        chart_data.set_index("datetime", inplace=True)
        fig = go.Figure(data=[go.Candlestick(x=chart_data.index, open=chart_data["Open"], high=chart_data["High"],
                                             low=chart_data["Low"], close=chart_data["Close"])])
        for signal, color in [("Buy", "green"), ("Sell", "red")]:
            df = clean_data[clean_data["Label"] == signal]
            fig.add_trace(go.Scatter(x=df["datetime"], y=df["Close"], mode="markers", name=signal, marker=dict(color=color, size=6)))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart error: {e}")
else:
    model = None
    st.warning("⚠️ No valid rows available for training.")
