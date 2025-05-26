# Full ClarityTrader app with Yahoo Finance & Twelve Data support, safe training, labeling, visualization

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import os
import io
from streamlit_autorefresh import st_autorefresh
from strategy_utils import generate_signal, run_backtest, train_model, bayesian_update_user
from live_data import fetch_latest_data
from send_slack_alert import send_slack_alert
from yahoo_data import fetch_yahoo_intraday
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

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
    hist_df.rename(columns={"Datetime": "datetime", "Close_SPY": "Close", "High_SPY": "High", "Low_SPY": "Low", "Volume_SPY": "Volume"}, inplace=True)

    if hist_df.empty:
        st.warning("⚠️ No data retrieved from Yahoo Finance.")
    else:
        hist_df["Momentum"] = hist_df["Close"] - hist_df["Close"].shift(5)
        hist_df["H-L"] = hist_df["High"] - hist_df["Low"]
        hist_df["H-PC"] = abs(hist_df["High"] - hist_df["Close"].shift(1))
        hist_df["L-PC"] = abs(hist_df["Low"] - hist_df["Close"].shift(1))
        hist_df["TR"] = hist_df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        hist_df["ATR"] = hist_df["TR"].rolling(14).mean()
        hist_df["RSI"] = 100 - (100 / (1 + (hist_df["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
                                           (-hist_df["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean()))))

        if "Volume" not in hist_df.columns or hist_df["Volume"].nunique() <= 1:
            hist_df["Volume"] = 1000000

        hist_df = hist_df.dropna()

        required_columns = ["datetime", "Open", "High", "Low", "Close", "RSI", "Momentum", "ATR", "Volume"]
        for col in required_columns:
            if col not in hist_df.columns:
                hist_df[col] = None
        ordered_cols = required_columns + [col for col in hist_df.columns if col not in required_columns]
        hist_df = hist_df[ordered_cols]

        hist_df["Label"] = np.where(hist_df["Close"].shift(-5) > hist_df["Close"], "Buy", "Sell")
        st.session_state.training_data = pd.concat([st.session_state.training_data, hist_df], ignore_index=True)
        st.success(f"✅ Pulled {len(hist_df)} rows from Yahoo Finance")
        timestamp = str(datetime.datetime.now())
        price = hist_df["Close"].iloc[-1] if "Close" in hist_df.columns and not hist_df["Close"].empty else 0

with st.expander("➕ Add Training Row Manually"):
    rsi = st.number_input("RSI", min_value=0.0, max_value=100.0, step=0.1)
    momentum = st.number_input("Momentum", step=0.01)
    atr = st.number_input("ATR", step=0.01)
    volume = st.number_input("Volume", step=1000)
    label = st.selectbox("Label", ["Buy", "Sell", "Hold"])
    if st.button("📌 Add to Training Data"):
        new_row = pd.DataFrame([{"RSI": rsi, "Momentum": momentum, "ATR": atr, "Volume": volume, "Label": label}])
        st.session_state.training_data = pd.concat([st.session_state.training_data, new_row], ignore_index=True)
        st.success("✅ Row added to training data.")

st.write("### 📊 Label Distribution")
label_counts = st.session_state.training_data["Label"].value_counts()
st.bar_chart(label_counts)
st.write("### 🧾 Current Training Data")
st.dataframe(st.session_state.training_data.tail(10))

csv_export = st.session_state.training_data.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Updated Training Data", csv_export, file_name="updated_training_data.csv")

threshold = st.slider("🎯 Confidence Threshold (%)", min_value=50, max_value=100, value=70, step=1)
apply_bayes = st.checkbox("Use Bayesian Forecasting", value=True)
if apply_bayes:
    bayesian_update_user()

clean_data = st.session_state.training_data.dropna(subset=["RSI", "Momentum", "ATR", "Volume", "Label"])
if not clean_data.empty:
    model = train_model(clean_data, apply_bayesian=apply_bayes)
    buf = io.BytesIO()
    pickle.dump(model, buf)
    st.session_state['model'] = buf.getvalue()
    st.success("✅ Model auto-trained")

    backtest_results = run_backtest(clean_data)
    st.json(backtest_results)

    y_true = clean_data["Label"]
    y_pred = model.predict(clean_data[["RSI", "Momentum", "ATR", "Volume"]])
    st.write("### 📊 Classification Report")
    st.text(classification_report(y_true, y_pred))
    st.write("### 📊 Confusion Matrix")
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Buy", "Sell"], yticklabels=["Buy", "Sell"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(fig)

    st.write("### 📈 Feature Importance")
    try:
        importances = model.feature_importances_
        features = ["RSI", "Momentum", "ATR", "Volume"]
        importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
        st.bar_chart(importance_df.set_index("Feature"))
    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")

    st.write("### 📉 OHLC Price Chart")
    try:
        ohlc_plot_data = clean_data[['datetime', 'Open', 'High', 'Low', 'Close']].copy()
        ohlc_plot_data['datetime'] = pd.to_datetime(ohlc_plot_data['datetime'])
        ohlc_plot_data.set_index('datetime', inplace=True)
        fig = go.Figure(data=[go.Candlestick(
            x=ohlc_plot_data.index,
            open=ohlc_plot_data["Open"],
            high=ohlc_plot_data["High"],
            low=ohlc_plot_data["Low"],
            close=ohlc_plot_data["Close"],
            name="Price")])
        buy_signals = clean_data[clean_data["Label"] == "Buy"]
        sell_signals = clean_data[clean_data["Label"] == "Sell"]
        fig.add_trace(go.Scatter(x=buy_signals["datetime"], y=buy_signals["Close"], mode="markers", name="Buy", marker=dict(color="green", size=8)))
        fig.add_trace(go.Scatter(x=sell_signals["datetime"], y=sell_signals["Close"], mode="markers", name="Sell", marker=dict(color="red", size=8)))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot OHLC chart: {e}")
else:
    model = None
    st.warning("⚠️ No valid rows available for training.")
