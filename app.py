import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Function to fetch data from Twelve Data
def fetch_twelve_data(symbol, api_key):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1m&outputsize=20&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data["values"])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

# Function to add custom features
def add_custom_features(df):
    # Add your feature engineering code here
    return df

# Function to train a model
def train_model(df):
    X = df[["RSI", "Momentum", "ATR", "Volume"]]
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to fetch Yahoo intraday data
def fetch_yahoo_intraday(ticker, period):
    # Fetch and preprocess Yahoo Finance data here
    pass

# Function to run backtest
def run_backtest(df):
    # Simple backtest logic here
    pass

# Main Streamlit App
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

# Sidebar options
with st.sidebar:
    action = st.radio("📡 Choose Action", ["🔁 Predict Signal from Twelve Data", "📥 Load & Train from Yahoo Finance"])

if action == "🔁 Predict Signal from Twelve Data":
    api_key = st.text_input("🔑 Twelve Data API Key", type="password")
    if api_key:
        symbol = "SPY"
        df = fetch_twelve_data(symbol, api_key)
        df = add_custom_features(df)
        df.to_csv("training_data.csv", index=False)

        st.write(f"### Latest Data")
        st.write(df.tail())

        model_path = "model.pkl"
        if not os.path.exists(model_path):
            st.warning("⚠️ No trained model available. Please train with Yahoo data first.")
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            X_live = df[["RSI", "Momentum", "ATR", "Volume"]]
            pred = model.predict(X_live)
            st.write(f"### Live Signal")
            st.text(pred[-1])

if action == "📥 Load & Train from Yahoo Finance":
    ticker = st.selectbox("Choose Ticker", ["SPY"])
    period = st.slider("Period", min_value="1d", max_value="1mo", value="1mo")
    
    if st.button("🧠 Train Model"):
        try:
            hist_df = fetch_yahoo_intraday(ticker, period)
            hist_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in hist_df.columns]
            hist_df.rename(columns={
                "Datetime": "datetime", "Close_SPY": "Close", "High_SPY": "High",
                "Low_SPY": "Low", "Volume_SPY": "Volume"
            }, inplace=True)

            hist_df["Momentum"] = hist_df["Close"] - hist_df["Close"].shift(5)
            hist_df["TR"] = hist_df[["High", "Low"]].max(axis=1) - hist_df[["High", "Low"]].min(axis=1)
            hist_df["ATR"] = hist_df["TR"].rolling(14).mean()
            hist_df["RSI"] = 100 - (100 / (1 + (
                hist_df["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
                -hist_df["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean()
            )))
            hist_df["Label"] = np.where(hist_df["Close"].shift(-5) > hist_df["Close"], "Buy", "Sell")
            hist_df["Volume"] = hist_df["Volume"].fillna(1000000)
            hist_df = hist_df.dropna()

            df = add_custom_features(hist_df)
            df.to_csv("training_data.csv", index=False)

            if len(df) < 30:
                raise ValueError("Not enough samples to train a model. Please select a longer period.")

            model = train_model(df)
            pickle.dump(model, open("model.pkl", "wb"))
            st.success("✅ Model trained and saved to model.pkl.")

            st.write("### 📊 Label Distribution")
            st.bar_chart(df["Label"].value_counts())

        except Exception as e:
            st.warning(f"⚠️ Training failed: {e}")

# Ensure model is not None after training
if os.path.exists("model.pkl"):
    with open("model.pkl", 'rb') as f:
        st.session_state.model = pickle.load(f)
    st.success("✅ Model loaded from disk.")
else:
    st.warning("⚠️ No trained model available. Please train with Yahoo data first.")

st.stop()  # Stop further execution if no valid model is found
