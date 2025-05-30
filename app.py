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
    df['Momentum'] = df['Close'].diff().rolling(5).mean()
    df['ATR'] = df['High'].diff().abs() + df['Low'].diff().abs() + (df['High'] - df['Low']).abs()
    df['ATR'] = df['ATR'].rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (
        df["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
        -hist_df["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean()
    )))
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
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={int(pd.Timestamp(f'now-{period}').timestamp())}&period2={int(pd.Timestamp.now().timestamp())}&interval=1m&events=history"
    response = requests.get(url)
    df = pd.read_csv(response.text, index_col='Date')
    return df

# Function to run backtest
def run_backtest(df):
    # Simple backtest logic here
    pass

# Streamlit App with Tabs
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

tab1, tab2 = st.tabs(["📈 Load & Train from Yahoo Finance", "🤖 Predict Signal from Twelve Data"])

with tab1:
    st.header("📈 Load & Train from Yahoo Finance")
    
    ticker = st.selectbox("Choose Ticker", ["SPY"])
    period_options = [1, 7, 30]  # Days
    period = st.slider("Period (days)", min_value=period_options[0], max_value=period_options[-1], value=period_options[2])
    api_key = st.text_input("🔑 Twelve Data API Key", type="password")

    if st.button("🧠 Train Model"):
        try:
            hist_df = fetch_yahoo_intraday(ticker, period)
            hist_df["Momentum"] = hist_df["Close"].diff().rolling(5).mean()
            hist_df["ATR"] = hist_df["High"].diff().abs() + hist_df["Low"].diff().abs() + (hist_df["High"] - df['Low']).abs()
            hist_df["ATR"] = hist_df["ATR"].rolling(14).mean()
            hist_df["RSI"] = 100 - (100 / (1 + (
                hist_df["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
                -hist_df["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean()
            )))
            hist_df["Label"] = np.where(hist_df["Close"].shift(-5) > hist_df["Close"], "Buy", "Sell")
            hist_df["Volume"] = hist_df["Volume"].fillna(1)
            hist_df = hist_df.dropna()

            if len(hist_df) < 30:
                raise ValueError("Not enough samples to train a model. Please select a longer period.")

            model = train_model(hist_df)
            pickle.dump(model, open("model.pkl", "wb"))
            st.success("✅ Model trained and saved to model.pkl.")

            st.write("### 📊 Label Distribution")
            st.bar_chart(hist_df["Label"].value_counts())

        except Exception as e:
            st.warning(f"⚠️ Training failed: {e}")

with tab2:
    st.header("🤖 Predict Signal from Twelve Data")
    
    api_key = st.text_input("🔑 Twelve Data API Key", type="password")
    
    if api_key:
        symbol = "SPY"
        df = fetch_twelve_data(symbol, api_key)
        df = add_custom_features(df)
        df.to_csv("training_data.csv", index=False)

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

# Ensure model is not None after training
if os.path.exists("model.pkl"):
    with open("model.pkl", 'rb') as f:
        st.session_state.model = pickle.load(f)
    st.success("✅ Model loaded from disk.")
else:
    st.warning("⚠️ No trained model available. Please train with Yahoo data first.")

st.stop()  # Stop further execution if no valid model is found
