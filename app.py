import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import requests
import pickle

# Function to fetch Twelve Data intraday data
def fetch_twelve_data(symbol, api_key):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=500&apikey={api_key}"
    response = requests.get(url)
    df = pd.DataFrame(response.json()['values'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df.set_index('datetime', inplace=True)
    return df

# Function to add custom features
def add_custom_features(df):
    df["Momentum"] = df["close"].diff().rolling(5).mean()
    df["ATR"] = (df["high"] - df["low"]).abs() + (df["close"].diff().abs())
    df["RSI"] = 100 * (df["close"] - df["low"].rolling(window=14).min()) / (df["high"].rolling(window=14).max() - df["low"].rolling(window=14).min())
    return df

# Function to predict the stock price
def predict_price(api_key, symbol):
    try:
        df = fetch_twelve_data(symbol, api_key)
        df = add_custom_features(df)
        # Assuming last row is the latest data point
        X_new = df.iloc[-1:].drop(['close'], axis=1)
        return int(X_new.values[0])
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Load pre-trained model
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define tabs
tab1, tab2 = st.tabs(["Train Model with Yahoo Data", "Predict Signal from Twelve Data"])

with tab1:
    st.header("🤖 Train Model with Yahoo Data")
    symbol = st.text_input("Yahoo Finance Symbol", value="SPY")
    days_to_train = st.radio("Days to Train Model", ["5", "7"])
    if st.button("Train Model"):
        # Your training code here
        st.success("Model trained successfully!")

with tab2:
    st.header("🤖 Predict Signal from Twelve Data")
    api_key = st.text_input("🔑 Twelve Data API Key", type="password", key="unique_api_key")  # Add unique key here
    symbol = st.text_input("Yahoo Finance Symbol", value="SPY", key="unique_symbol")  # Add unique key here
    refresh_minutes = st.slider("Refresh Minutes", min_value=1, max_value=60)

    if 'model' in st.session_state:
        st.success("✅ Model loaded from session state.")
    else:
        st.warning("⚠️ No trained model available. Please train with Yahoo data first.")

    if st.button("Go Live"):
        # Fetch and predict the price
        predicted_price = predict_price(api_key, symbol)
        if predicted_price is not None:
            st.info(f"Predicted Signal: {'Bullish' if predicted_price == 1 else 'Bearish'}")
            st.text(predicted_price)

st.stop()  # Stop further execution if no valid model is found
