import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
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
    # Ensure 'close', 'high', and 'low' columns are numeric
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])

    df["Momentum"] = df["close"].diff().rolling(5).mean()
    df["ATR"] = (df["high"] - df["low"]).abs() + (df["close"].diff().abs())
    
    # Calculate RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    if loss.sum() == 0:
        rs = pd.Series(0, index=df.index)  # Handle division by zero
    else:
        rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    return df

# Function to train the model
def train_model(api_key, symbol):
    try:
        df = fetch_twelve_data(symbol, api_key)
        df = add_custom_features(df)
        
        # Prepare data for training
        X = df.drop(['close', 'symbol'], axis=1).dropna()
        y = (df['close'].diff() > 0).astype(int)  # Bullish if price increases, Bearish otherwise
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    except Exception as e:
        st.error(f"An error occurred during training: {e}")
        return None, None, None

# Function to predict the stock price
def predict_price(model, api_key, symbol):
    try:
        df = fetch_twelve_data(symbol, api_key)
        df = add_custom_features(df)
        
        # Get the latest data point
        X_new = df.iloc[-1:].drop(['close', 'symbol'], axis=1).values.reshape(1, -1)  # Ensure correct shape for prediction
        
        # Make a prediction
        predicted_price = model.predict(X_new)[0]
        return int(predicted_price)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Load pre-trained model and test data (if available)
model, X_test, y_test = None, None, None
if "model" not in st.session_state:
    st.warning("⚠️ No trained model available. Please train with Yahoo data first.")
else:
    model, X_test, y_test = st.session_state["model"], st.session_state["X_test"], st.session_state["y_test"]

# Define tabs
tab1, tab2 = st.tabs(["Train Model with Yahoo Data", "Predict Signal from Twelve Data"])

with tab1:
    st.header("🤖 Train Model with Yahoo Data")
    symbol = st.text_input("Yahoo Finance Symbol", value="SPY")
    days_to_train = st.radio("Days to Train Model", ["5", "7"])
    if st.button("Train Model"):
        model, X_test, y_test = train_model("", symbol)
        if model is not None:
            st.success("Model trained successfully!")
            st.session_state["model"] = model
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test

with tab2:
    st.header("🤖 Predict Signal from Twelve Data")
    api_key = st.text_input("🔑 Twelve Data API Key", type="password", key="unique_api_key")  # Add unique key here
    symbol = st.text_input("Yahoo Finance Symbol", value="SPY", key="unique_symbol")  # Add unique key here
    
    if model is None:
        st.warning("⚠️ No trained model available. Please train with Yahoo data first.")
    else:
        if st.button("Go Live"):
            predicted_price = predict_price(model, api_key, symbol)
            if predicted_price is not None:
                st.info(f"Predicted Signal: {'Bullish' if predicted_price == 1 else 'Bearish'}")
                st.text(predicted_price)

st.stop()  # Stop further execution if no valid model is found
