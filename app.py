import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import yfinance as yf

# Function to fetch data from Yahoo Finance
def fetch_yahoo_data(symbol):
    df = yf.download(symbol, period="1d")
    return df

# Function to add custom features
def add_custom_features(df):
    # Ensure 'Close', 'High', and 'Low' columns are numeric
    df['Close'] = pd.to_numeric(df['Close'])
    df['High'] = pd.to_numeric(df['High'])
    df['Low'] = pd.to_numeric(df['Low'])

    df["Momentum"] = df["Close"].diff().rolling(5).mean()
    df["ATR"] = (df["High"] - df["Low"]).abs() + (df["Close"].diff().abs())
    
    # Calculate RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    if loss.sum() == 0:
        rs = pd.Series(0, index=df.index)  # Handle division by zero
    else:
        rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    return df

# Function to train the model
def train_model(df):
    try:
        X = df.drop(['Close', 'Open', 'High', 'Low', 'Volume'], axis=1).dropna()
        y = (df['Close'].diff() > 0).astype(int)  # Bullish if price increases, Bearish otherwise
        
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
def predict_price(model, df):
    try:
        # Get the latest data point
        X_new = df.iloc[-1:].drop(['Close', 'Open', 'High', 'Low', 'Volume'], axis=1).values.reshape(1, -1)  # Ensure correct shape for prediction
        
        # Make a prediction
        predicted_price = model.predict(X_new)[0]
        return int(predicted_price)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Define tabs
tab1, tab2 = st.tabs(["Train Model with Yahoo Data", "Predict Signal from Current Data"])

with tab1:
    st.header("🤖 Train Model with Yahoo Data")
    symbol = st.text_input("Yahoo Finance Symbol", value="SPY")
    days_to_train = st.radio("Days to Train Model", ["5", "7"])
    if st.button("Train Model"):
        df = fetch_yahoo_data(symbol)
        df = add_custom_features(df)
        
        model, X_test, y_test = train_model(df)
        if model is not None:
            st.success("Model trained successfully!")
            st.session_state["model"] = model
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test

with tab2:
    st.header("🤖 Predict Signal from Current Data")
    symbol = st.text_input("Yahoo Finance Symbol", value="SPY", key="unique_symbol")  # Add unique key here
    
    if "model" not in st.session_state:
        st.warning("⚠️ No trained model available. Please train with Yahoo data first.")
    else:
        if st.button("Go Live"):
            df = fetch_yahoo_data(symbol)
            df = add_custom_features(df)
            
            predicted_price = predict_price(st.session_state["model"], df)
            if predicted_price is not None:
                st.info(f"Predicted Signal: {'Bullish' if predicted_price == 1 else 'Bearish'}")
                st.text(predicted_price)

st.stop()  # Stop further execution if no valid model is found
