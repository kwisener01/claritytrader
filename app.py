import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import requests

# Function to fetch historical data from Yahoo Finance
def fetch_yahoo_intraday(symbol, period):
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1=0&period2={int(time.time())}&interval=1m&events=history"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    df = pd.read_csv(pd.compat.StringIO(response.text))
    return df

# Function to calculate features
def add_custom_features(df):
    df["Momentum"] = df["Close"].diff(5)
    df["TR"] = df[["High", "Low"]].max(axis=1) - df[["High", "Low"]].min(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (
        df["Close"].diff().where(lambda x: x > 0, 0).rolling(14).mean() /
        -df["Close"].diff().where(lambda x: x < 0, 0).rolling(14).mean()
    )))
    df["Label"] = np.where(df["Close"].shift(-5) > df["Close"], "Buy", "Sell")
    df["Volume"] = df["Volume"].fillna(1000000)
    df = df.dropna()
    return df

# Function to train the model
def train_model(df):
    X = df[["RSI", "Momentum", "ATR", "Volume"]]
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    return model

# Function to fetch live data from Twelve Data
def fetch_twelve_data(symbol, api_key):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1m&outputsize=20&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data["values"])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df

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

        if 'model' in st.session_state and st.session_state.model is not None:
            X_live = df[["RSI", "Momentum", "ATR", "Volume"]]
            pred = st.session_state.model.predict(X_live)
            st.write(f"### Live Signal")
            st.text(pred[-1])
        else:
            st.warning("⚠️ No trained model available. Please train with Yahoo data first.")

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
            st.session_state.training_data = df

            if len(df) < 30:
                raise ValueError("Not enough samples to train a model. Please select a longer period.")

            model = train_model(df)
            st.session_state.model = model
            pickle.dump(model, open("model.pkl", "wb"))
            st.success("✅ Model trained and saved to model.pkl.")

            st.write("### 📊 Label Distribution")
            st.bar_chart(df["Label"].value_counts())

            X = df[["RSI", "Momentum", "ATR", "Volume"]]
            y = df["Label"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            y_pred = model.predict(X_test)

            st.write("### 🧪 Backtest")
            st.json(run_backtest(df))

            st.write("### 📊 Classification Report")
            from sklearn.metrics import classification_report
            st.text(classification_report(y_test, y_pred))

            st.write("### 📊 Confusion Matrix")
            from sklearn.metrics import confusion_matrix, plot_confusion_matrix
            conf_matrix = confusion_matrix(y_test, y_pred, labels=["Buy", "Sell"])
            fig, ax = plt.subplots()
            im = ax.imshow(conf_matrix, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Buy", "Sell"])
            ax.set_yticklabels(["Buy", "Sell"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="white")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ Training failed: {e}")

# Ensure model is not None after training
if 'model' in st.session_state and st.session_state.model is not None:
    st.success("✅ Model available.")
else:
    st.warning("⚠️ No trained model available. Please train with Yahoo data first.")

st.stop()  # Stop further execution if no valid model is found
