import streamlit as st
import pandas as pd
import pickle
from strategy_utils import generate_signal, run_backtest, train_model, bayesian_update_user
from live_data import fetch_latest_data

st.set_page_config(page_title="ClarityTrader Signal", layout="centered")
st.title("🧠 ClarityTrader – Emotion-Free Signal Generator")

# Upload data or load default training set
uploaded_file = st.file_uploader("📤 Upload CSV or use example data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("spy_training_data.csv")



st.write("### 📡 Live Signal (1-min Data Feed)")

col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Choose Ticker (ETF Proxy)", ["SPY", "QQQ", "DIA", "IWM"])
with col2:
    api_key = st.text_input("🔑 Twelve Data API Key", type="password")



# Confidence threshold control
threshold = st.slider("🎯 Confidence Threshold (%)", min_value=50, max_value=100, value=70, step=1)

st.write("### 📊 Preview Data", df.head())

# Bayesian option
apply_bayes = st.checkbox("Use Bayesian Forecasting", value=True)
if apply_bayes:
    bayesian_update_user()

# Train model
if st.button("🛠️ Train Model Now"):
    model = train_model(df, apply_bayesian=apply_bayes)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("✅ Model trained and saved as model.pkl")

# Try loading model if exists
try:
    model = pickle.load(open("model.pkl", "rb"))
    st.success("📦 Model loaded successfully.")
except:
    model = None
    st.warning("⚠️ No trained model found. Please train it first.")

# Predict signal from latest row
st.write("### 🧠 Generate Signal from Latest Row")
row = df.iloc[-1].drop("Label", errors="ignore").to_dict()

if model:
    input_df = pd.DataFrame([row])[["RSI", "Momentum", "ATR", "Volume"]]
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    confidence = round(100 * max(proba), 2)
    if confidence >= threshold:
        st.metric(label="Predicted Signal", value=pred)
        st.write(f"🧠 Confidence: **{confidence}%**")
    else:
        st.warning(f"No signal. Confidence too low ({confidence}%)")
else:
    pred = generate_signal(row)
    st.metric(label="Predicted Signal (Rule-Based)", value=pred)

## Live Signal from SPY using Twelve Data
#st.write("### 📡 Live Signal (SPY via Twelve Data)")
#api_key = st.text_input("🔑 Enter your Twelve Data API Key", type="password")

if st.button("🔍 Get Live Signal"):
    if model is None:
        st.error("⚠️ Train or load a model before using live signals.")
    elif not api_key:
        st.warning("Please enter your API key.")
    else:
        live_row = fetch_latest_data(symbol=ticker, api_key=api_key)
        if "error" in live_row:
            st.error(f"API Error: {live_row['error']}")
        else:
            live_input = pd.DataFrame([{
                "RSI": live_row["RSI"],
                "Momentum": live_row["Momentum"],
                "ATR": live_row["ATR"],
                "Volume": live_row["Volume"]
            }])
            pred = model.predict(live_input)[0]
            proba = model.predict_proba(live_input)[0]
            confidence = round(100 * max(proba), 2)

            st.markdown("---")
            st.markdown(f"🧠 **LIVE SIGNAL for {ticker}**  \n🕒 Timestamp: `{live_row['datetime']}`")

            if confidence >= threshold:
                st.metric(label="Signal", value=pred)
                st.metric(label="Live Price", value=f"${live_row['close']:.2f}")
                st.metric(label="Confidence", value=f"{confidence}%")
            else:
                st.warning(f"🧠 No signal. Confidence too low ({confidence}%)")

with st.expander("🧾 ClarityTrader Market Open Checklist", expanded=False):
    st.markdown("### 1. 🔧 Pre-Market Prep (8:30–9:25 AM ET)")
    st.markdown("- [ ] Open ClarityTrader in your browser")
    st.markdown("- [ ] Enter Twelve Data API key")
    st.markdown("- [ ] Upload new training data (optional)")
    st.markdown("- [ ] Retrain model with 🛠️ button (if needed)")
    st.markdown("- [ ] Set confidence threshold (e.g. 70–85%)")
    st.markdown("- [ ] Enable Bayesian Forecasting (optional)")

    st.markdown("### 2. ⏱ Market Open (9:30 AM ET)")
    st.markdown("- [ ] Wait for first 1-min candle (9:30)")
    st.markdown("- [ ] Click 🔍 Get Live Signal")
    st.markdown("- [ ] Optionally turn on auto-refresh")
    st.markdown("- [ ] Review:")
    st.markdown("   - ✅ Signal (Buy/Sell/Hold)\n   - ✅ Confidence %\n   - ✅ Price (rounded)\n   - ✅ Timestamp")

    st.markdown("### 3. 🧪 Execution Plan")
    st.markdown("- [ ] Trade only signals > threshold")
    st.markdown("- [ ] Log trades (manual or auto-logger)")
    st.markdown("- [ ] Skip low-confidence or unclear trades")
    st.markdown("- [ ] Stay emotion-free: follow logic")

    st.markdown("### 4. 📈 Post-Session (12:00 or 4:00 PM)")
    st.markdown("- [ ] Backtest with adjustable time window")
    st.markdown("- [ ] Review win rate + notes")
    st.markdown("- [ ] Save model if needed")

    st.markdown("### ✨ Daily Habits")
    st.markdown("- [ ] Count signals followed vs skipped")
    st.markdown("- [ ] Note emotional overrides")
    st.markdown("- [ ] Aim for consistency, not perfection")


st.write("### ⏱ Backtest Time Window")
start_idx = st.number_input("Start Row", min_value=0, max_value=len(df)-2, value=0)
end_idx = st.number_input("End Row", min_value=start_idx+1, max_value=len(df), value=len(df))

df_window = df.iloc[int(start_idx):int(end_idx)]

# Backtest summary
st.write("### 📈 Backtest Strategy Results")
backtest_results = run_backtest(df)
st.write(backtest_results)
