import streamlit as st

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

# Ensure model is not None after training
if 'model' in st.session_state:
    st.success("✅ Model loaded from session state.")
else:
    st.warning("⚠️ No trained model available. Please train with Yahoo data first.")

st.stop()  # Stop further execution if no valid model is found
