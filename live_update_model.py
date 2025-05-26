import pandas as pd
import pickle
from strategy_utils import add_custom_features
from train_model import train_model
from live_data import fetch_latest_data

# Load previous training data and model
try:
    df = pd.read_csv("spy_training_data.csv")
except:
    df = pd.DataFrame()

new_row = fetch_latest_data("SPY", api_key="YOUR_API_KEY")
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
df = add_custom_features(df).dropna()

# Keep last 30,000 rows max
if len(df) > 30000:
    df = df[-30000:]

model = train_model(df)
pickle.dump(model, open("model.pkl", "wb"))
df.to_csv("spy_training_data.csv", index=False)
print("✅ Model updated and saved.")
