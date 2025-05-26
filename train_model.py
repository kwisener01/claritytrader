import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load training data
df = pd.read_csv("spy_training_data.csv")

# Define features and target
X = df[["RSI", "Momentum", "ATR", "Volume"]]
y = df["Label"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model to file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to model.pkl")
