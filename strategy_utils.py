import pandas as pd

def generate_signal(row):
    # Basic rule-based logic
    if row["RSI"] < 30 and row["Momentum"] > 0:
        return "Buy"
    elif row["RSI"] > 70 and row["Momentum"] < 0:
        return "Sell"
    else:
        return "Hold"

def run_backtest(df):
    signals = []
    for _, row in df.iterrows():
        signals.append(generate_signal(row))
    df["Signal"] = signals
    buys = df[df["Signal"] == "Buy"]
    sells = df[df["Signal"] == "Sell"]
    return {
        "Total Signals": len(df),
        "Buy Signals": len(buys),
        "Sell Signals": len(sells),
        "Hold Signals": len(df) - len(buys) - len(sells)
    }


from sklearn.ensemble import RandomForestClassifier

def train_model(df):
    X = df[["RSI", "Momentum", "ATR", "Volume"]]
    y = df["Label"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
