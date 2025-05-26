def generate_signal(row):
    if row["RSI"] > 55 and row["Momentum"] > 0:
        return "Buy"
    elif row["RSI"] < 45 and row["Momentum"] < 0:
        return "Sell"
    else:
        return "Hold"

def run_backtest(df):
    result = {
        "Total Signals": len(df),
        "Buy Signals": (df["Label"] == "Buy").sum(),
        "Sell Signals": (df["Label"] == "Sell").sum(),
        "Hold Signals": (df["Label"] == "Hold").sum(),
        "Backtest Range": f"0 to {len(df)-1}",
    }
    filtered = df[df["Label"].isin(["Buy", "Sell"])]
    if not filtered.empty:
        result["Win Rate (%)"] = round((filtered["Label"] == filtered["Label"]).mean() * 100, 2)  # placeholder logic
    else:
        result["Win Rate (%)"] = 0.0
    return result

def add_custom_features(df):
    df["Accel"] = df["Momentum"].diff()
    df["VolSpike"] = df["Volume"] / df["Volume"].rolling(20).mean()
    return df
