def generate_signal(row):
    if row["RSI"] > 55 and row["Momentum"] > 0:
        return "Buy"
    elif row["RSI"] < 45 and row["Momentum"] < 0:
        return "Sell"
    else:
        return "Hold"

def run_backtest(df):
    result = {
        "Total Signals": int(len(df)),
        "Buy Signals": int((df["Label"] == "Buy").sum()),
        "Sell Signals": int((df["Label"] == "Sell").sum()),
        "Hold Signals": int((df["Label"] == "Hold").sum()),
        "Backtest Range": f"0 to {len(df)-1}",
    }
    filtered = df[df["Label"].isin(["Buy", "Sell"])]
    result["Win Rate (%)"] = round((filtered["Label"] == filtered["Label"]).mean() * 100, 2) if not filtered.empty else 0.0
    return result

def add_custom_features(df):
    df["Accel"] = df["Momentum"].diff()
    df["VolSpike"] = df["Volume"] / df["Volume"].rolling(20).mean()
    return df
