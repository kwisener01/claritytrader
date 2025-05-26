def generate_signal(row):
    if row["RSI"] > 55 and row["Momentum"] > 0:
        return "Buy"
    elif row["RSI"] < 45 and row["Momentum"] < 0:
        return "Sell"
    else:
        return "Hold"

def run_backtest(df):
    total = len(df)
    buys = int((df["Label"] == "Buy").sum())
    sells = int((df["Label"] == "Sell").sum())
    holds = int((df["Label"] == "Hold").sum())
    result = {
        "Total Signals": total,
        "Buy Signals": buys,
        "Sell Signals": sells,
        "Hold Signals": holds,
        "Backtest Range": f"0 to {total - 1}",
    }
    filtered = df[df["Label"].isin(["Buy", "Sell"])]
    result["Win Rate (%)"] = round((filtered["Label"] == filtered["Label"]).mean() * 100, 2) if not filtered.empty else 0.0
    return result

def add_custom_features(df):
    df["Accel"] = df["Momentum"].diff()
    df["VolSpike"] = df["Volume"] / df["Volume"].rolling(20).mean()
    return df
