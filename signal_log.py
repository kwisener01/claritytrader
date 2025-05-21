
import csv
import os
from datetime import datetime

def log_signal(ticker, signal, confidence, price, timestamp):
    filename = "signal_log.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Ticker", "Signal", "Confidence", "Price"])
        writer.writerow([timestamp, ticker, signal, confidence, price])
