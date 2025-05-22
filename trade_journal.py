
import csv
import os

def log_journal(timestamp, ticker, signal, reason, emotion, reflection, filename=None):
    filename = "trade_journal.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Ticker", "Signal", "Reason", "Emotion", "Reflection", "Attachment"])
        writer.writerow([timestamp, ticker, signal, reason, emotion, reflection, filename])
