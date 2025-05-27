
import requests

def send_slack_alert(signal, price, confidence, ticker="SPY"):
    import requests
    url = "https://hooks.slack.com/services/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"
    text = f"{signal} signal for {ticker} at ${price:.2f} — Confidence: {confidence}%"
    payload = {
        "channel": "#signals",
        "text": text
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Slack alert failed:", e)
