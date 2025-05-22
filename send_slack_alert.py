
import requests

def send_slack_alert(webhook_url, ticker, signal, confidence, price, timestamp):
    message = {
        "text": f":rotating_light: *ClarityTrader Alert*\n"
                f"• *Ticker:* `{ticker}`\n"
                f"• *Signal:* *{signal.upper()}*\n"
                f"• *Confidence:* `{confidence}%`\n"
                f"• *Price:* `${price}`\n"
                f"• *Time:* `{timestamp}`"
    }

    try:
        response = requests.post(webhook_url, json=message)
        if response.status_code != 200:
            raise ValueError(f"Request to Slack returned error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Slack alert failed: {e}")
