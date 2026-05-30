"""
Stock Briefer Telegram Notifier

Sends two types of notifications:
1. Daily morning summary at 9:30 AM Pacific (run via scheduler)
2. Price alerts when tracked stocks hit buy price (run on a frequent interval)

Environment variables required:
  TELEGRAM_BOT_TOKEN  - Bot token from @BotFather
  JSONBIN_BIN_ID      - JSONBin bin ID (same as Streamlit app)
  JSONBIN_KEY         - JSONBin API key (same as Streamlit app)

Usage:
  python notifier.py --morning     Send daily morning summary to all linked users
  python notifier.py --alerts      Check tracked stocks and send buy-price alerts
"""

import os
import sys
import argparse
import requests
import yfinance as yf
from datetime import datetime, timedelta

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
JSONBIN_BIN_ID = os.environ["JSONBIN_BIN_ID"]
JSONBIN_KEY = os.environ["JSONBIN_KEY"]

INDICES = {'^DJI': 'Dow Jones', '^GSPC': 'S&P 500', '^IXIC': 'NASDAQ'}


def send_telegram_message(chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    resp = requests.post(url, json=payload, timeout=10)
    if resp.status_code != 200:
        print(f"  [WARN] Failed to send to {chat_id}: {resp.text}")
    return resp.status_code == 200


def load_users() -> dict:
    url = f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}/latest"
    headers = {"X-Access-Key": JSONBIN_KEY}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()["record"]


def save_users(data: dict):
    url = f"https://api.jsonbin.io/v3/b/{JSONBIN_BIN_ID}"
    headers = {"Content-Type": "application/json", "X-Access-Key": JSONBIN_KEY}
    resp = requests.put(url, json=data, headers=headers, timeout=10)
    resp.raise_for_status()


def get_fear_and_greed() -> tuple:
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        score = int(data["fear_and_greed_historical"]["data"][-1]["y"])
        if score <= 24:
            rating = "Extreme Fear"
        elif score <= 44:
            rating = "Fear"
        elif score <= 54:
            rating = "Neutral"
        elif score <= 74:
            rating = "Greed"
        else:
            rating = "Extreme Greed"
        return score, rating
    except Exception:
        return None, None


def get_index_data() -> list:
    tickers_str = " ".join(INDICES.keys())
    try:
        df = yf.download(tickers_str, period="5d", progress=False)
    except Exception:
        return []

    results = []
    closes = df["Close"] if "Close" in df.columns else df
    for ticker, name in INDICES.items():
        try:
            series = closes[ticker] if ticker in closes.columns else closes
            current = series.iloc[-1]
            prev = series.iloc[-2] if len(series) >= 2 else current
            pct = ((current - prev) / prev) * 100
            results.append({"name": name, "price": current, "pct": pct})
        except Exception:
            continue
    return results


def get_portfolio_movers(tickers: list) -> tuple:
    """Returns (top_gainers, top_losers) as lists of (ticker, pct_change)."""
    if not tickers:
        return [], []

    tickers_str = " ".join(tickers)
    try:
        df = yf.download(tickers_str, period="5d", progress=False)
    except Exception:
        return [], []

    changes = []
    if isinstance(df.columns, (list,)) or (hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1):
        closes = df["Close"]
    else:
        closes = df

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                series = closes if isinstance(closes, type(df["Close"])) else closes
                if hasattr(series, "columns"):
                    series = series[series.columns[0]]
            else:
                series = closes[ticker]

            current = series.iloc[-1]
            prev = series.iloc[-2] if len(series) >= 2 else current
            pct = ((current - prev) / prev) * 100
            changes.append((ticker, float(pct)))
        except Exception:
            continue

    changes.sort(key=lambda x: x[1], reverse=True)
    top_gainers = changes[:3]
    top_losers = changes[-3:][::-1] if len(changes) >= 3 else list(reversed(changes[-len(changes):]))
    # Reverse losers so worst is first
    top_losers = sorted(top_losers, key=lambda x: x[1])
    return top_gainers, top_losers


def get_upcoming_earnings(tickers: list) -> list:
    """Returns list of (ticker, earnings_date) for stocks with earnings within 7 days."""
    upcoming = []
    today = datetime.now().date()
    cutoff = today + timedelta(days=7)

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            cal = stock.calendar
            if cal and "Earnings Date" in cal:
                earn_dates = cal["Earnings Date"]
                if earn_dates:
                    earn_date = earn_dates[0]
                    if hasattr(earn_date, "date"):
                        earn_date = earn_date.date()
                    if today <= earn_date <= cutoff:
                        upcoming.append((ticker, earn_date))
        except Exception:
            continue
    return upcoming


def get_tracked_stock_prices(tracked_stocks: dict) -> list:
    """Returns list of (ticker, current_price, buy_price) where current <= buy."""
    alerts = []
    for ticker, buy_price in tracked_stocks.items():
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
            if current is not None and float(current) <= float(buy_price):
                alerts.append((ticker, float(current), float(buy_price)))
        except Exception:
            continue
    return alerts


def build_morning_message(user: str, user_data: dict) -> str:
    lines = [f"<b>☀️ Good Morning, {user}!</b>", ""]

    # 1. Market Indices
    indices = get_index_data()
    if indices:
        lines.append("<b>📊 Market Indices</b>")
        for idx in indices:
            arrow = "🟢" if idx["pct"] >= 0 else "🔴"
            lines.append(f"  {arrow} {idx['name']}: {idx['price']:,.2f} ({idx['pct']:+.2f}%)")
        lines.append("")

    # 2. Fear & Greed
    score, rating = get_fear_and_greed()
    if score is not None:
        lines.append(f"<b>😨 Fear & Greed:</b> {score} ({rating})")
        lines.append("")

    # 3. Portfolio movers
    stocks_dict = user_data.get("stocks", {})
    portfolio_tickers = list(stocks_dict.keys())
    if portfolio_tickers:
        gainers, losers = get_portfolio_movers(portfolio_tickers)
        if gainers:
            lines.append("<b>📈 Top Gainers</b>")
            for t, pct in gainers:
                lines.append(f"  🟢 {t}: {pct:+.2f}%")
            lines.append("")
        if losers:
            lines.append("<b>📉 Top Losers</b>")
            for t, pct in losers:
                lines.append(f"  🔴 {t}: {pct:+.2f}%")
            lines.append("")

        # 4. Upcoming earnings
        earnings = get_upcoming_earnings(portfolio_tickers)
        if earnings:
            lines.append("<b>📅 Earnings This Week</b>")
            for t, d in earnings:
                days_away = (d - datetime.now().date()).days
                day_str = "Today" if days_away == 0 else f"in {days_away}d"
                lines.append(f"  {t} — {d.strftime('%b %d')} ({day_str})")
            lines.append("")

    # 5. Tracked stocks at or below buy price
    tracked_stocks = user_data.get("tracked_stocks", {})
    if tracked_stocks:
        alerts = get_tracked_stock_prices(tracked_stocks)
        if alerts:
            lines.append("<b>🎯 Buy Price Alerts</b>")
            for t, current, buy in alerts:
                lines.append(f"  ⚡ {t}: ${current:.2f} (buy target: ${buy:.2f})")
            lines.append("")

    if lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def run_morning_summary():
    print("Running morning summary...")
    users_data = load_users()

    for user, user_data in users_data.items():
        if isinstance(user_data, list):
            continue
        chat_id = user_data.get("telegram_chat_id", "")
        if not chat_id:
            print(f"  Skipping {user} (no Telegram linked)")
            continue

        print(f"  Sending to {user}...")
        message = build_morning_message(user, user_data)
        send_telegram_message(chat_id, message)

    print("Done.")


def run_price_alerts():
    print("Checking price alerts...")
    users_data = load_users()
    changed = False

    for user, user_data in users_data.items():
        if isinstance(user_data, list):
            continue
        chat_id = user_data.get("telegram_chat_id", "")
        if not chat_id:
            continue

        tracked_stocks = user_data.get("tracked_stocks", {})
        if not tracked_stocks:
            continue

        already_alerted = user_data.get("_alerted_stocks", {})
        alerts = get_tracked_stock_prices(tracked_stocks)

        for ticker, current, buy in alerts:
            last_alert_date = already_alerted.get(ticker)
            today_str = datetime.now().strftime("%Y-%m-%d")

            if last_alert_date == today_str:
                continue

            message = (
                f"<b>🎯 Price Alert: {ticker}</b>\n\n"
                f"Current price: <b>${current:.2f}</b>\n"
                f"Your buy price: ${buy:.2f}\n\n"
                f"<i>{ticker} has reached or fallen below your target.</i>"
            )
            if send_telegram_message(chat_id, message):
                print(f"  Alert sent to {user} for {ticker} (${current:.2f} <= ${buy:.2f})")
                already_alerted[ticker] = today_str
                changed = True

        # Clean up alerts for stocks no longer tracked or above buy price
        alert_tickers_today = {t for t, _, _ in alerts}
        for t in list(already_alerted.keys()):
            if t not in tracked_stocks:
                del already_alerted[t]
                changed = True

        if already_alerted:
            users_data[user]["_alerted_stocks"] = already_alerted
        elif "_alerted_stocks" in users_data[user]:
            del users_data[user]["_alerted_stocks"]
            changed = True

    if changed:
        save_users(users_data)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Briefer Telegram Notifier")
    parser.add_argument("--morning", action="store_true", help="Send daily morning summary")
    parser.add_argument("--alerts", action="store_true", help="Check and send buy-price alerts")
    args = parser.parse_args()

    if not args.morning and not args.alerts:
        print("Usage: python notifier.py --morning | --alerts")
        sys.exit(1)

    if args.morning:
        run_morning_summary()
    if args.alerts:
        run_price_alerts()
