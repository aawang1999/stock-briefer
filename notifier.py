"""
Stock Briefer Telegram Notifier

Modes:
  python notifier.py --summary     Send summary to all linked users
  python notifier.py --alerts      Check tracked stocks and send buy-price alerts
  python notifier.py --bot         Run interactive Telegram bot (responds to /summary, /alerts)

Environment variables required:
  TELEGRAM_BOT_TOKEN  - Bot token from @BotFather
  JSONBIN_BIN_ID      - JSONBin bin ID (same as Streamlit app)
  JSONBIN_KEY         - JSONBin API key (same as Streamlit app)
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
    top_losers = sorted(changes[-3:], key=lambda x: x[1])
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


def build_summary_message(user: str, user_data: dict) -> str:
    lines = [f"<b>📋 Stock Summary for {user}</b>", ""]

    indices = get_index_data()
    if indices:
        lines.append("<b>📊 Market Indices</b>")
        for idx in indices:
            arrow = "🟢" if idx["pct"] >= 0 else "🔴"
            lines.append(f"  {arrow} {idx['name']}: {idx['price']:,.2f} ({idx['pct']:+.2f}%)")
        lines.append("")

    score, rating = get_fear_and_greed()
    if score is not None:
        lines.append(f"<b>😨 Fear & Greed:</b> {score} ({rating})")
        lines.append("")

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

        earnings = get_upcoming_earnings(portfolio_tickers)
        if earnings:
            lines.append("<b>📅 Earnings This Week</b>")
            for t, d in earnings:
                days_away = (d - datetime.now().date()).days
                day_str = "Today" if days_away == 0 else f"in {days_away}d"
                lines.append(f"  {t} — {d.strftime('%b %d')} ({day_str})")
            lines.append("")

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


def _alert_is_within_cooldown(already_alerted: dict, ticker: str) -> bool:
    """Returns True if an alert was sent for this ticker within the last 24 hours."""
    last_alert_iso = already_alerted.get(ticker)
    if not last_alert_iso:
        return False
    try:
        last_alert_time = datetime.fromisoformat(last_alert_iso)
        return (datetime.now() - last_alert_time) < timedelta(hours=24)
    except (ValueError, TypeError):
        return False


def run_summary():
    print("Running summary...")
    users_data = load_users()

    for user, user_data in users_data.items():
        if isinstance(user_data, list):
            continue
        chat_id = user_data.get("telegram_chat_id", "")
        if not chat_id:
            print(f"  Skipping {user} (no Telegram linked)")
            continue

        print(f"  Sending to {user}...")
        message = build_summary_message(user, user_data)
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
            if _alert_is_within_cooldown(already_alerted, ticker):
                continue

            message = (
                f"<b>🎯 Price Alert: {ticker}</b>\n\n"
                f"Current price: <b>${current:.2f}</b>\n"
                f"Your buy price: ${buy:.2f}\n\n"
                f"<i>{ticker} has reached or fallen below your target.</i>"
            )
            if send_telegram_message(chat_id, message):
                print(f"  Alert sent to {user} for {ticker} (${current:.2f} <= ${buy:.2f})")
                already_alerted[ticker] = datetime.now().isoformat()
                changed = True

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


# --- Interactive Telegram Bot ---

def _find_user_by_chat_id(users_data: dict, chat_id: str):
    """Find the username associated with a chat_id."""
    for user, user_data in users_data.items():
        if isinstance(user_data, list):
            continue
        if str(user_data.get("telegram_chat_id", "")) == str(chat_id):
            return user, user_data
    return None, None


def run_bot():
    """Run the Telegram bot with long polling, responding to /summary and /alerts."""
    from telegram import Update, BotCommand
    from telegram.ext import Application, CommandHandler, ContextTypes

    async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        users_data = load_users()
        user, _ = _find_user_by_chat_id(users_data, str(chat_id))
        if user:
            await update.message.reply_text(
                f"✅ Connected as {user}.\n\n"
                "Commands:\n"
                "/summary — Get your stock summary\n"
                "/alerts — Check for buy-price alerts"
            )
        else:
            await update.message.reply_text(
                f"Your chat ID is: <code>{chat_id}</code>\n\n"
                "Enter this in the Stock Briefer app sidebar under "
                "Telegram Notifications to link your account.",
                parse_mode="HTML"
            )

    async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = str(update.effective_chat.id)
        users_data = load_users()
        user, user_data = _find_user_by_chat_id(users_data, chat_id)
        if not user:
            await update.message.reply_text("⚠️ Account not linked. Send /start for instructions.")
            return

        await update.message.reply_text("⏳ Fetching your summary...")
        message = build_summary_message(user, user_data)
        await update.message.reply_text(message, parse_mode="HTML")

    async def cmd_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = str(update.effective_chat.id)
        users_data = load_users()
        user, user_data = _find_user_by_chat_id(users_data, chat_id)
        if not user:
            await update.message.reply_text("⚠️ Account not linked. Send /start for instructions.")
            return

        tracked_stocks = user_data.get("tracked_stocks", {})
        if not tracked_stocks:
            await update.message.reply_text("You have no tracked stocks.")
            return

        await update.message.reply_text("⏳ Checking prices...")
        alerts = get_tracked_stock_prices(tracked_stocks)
        if not alerts:
            await update.message.reply_text("✅ No tracked stocks are at or below your buy price.")
        else:
            lines = ["<b>🎯 Buy Price Alerts</b>", ""]
            for t, current, buy in alerts:
                lines.append(f"⚡ <b>{t}</b>: ${current:.2f} (buy target: ${buy:.2f})")
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def post_init(application: Application):
        await application.bot.set_my_commands([
            BotCommand("summary", "Get your stock summary"),
            BotCommand("alerts", "Check buy-price alerts"),
            BotCommand("start", "Link your account"),
        ])

    print("Starting Stock Briefer bot (polling)...")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CommandHandler("alerts", cmd_alerts))
    app.run_polling(allowed_updates=["message"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Briefer Telegram Notifier")
    parser.add_argument("--summary", action="store_true", help="Send summary to all linked users")
    parser.add_argument("--alerts", action="store_true", help="Check and send buy-price alerts")
    parser.add_argument("--bot", action="store_true", help="Run interactive Telegram bot")
    args = parser.parse_args()

    if not args.summary and not args.alerts and not args.bot:
        print("Usage: python notifier.py --summary | --alerts | --bot")
        sys.exit(1)

    if args.bot:
        run_bot()
    if args.summary:
        run_summary()
    if args.alerts:
        run_price_alerts()
