# Telegram Notifications Setup

## 1. Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts to name your bot
3. BotFather will give you a **bot token** (looks like `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)
4. **Keep this token secret** — it controls your bot

## 2. Get Your Chat ID

1. Open your new bot in Telegram and send `/start`
2. Visit `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates` in a browser
3. Find `"chat":{"id":123456789}` in the response — that number is your chat ID
4. Enter this chat ID in the Stock Briefer sidebar under **Telegram Notifications**

## 3. Configure Environment Variables

The notifier script needs three environment variables:

```bash
export TELEGRAM_BOT_TOKEN="your-bot-token-here"
export JSONBIN_BIN_ID="your-jsonbin-id"      # same as in .streamlit/secrets.toml
export JSONBIN_KEY="your-jsonbin-key"          # same as in .streamlit/secrets.toml
```

## 4. Run the Notifier

```bash
# Send the daily morning summary
python notifier.py --morning

# Check tracked stocks and send buy-price alerts
python notifier.py --alerts
```

## 5. Schedule It (GitHub Actions Example)

Create `.github/workflows/notify.yml`:

```yaml
name: Stock Notifications

on:
  schedule:
    # Morning summary at 9:30 AM Pacific (16:30 UTC)
    - cron: '30 16 * * 1-5'
    # Price alerts every 15 minutes during market hours (6:30 AM - 1:00 PM Pacific = 13:30-20:00 UTC)
    - cron: '*/15 13-20 * * 1-5'

  workflow_dispatch:  # manual trigger

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - run: pip install requests yfinance

      - name: Morning summary
        if: github.event.schedule == '30 16 * * 1-5' || github.event_name == 'workflow_dispatch'
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          JSONBIN_BIN_ID: ${{ secrets.JSONBIN_BIN_ID }}
          JSONBIN_KEY: ${{ secrets.JSONBIN_KEY }}
        run: python notifier.py --morning

      - name: Price alerts
        if: github.event.schedule == '*/15 13-20 * * 1-5'
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          JSONBIN_BIN_ID: ${{ secrets.JSONBIN_BIN_ID }}
          JSONBIN_KEY: ${{ secrets.JSONBIN_KEY }}
        run: python notifier.py --alerts
```

Then add `TELEGRAM_BOT_TOKEN`, `JSONBIN_BIN_ID`, and `JSONBIN_KEY` as
repository secrets in GitHub (Settings → Secrets and variables → Actions).

## 5b. Schedule It (cron on a server)

```crontab
# Morning summary at 9:30 AM Pacific (adjust for your server's timezone)
30 9 * * 1-5 cd /path/to/stock-briefer && python notifier.py --morning

# Price alerts every 15 min during market hours
*/15 6-13 * * 1-5 cd /path/to/stock-briefer && python notifier.py --alerts
```

## Security Notes

- The bot token is stored only in environment variables / GitHub secrets
- User chat IDs are stored in JSONBin (same as other app data) and displayed
  as a password field in the Streamlit UI
- The bot can only send messages — it does not read your Telegram messages
- No personal Telegram handles or phone numbers are collected or stored
