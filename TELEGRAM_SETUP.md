# Telegram Notifications Setup

## 1. Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts to name your bot
3. BotFather will give you a **bot token** (looks like `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)
4. **Keep this token secret** — it controls your bot

## 2. Get Your Chat ID

1. Open your new bot in Telegram and send `/start`
2. The bot will reply with your chat ID
3. Enter this chat ID in the Stock Briefer sidebar under **Telegram Notifications**

## 3. Configure Environment Variables

The notifier script needs three environment variables:

```bash
export TELEGRAM_BOT_TOKEN="your-bot-token-here"
export JSONBIN_BIN_ID="your-jsonbin-id"
export JSONBIN_KEY="your-jsonbin-key"
```

## 4. Running the Notifier

```bash
# Send summary to all linked users (scheduled daily)
python notifier.py --summary

# Check tracked stocks and send buy-price alerts
python notifier.py --alerts

# Check portfolio stocks and send RSI (10d) >= 90 alerts
python notifier.py --rsi-alerts

# Process pending Telegram commands once (used by GitHub Actions)
python notifier.py --poll

# Run the interactive bot locally or on Render (optional; do not run with --poll)
python notifier.py --bot
```

### Bot Commands (available in Telegram)

| Command | Description |
|---------|-------------|
| `/start` | Link your account (shows your chat ID) |
| `/summary` | Get your stock summary on demand |
| `/alerts` | Check if any tracked stocks hit your buy price |

## 5. Deployment (GitHub Actions)

Configured in `.github/workflows/stock-notifications.yml`:
- **Summary** at 9:30 AM Pacific, Mon–Fri
- **Price alerts** and **RSI alerts** every hour during market hours
- **Bot commands** (`/summary`, `/alerts`) polled every 10 minutes

Secrets needed in GitHub (Settings → Secrets → Actions):
- `TELEGRAM_BOT_TOKEN`
- `JSONBIN_BIN_ID`
- `JSONBIN_KEY`

### Important: workflow inactivity

GitHub **automatically disables** scheduled workflows after ~60 days with no repository activity
(no commits/PRs). If messages stop arriving:

1. Open https://github.com/aawang1999/stock-briefer/actions
2. Click **Stock Notifications**
3. If you see a banner saying the workflow is disabled, click **Enable workflow**
4. Click **Run workflow** → choose `summary` (or `all`) to verify

Making an occasional commit to the repo (or manually running the workflow) prevents this.

## Price Alert Behavior

- When a tracked stock reaches or falls below your buy price, an alert is sent on the next hourly check
- After sending an alert for a stock, no repeat alert is sent until **24 hours** have passed
- You can always check manually with `/alerts` in Telegram regardless of cooldown

## RSI Alert Behavior

- When a portfolio stock's RSI (10d) meets or exceeds **90**, an alert is sent on the next hourly check
- The message includes the ticker, that day's RSI (10d), and how many consecutive trading days it has stayed at or above 90
- After sending an RSI alert for a stock, no repeat RSI alert is sent until **24 hours** have passed
- RSI alerts have their own cooldown, separate from price alerts

## Security Notes

- The bot token is stored only in environment variables / platform secrets
- User chat IDs are stored in JSONBin (same as other app data) and entered via a password field
- The bot only responds to users whose chat ID is linked in the app
- No personal Telegram handles or phone numbers are collected or stored
