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

# Run the interactive bot (responds to /summary and /alerts in Telegram)
python notifier.py --bot
```

### Bot Commands (available in Telegram)

| Command | Description |
|---------|-------------|
| `/start` | Link your account (shows your chat ID) |
| `/summary` | Get your stock summary on demand |
| `/alerts` | Check if any tracked stocks hit your buy price |

## 5. Deployment

The system has two parts:

### A. Scheduled jobs (GitHub Actions)

Already configured in `.github/workflows/notify.yml`:
- **Summary** at 9:30 AM Pacific, Mon–Fri
- **Price alerts** every 15 min during market hours

Secrets needed in GitHub (Settings → Secrets → Actions):
- `TELEGRAM_BOT_TOKEN`
- `JSONBIN_BIN_ID`
- `JSONBIN_KEY`

### B. Interactive bot (Render — free tier)

The bot needs to run continuously to respond to commands in real-time.

1. Go to [render.com](https://render.com) and sign up (free)
2. Click **New** → **Background Worker**
3. Connect your GitHub repo (`aawang1999/stock-briefer`)
4. Render will auto-detect `render.yaml` — confirm the settings
5. Add environment variables: `TELEGRAM_BOT_TOKEN`, `JSONBIN_BIN_ID`, `JSONBIN_KEY`
6. Deploy

Alternatively, run it locally:
```bash
export TELEGRAM_BOT_TOKEN="..."
export JSONBIN_BIN_ID="..."
export JSONBIN_KEY="..."
python notifier.py --bot
```

## Price Alert Behavior

- When a tracked stock reaches or falls below your buy price, an alert is sent immediately
- After sending an alert for a stock, no repeat alert is sent until **24 hours** have passed
- You can always check manually with `/alerts` in Telegram regardless of cooldown

## Security Notes

- The bot token is stored only in environment variables / platform secrets
- User chat IDs are stored in JSONBin (same as other app data) and entered via a password field
- The bot only responds to users whose chat ID is linked in the app
- No personal Telegram handles or phone numbers are collected or stored
