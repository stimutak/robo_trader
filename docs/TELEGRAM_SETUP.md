# Telegram Bot Setup Guide

Receive real-time trading alerts on your phone via Telegram.

## Features

📊 **Trading Alerts**
- Trade executions with AI confidence
- Daily P&L summaries
- Position updates

🔥 **Market Events**
- High-impact news
- Options flow signals
- SEC filings & earnings

⚠️ **System Monitoring**
- Error notifications
- System start/stop alerts
- Critical warnings

## Quick Setup (10 minutes)

### Step 1: Create Your Bot

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Choose a name (e.g., "My Trading Bot")
4. Choose a username (e.g., `my_trading_bot`)
5. Save the token you receive (looks like: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Step 2: Get Your Chat ID

1. Message your new bot (search for its username)
2. Send any message like "Hello"
3. Visit this URL in your browser (replace TOKEN with your bot token):
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
4. Find your chat ID in the response:
   ```json
   {
     "message": {
       "chat": {
         "id": 123456789  // This is your chat ID
       }
     }
   }
   ```

### Step 3: Configure Environment

Add to your `.env` file:
```bash
# Telegram Notifications
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### Step 4: Test It

```python
# Quick test script
from robo_trader.telegram_bot import notifier

# Send test message
notifier.send_message("🎉 Telegram notifications are working!")
```

Or test from command line:
```bash
python -c "from robo_trader.telegram_bot import notifier; notifier.send_startup_message()"
```

## Integration with Trading System

The Telegram bot automatically sends alerts for:

### Trade Executions
```python
# Automatically sent when trades execute
🟢 Trade Executed
Symbol: AAPL
Action: BUY
Quantity: 100
Price: $175.50
AI Confidence: 85.2%
```

### Daily Summaries
```python
# Sent at market close
📈 Daily Trading Summary
P&L: $1,234.56
Trades: 15
Win Rate: 73.3%
Best Trade: $456.78
```

### Market Alerts
```python
# High-impact events
🔥 Market Event
Symbol: NVDA
Event: Earnings Beat
Impact: 95/100
```

## Advanced Configuration

### Selective Notifications

Control which alerts you receive by modifying `ai_runner.py`:

```python
# Only send alerts for high-confidence trades
if trade['confidence'] > 80:
    send_trade_notification(trade)

# Only send critical errors
send_error_notification(error, critical=True)
```

### Custom Alert Formats

Modify `telegram_bot.py` to customize message formats:

```python
def send_trade_alert(self, trade):
    # Add custom fields, emojis, or formatting
    message = f"Your custom format here"
    return self.send_message(message)
```

### Multiple Recipients

To send to multiple people, create a Telegram group:
1. Create a new group in Telegram
2. Add your bot to the group
3. Get the group chat ID (will be negative)
4. Use group chat ID in TELEGRAM_CHAT_ID

## Troubleshooting

### Bot Not Responding?
- Ensure you messaged the bot first
- Check token is correct (no extra spaces)
- Verify chat ID is correct

### No Notifications?
- Check `.env` file has correct values
- Restart trading system after adding credentials
- Check `ai_trading.log` for Telegram errors

### Test Commands

```bash
# Test connection
curl -X POST "https://api.telegram.org/bot<TOKEN>/sendMessage" \
  -H "Content-Type: application/json" \
  -d '{"chat_id": "<CHAT_ID>", "text": "Test message"}'

# Check bot info
curl "https://api.telegram.org/bot<TOKEN>/getMe"
```

## Security Notes

⚠️ **Keep Your Token Secret**
- Never commit bot token to git
- Keep it in `.env` file only
- Regenerate if compromised

🔒 **Private Bot**
- Your bot only responds to your chat ID
- Others can't use your bot
- Messages are encrypted in transit

## Example Notifications

### Trade Alert
```
🟢 Trade Executed
Symbol: TSLA
Action: BUY
Quantity: 50
Price: $245.30
AI Confidence: 92.1%
Reason: Bullish options flow detected
Time: 10:35:42
```

### Error Alert
```
🚨 CRITICAL ERROR
Failed to connect to IB API
Connection timeout after 30 seconds
Time: 09:30:15
```

### Market Event
```
⚡ SEC Filing
Symbol: AAPL
Event: Form 8-K Filed
Impact: 80/100
Details: Material agreement announced
Time: 14:22:18
```

## Tips

1. **Mute During Development**: Mute notifications while testing
2. **Custom Sounds**: Set custom notification sounds per chat
3. **Do Not Disturb**: Set quiet hours in Telegram settings
4. **Pin Important**: Pin your bot chat for quick access
5. **Commands**: Add bot commands for status checks

## Future Enhancements

- [ ] Two-way communication (send commands to bot)
- [ ] Rich media (charts, graphs)
- [ ] Voice alerts for critical events
- [ ] Inline keyboards for quick actions
- [ ] Schedule reports (weekly, monthly)

---

Need help? Check Telegram Bot API docs: https://core.telegram.org/bots/api