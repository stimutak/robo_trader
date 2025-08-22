# Remote Access Guide - Robo Trader Dashboard

Access your trading bot dashboard securely from anywhere using Tailscale.

## Quick Start (5 minutes)

### 1. Install Tailscale on Both Machines

**On your home machine (where the bot runs):**
```bash
./setup_tailscale.sh
```

**On your MacBook Pro:**
```bash
brew install tailscale
tailscale up
```

### 2. Start the Trading System
```bash
./restart_trading.sh
```

### 3. Access the Dashboard

From your MacBook, open:
- `http://[tailscale-ip]:5555` - Using Tailscale IP
- `http://[hostname]:5555` - Using MagicDNS (easier!)

To find your Tailscale IP:
```bash
tailscale ip -4
```

## Optional: Enable Authentication

For extra security, enable basic authentication:

### 1. Generate Password Hash
```bash
python generate_password_hash.py
# Enter your desired password when prompted
```

### 2. Update .env File
Add the generated hash to your `.env`:
```bash
DASH_AUTH_ENABLED=true
DASH_USER=your_username
DASH_PASS_HASH=your_generated_hash_here
```

### 3. Restart the Dashboard
```bash
./restart_trading.sh
```

Now you'll be prompted for username/password when accessing the dashboard.

## Mobile Access (iPhone/iPad)

1. Install Tailscale app from App Store
2. Sign in with the same account
3. Open Safari and navigate to `http://[hostname]:5555`
4. Save to home screen for quick access

## Security Features

✅ **End-to-End Encryption**: All traffic encrypted via WireGuard  
✅ **Private Network**: No ports exposed to internet  
✅ **Device Authorization**: Only your authorized devices can connect  
✅ **Optional Auth**: Additional username/password protection  
✅ **No Configuration**: Zero network setup required  

## Health Monitoring

Check if the system is running remotely:
```bash
curl http://[hostname]:5555/api/health
```

Returns:
```json
{
  "status": "healthy",
  "trading_active": true,
  "timestamp": "2025-08-22T10:30:00"
}
```

## SSH Access (Bonus)

Tailscale also enables SSH without port forwarding:

```bash
# Connect to your home machine
ssh user@[hostname]

# Monitor logs
tail -f ~/robo_trader/ai_trading.log

# Check trading status
cd ~/robo_trader
python -c "from robo_trader.database import TradingDatabase; db = TradingDatabase(); print(db.get_today_pnl())"
```

## Troubleshooting

### Can't Connect?
1. Ensure Tailscale is running on both machines:
   ```bash
   tailscale status
   ```

2. Check if machines can ping each other:
   ```bash
   tailscale ping [hostname]
   ```

3. Verify dashboard is running:
   ```bash
   curl localhost:5555/api/health  # On home machine
   ```

### Authentication Issues?
- Ensure `.env` has correct `DASH_PASS_HASH`
- Username is case-sensitive
- Try disabling auth temporarily: `DASH_AUTH_ENABLED=false`

### Performance Tips
- Tailscale creates direct peer-to-peer connections when possible
- If connection seems slow, check: `tailscale netcheck`
- Consider enabling Tailscale's MagicDNS for easier access

## Advanced Configuration

### Restrict Access by Device
In Tailscale admin panel, create ACLs to limit which devices can access port 5555:

```json
{
  "acls": [
    {
      "action": "accept",
      "users": ["your-email@example.com"],
      "ports": ["trading-bot:5555"]
    }
  ]
}
```

### Automatic Start on Boot
Add to crontab on your home machine:
```bash
@reboot cd /home/user/robo_trader && ./restart_trading.sh
```

## Why Tailscale?

| Feature | Tailscale | Traditional VPN | Port Forwarding |
|---------|-----------|-----------------|-----------------|
| Setup Time | 5 min | 1-2 hours | 30 min |
| Security | Excellent | Good | Poor |
| NAT Traversal | Automatic | Manual | Manual |
| Mobile Support | Native | Complex | Limited |
| Cost | Free | Often Paid | Free |

## Need Help?

- Tailscale Docs: https://tailscale.com/kb/
- Dashboard Issues: Check `dashboard.log`
- Trading Issues: Check `ai_trading.log`
- Network Issues: Run `tailscale netcheck`

---

**Remember**: Never expose port 5555 directly to the internet. Always use Tailscale or similar VPN solution for remote access.