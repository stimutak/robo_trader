## IBKR Setup (Paper Mode)

Robo Trader connects to IBKR via `ib_insync`. Start in paper mode only.

### 1) Install TWS (Paper)
- Download and install Interactive Brokers Trader Workstation (TWS) for macOS.
- Use a paper trading account.

### 2) Enable API in TWS
- TWS: Configure → API → Settings:
  - Enable “ActiveX and Socket Clients”
  - Turn on “Read-Only API” initially
  - Socket Port: `7497` (paper default)
  - Trusted IPs: add `127.0.0.1` or your machine IP
  - Keep pacing rates conservative; do not increase unless necessary

### 3) .env Configuration
Create `.env` from `.env.example` and set:
```
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=123
TRADING_MODE=paper
```

### 4) Connectivity Sanity Check
Run the example runner in paper mode:
```bash
python -m robo_trader.runner
```
This should fetch historical bars for configured symbols and simulate paper orders under risk checks.

### Notes
- Respect IBKR API pacing. Batch requests reasonably; prefer historical bars with RTH (`useRTH=True`).
- Never enable live mode without completing the safeguards in `../live-safeguards.md`.


