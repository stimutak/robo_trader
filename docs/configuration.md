## Configuration

All runtime configuration is loaded via environment variables in `robo_trader/config.py`. Use `.env` with `python-dotenv`.

### Required
- `IBKR_HOST` (default `127.0.0.1`)
- `IBKR_PORT` (default `7497` for paper)
- `IBKR_CLIENT_ID` (integer client ID)

### Trading Mode
- `TRADING_MODE`:
  - `paper` (default)
  - `live` (opt-in only; see live safeguards)

### Symbols
- `SYMBOLS` comma-separated list, default: `AAPL,MSFT,SPY`

### Risk Parameters
- `MAX_DAILY_LOSS` (default `1000`)
- `MAX_POSITION_RISK_PCT` (default `0.01`)
- `MAX_SYMBOL_EXPOSURE_PCT` (default `0.2`)
- `MAX_LEVERAGE` (default `2.0`)

### Paper Equity
- `DEFAULT_CASH` (default `100000`): starting equity for paper simulation

### Validation
On startup, ensure variables are present and parseable. Fail fast with clear messages.


