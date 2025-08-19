## Robo Trader

Safe, testable, risk-managed trading scaffold that connects to IBKR for market data and runs in paper mode by default. Emphasizes capital preservation, clarity, and reproducibility.

### Features
- Paper trading default; live gated behind config and risk checks
- Env-driven config via `.env` (see `.env.example`)
- Risk controls: daily loss cap, per-symbol exposure, leverage limit
- Simple SMA crossover strategy example
- Async IBKR client wrapper
- Tests with `pytest`

### Project Layout
```
robo_trader/
├── robo_trader/
│   ├── config.py              # Env-driven configuration
│   ├── ibkr_client.py         # Async ib_insync client wrapper
│   ├── execution.py           # Paper execution simulator
│   ├── risk.py                # Position sizing & exposure checks
│   ├── strategies.py          # SMA crossover example
│   └── runner.py              # Example orchestrator (paper only)
├── tests/
│   ├── test_risk.py
│   └── test_strategies.py
├── requirements.txt
└── pyproject.toml
```

### Quickstart
```bash
# 1) Create venv
python3 -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .

# 3) Configure env
cp .env.example .env
# edit .env as needed (IBKR host/port/client id)

# 4) Run tests
pytest -q

# 5) Run example paper loop
python -m robo_trader.runner
```

### Configuration
All values are read in `robo_trader/config.py` via environment variables. Defaults are conservative.

Required for IBKR connectivity:
- `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`

Risk and trading mode:
- `TRADING_MODE` defaults to `paper`
- `MAX_DAILY_LOSS`, `MAX_POSITION_RISK_PCT`, `MAX_SYMBOL_EXPOSURE_PCT`, `MAX_LEVERAGE`

### Safety & Constraints
- Never place live orders by default. Live requires `TRADING_MODE=live` plus explicit approval and unchanged risk checks.
- Do not commit secrets. Use `.env` and ensure `.env` is gitignored.
- Keep tests green; extend tests when adding features.

### CI
GitHub Actions workflow at `.github/workflows/ci.yml` runs tests on push/PR.


