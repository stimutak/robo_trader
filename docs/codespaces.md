# GitHub Codespaces Setup

Use Codespaces for a one-click cloud dev environment that mirrors production.

## Quick Start

1) Open the repo in GitHub → Code → Codespaces → Create codespace on main.
2) The devcontainer builds from the repo `Dockerfile` and sets user `trader`.
3) Post-create installs dev deps and `-e .` and auto-copies `.env.example` → `.env` if missing.

## Ports

- 5555: Dash UI (`python app.py` uses `DASH_PORT`, default 5555)
- 8765: WebSocket server
- 9090: Metrics endpoint

Codespaces will auto-forward these. Mark them Public/Private as you prefer.

## Env & Secrets

- Do not commit `.env`. The container will create a local `.env` from `.env.example`.
- For sensitive values, set Codespaces secrets in repo/org → Codespaces → Secrets, e.g.:
  - `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`, `IBKR_ACCOUNT`
  - Any API keys listed in `.env.example`
- Codespaces injects these as environment variables inside the container on start.

## Common Commands

```bash
# Install additional tools (already handled post-create)
pip install -r requirements-dev.txt && pip install -e .

# Run tests
pytest -q

# Run trader
python -m robo_trader.runner_async --symbols AAPL,MSFT

# Run dashboard
python app.py  # forwards port 5555
```

## Notes

- The dev container uses the non-root user `trader` and installs Python packages in user site.
- The container mirrors production Python (3.11-slim). No virtualenv needed inside container.
- Default safety: `EXECUTION_MODE=paper` is set via container env.

