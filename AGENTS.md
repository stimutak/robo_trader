# Repository Guidelines

## Project Structure & Modules
- Source: `robo_trader/` (core trading, ML, features, analytics, backtesting).
- Entry points: `robo_trader/runner_async.py`, dashboard in `app.py`.
- Tests: `tests/` plus top-level `test_*.py` files.
- Artifacts: `trained_models/`, `performance_results/`, logs and `.db` files (gitignored).
- Config: `.env` (use `.env.example` as a template).

## Build, Test, and Dev Commands
- Create venv: `make venv` then `. .venv/bin/activate`.
- Install: `make install` (installs `requirements.txt` and package in editable mode).
- Test suite: `make test` or `pytest -q`.
- Run trader: `python -m robo_trader.runner_async --symbols AAPL,MSFT`.
- Run dashboard: `python app.py` (uses `DASH_PORT`, default 5555).

## Coding Style & Naming
- Language: Python 3.10+ with type hints where practical.
- Formatting: `black .` (default settings) and `isort .`.
- Linting: `flake8 robo_trader tests` and security checks via `bandit -r robo_trader`.
- Typing: `mypy robo_trader` (prefer explicit types at boundaries).
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Indentation: 4 spaces; keep functions focused and side-effect aware.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` (import mode: `importlib`).
- Location: add tests under `tests/` and follow `test_*.py`, `Test*` class, `test_*` function patterns.
- Scope: include unit tests for new logic and regression tests for bug fixes; add async tests where relevant.
- Running examples: `pytest -k feature_pipeline -q`, `pytest tests/backtesting -q`.

## Commit & Pull Requests
- Commit style: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, optional scopes) as seen in history.
- PR requirements:
  - Clear description, linked issues, and rationale.
  - Before/after notes (and screenshots/log snippets for dashboard or perf changes).
  - Checklist: tests added/updated, `black`/`isort`/`flake8`/`mypy` pass, no secrets or large artifacts.

## Security & Configuration
- Never commit secrets. Keep credentials in `.env`; copy from `.env.example`.
- Default to paper trading: set `EXECUTION_MODE=paper`. Confirm IBKR host/port before live runs.
- Large logs/DBs are ignored by `.gitignore`; avoid adding artifacts to VCS.
