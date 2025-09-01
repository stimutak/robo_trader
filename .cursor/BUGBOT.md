### BugBot Guide for `robo_trader`

This repository is a safe, testable IBKR paper-trading framework. BugBot should prioritize capital preservation, determinism, and keeping the test suite green.

### Guardrails (non-negotiable)
- Do not enable live trading or connect to real accounts. Paper-only by default.
- Never commit secrets. Configuration must come from environment variables (see `robo_trader/config.py`).
- Do not bypass or weaken risk checks in `robo_trader/risk.py`.
- Prefer clear, minimal edits over large refactors. Update code in place; avoid duplicate files.

### Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-dev.txt || true
pip install -e .
```

### Core Commands
- Run targeted fast tests:
```bash
pytest -q -k "risk or strategies or retry or portfolio"
```
- Run full test suite with coverage:
```bash
pytest tests/ -v --cov=robo_trader --cov-report=term
```
- Lint and type-check:
```bash
flake8 robo_trader/
mypy robo_trader/ --ignore-missing-imports --no-strict-optional
```
- Format:
```bash
black robo_trader/ --line-length=100
isort robo_trader/ --profile black --line-length=100
```
- Security checks (non-blocking):
```bash
bandit -r robo_trader/ -f json -o bandit-report.json || true
safety check --json --output safety-report.json || true
```
- Makefile equivalents (preferred when available):
```bash
make install
make check          # format + lint + tests
make test-all       # all tests + coverage
make security
```

### Running the paper trader (for reproducibility only)
```bash
python -m robo_trader.runner_async --symbols AAPL,MSFT,SPY
```
Notes: execution must remain paper mode; respect pacing limits when applicable. Do not add networked/live paths in automated fixes.

### Paths/Artifacts to Ignore
- Do not scan or modify these unless a test failure points here:
  - `clientportal.gw/`, `root/`, `build/`, `dist/`, `.venv/`, `robo_trader.egg-info/`
  - Binary/artifact/log/db files: `*.jar`, `*.db`, `*.log`, `logs/`, `feature_store.db`, `trading.db`, `feature_store.*`

### Determinism & Testing
- Prefer deterministic logic; seed where relevant: `numpy.random.seed(42)`.
- Keep model/feature code reproducible given the same inputs. Avoid nondeterministic multi-threading in tests.
- For ML-heavy tests, prefer smaller synthetic inputs and disable expensive hyperparameter tuning unless explicitly tested.

### Triage Order for Failures
1. Unit tests in `tests/` (risk, strategies, retry, portfolio, production, analytics, ml).
2. Type errors (mypy) and lint issues (flake8).
3. Security warnings (bandit/safety) – fix if clearly actionable.

### Repo Conventions Recap
- Python >= 3.10. Use explicit type hints and early returns.
- Keep paper trading as the default; live paths must be gated and are out-of-scope for automated fixes.
- Prefer structured logging patterns; avoid `print` in library code.

### Common Fix Patterns
- Tighten input validation and edge-case handling (empty frames, NaNs, alignment).
- Ensure feature and model pipelines are deterministic and have stable interfaces.
- Maintain or extend tests when changing risk, sizing, or strategy behavior.

### Quick References
- Tests live in `tests/` and use `pytest` with `--import-mode=importlib`.
- Feature pipeline: `robo_trader/features/feature_pipeline.py`
- Risk logic: `robo_trader/risk.py`
- ML trainer: `robo_trader/ml/model_trainer.py`
- Async runner: `robo_trader/runner_async.py`

If in doubt, keep changes minimal, maintain safety constraints, and ensure tests pass locally before concluding a fix.


