.PHONY: test install clean lint format check security pre-commit all venv

# Create virtual environment
venv:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install -U pip setuptools wheel

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

# Install with venv
install-venv:
	. .venv/bin/activate && pip install -r requirements.txt && pip install -r requirements-dev.txt && pip install -e .

# Install pre-commit hooks
install-hooks:
	pre-commit install

# Run tests
test:
	pytest tests/test_portfolio.py tests/test_retry.py -v

# Run tests with venv
test-venv:
	. .venv/bin/activate && pytest tests/test_portfolio.py tests/test_retry.py -v

# Run all tests with coverage
test-all:
	pytest tests/ -v --cov=robo_trader --cov-report=html --cov-report=term

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Lint code
lint:
	flake8 robo_trader/
	mypy robo_trader/ --ignore-missing-imports --no-strict-optional

# Format code
format:
	black robo_trader/ --line-length=100
	isort robo_trader/ --profile black --line-length=100

# Check code formatting
check-format:
	black --check --diff robo_trader/
	isort --check-only --diff robo_trader/

# Security checks
security:
	bandit -r robo_trader/ -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true

# Run pre-commit on all files
pre-commit:
	pre-commit run --all-files

# Full check (format, lint, test)
check: format lint test

# Run everything
all: clean install format lint security test

# Development setup
dev-setup: install install-hooks
	@echo "Development environment setup complete!"
	@echo "Run 'make check' to verify everything works"

# Run the trading system
run:
	python -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA

# Run with venv
run-venv:
	. .venv/bin/activate && python -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA

# Run BugBot
bugbot:
	./scripts/run_bugbot.sh

# Run BugBot with specific config
bugbot-dev:
	python3 scripts/bug_detector.py --scan --config development --output bug-report-dev.json

# Run BugBot with production config
bugbot-prod:
	python3 scripts/bug_detector.py --scan --config production --tools mypy,bandit,flake8 --output bug-report-prod.json

# Run BugBot in watch mode
bugbot-watch:
	python3 scripts/bug_detector.py --watch --config development


