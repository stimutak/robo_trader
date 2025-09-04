# GEMINI.md

## Project Overview

This project is a sophisticated, ML-driven algorithmic trading platform named "Robo Trader". It is designed for production-grade use with Interactive Brokers (IBKR) integration, featuring an advanced Machine Learning infrastructure, an asynchronous architecture for high performance, and comprehensive risk management features. The system defaults to paper trading and includes strict capital preservation controls.

The project is developed in Python and is currently in Phase 3 of its development roadmap, focusing on advanced strategy development with an ML-enhanced framework.

The architecture is built around an asynchronous, event-driven model, using `asyncio` to handle concurrent operations. The `runner_async.py` is the core of the application, orchestrating the trading process. It processes multiple symbols in parallel, fetching data, generating signals, and executing trades. It uses a connection pool to manage connections to the Interactive Brokers API, ensuring efficient and robust communication.

The platform supports multiple trading strategies, including a simple SMA crossover, a Machine Learning-based strategy, and an enhanced ML strategy with regime detection and multi-timeframe analysis.

## Building and Running

### Prerequisites

*   Python 3.9+ (tested on 3.13)
*   Interactive Brokers TWS or IB Gateway
*   IBKR Paper Trading Account

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/stimutak/robo_trader.git
    cd robo_trader
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
4.  **Configure the environment:**
    ```bash
    cp .env.example .env
    ```
    Then, edit the `.env` file with your IBKR credentials and risk settings.

### Running the System

*   **Run the async trading system with parallel processing:**
    ```bash
    python -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA
    ```
*   **Run with the ML Enhanced Strategy:**
    ```bash
    python -m robo_trader.runner_async --symbols AAPL,NVDA --use-ml-enhanced
    ```
*   **Run with Smart Execution:**
    ```bash
    python -m robo_trader.runner_async --symbols AAPL,NVDA --use-smart-execution
    ```
*   **Run the monitoring dashboard (on port 5555):**
    ```bash
    export DASH_PORT=5555
    python app.py
    ```

## Recent Updates (2025-09-04)

### Dashboard Enhancements
*   **Market Status Indicator:** Real-time market status display in dashboard header (Regular/Pre-Market/After-Hours/Closed)
*   **Performance Monitor Integration:** Live metrics display showing latency, throughput, and trading activity
*   **API Endpoints:** New `/api/market-status` and `/api/performance-monitor` endpoints for real-time data

### Database Recovery
*   **Recovery Scripts:** `init_database.py` for sample data initialization, `recover_database.py` for backup recovery
*   **Locking Resolution:** Fixed persistent SQLite locking issues that prevented trading system from running

## Development Conventions

The project enforces a strict code style and quality standards using a variety of tools, as defined in the `.pre-commit-config.yaml` and `pyproject.toml` files.

*   **Formatting:** `black` is used for code formatting with a line length of 100 characters.
*   **Import Sorting:** `isort` is used to sort imports, following the `black` profile.
*   **Linting:** `flake8` is used for linting with a max line length of 100.
*   **Type Checking:** `mypy` is used for static type checking.
*   **Security:** `bandit` is used for identifying common security issues.

These checks are enforced via pre-commit hooks, ensuring that all committed code adheres to the project's standards.

The codebase is written using asynchronous programming with `asyncio`. All I/O operations, such as API requests and database queries, are non-blocking.

## Testing

The project has a comprehensive test suite.

*   **Run all tests:**
    ```bash
    pytest
    ```
*   **Run specific test suites:**
    ```bash
    # Phase 1 tests (infrastructure)
    python test_phase1_complete.py

    # ML pipeline tests
    python test_ml_pipeline.py

    # Model training tests
    python test_m3_complete.py

    # Performance analytics tests
    python test_m4_performance.py
    ```

## Key Files and Directories

*   `robo_trader/`: The main source code for the application.
    *   `runner_async.py`: The core asynchronous and parallel trading system.
    *   `clients/async_ibkr_client.py`: A robust, asynchronous client for the Interactive Brokers API with connection pooling and retry logic.
    *   `config.py`: A comprehensive, Pydantic-based configuration system with environment-specific settings.
    *   `features/`: Machine Learning feature engineering pipeline.
    *   `ml/`: Machine Learning model training and selection.
    *   `backtesting/`: The walk-forward backtesting framework.
    *   `analytics/`: Performance analytics and metrics.
    *   `portfolio/`: Multi-strategy portfolio management.
    *   `websocket_server.py`: Real-time updates via websockets.
*   `app.py`: A Flask application that serves a real-time monitoring dashboard.
*   `tests/`: The test suites for the project.
*   `performance_results/`: Directory for storing strategy performance JSON files.
*   `trained_models/`: Directory for storing saved Machine Learning models.
*   `IMPLEMENTATION_PLAN.md`: The development roadmap for the project.
*   `.env.example`: Example environment file.
*   `requirements.txt`: The project's Python dependencies.
*   `pyproject.toml`: Project metadata and tool configuration.
*   `.pre-commit-config.yaml`: Configuration for pre-commit hooks.