"""Tests for Flask API multi-portfolio endpoints.

Tests the validate_portfolio decorator and endpoint behavior with
various portfolio_id values (valid, invalid, non-existent).
"""

import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def client():
    """Create a Flask test client with auth disabled."""
    # Ensure auth is disabled for tests
    with patch.dict(os.environ, {"DASH_AUTH_ENABLED": "false"}, clear=False):
        from app import app

        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


# ──────────────────────────────────────────────
# Validation Tests (400 responses)
# These are rejected by the validate_portfolio decorator
# before any DB access, so no mocking needed.
# ──────────────────────────────────────────────


class TestPortfolioIdValidation400:
    """Test that invalid portfolio_id values are rejected with 400."""

    ENDPOINTS = [
        "/api/status",
        "/api/pnl",
        "/api/positions",
        "/api/watchlist",
        "/api/performance",
        "/api/equity-curve",
        "/api/trades",
        "/api/strategies/status",
    ]

    def test_special_chars_rejected(self, client):
        """Portfolio ID with special characters returns 400."""
        for endpoint in self.ENDPOINTS:
            resp = client.get(f"{endpoint}?portfolio_id=bad!id")
            assert resp.status_code == 400, f"{endpoint} did not return 400 for 'bad!id'"
            data = resp.get_json()
            assert "error" in data

    def test_spaces_rejected(self, client):
        """Portfolio ID with spaces returns 400."""
        resp = client.get("/api/positions?portfolio_id=has spaces")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_semicolon_rejected(self, client):
        """Portfolio ID with semicolons returns 400."""
        resp = client.get("/api/positions?portfolio_id=semi;colon")
        assert resp.status_code == 400

    def test_sql_injection_rejected(self, client):
        """SQL injection payloads return 400."""
        injections = [
            "'; DROP TABLE trades; --",
            "1 OR 1=1",
            "default' UNION SELECT * FROM account --",
            "x'; DELETE FROM positions; --",
        ]
        for payload in injections:
            resp = client.get(f"/api/positions?portfolio_id={payload}")
            assert resp.status_code == 400, f"SQL injection not blocked: {payload}"

    def test_too_long_rejected(self, client):
        """Portfolio ID exceeding 64 characters returns 400."""
        long_id = "a" * 65
        resp = client.get(f"/api/positions?portfolio_id={long_id}")
        assert resp.status_code == 400

    def test_dot_rejected(self, client):
        """Portfolio ID with dots returns 400."""
        resp = client.get("/api/trades?portfolio_id=has.dot")
        assert resp.status_code == 400

    def test_quote_rejected(self, client):
        """Portfolio ID with quotes returns 400."""
        resp = client.get("/api/trades?portfolio_id=has'quote")
        assert resp.status_code == 400

    def test_all_endpoints_reject_invalid(self, client):
        """All 8 decorated endpoints reject invalid portfolio_id."""
        for endpoint in self.ENDPOINTS:
            resp = client.get(f"{endpoint}?portfolio_id=inv@lid!")
            assert resp.status_code == 400, f"{endpoint} did not reject 'inv@lid!'"


# ──────────────────────────────────────────────
# Non-existent Portfolio Tests (404 responses)
# The decorator checks sqlite3 for portfolio existence.
# We mock sqlite3.connect to control what the decorator sees.
# ──────────────────────────────────────────────


class TestNonExistentPortfolio404:
    """Test that valid-format but non-existent portfolios return 404."""

    @pytest.fixture
    def trading_db(self, tmp_path, monkeypatch):
        """Create trading_data.db in a temp dir and chdir there.

        The validate_portfolio decorator opens Path("trading_data.db") directly,
        so we place a real DB file in the working directory.
        """
        db_path = tmp_path / "trading_data.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE account (
                portfolio_id TEXT NOT NULL,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE portfolios (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                starting_cash REAL DEFAULT 100000
            )
        """)
        conn.execute(
            "INSERT INTO account (portfolio_id, cash, equity) VALUES ('default', 100000, 100000)"
        )
        conn.commit()
        conn.close()
        monkeypatch.chdir(tmp_path)
        return db_path

    def test_nonexistent_portfolio_returns_404(self, client, trading_db):
        """A valid-format portfolio_id that doesn't exist in DB returns 404."""
        resp = client.get("/api/positions?portfolio_id=nonexistent")
        assert resp.status_code == 404
        data = resp.get_json()
        assert "not found" in data["error"].lower()

    def test_default_portfolio_not_checked_for_existence(self, client):
        """The 'default' portfolio skips the DB existence check entirely."""
        mock_reader = MagicMock()
        mock_reader.get_positions.return_value = []
        mock_reader.get_signals.return_value = []
        with patch("sync_db_reader.SyncDatabaseReader", return_value=mock_reader):
            resp = client.get("/api/positions?portfolio_id=default")
            assert resp.status_code != 404


# ──────────────────────────────────────────────
# Default Behavior Tests (200 responses)
# These need SyncDatabaseReader mocked to avoid
# requiring a real database.
# ──────────────────────────────────────────────


class TestDefaultPortfolioBehavior:
    """Test that endpoints work with default portfolio_id."""

    @pytest.fixture
    def mock_db_reader(self):
        """Create a mock SyncDatabaseReader with sensible defaults."""
        reader = MagicMock()
        reader.get_positions.return_value = []
        reader.get_recent_trades.return_value = []
        reader.get_account_info.return_value = {
            "cash": 100000,
            "equity": 100000,
            "daily_pnl": 0,
            "realized_pnl": 0,
            "unrealized_pnl": 0,
        }
        reader.get_equity_history.return_value = []
        reader.get_signals.return_value = []
        reader.get_portfolios.return_value = [
            {"id": "default", "name": "Default Portfolio", "starting_cash": 100000}
        ]
        return reader

    def test_positions_default_returns_200(self, client, mock_db_reader):
        """GET /api/positions without portfolio_id returns 200."""
        with patch("sync_db_reader.SyncDatabaseReader", return_value=mock_db_reader):
            resp = client.get("/api/positions")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "positions" in data

    def test_trades_default_returns_200(self, client, mock_db_reader):
        """GET /api/trades without portfolio_id returns 200."""
        with patch("sync_db_reader.SyncDatabaseReader", return_value=mock_db_reader):
            resp = client.get("/api/trades")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "trades" in data

    def test_pnl_default_returns_200(self, client, mock_db_reader):
        """GET /api/pnl without portfolio_id returns 200."""
        with patch("sync_db_reader.SyncDatabaseReader", return_value=mock_db_reader):
            resp = client.get("/api/pnl")
            assert resp.status_code == 200
            data = resp.get_json()
            # PnL endpoint returns equity, cash, etc.
            assert "equity" in data or "total" in data

    def test_equity_curve_default_returns_200(self, client, mock_db_reader):
        """GET /api/equity-curve without portfolio_id returns 200."""
        with patch("sync_db_reader.SyncDatabaseReader", return_value=mock_db_reader):
            resp = client.get("/api/equity-curve")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "labels" in data or "portfolio_values" in data


# ──────────────────────────────────────────────
# Portfolio Scoping Tests
# Verify portfolio_id is passed through to DB reader.
# ──────────────────────────────────────────────


class TestPortfolioScoping:
    """Test that portfolio_id is correctly passed to the DB reader."""

    @pytest.fixture
    def mock_db_reader(self):
        reader = MagicMock()
        reader.get_positions.return_value = [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "avg_cost": 180.0,
                "market_price": 185.0,
                "timestamp": "2026-02-06",
            }
        ]
        reader.get_recent_trades.return_value = []
        reader.get_account_info.return_value = {
            "cash": 50000,
            "equity": 55000,
            "daily_pnl": 0,
            "realized_pnl": 0,
            "unrealized_pnl": 0,
        }
        reader.get_equity_history.return_value = []
        reader.get_signals.return_value = []
        reader.get_portfolios.return_value = [
            {"id": "default", "name": "Default Portfolio", "starting_cash": 100000}
        ]
        return reader

    @pytest.fixture
    def trading_db_with_portfolio(self, tmp_path, monkeypatch):
        """Create trading_data.db with a non-default portfolio in a temp dir."""
        db_path = tmp_path / "trading_data.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE account (
                portfolio_id TEXT NOT NULL,
                cash REAL NOT NULL,
                equity REAL NOT NULL,
                daily_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE portfolios (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                starting_cash REAL DEFAULT 100000
            )
        """)
        conn.execute(
            "INSERT INTO account (portfolio_id, cash, equity) VALUES ('default', 100000, 100000)"
        )
        conn.execute(
            "INSERT INTO account (portfolio_id, cash, equity) VALUES ('aggressive', 50000, 55000)"
        )
        conn.commit()
        conn.close()
        monkeypatch.chdir(tmp_path)
        return db_path

    def test_positions_with_explicit_default(self, client, mock_db_reader):
        """GET /api/positions?portfolio_id=default returns 200."""
        with patch("sync_db_reader.SyncDatabaseReader", return_value=mock_db_reader):
            resp = client.get("/api/positions?portfolio_id=default")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "positions" in data

    def test_trades_with_explicit_default(self, client, mock_db_reader):
        """GET /api/trades?portfolio_id=default returns 200."""
        with patch("sync_db_reader.SyncDatabaseReader", return_value=mock_db_reader):
            resp = client.get("/api/trades?portfolio_id=default")
            assert resp.status_code == 200

    def test_positions_passes_portfolio_id_to_reader(
        self, client, mock_db_reader, trading_db_with_portfolio
    ):
        """Endpoint passes portfolio_id to SyncDatabaseReader.get_positions."""
        with patch("sync_db_reader.SyncDatabaseReader", return_value=mock_db_reader):
            resp = client.get("/api/positions?portfolio_id=aggressive")
            assert resp.status_code == 200
            mock_db_reader.get_positions.assert_called_with(portfolio_id="aggressive")

    def test_trades_passes_portfolio_id_to_reader(
        self, client, mock_db_reader, trading_db_with_portfolio
    ):
        """Endpoint passes portfolio_id to SyncDatabaseReader.get_recent_trades."""
        with patch("sync_db_reader.SyncDatabaseReader", return_value=mock_db_reader):
            resp = client.get("/api/trades?portfolio_id=aggressive")
            assert resp.status_code == 200
            mock_db_reader.get_recent_trades.assert_called()
            # Check portfolio_id was in the call kwargs
            call_kwargs = mock_db_reader.get_recent_trades.call_args
            assert call_kwargs.kwargs.get("portfolio_id") == "aggressive" or (
                len(call_kwargs.args) == 0 and "aggressive" in str(call_kwargs)
            )


# ──────────────────────────────────────────────
# Valid portfolio_id format edge cases
# ──────────────────────────────────────────────


class TestValidPortfolioIdFormats:
    """Test that valid portfolio_id formats pass the decorator."""

    VALID_IDS = [
        "default",
        "aggressive",
        "my-portfolio",
        "port_1",
        "A",
        "abc123",
        "a" * 64,  # max length
        "user123_portfolio456",
    ]

    def test_valid_ids_not_rejected_as_400(self, client):
        """Valid portfolio IDs should not get a 400 response.

        They may get 404 (non-existent) or 200 (if default), but never 400.
        """
        for pid in self.VALID_IDS:
            resp = client.get(f"/api/positions?portfolio_id={pid}")
            assert resp.status_code != 400, f"Valid portfolio_id '{pid}' was rejected as 400"

    def test_no_portfolio_id_defaults_to_default(self, client):
        """Omitting portfolio_id should default to 'default' (not 400)."""
        mock_reader = MagicMock()
        mock_reader.get_positions.return_value = []
        mock_reader.get_signals.return_value = []
        with patch("sync_db_reader.SyncDatabaseReader", return_value=mock_reader):
            resp = client.get("/api/positions")
            assert resp.status_code != 400
