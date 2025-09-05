#!/usr/bin/env python3
"""Simple database recovery using sync SQLite"""

import os
import sqlite3
from datetime import datetime

# Remove any existing database files
for f in [
    "trading_data.db",
    "trading_data.db-journal",
    "trading_data.db-wal",
    "trading_data.db-shm",
]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed {f}")

print("\nCreating fresh database...")

# Create new database
conn = sqlite3.connect("trading_data.db")
cursor = conn.cursor()

# Create all tables
cursor.executescript(
    """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    avg_cost REAL NOT NULL,
    market_price REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    slippage REAL DEFAULT 0,
    commission REAL DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY,
    cash REAL NOT NULL,
    equity REAL NOT NULL,
    daily_pnl REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    unrealized_pnl REAL DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER,
    timestamp DATETIME NOT NULL,
    UNIQUE(symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    strength REAL,
    metadata TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ticks (
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    bid REAL,
    ask REAL,
    last REAL,
    bid_size INTEGER,
    ask_size INTEGER,
    last_size INTEGER,
    volume INTEGER,
    PRIMARY KEY (timestamp, symbol)
);

CREATE TABLE IF NOT EXISTS features (
    timestamp DATETIME NOT NULL,
    symbol TEXT NOT NULL,
    rsi REAL,
    macd_line REAL,
    macd_signal REAL,
    macd_histogram REAL,
    bb_upper REAL,
    bb_middle REAL,
    bb_lower REAL,
    atr REAL,
    vwap REAL,
    obv REAL,
    sma_20 REAL,
    sma_50 REAL,
    sma_200 REAL,
    volume_ratio REAL,
    spread_bps REAL,
    trend_strength REAL,
    mean_reversion_signal REAL,
    breakout_signal REAL,
    PRIMARY KEY (timestamp, symbol)
);

INSERT OR IGNORE INTO account (id, cash, equity) VALUES (1, 100000, 100000);
"""
)

print("✅ Created all tables")

# Now try to recover data from backup
try:
    backup_conn = sqlite3.connect("trading_data_backup_20250901_081014.db")

    # Recover trades
    try:
        trades = backup_conn.execute("SELECT * FROM trades").fetchall()
        for trade in trades:
            # Backup order: id, symbol, side, quantity, price, timestamp, slippage, commission
            # Need to reorder to: id, symbol, side, quantity, price, slippage, commission, timestamp
            cursor.execute(
                """
                INSERT INTO trades (id, symbol, side, quantity, price, slippage, commission, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (trade[0], trade[1], trade[2], trade[3], trade[4], trade[6], trade[7], trade[5]),
            )
        print(f"✅ Restored {len(trades)} trades")
    except Exception as e:
        print(f"❌ Could not restore trades: {e}")

    # Recover positions
    try:
        positions = backup_conn.execute("SELECT * FROM positions").fetchall()
        for pos in positions:
            cursor.execute(
                """
                INSERT OR REPLACE INTO positions (id, symbol, quantity, avg_cost, market_price, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                pos,
            )
        print(f"✅ Restored {len(positions)} positions")
    except Exception as e:
        print(f"❌ Could not restore positions: {e}")

    # Recover account
    try:
        account = backup_conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
        if account:
            cursor.execute(
                """
                UPDATE account SET cash = ?, equity = ?, daily_pnl = ?, realized_pnl = ?, unrealized_pnl = ?
                WHERE id = 1
            """,
                (
                    account[1],
                    account[2],
                    account[3] if len(account) > 3 else 0,
                    account[4] if len(account) > 4 else 0,
                    account[5] if len(account) > 5 else 0,
                ),
            )
            print(f"✅ Restored account info")
    except Exception as e:
        print(f"❌ Could not restore account: {e}")

    backup_conn.close()

except Exception as e:
    print(f"❌ Could not open backup: {e}")

# Commit and close
conn.commit()
conn.close()

print("\n✅ Database recovery complete!")
print("Database saved to: trading_data.db")
