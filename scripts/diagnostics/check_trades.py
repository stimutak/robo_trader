#!/usr/bin/env python3
import sqlite3
from datetime import datetime

conn = sqlite3.connect("trading.db")
cursor = conn.cursor()

# Get last 10 trades
print("Checking last trades...")
cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10")
trades = cursor.fetchall()

if trades:
    print(f"\nLast {len(trades)} trades:")
    for trade in trades:
        # Assuming columns: id, symbol, side, quantity, price, timestamp, ...
        print(f"  {trade[1]} {trade[2]}: {trade[3]} shares @ ${trade[4]:.2f} on {trade[5]}")

    # Parse the timestamp of the last trade
    last_trade_time = trades[0][5]
    print(f"\nLast trade was on: {last_trade_time}")
else:
    print("No trades found in database")

# Check positions
cursor.execute("SELECT * FROM positions WHERE quantity != 0")
positions = cursor.fetchall()
print(f"\nActive positions: {len(positions)}")
for pos in positions[:5]:  # Show first 5
    print(f"  {pos[1]}: {pos[2]} shares @ avg ${pos[3]:.2f}")

# Check if trading_enabled flag exists
cursor.execute("SELECT * FROM system_status WHERE key='trading_enabled' LIMIT 1")
status = cursor.fetchone()
if status:
    print(f"\nTrading enabled: {status}")

conn.close()
