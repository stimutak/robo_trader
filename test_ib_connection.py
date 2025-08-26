#!/usr/bin/env python3
"""Test IB connection and fetch some data"""

import asyncio
import pytest
from ib_insync import IB, Stock
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

@pytest.mark.asyncio
async def test_connection():
    ib = IB()
    try:
        # Try different client IDs until we find one that works
        for client_id in range(10, 20):
            try:
                print(f"Trying to connect with client ID {client_id}...")
                await ib.connectAsync('127.0.0.1', 7497, clientId=client_id)
                print(f"Connected successfully with client ID {client_id}!")
                break
            except Exception as e:
                print(f"Client ID {client_id} failed: {e}")
                continue
        
        if not ib.isConnected():
            print("Failed to connect to IB Gateway/TWS")
            return
        
        # Test fetching data for SPY
        contract = Stock('SPY', 'SMART', 'USD')
        ib.qualifyContracts(contract)
        
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        print(f"Fetched {len(bars)} bars for SPY")
        if bars:
            print(f"Latest bar: {bars[-1]}")
            
        # Update database with test data
        import sqlite3
        conn = sqlite3.connect('trading_data.db')
        cursor = conn.cursor()
        
        # Test write to database
        cursor.execute("""
            INSERT OR REPLACE INTO account (timestamp, cash, equity, open_pnl, closed_pnl)
            VALUES (datetime('now'), 100000, 100000, 0, 0)
        """)
        conn.commit()
        print("Successfully wrote to database")
        conn.close()
        
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("Disconnected from IB")

if __name__ == '__main__':
    asyncio.run(test_connection())