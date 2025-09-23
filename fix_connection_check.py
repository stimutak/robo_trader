#!/usr/bin/env python3
"""
Fix for IBKR connection verification.
Adds proper connection checking to prevent "Trading Active" without IBKR.
"""


def check_ibkr_connection():
    """Check if IBKR is actually connected."""
    try:
        from ib_async import IB

        ib = IB()

        # Try to connect to TWS/Gateway
        # TWS uses port 7497 for paper, 7496 for live
        # Gateway uses port 4002 for paper, 4001 for live

        for port in [7497, 4002, 7496, 4001]:
            try:
                ib.connect("127.0.0.1", port, clientId=999)
                if ib.isConnected():
                    print(f"✅ IBKR connected on port {port}")
                    ib.disconnect()
                    return True
            except Exception:
                continue

        print("❌ IBKR not connected - TWS/Gateway not running")
        return False

    except ImportError:
        print("❌ ib_insync not installed")
        return False


def fix_dashboard_api():
    """Fix the /api/status endpoint to check real connection."""

    fix_code = '''
# Replace line 2424 in app.py
# OLD: "connected": True,  # Hardcoded!
# NEW: "connected": check_ibkr_connection(),

# Add this function to app.py:
def check_ibkr_connection():
    """Check IBKR connection status."""
    try:
        # Check if any runner processes have active IBKR connections
        from robo_trader.clients.async_ibkr_client import AsyncIBKRClient
        client = AsyncIBKRClient()
        # Try to get connection from pool
        if client.pool and client.pool.pool:
            for conn in client.pool.pool:
                if conn and conn.is_connected():
                    return True
    except:
        pass
    return False
'''
    print(fix_code)


def fix_paper_executor():
    """Fix PaperExecutor to verify IBKR connection."""

    fix_code = '''
# In robo_trader/execution.py, modify PaperExecutor.place_order():

def place_order(self, order: Order) -> ExecutionResult:
    """Place a paper order."""
    
    # ADD: Check IBKR connection for paper trading
    if not self._check_ibkr_connection():
        return ExecutionResult(
            False, 
            "IBKR not connected - cannot execute paper trades"
        )
    
    # Rest of existing code...
    
def _check_ibkr_connection(self) -> bool:
    """Verify IBKR is connected for paper trading."""
    # Paper trading still needs IBKR for market data
    # Check if client pool has active connections
    try:
        from robo_trader.clients.async_ibkr_client import AsyncIBKRClient
        client = AsyncIBKRClient()
        return client.pool and any(
            conn.is_connected() for conn in client.pool.pool
        )
    except:
        return False
'''
    print(fix_code)


if __name__ == "__main__":
    print("=" * 60)
    print("IBKR CONNECTION CHECK FIX")
    print("=" * 60)

    # Check current connection
    print("\nCurrent Status:")
    is_connected = check_ibkr_connection()

    if not is_connected:
        print("\n⚠️  IBKR is not connected!")
        print("Please start TWS or IB Gateway before trading.")

    print("\n" + "=" * 60)
    print("FIXES TO APPLY:")
    print("=" * 60)

    print("\n1. Fix Dashboard API:")
    fix_dashboard_api()

    print("\n2. Fix PaperExecutor:")
    fix_paper_executor()

    print("\n" + "=" * 60)
    print("WHY NO TRADES SINCE JAN 9:")
    print("=" * 60)
    print(
        """
    1. IBKR connection was lost (TWS not running)
    2. System continued reporting "Trading Active" (hardcoded)
    3. Without IBKR connection:
       - No real market data received
       - Strategies can't generate signals
       - No trades executed (even paper trades need data)
    4. The system was in "zombie mode" - running but non-functional
    
    TO FIX:
    1. Start TWS or IB Gateway
    2. Apply the connection check fixes above
    3. Restart the trading system
    """
    )
