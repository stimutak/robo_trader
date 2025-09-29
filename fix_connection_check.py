#!/usr/bin/env python3
"""
Fix for IBKR connection verification.
Adds proper connection checking to prevent "Trading Active" without IBKR.
"""

import asyncio


def check_ibkr_connection():
    """Check if IBKR is actually connected."""
    try:
        from robo_trader.connection_manager import ConnectionManager

        mgr = ConnectionManager()
        try:
            ib = asyncio.get_event_loop().run_until_complete(mgr.connect())
            if ib and ib.isConnected():
                print("✅ IBKR connected")
                return True
        except Exception:
            pass
        finally:
            try:
                asyncio.get_event_loop().run_until_complete(mgr.disconnect())
            except Exception:
                pass

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
        from robo_trader.connection_manager import ConnectionManager
        mgr = ConnectionManager()
        ib = asyncio.get_event_loop().run_until_complete(mgr.connect())
        return bool(ib and ib.isConnected())
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
        # Deprecated client pool check removed
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
