#!/usr/bin/env python3
"""
IBKR Connection Test Script
Tests connection to TWS/IB Gateway BEFORE starting trading
This MUST pass before allowing trading to start
"""

import asyncio
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

try:
    from ib_insync import IB, util
    from ib_insync.contract import Stock
except ImportError:
    print("ERROR: ib_insync not installed. Run: pip3 install ib_insync")
    sys.exit(1)


def test_port_open(host: str = "127.0.0.1", port: int = 7497, timeout: int = 5) -> bool:
    """Test if TWS/IB Gateway port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Port test failed: {e}")
        return False


async def test_ibkr_connection(
    client_id: int = 420, port: int = 7497, timeout: int = 30
) -> Tuple[bool, Optional[str]]:
    """
    Test IBKR API connection
    Returns: (success: bool, error_message: Optional[str])
    """
    ib = IB()

    try:
        # First check if port is open
        if not test_port_open(port=port):
            return False, f"Port {port} is not open - TWS/IB Gateway not running"

        print(f"✓ Port {port} is open")

        # Try to connect
        print(f"Connecting to IBKR on port {port} with client ID {client_id}...")
        await asyncio.wait_for(
            ib.connectAsync(host="127.0.0.1", port=port, clientId=client_id), timeout=timeout
        )

        # Verify connection is active
        if not ib.isConnected():
            return False, "Connected but connection is not active"

        print("✓ Connected to IBKR")

        # Test getting account info
        accounts = ib.managedAccounts()
        if not accounts:
            return False, "No managed accounts found"

        print(f"✓ Found accounts: {accounts}")

        # Test requesting market data for a simple symbol
        test_contract = Stock("AAPL", "SMART", "USD")
        ticker = ib.reqMktData(test_contract, "", False, False)

        # Wait briefly for data
        await asyncio.sleep(2)

        # Check if we got any price data
        if ticker.marketPrice() and ticker.marketPrice() > 0:
            print(f"✓ Market data working - AAPL price: ${ticker.marketPrice()}")
        else:
            print("⚠ Warning: No market data received (markets may be closed)")

        # Cancel market data subscription
        ib.cancelMktData(test_contract)

        # Disconnect cleanly
        ib.disconnect()

        return True, None

    except asyncio.TimeoutError:
        return False, f"Connection timeout after {timeout} seconds"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"
    finally:
        if ib.isConnected():
            ib.disconnect()


def write_connection_status(success: bool, message: str = ""):
    """Write connection status to file for other scripts to check"""
    status_file = Path("/tmp/ibkr_connection_status.txt")
    timestamp = datetime.now().isoformat()

    with open(status_file, "w") as f:
        f.write(f"{timestamp}\n")
        f.write(f"SUCCESS={success}\n")
        if message:
            f.write(f"MESSAGE={message}\n")

    print(f"Status written to {status_file}")


def main():
    """Main test function"""
    print("=" * 60)
    print("IBKR CONNECTION TEST")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check for custom port/client ID
    port = 7497  # Default TWS paper trading port
    client_id = 420  # Default client ID

    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            client_id = int(sys.argv[2])
        except ValueError:
            print(f"Invalid client ID: {sys.argv[2]}")
            sys.exit(1)

    print(f"Testing connection to port {port} with client ID {client_id}")
    print()

    # Run the async test
    success, error = asyncio.run(test_ibkr_connection(client_id, port))

    print()
    print("=" * 60)

    if success:
        print("✅ IBKR CONNECTION TEST PASSED")
        print("System is ready for trading")
        write_connection_status(True, "Connection test passed")
        sys.exit(0)
    else:
        print("❌ IBKR CONNECTION TEST FAILED")
        print(f"Error: {error}")
        print()
        print("TROUBLESHOOTING STEPS:")
        print("1. Ensure TWS or IB Gateway is running")
        print("2. Check TWS Configuration:")
        print("   - File → Global Configuration → API → Settings")
        print("   - Enable 'Enable ActiveX and Socket Clients'")
        print("   - Enable 'Allow connections from localhost only'")
        print("   - Socket port should be 7497 (paper) or 7496 (live)")
        print("3. Check if another client is using the same client ID")
        print("4. Try a different client ID (e.g., 421, 422, etc.)")
        print("5. Restart TWS/IB Gateway")
        write_connection_status(False, error or "Unknown error")
        sys.exit(1)


if __name__ == "__main__":
    main()
