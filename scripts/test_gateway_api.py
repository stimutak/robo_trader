#!/usr/bin/env python3
"""
Quick Gateway API connectivity test.

Tests if Gateway responds to IBKR API handshake.
Properly disconnects using safe_disconnect() to avoid creating zombie connections.

Exit codes:
  0 = Success (API responding)
  1 = Handshake timeout (API not responding)
  2 = Connection refused (port not open)
"""

import os
import socket
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CRITICAL: Enable force disconnect for this test script
# Without this, ib.disconnect() is a no-op and creates zombies
os.environ["IBKR_FORCE_DISCONNECT"] = "1"


def test_api_handshake(host="127.0.0.1", port=4002, timeout=15):
    """Test if Gateway responds to API handshake."""
    try:
        from ib_async import IB

        from robo_trader.utils.ibkr_safe import safe_disconnect

        ib = IB()

        # Set shorter timeout for quick test
        try:
            ib.connect(host, port, clientId=99999, readonly=True, timeout=timeout)

            if ib.isConnected():
                print(f"SUCCESS: Connected to Gateway API at {host}:{port}")
                accounts = ib.managedAccounts()
                if accounts:
                    print(f"  Accounts: {accounts}")

                # Properly disconnect using safe_disconnect (with force flag set above)
                safe_disconnect(ib)
                return 0
            else:
                print(f"FAILED: Connection not established")
                return 1

        except Exception as e:
            err_str = str(e).lower()
            if "timeout" in err_str:
                print(f"TIMEOUT: Gateway not responding to API handshake")
                print(f"  Gateway may be stuck - restart recommended")
                return 1
            elif "connection refused" in err_str:
                print(f"REFUSED: Port {port} not accepting connections")
                return 2
            else:
                print(f"ERROR: {e}")
                return 1
        finally:
            # Ensure disconnect
            try:
                if ib.isConnected():
                    safe_disconnect(ib)
            except Exception:
                pass

    except ImportError as e:
        # Fallback: Just test if port accepts connection and sends data
        print(f"WARNING: Could not import ib_async/safe_disconnect: {e}")
        print("Falling back to socket test...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))

            # Wait briefly for any response
            sock.settimeout(2)
            try:
                data = sock.recv(1024)
                if data:
                    print(f"SUCCESS: Gateway port {port} is responding")
                    sock.close()
                    return 0
            except socket.timeout:
                pass

            sock.close()
            print(f"UNKNOWN: Port open but no response (ib_async not available)")
            return 1

        except socket.timeout:
            print(f"TIMEOUT: Gateway not responding")
            return 1
        except ConnectionRefusedError:
            print(f"REFUSED: Port {port} not accepting connections")
            return 2
        except Exception as e:
            print(f"ERROR: {e}")
            return 1


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4002
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    sys.exit(test_api_handshake(port=port, timeout=timeout))
