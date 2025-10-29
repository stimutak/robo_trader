#!/usr/bin/env python3
"""
Gateway Connection Health Checker

Checks for zombie CLOSE_WAIT connections and verifies Gateway is listening.
Useful for troubleshooting connection issues before starting the trader.
"""
import subprocess
import sys


def check_zombies(port: int = 4002) -> bool:
    """
    Check for zombie CLOSE_WAIT connections.

    Args:
        port: IBKR Gateway port to check (default: 4002 for paper)

    Returns:
        True if no zombies detected, False if zombies found
    """
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:CLOSE_WAIT"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.stdout.strip():
            print(f"⚠ Found zombie CLOSE_WAIT connections on port {port}:")
            print(result.stdout)
            print("\nThese connections block new API connections.")
            print("Solutions:")
            print("  1. Restart Gateway (File → Exit → Restart with 2FA)")
            print("  2. Kill zombie processes (if owned by Python):")
            print(f"     lsof -ti tcp:{port} -sTCP:CLOSE_WAIT | xargs kill -9")
            return False
        else:
            print(f"✓ No zombie connections on port {port}")
            return True

    except subprocess.TimeoutExpired:
        print("⚠ lsof command timed out")
        return True
    except FileNotFoundError:
        print("⚠ lsof not available on this system")
        return True
    except Exception as e:
        print(f"Error checking zombies: {e}")
        return True


def check_gateway_listening(port: int = 4002) -> bool:
    """
    Check if Gateway is listening on the port.

    Args:
        port: IBKR Gateway port to check (default: 4002 for paper)

    Returns:
        True if Gateway is listening, False otherwise
    """
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if "JavaAppli" in result.stdout:
            # Extract PID
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) > 1:
                    pid = parts[1]
                    print(f"✓ Gateway listening on port {port} (PID: {pid})")
                else:
                    print(f"✓ Gateway listening on port {port}")
            else:
                print(f"✓ Gateway listening on port {port}")
            return True
        else:
            print(f"✗ Gateway NOT listening on port {port}")
            print("\nGateway may not be running or API settings may be incorrect.")
            print("Check:")
            print("  1. Gateway is running")
            print("  2. File → Global Configuration → API → Settings")
            print("  3. ☑️ Enable ActiveX and Socket Clients")
            print("  4. Socket port matches (4002 for paper, 4001 for live)")
            return False

    except subprocess.TimeoutExpired:
        print("⚠ lsof command timed out")
        return False
    except FileNotFoundError:
        print("⚠ lsof not available on this system")
        return False
    except Exception as e:
        print(f"Error checking Gateway: {e}")
        return False


def count_connections(port: int = 4002) -> int:
    """
    Count total number of connections to the port.

    Args:
        port: IBKR Gateway port to check

    Returns:
        Number of connections found
    """
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        lines = result.stdout.strip().split("\n")
        # Subtract 1 for header line
        count = max(0, len(lines) - 1) if lines[0] else 0

        if count > 0:
            print(f"\nTotal connections on port {port}: {count}")
            if count > 5:
                print("⚠ High number of connections detected")
                print("  Gateway may be approaching connection limit")

        return count

    except Exception:
        return 0


def main():
    """Run all connection health checks."""
    # Parse port from command line
    port = 4002  # Default to paper trading port
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            print("Usage: python3 check_gateway_connections.py [port]")
            sys.exit(1)

    print(f"=== Gateway Connection Health Check (Port {port}) ===\n")

    # Run checks
    gateway_ok = check_gateway_listening(port)
    print()
    zombies_ok = check_zombies(port)
    print()
    count_connections(port)

    # Summary
    print("\n=== Summary ===")
    if gateway_ok and zombies_ok:
        print("✓ Gateway connection health: GOOD")
        print("  Trading system should be able to connect.")
        sys.exit(0)
    else:
        print("✗ Gateway connection health: ISSUES DETECTED")
        if not gateway_ok:
            print("  - Gateway not listening or API settings incorrect")
        if not zombies_ok:
            print("  - Zombie connections blocking new connections")
        print("\nResolve issues before starting the trading system.")
        sys.exit(1)


if __name__ == "__main__":
    main()
