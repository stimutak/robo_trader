#!/usr/bin/env python3
"""
Enhanced IBKR Gateway API Diagnostic Tool

Performs comprehensive diagnostics to identify why API handshakes are timing out.
Checks Gateway process, port status, TCP connectivity, API handshake, and provides
actionable recommendations.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from ib_async import IB


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_result(check: str, passed: bool, details: str = "") -> None:
    """Print a check result with status icon."""
    icon = "✅" if passed else "❌"
    print(f"{icon} {check}")
    if details:
        print(f"   {details}")


def check_gateway_process() -> tuple[bool, str]:
    """Check if IB Gateway process is running."""
    print_section("1. Gateway Process Check")

    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)

        gateway_lines = [
            line
            for line in result.stdout.split("\n")
            if "gateway" in line.lower() or "tws" in line.lower()
            if "grep" not in line.lower()
        ]

        if gateway_lines:
            print_result("Gateway process running", True)
            for line in gateway_lines:
                parts = line.split()
                if len(parts) >= 2:
                    print(f"   PID: {parts[1]}")
            return True, "Gateway process found"
        else:
            print_result("Gateway process running", False, "No Gateway/TWS process found")
            return False, "Gateway not running"

    except Exception as e:
        print_result("Gateway process check", False, f"Error: {e}")
        return False, f"Check failed: {e}"


def check_port_listening(port: int = 4002) -> tuple[bool, str]:
    """Check if Gateway port is listening."""
    print_section("2. Port Status Check")

    try:
        result = subprocess.run(["netstat", "-an"], capture_output=True, text=True, timeout=5)

        listening = False
        for line in result.stdout.split("\n"):
            if str(port) in line and "LISTEN" in line:
                listening = True
                print_result(f"Port {port} listening", True, line.strip())
                break

        if not listening:
            print_result(f"Port {port} listening", False, f"Port {port} not in LISTEN state")
            return False, f"Port {port} not listening"

        return True, f"Port {port} listening"

    except Exception as e:
        print_result("Port status check", False, f"Error: {e}")
        return False, f"Check failed: {e}"


def check_tcp_connection(
    host: str = "127.0.0.1", port: int = 4002, timeout: float = 5.0
) -> tuple[bool, str]:
    """Test raw TCP socket connection."""
    print_section("3. TCP Socket Connection Test")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        start_time = time.time()
        result = sock.connect_ex((host, port))
        connect_time = time.time() - start_time
        sock.close()

        if result == 0:
            print_result("TCP connection", True, f"Connected in {connect_time*1000:.1f}ms")
            return True, f"TCP connection successful ({connect_time*1000:.1f}ms)"
        else:
            print_result("TCP connection", False, f"Connection failed with error code {result}")
            return False, f"TCP connection failed (error {result})"

    except Exception as e:
        print_result("TCP connection", False, f"Error: {e}")
        return False, f"TCP connection error: {e}"


async def check_api_handshake(
    host: str = "127.0.0.1", port: int = 4002, client_id: int = 1, timeout: float = 15.0
) -> tuple[bool, str]:
    """Test IBKR API handshake."""
    print_section(f"4. API Handshake Test (Client ID: {client_id})")

    ib = IB()

    try:
        start_time = time.time()
        print(f"   Starting handshake at {time.strftime('%H:%M:%S')}...")

        await asyncio.wait_for(
            ib.connectAsync(
                host=host,
                port=port,
                clientId=client_id,
                timeout=timeout,
                readonly=True,
            ),
            timeout=timeout + 2,
        )

        connect_time = time.time() - start_time

        # Verify connection
        if not ib.isConnected():
            print_result("API handshake", False, "Connected but isConnected() returned False")
            return False, "Connection verification failed"

        # Get accounts
        accounts = ib.managedAccounts()

        # Get server version
        server_version = ib.serverVersion()

        print_result("API handshake", True, f"Connected in {connect_time:.2f}s")
        print(f"   Server version: {server_version}")
        print(f"   Managed accounts: {accounts}")

        return True, f"API handshake successful ({connect_time:.2f}s)"

    except asyncio.TimeoutError:
        connect_time = time.time() - start_time
        print_result("API handshake", False, f"TIMEOUT after {connect_time:.2f}s")
        return False, f"API handshake timeout ({connect_time:.2f}s)"

    except Exception as e:
        connect_time = time.time() - start_time
        print_result("API handshake", False, f"{type(e).__name__}: {e}")
        return False, f"API handshake failed: {e}"

    finally:
        if ib.isConnected():
            ib.disconnect()


def check_zombie_connections(port: int = 4002) -> tuple[int, str]:
    """Check for zombie CLOSE_WAIT connections."""
    print_section("5. Zombie Connection Check")

    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:CLOSE_WAIT"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        zombie_count = len(
            [
                line
                for line in result.stdout.split("\n")
                if line.strip() and not line.startswith("COMMAND")
            ]
        )

        if zombie_count == 0:
            print_result("Zombie connections", True, "No CLOSE_WAIT connections found")
            return 0, "No zombies"
        else:
            print_result(
                "Zombie connections", False, f"Found {zombie_count} CLOSE_WAIT connection(s)"
            )
            print("\n   Zombie connections:")
            for line in result.stdout.split("\n"):
                if line.strip() and not line.startswith("COMMAND"):
                    print(f"   {line}")
            return zombie_count, f"{zombie_count} zombie connections"

    except FileNotFoundError:
        print_result("Zombie connections", True, "lsof not available (skipping)")
        return 0, "lsof not available"
    except Exception as e:
        print_result("Zombie connection check", False, f"Error: {e}")
        return 0, f"Check failed: {e}"


def check_gateway_api_logs() -> tuple[bool, str]:
    """Check for Gateway API log files."""
    print_section("6. Gateway API Logs Check")

    log_paths = [
        Path.home() / "Jts" / "api_logs",
        Path.home() / "Jts",
    ]

    for log_path in log_paths:
        if log_path.exists():
            log_files = list(log_path.glob("*.log"))
            if log_files:
                print_result(
                    "API logs found", True, f"Found {len(log_files)} log file(s) in {log_path}"
                )

                # Show most recent log file
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                print(f"   Latest: {latest_log.name}")
                print(f"   Modified: {time.ctime(latest_log.stat().st_mtime)}")

                # Try to read last few lines
                try:
                    with open(latest_log, "r") as f:
                        lines = f.readlines()
                        if lines:
                            print(f"\n   Last 5 lines:")
                            for line in lines[-5:]:
                                print(f"   {line.rstrip()}")
                except Exception as e:
                    print(f"   Could not read log: {e}")

                return True, f"API logs available at {log_path}"

    print_result("API logs found", False, "No API log files found (API logging may not be enabled)")
    return False, "No API logs found"


def check_env_config() -> tuple[bool, str]:
    """Check .env configuration."""
    print_section("7. Configuration Check")

    env_file = Path(".env")
    if not env_file.exists():
        print_result(".env file", False, "File not found")
        return False, ".env not found"

    print_result(".env file", True, "Found")

    # Read relevant settings
    config = {}
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.split("#")[0].strip()  # Remove inline comments
                    if key.startswith("IBKR_"):
                        config[key] = value

    print("\n   IBKR Configuration:")
    for key, value in sorted(config.items()):
        print(f"   {key} = {value}")

    # Validate
    issues = []
    if config.get("IBKR_PORT") != "4002":
        issues.append(f"Port is {config.get('IBKR_PORT')}, expected 4002 for Gateway paper")

    if config.get("IBKR_HOST") != "127.0.0.1":
        issues.append(f"Host is {config.get('IBKR_HOST')}, expected 127.0.0.1")

    if issues:
        print("\n   ⚠️  Configuration issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False, "Configuration issues found"

    return True, "Configuration looks good"


async def main():
    """Run comprehensive diagnostics."""
    print("\n" + "=" * 70)
    print("  IBKR GATEWAY API DIAGNOSTIC TOOL")
    print("  Enhanced diagnostics for API handshake timeout issues")
    print("=" * 70)

    # Run all checks
    results = {}

    results["process"] = check_gateway_process()
    results["port"] = check_port_listening(port=4002)
    results["tcp"] = check_tcp_connection(host="127.0.0.1", port=4002)
    results["api"] = await check_api_handshake(host="127.0.0.1", port=4002, client_id=1)
    results["zombies"] = check_zombie_connections(port=4002)
    results["logs"] = check_gateway_api_logs()
    results["config"] = check_env_config()

    # Summary
    print_section("DIAGNOSTIC SUMMARY")

    all_passed = all(
        result[0]
        for result in [results["process"], results["port"], results["tcp"], results["api"]]
    )

    print(f"\nGateway Process:     {'✅' if results['process'][0] else '❌'}")
    print(f"Port Listening:      {'✅' if results['port'][0] else '❌'}")
    print(f"TCP Connection:      {'✅' if results['tcp'][0] else '❌'}")
    print(f"API Handshake:       {'✅' if results['api'][0] else '❌'}")
    zombie_count = results["zombies"][0]
    zombie_status = "✅" if zombie_count == 0 else f"⚠️  {zombie_count} found"
    print(f"Zombie Connections:  {zombie_status}")
    print(f"API Logs:            {'✅' if results['logs'][0] else '⚠️  Not enabled'}")
    print(f"Configuration:       {'✅' if results['config'][0] else '⚠️  Issues found'}")

    # Recommendations
    print_section("RECOMMENDATIONS")

    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\nYour IBKR Gateway API is working correctly.")
        print("You can now run the trading system:")
        print("  python3 -m robo_trader.runner_async --symbols AAPL")

    else:
        print("❌ ISSUES DETECTED\n")

        if not results["process"][0]:
            print("1. START IB GATEWAY")
            print("   → Launch IB Gateway application")
            print("   → Login with your IBKR credentials + 2FA")
            print()

        elif not results["port"][0]:
            print("1. CHECK GATEWAY PORT CONFIGURATION")
            print("   → In Gateway: File → Global Configuration → API → Settings")
            print("   → Verify Socket port is set to 4002 (paper) or 4001 (live)")
            print()

        elif not results["tcp"][0]:
            print("1. CHECK FIREWALL/NETWORK SETTINGS")
            print("   → Verify no firewall is blocking port 4002")
            print("   → Check macOS Security & Privacy settings")
            print()

        elif not results["api"][0]:
            print("1. CONFIGURE GATEWAY API SETTINGS (MOST LIKELY ISSUE)")
            print("   → In Gateway: File → Global Configuration → API → Settings")
            print("   → ☑️  Enable 'Enable ActiveX and Socket Clients'")
            print("   → Add 127.0.0.1 to 'Trusted IPs'")
            print("   → Set Socket port to 4002")
            print("   → Click Apply and OK")
            print("   → DO NOT restart Gateway (settings apply immediately)")
            print()
            print("2. IF STILL FAILING: RESTART GATEWAY")
            print("   → Close Gateway completely")
            print("   → Relaunch and login (requires 2FA)")
            print("   → Verify API settings again")
            print()

        if results["zombies"][0] > 0:
            print("3. CLEAN UP ZOMBIE CONNECTIONS")
            print(
                "   → Run: python3 -c 'from robo_trader.utils.robust_connection import kill_tws_zombie_connections; kill_tws_zombie_connections(4002)'"
            )
            print()

        if not results["logs"][0]:
            print("4. ENABLE API LOGGING (for debugging)")
            print("   → In Gateway: File → Global Configuration → API → Settings")
            print("   → ☑️  Create API message log file")
            print("   → Logs will be in ~/Jts/api_logs/")
            print()

    print("\n" + "=" * 70)
    print("For detailed remediation plan, see:")
    print("  IBKR_GATEWAY_TIMEOUT_REMEDIATION_PLAN.md")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
