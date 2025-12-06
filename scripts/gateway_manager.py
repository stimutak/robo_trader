#!/usr/bin/env python3
"""
Gateway Manager - Cross-platform IB Gateway management using IBC

This script provides unified Gateway management for RoboTrader across
macOS and Windows platforms. It uses IBC (IB Controller) to:
- Start Gateway with automated login
- Stop Gateway gracefully
- Restart Gateway (clears zombie connections)
- Check Gateway status

Usage:
    python3 gateway_manager.py start [--paper|--live]
    python3 gateway_manager.py stop
    python3 gateway_manager.py restart [--paper|--live]
    python3 gateway_manager.py status
    python3 gateway_manager.py clear-zombies

Environment Variables:
    IBKR_USERNAME - Your IBKR username (or set in config.ini)
    IBKR_PASSWORD - Your IBKR password (or set in config.ini)
    GATEWAY_VERSION - Gateway version to use (default: auto-detect latest)
"""

import argparse
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Determine paths based on platform
PLATFORM = platform.system()  # 'Darwin' for macOS, 'Windows' for Windows

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# IBC paths - config stored in project for portability
if PLATFORM == "Darwin":
    IBC_DIR = PROJECT_ROOT / "IBCMacos-3"
    IBC_CONFIG = PROJECT_ROOT / "config" / "ibc" / "config.ini"
    IBC_LOGS = PROJECT_ROOT / "config" / "ibc" / "logs"
    GATEWAY_BASE = Path.home() / "Applications"
    GATEWAY_SETTINGS = Path.home() / "Jts"
elif PLATFORM == "Windows":
    IBC_DIR = PROJECT_ROOT / "IBCWin-3"
    IBC_CONFIG = PROJECT_ROOT / "config" / "ibc" / "config.ini"
    IBC_LOGS = PROJECT_ROOT / "config" / "ibc" / "logs"
    GATEWAY_BASE = Path("C:/Jts")
    GATEWAY_SETTINGS = Path.home() / "Jts"
else:
    print(f"Unsupported platform: {PLATFORM}")
    sys.exit(1)

# API ports
PAPER_PORT = 4002
LIVE_PORT = 4001


def find_gateway_version() -> Optional[str]:
    """Find the installed Gateway version."""
    if PLATFORM == "Darwin":
        # Look for "IB Gateway X.XX" folders
        if GATEWAY_BASE.exists():
            versions = []
            for item in GATEWAY_BASE.iterdir():
                if item.is_dir() and item.name.startswith("IB Gateway"):
                    # Extract version like "10.41" from "IB Gateway 10.41"
                    version = item.name.replace("IB Gateway ", "")
                    versions.append(version)
            if versions:
                # Return the latest version
                versions.sort(key=lambda v: [int(x) for x in v.split(".")])
                return versions[-1]
    elif PLATFORM == "Windows":
        # Look for "ibgateway-XXX" folders
        if GATEWAY_BASE.exists():
            versions = []
            for item in GATEWAY_BASE.iterdir():
                if item.is_dir() and item.name.startswith("ibgateway-"):
                    version = item.name.replace("ibgateway-", "")
                    versions.append(version)
            if versions:
                versions.sort()
                return versions[-1]
    return None


def is_gateway_running() -> bool:
    """Check if Gateway process is running."""
    try:
        if PLATFORM == "Darwin":
            # Check for IBC-launched Gateway (IbcGateway) or direct Gateway (IB Gateway)
            result = subprocess.run(
                ["pgrep", "-f", "IbcGateway|IB Gateway"], capture_output=True, text=True
            )
            return result.returncode == 0
        elif PLATFORM == "Windows":
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq java.exe"], capture_output=True, text=True
            )
            return "ibgateway" in result.stdout.lower()
    except Exception:
        pass
    return False


def is_api_port_listening(port: int = PAPER_PORT) -> bool:
    """Check if the API port is accepting connections."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_zombie_connections(port: int = PAPER_PORT) -> list:
    """Get list of zombie CLOSE_WAIT connections on the API port."""
    zombies = []
    try:
        if PLATFORM == "Darwin":
            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:CLOSE_WAIT"], capture_output=True, text=True
            )
            if result.stdout:
                for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                    parts = line.split()
                    if len(parts) >= 2:
                        zombies.append({"command": parts[0], "pid": parts[1]})
        elif PLATFORM == "Windows":
            result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "CLOSE_WAIT" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        zombies.append({"pid": parts[-1]})
    except Exception as e:
        print(f"Error checking zombies: {e}")
    return zombies


def clear_python_zombies(port: int = PAPER_PORT) -> int:
    """Kill Python-owned zombie connections. Returns count killed."""
    killed = 0
    try:
        if PLATFORM == "Darwin":
            # Find Python processes with CLOSE_WAIT on the port
            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:CLOSE_WAIT"], capture_output=True, text=True
            )
            if result.stdout:
                for line in result.stdout.strip().split("\n")[1:]:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].lower().startswith("python"):
                        pid = int(parts[1])
                        try:
                            os.kill(pid, signal.SIGKILL)
                            killed += 1
                            print(f"  Killed Python zombie PID {pid}")
                        except ProcessLookupError:
                            pass
    except Exception as e:
        print(f"Error clearing zombies: {e}")
    return killed


def start_gateway(trading_mode: str = "paper", version: Optional[str] = None) -> bool:
    """Start Gateway using IBC."""
    print(f"\n{'='*60}")
    print(f"Starting IB Gateway ({trading_mode} mode)")
    print(f"{'='*60}\n")

    # Check if already running
    if is_gateway_running():
        print("Gateway is already running.")
        port = PAPER_PORT if trading_mode == "paper" else LIVE_PORT
        if is_api_port_listening(port):
            print(f"API port {port} is accepting connections.")
            return True
        else:
            print(f"WARNING: Gateway running but port {port} not listening.")
            print("Consider restarting Gateway.")
            return False

    # Find Gateway version
    if not version:
        version = os.environ.get("GATEWAY_VERSION") or find_gateway_version()
    if not version:
        print("ERROR: Could not find installed Gateway version.")
        print(f"Please install IB Gateway in {GATEWAY_BASE}")
        return False

    print(f"Using Gateway version: {version}")

    # Ensure IBC directories exist
    IBC_LOGS.mkdir(parents=True, exist_ok=True)

    # Check IBC config
    if not IBC_CONFIG.exists():
        print(f"ERROR: IBC config not found at {IBC_CONFIG}")
        print("Run: python3 scripts/gateway_manager.py setup")
        return False

    # Build environment
    env = os.environ.copy()
    env["TWS_MAJOR_VRSN"] = version
    env["IBC_INI"] = str(IBC_CONFIG)
    env["TRADING_MODE"] = trading_mode
    env["TWOFA_TIMEOUT_ACTION"] = "restart"
    env["IBC_PATH"] = str(IBC_DIR)
    env["TWS_PATH"] = str(GATEWAY_BASE)
    env["LOG_PATH"] = str(IBC_LOGS)

    # Pass credentials from environment if set
    if os.environ.get("IBKR_USERNAME"):
        env["TWSUSERID"] = os.environ["IBKR_USERNAME"]
    if os.environ.get("IBKR_PASSWORD"):
        env["TWSPASSWORD"] = os.environ["IBKR_PASSWORD"]

    # Start Gateway
    if PLATFORM == "Darwin":
        script = IBC_DIR / "gatewaystartmacos.sh"
        if not script.exists():
            print(f"ERROR: IBC script not found: {script}")
            return False

        # Make executable
        script.chmod(0o755)
        (IBC_DIR / "scripts" / "displaybannerandlaunch.sh").chmod(0o755)
        (IBC_DIR / "scripts" / "ibcstart.sh").chmod(0o755)

        print("Starting Gateway via IBC...")
        print("(A new Terminal window will open)")
        print("")
        print("After Gateway starts and you complete 2FA:")
        print(f"  - Wait for Gateway to show 'IB Gateway - READY'")
        print(f"  - Then run: ./START_TRADER.sh")
        print("")

        # Run the start script
        subprocess.Popen([str(script)], env=env, cwd=str(IBC_DIR))

    elif PLATFORM == "Windows":
        script = IBC_DIR / "StartGateway.bat"
        if not script.exists():
            print(f"ERROR: IBC script not found: {script}")
            return False

        print("Starting Gateway via IBC...")
        subprocess.Popen(
            ["cmd", "/c", str(script)],
            env=env,
            cwd=str(IBC_DIR),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )

    # Wait for startup
    print("\nWaiting for Gateway to start...")
    port = PAPER_PORT if trading_mode == "paper" else LIVE_PORT
    for i in range(60):  # Wait up to 60 seconds
        if is_api_port_listening(port):
            print(f"\nGateway started successfully! API port {port} is ready.")
            return True
        time.sleep(1)
        if i % 10 == 0:
            print(f"  Waiting... ({i}s)")

    print("\nGateway start initiated. Complete 2FA in the Gateway window.")
    return True


def stop_gateway() -> bool:
    """Stop Gateway gracefully using IBC command server."""
    print("\n" + "=" * 60)
    print("Stopping IB Gateway")
    print("=" * 60 + "\n")

    if not is_gateway_running():
        print("Gateway is not running.")
        return True

    # Try IBC command server first (graceful shutdown)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(("127.0.0.1", 7462))  # IBC command port
        sock.send(b"STOP\n")
        response = sock.recv(1024)
        sock.close()
        print(f"IBC response: {response.decode().strip()}")
        print("Gateway stopping gracefully...")

        # Wait for shutdown
        for i in range(30):
            if not is_gateway_running():
                print("Gateway stopped successfully.")
                return True
            time.sleep(1)

    except Exception as e:
        print(f"IBC command server not available: {e}")
        print("Falling back to process termination...")

    # Fallback: kill the process
    try:
        if PLATFORM == "Darwin":
            # Kill IBC-launched Gateway (IbcGateway) or direct Gateway (IB Gateway)
            subprocess.run(["pkill", "-f", "IbcGateway"], check=False)
            subprocess.run(["pkill", "-f", "IB Gateway"], check=False)
        elif PLATFORM == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/IM", "java.exe", "/FI", "WINDOWTITLE eq *IB Gateway*"],
                check=False,
            )

        time.sleep(2)
        if not is_gateway_running():
            print("Gateway terminated.")
            return True

    except Exception as e:
        print(f"Error stopping Gateway: {e}")

    return False


def restart_gateway(trading_mode: str = "paper") -> bool:
    """Restart Gateway (clears all zombie connections)."""
    print("\n" + "=" * 60)
    print("Restarting IB Gateway")
    print("=" * 60 + "\n")

    print("This will clear all zombie connections.")
    print("You will need to complete 2FA again.\n")

    # Stop Gateway
    stop_gateway()

    # Wait a moment
    time.sleep(3)

    # Clear any remaining zombies
    port = PAPER_PORT if trading_mode == "paper" else LIVE_PORT
    zombies = get_zombie_connections(port)
    if zombies:
        print(f"Found {len(zombies)} zombie connection(s) after stop.")
        print("These will be cleared when Gateway restarts.")

    # Start Gateway
    return start_gateway(trading_mode)


def show_status():
    """Show Gateway status information."""
    print("\n" + "=" * 60)
    print("IB Gateway Status")
    print("=" * 60 + "\n")

    # Process status
    running = is_gateway_running()
    print(f"Gateway Process: {'RUNNING' if running else 'NOT RUNNING'}")

    # Port status
    for name, port in [("Paper", PAPER_PORT), ("Live", LIVE_PORT)]:
        listening = is_api_port_listening(port)
        zombies = get_zombie_connections(port)
        status = "LISTENING" if listening else "NOT LISTENING"
        print(f"{name} API Port ({port}): {status}")
        if zombies:
            gateway_zombies = [
                z for z in zombies if not z.get("command", "").lower().startswith("python")
            ]
            python_zombies = [
                z for z in zombies if z.get("command", "").lower().startswith("python")
            ]
            if gateway_zombies:
                print(f"  WARNING: {len(gateway_zombies)} Gateway zombie(s) - RESTART GATEWAY")
            if python_zombies:
                print(f"  WARNING: {len(python_zombies)} Python zombie(s) - can be cleared")

    # IBC status
    print(f"\nIBC Config: {IBC_CONFIG}")
    print(f"IBC Config Exists: {IBC_CONFIG.exists()}")

    # Gateway version
    version = find_gateway_version()
    print(f"Gateway Version: {version or 'NOT FOUND'}")

    # Platform info
    print(f"\nPlatform: {PLATFORM}")
    print(f"IBC Directory: {IBC_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="IB Gateway Manager for RoboTrader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    gateway_manager.py start --paper    Start Gateway in paper trading mode
    gateway_manager.py restart          Restart Gateway (clears zombies)
    gateway_manager.py status           Show Gateway status
    gateway_manager.py clear-zombies    Kill Python zombie connections
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start Gateway")
    start_parser.add_argument(
        "--paper", action="store_true", default=True, help="Use paper trading mode (default)"
    )
    start_parser.add_argument("--live", action="store_true", help="Use live trading mode")
    start_parser.add_argument("--version", help="Gateway version to use")

    # Stop command
    subparsers.add_parser("stop", help="Stop Gateway")

    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart Gateway")
    restart_parser.add_argument(
        "--paper", action="store_true", default=True, help="Use paper trading mode (default)"
    )
    restart_parser.add_argument("--live", action="store_true", help="Use live trading mode")

    # Status command
    subparsers.add_parser("status", help="Show Gateway status")

    # Clear zombies command
    clear_parser = subparsers.add_parser("clear-zombies", help="Clear Python zombie connections")
    clear_parser.add_argument("--port", type=int, default=PAPER_PORT, help="API port to check")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "start":
        mode = "live" if args.live else "paper"
        success = start_gateway(mode, args.version)
        return 0 if success else 1

    elif args.command == "stop":
        success = stop_gateway()
        return 0 if success else 1

    elif args.command == "restart":
        mode = "live" if args.live else "paper"
        success = restart_gateway(mode)
        return 0 if success else 1

    elif args.command == "status":
        show_status()
        return 0

    elif args.command == "clear-zombies":
        port = args.port
        zombies = get_zombie_connections(port)
        if not zombies:
            print(f"No zombie connections found on port {port}")
            return 0

        print(f"Found {len(zombies)} zombie connection(s) on port {port}")
        killed = clear_python_zombies(port)
        print(f"Killed {killed} Python zombie(s)")

        remaining = get_zombie_connections(port)
        gateway_zombies = len(remaining) - killed
        if gateway_zombies > 0:
            print(f"\nWARNING: {gateway_zombies} Gateway zombie(s) remain.")
            print("These can only be cleared by restarting Gateway.")
            print("Run: python3 scripts/gateway_manager.py restart")
            return 1

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
