#!/usr/bin/env python3
"""
Gateway Manager - Cross-platform IB Gateway management using IBC

This script provides unified Gateway management for RoboTrader across
macOS and Windows platforms. It uses IBC (IB Controller) to:
- Start Gateway with automated login
- Stop Gateway gracefully
- Restart Gateway (clears zombie connections)
- Check Gateway status

Operator usage:
    python3 gateway_manager.py status

Gateway lifecycle commands are reserved for supervised startup and internal
persistent-connection recovery. Operators must use ./START_TRADER.sh.

Environment Variables:
    IBKR_USERNAME - Your IBKR username (or set in config.ini)
    IBKR_PASSWORD - Your IBKR password (or set in config.ini)
    GATEWAY_VERSION - Gateway version to use (default: auto-detect latest)
"""

import argparse
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# NEW-IB-H1.1 compatibility detector retained for focused regex regression
# tests. Gateway lifecycle decisions use _ibc_safety_config_error below so
# duplicate or conflicting assignments cannot pass merely because one is good.
_READONLY_API_RE = re.compile(
    r"^[ \t]*readonlyapi[ \t]*=[ \t]*yes[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_IBC_ASSIGNMENT_RE = re.compile(r"^([^=]+)=(.*)$")

# Determine paths based on platform
PLATFORM = platform.system()  # 'Darwin' for macOS, 'Windows' for Windows

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_trader.runtime_lifecycle_lock import (  # noqa: E402
    RuntimeLifecycleLock,
    runtime_lifecycle_lock_path,
)

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
    # Unsupported platform (e.g. Linux CI). Keep the module importable so the
    # platform-independent regexes (e.g. _READONLY_API_RE) can be unit-tested;
    # main() refuses to actually run on unsupported platforms (see below).
    IBC_DIR = PROJECT_ROOT / "IBC"
    IBC_CONFIG = PROJECT_ROOT / "config" / "ibc" / "config.ini"
    IBC_LOGS = PROJECT_ROOT / "config" / "ibc" / "logs"
    GATEWAY_BASE = Path.home() / "Applications"
    GATEWAY_SETTINGS = Path.home() / "Jts"

# API ports
PAPER_PORT = 4002
LIVE_PORT = 4001

# Lifecycle commands exist only for the runner's persistent-connection recovery
# path; they are not an operator API. Recovery supplies an explicit marker and
# the child independently verifies that its parent is the trading runner.
_INTERNAL_LIFECYCLE_ENV = "ROBOTRADER_INTERNAL_GATEWAY_RECOVERY"
_INTERNAL_LIFECYCLE_VALUE = "1"
_LIFECYCLE_COMMANDS = frozenset({"start", "stop", "restart", "clear-zombies"})
_RUNTIME_LIFECYCLE_LOCK_PATH = runtime_lifecycle_lock_path()


def _parent_is_runner() -> bool:
    """Return whether the direct parent is the supervised trading runner."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(os.getppid()), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    if result.returncode != 0:
        return False
    return bool(
        re.search(
            r"(?:^|[ /])(?:robo_trader[./]runner_async|runner_async[.]py)(?:[ ]|$)",
            result.stdout,
        )
    )


def _internal_lifecycle_authorized() -> bool:
    """Return whether this process was launched by supervised recovery."""
    return (
        os.environ.get(_INTERNAL_LIFECYCLE_ENV) == _INTERNAL_LIFECYCLE_VALUE and _parent_is_runner()
    )


def _ibc_safety_config_error(config_text: str) -> Optional[str]:
    """Return a fail-closed error for an ambiguous IBC safety configuration."""
    assignments: dict[str, list[str]] = {}
    for raw_line in config_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        match = _IBC_ASSIGNMENT_RE.fullmatch(line)
        if match is None:
            continue
        key = match.group(1).strip().casefold()
        value = match.group(2).strip().casefold()
        assignments.setdefault(key, []).append(value)

    requirements = (
        ("readonlyapi", "ReadOnlyApi", "yes"),
        ("tradingmode", "TradingMode", "paper"),
    )
    errors = []
    for normalized_key, display_key, expected_value in requirements:
        values = assignments.get(normalized_key, [])
        if len(values) != 1 or values[0] != expected_value:
            errors.append(
                f"exactly one active {display_key}={expected_value} assignment is "
                f"required; found {len(values)} {display_key} assignment(s)"
            )
    return "; ".join(errors) if errors else None


def _ibc_safety_file_error(config_path: Optional[Path] = None) -> Optional[str]:
    """Read and validate the active IBC configuration without leaking it."""
    if config_path is None:
        config_path = IBC_CONFIG
    if not config_path.exists():
        return f"IBC config not found at {config_path}"
    try:
        config_text = config_path.read_text()
    except OSError as exc:
        return f"cannot read IBC config {config_path}: {exc}"
    config_error = _ibc_safety_config_error(config_text)
    if config_error:
        return f"invalid IBC paper/read-only configuration: {config_error}"
    return None


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
    """Check if the API port is accepting connections using lsof (no zombies)."""
    try:
        if PLATFORM == "Darwin":
            # Use lsof to check for LISTEN state - does NOT create zombie connections
            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        elif PLATFORM == "Windows":
            result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
            return f":{port}" in result.stdout and "LISTENING" in result.stdout
        else:
            # Fallback for other platforms - use lsof
            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and len(result.stdout.strip()) > 0
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
    if trading_mode != "paper":
        print("ERROR: live Gateway startup is disabled during remediation.")
        return False
    print(f"\n{'='*60}")
    print(f"Starting IB Gateway ({trading_mode} mode)")
    print(f"{'='*60}\n")

    config_error = _ibc_safety_file_error()
    if config_error:
        print(f"ERROR: {config_error}. Refusing to start Gateway.")
        return False

    # Check if already running
    if is_gateway_running():
        print("Gateway is already running.")
        port = PAPER_PORT if trading_mode == "paper" else LIVE_PORT
        if is_api_port_listening(port):
            print(f"API port {port} is accepting connections.")
            return True
        else:
            print(f"WARNING: Gateway running but port {port} not listening.")
            print("Run ./START_TRADER.sh to restore the supervised stack.")
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

    # Build environment
    # IBN-H2 (followup audit): full os.environ.copy() propagates ALL parent-process
    # secrets (API keys, tokens, dashboard hashes) to the IBC subprocess that
    # doesn't need them. Build a minimal allowlist of vars IBC actually requires.
    _IBC_ENV_ALLOWLIST = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "DISPLAY",
        "JAVA_HOME",
    }
    env = {k: v for k, v in os.environ.items() if k in _IBC_ENV_ALLOWLIST}
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
    if trading_mode != "paper":
        print("ERROR: live Gateway restart is disabled during remediation.")
        return False
    config_error = _ibc_safety_file_error()
    if config_error:
        print(f"ERROR: {config_error}. Refusing to restart Gateway.")
        return False
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
                print(
                    f"  WARNING: {len(gateway_zombies)} Gateway zombie(s) - "
                    "run ./START_TRADER.sh"
                )
            if python_zombies:
                print(f"  WARNING: {len(python_zombies)} Python zombie(s) - can be cleared")

    # IBC status
    print(f"\nIBC Config: {IBC_CONFIG}")
    print(f"IBC Config Exists: {IBC_CONFIG.exists()}")

    # Report the same cardinality-aware safety contract used by start/restart.
    if IBC_CONFIG.exists():
        try:
            config_text = IBC_CONFIG.read_text(errors="replace")
            config_error = _ibc_safety_config_error(config_text)
            if config_error is None:
                print("IBC Safety: ReadOnlyApi=yes, TradingMode=paper (unambiguous)")
            else:
                print(f"IBC Safety: INVALID - {config_error}")
        except Exception as e:
            print(f"IBC Safety: WARNING - could not read config: {e}")

    # Gateway version
    version = find_gateway_version()
    print(f"Gateway Version: {version or 'NOT FOUND'}")

    # Platform info
    print(f"\nPlatform: {PLATFORM}")
    print(f"IBC Directory: {IBC_DIR}")


def _dispatch_command(args: argparse.Namespace) -> int:
    """Execute an already-authorized Gateway command."""
    if args.command == "start":
        mode = "live" if args.live else "paper"
        return 0 if start_gateway(mode, args.version) else 1

    if args.command == "stop":
        return 0 if stop_gateway() else 1

    if args.command == "restart":
        mode = "live" if args.live else "paper"
        return 0 if restart_gateway(mode) else 1

    if args.command == "status":
        show_status()
        return 0

    if args.command == "clear-zombies":
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
            print("Run ./START_TRADER.sh to restore the supervised stack.")
            return 1

    return 0


def main():
    # Refuse operator lifecycle entry before argparse handles subcommand help
    # or validation. Even `gateway_manager.py restart --help` must point to the
    # sole operator lifecycle entry rather than advertise component controls.
    requested_command = sys.argv[1] if len(sys.argv) > 1 else None
    if requested_command in _LIFECYCLE_COMMANDS and not _internal_lifecycle_authorized():
        print(
            f"Refusing direct Gateway lifecycle command '{requested_command}'.\n"
            "Use ./START_TRADER.sh so preflight, paper/read-only checks, "
            "and the full supervised stack lifecycle are enforced.",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(
        description=(
            "IB Gateway status for RoboTrader. "
            "Use ./START_TRADER.sh for all lifecycle operations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    gateway_manager.py status           Show Gateway status

Gateway start, stop, restart, and zombie cleanup are managed by:
    ./START_TRADER.sh
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.add_parser("status", help="Show Gateway status")

    # Keep lifecycle commands entirely out of operator help. They are parsed
    # only after the marker and parent-process checks above authorize recovery.
    if requested_command in _LIFECYCLE_COMMANDS:
        start_parser = subparsers.add_parser("start", help=argparse.SUPPRESS)
        start_parser.add_argument("--paper", action="store_true", default=True)
        start_parser.add_argument("--live", action="store_true")
        start_parser.add_argument("--version")

        subparsers.add_parser("stop", help=argparse.SUPPRESS)

        restart_parser = subparsers.add_parser("restart", help=argparse.SUPPRESS)
        restart_parser.add_argument("--paper", action="store_true", default=True)
        restart_parser.add_argument("--live", action="store_true")

        clear_parser = subparsers.add_parser("clear-zombies", help=argparse.SUPPRESS)
        clear_parser.add_argument("--port", type=int, default=PAPER_PORT)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Keep the module importable on unsupported platforms (e.g. Linux CI),
    # while refusing any authorized lifecycle or status implementation there.
    if PLATFORM not in ("Darwin", "Windows"):
        print(f"Unsupported platform: {PLATFORM}")
        return 1

    if args.command not in _LIFECYCLE_COMMANDS:
        return _dispatch_command(args)

    lifecycle_lock = RuntimeLifecycleLock(_RUNTIME_LIFECYCLE_LOCK_PATH)
    if not lifecycle_lock.acquire():
        print(
            "Refusing concurrent Gateway recovery: another supervised runtime "
            f"lifecycle owns {_RUNTIME_LIFECYCLE_LOCK_PATH}.",
            file=sys.stderr,
        )
        return 75
    try:
        return _dispatch_command(args)
    finally:
        lifecycle_lock.release()


if __name__ == "__main__":
    sys.exit(main())
