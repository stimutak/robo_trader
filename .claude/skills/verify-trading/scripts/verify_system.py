#!/usr/bin/env python3
"""
Trading System Verification Script

Runs comprehensive health checks on the trading system.
Output is JSON for easy parsing by Claude.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


# Find project root by searching for marker files
def find_project_root() -> Path:
    """Find project root by looking for marker files (.git, pyproject.toml, CLAUDE.md)."""
    current = Path(__file__).resolve().parent
    markers = [".git", "pyproject.toml", "CLAUDE.md"]

    for _ in range(10):  # Max 10 levels up
        for marker in markers:
            if (current / marker).exists():
                return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent

    # Fallback to old method if markers not found
    return Path(__file__).parent.parent.parent.parent.parent


PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)

# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))


def run_cmd(cmd: str, timeout: int = 10) -> tuple[int, str]:
    """Run a shell command and return (returncode, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )  # nosec B602 - commands are hardcoded, not user input
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, str(e)


def check_processes() -> dict:
    """Check if required processes are running."""
    code, output = run_cmd(
        "ps aux | grep -E '(runner_async|app.py|websocket_server)' | grep -v grep"
    )

    runner = "runner_async" in output
    websocket = "websocket_server" in output
    dashboard = "app.py" in output

    return {
        "status": "pass" if runner else "fail",
        "runner_async": runner,
        "websocket_server": websocket,
        "dashboard": dashboard,
        "critical": not runner,  # Runner is critical
    }


def check_gateway() -> dict:
    """Check Gateway status."""
    code, output = run_cmd("python3 scripts/gateway_manager.py status 2>&1")

    running = "running" in output.lower() or "listening" in output.lower()
    return {"status": "pass" if running else "fail", "output": output.strip()[:200]}


def check_zombies() -> dict:
    """Check for zombie CLOSE_WAIT connections."""
    code, output = run_cmd("lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT 2>/dev/null | wc -l")

    try:
        count = int(output.strip()) if output and output.strip() else 0
    except (ValueError, AttributeError):
        count = 0

    return {"status": "pass" if count == 0 else "fail", "count": count}


def check_risk_params() -> dict:
    """Check .env for required risk parameters."""
    env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        return {"status": "fail", "error": ".env file not found"}

    try:
        env_content = env_path.read_text()
    except (PermissionError, UnicodeDecodeError, OSError) as e:
        return {"status": "fail", "error": f"Cannot read .env: {e}"}

    # Use actual env var names from .env file
    required = ["RISK_MAX_OPEN_POSITIONS", "STOP_LOSS_PERCENT", "EXECUTION_MODE"]
    found = {}

    for param in required:
        for line in env_content.split("\n"):
            if line.startswith(f"{param}="):
                found[param] = line.split("=", 1)[1].strip()
                break
        else:
            found[param] = None

    all_present = all(v is not None for v in found.values())

    return {"status": "pass" if all_present else "warn", "params": found}


def check_market_hours() -> dict:
    """Check market hours logic."""
    # Try using venv python first, fall back to system python
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
    python_cmd = str(venv_python) if venv_python.exists() else "python3"

    code, output = run_cmd(
        f'cd {PROJECT_ROOT} && {python_cmd} -c "'
        "import sys; sys.path.insert(0, '.'); "
        "from robo_trader.market_hours import is_market_open, get_market_session; "
        "import json; print(json.dumps({'open': is_market_open(), 'session': get_market_session()}))\""
    )

    if code == 0:
        try:
            data = json.loads(output.strip())
            return {"status": "pass", **data}
        except json.JSONDecodeError:
            pass

    # Gracefully handle missing dependencies
    if "ModuleNotFoundError" in output:
        return {"status": "warn", "note": "Missing dependencies (run in venv)"}

    return {"status": "fail", "error": output.strip()[:200]}


def check_recent_errors() -> dict:
    """Check for recent errors in logs."""
    log_path = PROJECT_ROOT / "robo_trader.log"

    if not log_path.exists():
        return {"status": "pass", "count": 0, "note": "No log file"}

    code, output = run_cmd(
        "tail -100 robo_trader.log | grep -cE '(ERROR|CRITICAL|Exception)' || echo 0"
    )

    try:
        count = int(output.strip())
    except ValueError:
        count = 0

    return {
        "status": "pass" if count < 5 else "warn" if count < 20 else "fail",
        "count": count,
    }


def main():
    """Run all checks and output results."""
    results = {
        "processes": check_processes(),
        "gateway": check_gateway(),
        "zombies": check_zombies(),
        "risk_params": check_risk_params(),
        "market_hours": check_market_hours(),
        "recent_errors": check_recent_errors(),
    }

    # Determine overall status
    critical_fail = results["processes"].get("critical", False)
    any_fail = any(r.get("status") == "fail" for r in results.values())
    any_warn = any(r.get("status") == "warn" for r in results.values())

    if critical_fail:
        overall = "NOT_READY_CRITICAL"
    elif any_fail:
        overall = "NOT_READY"
    elif any_warn:
        overall = "READY_WITH_WARNINGS"
    else:
        overall = "READY"

    results["overall"] = overall

    print(json.dumps(results, indent=2))

    # Exit code based on status
    if overall.startswith("NOT_READY"):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
