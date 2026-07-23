#!/usr/bin/env python3
"""Quarantined legacy IBKR monitor.

This utility used to probe a TWS-era port and restart the runner outside the
supervised startup path.  That topology is unsafe and unsupported.
"""

import sys


def main() -> int:
    """Refuse to create an alternate monitoring or restart authority."""
    print(
        "DISABLED: the legacy IBKR connection monitor is quarantined. "
        "Use ./START_TRADER.sh for the supervised paper-only system; "
        "use 'python3 scripts/gateway_manager.py status' for read-only diagnostics.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
