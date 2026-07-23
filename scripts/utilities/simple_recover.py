#!/usr/bin/env python3
"""Quarantined legacy database reset utility.

This script previously deleted the active SQLite database and its WAL/SHM
sidecars at import time.  That behavior is permanently unsafe for user trading
history.  A future recovery tool must provide an online backup, preview, typed
authorization, transactionality, and post-restore verification.
"""

import sys

MESSAGE = (
    "DISABLED: simple_recover.py cannot modify trading data. "
    "Use a reviewed recovery workflow with an explicit backup and user approval."
)


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
