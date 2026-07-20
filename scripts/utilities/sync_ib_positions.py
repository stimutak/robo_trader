#!/usr/bin/env python3
"""Quarantined legacy IBKR position replacement utility.

The former implementation deleted every local position before inserting the
latest broker response.  It was neither portfolio-scoped nor safe when the
broker response was empty or partial.  Position reconciliation must remain a
read-only diff until the durable ledger/reconciliation PRs are complete.
"""

import sys

MESSAGE = (
    "DISABLED: sync_ib_positions.py cannot overwrite local positions. "
    "Use read-only broker/database reconciliation and resolve differences explicitly."
)


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
