#!/usr/bin/env python3
"""Quarantined legacy database recovery utility.

The former implementation removed the active database before attempting a
partial restore.  It is retained as an inert compatibility entrypoint until a
WAL-safe, previewable, user-authorized recovery workflow is implemented.
"""

import sys

MESSAGE = (
    "DISABLED: recover_database.py cannot replace the active database. "
    "Create and verify a backup, then use a reviewed recovery procedure."
)


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
