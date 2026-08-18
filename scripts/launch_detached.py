#!/usr/bin/env python3
"""Exec a supervised child in a new POSIX session."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        print("usage: launch_detached.py WORKING_DIRECTORY COMMAND [ARG ...]", file=sys.stderr)
        return 64

    working_directory, *command = args
    os.chdir(working_directory)
    os.setsid()
    os.execv(command[0], command)
    return 70  # pragma: no cover - os.execv replaces this process on success.


if __name__ == "__main__":
    raise SystemExit(main())
