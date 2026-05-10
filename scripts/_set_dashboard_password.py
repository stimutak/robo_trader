"""Set DASH_PASS_HASH in .env from a password read via getpass.

The password is never printed and never echoed. It is hashed with SHA-256
and only the hex digest is written. Run from project root:

    .venv/bin/python scripts/_set_dashboard_password.py
"""

from __future__ import annotations

import getpass
import hashlib
import os
import sys
from pathlib import Path

ENV_PATH = Path(".env")


def main() -> int:
    if not ENV_PATH.exists():
        print(f"ERROR: {ENV_PATH} not found. Run from project root.", file=sys.stderr)
        return 1

    pw1 = getpass.getpass("Dashboard password: ")
    if not pw1:
        print("ERROR: empty password rejected.", file=sys.stderr)
        return 2
    if len(pw1) < 8:
        print("ERROR: password must be at least 8 characters.", file=sys.stderr)
        return 2
    pw2 = getpass.getpass("Confirm password: ")
    if pw1 != pw2:
        print("ERROR: passwords do not match.", file=sys.stderr)
        return 2

    digest = hashlib.sha256(pw1.encode("utf-8")).hexdigest()

    # Read .env, replace DASH_PASS_HASH line.
    text = ENV_PATH.read_text()
    new_lines = []
    found = False
    for line in text.splitlines():
        if line.strip().startswith("DASH_PASS_HASH="):
            new_lines.append(f"DASH_PASS_HASH={digest}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"DASH_PASS_HASH={digest}")

    ENV_PATH.write_text(
        "\n".join(new_lines) + ("\n" if text.endswith("\n") or not text else "")
    )
    try:
        os.chmod(ENV_PATH, 0o600)
    except OSError:
        pass

    print(f"DASH_PASS_HASH written to {ENV_PATH.resolve()} (mode 0600).")
    print("Login at the dashboard with this password; the plaintext is gone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
