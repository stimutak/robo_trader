"""Set DASH_PASS_HASH in .env from a password read via getpass.

The password is never printed and never echoed. SEC-B1: it is hashed with
bcrypt (salted, work-factor) and stored with a ``bcrypt$`` scheme prefix
so check_auth in app.py knows which verifier to use. Run from project root:

    .venv/bin/python scripts/_set_dashboard_password.py

Falls back to legacy unsalted SHA-256 only if passlib/bcrypt is not
installed AND --legacy-sha256 is explicitly passed.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import sys
from pathlib import Path

ENV_PATH = Path(".env")


def _bcrypt_hash(password: str) -> str | None:
    """Return a 'bcrypt$<modular-crypt-string>' hash, or None if passlib is missing."""
    try:
        from passlib.hash import bcrypt
    except ImportError:
        return None
    # cost=12 is OWASP-recommended for 2024+.
    return "bcrypt$" + bcrypt.using(rounds=12).hash(password)


def main() -> int:
    parser = argparse.ArgumentParser(description="Set the dashboard password in .env")
    parser.add_argument(
        "--legacy-sha256",
        action="store_true",
        help="Write unsalted SHA-256 instead of bcrypt. Strongly discouraged.",
    )
    args = parser.parse_args()

    if not ENV_PATH.exists():
        print(f"ERROR: {ENV_PATH} not found. Run from project root.", file=sys.stderr)
        return 1

    pw1 = getpass.getpass("Dashboard password: ")
    if not pw1:
        print("ERROR: empty password rejected.", file=sys.stderr)
        return 2
    if len(pw1) < 12:
        print(
            "ERROR: password must be at least 12 characters (bcrypt cost factor 12).",
            file=sys.stderr,
        )
        return 2
    pw2 = getpass.getpass("Confirm password: ")
    if pw1 != pw2:
        print("ERROR: passwords do not match.", file=sys.stderr)
        return 2

    if args.legacy_sha256:
        digest = hashlib.sha256(pw1.encode("utf-8")).hexdigest()
        scheme = "legacy SHA-256 (unsalted)"
    else:
        digest = _bcrypt_hash(pw1)
        if digest is None:
            print(
                "ERROR: passlib/bcrypt not installed. Install with "
                "'pip install \"passlib[bcrypt]\" bcrypt' or pass --legacy-sha256.",
                file=sys.stderr,
            )
            return 3
        scheme = "bcrypt (cost 12)"

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

    ENV_PATH.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") or not text else ""))
    try:
        os.chmod(ENV_PATH, 0o600)
    except OSError:
        pass

    print(f"DASH_PASS_HASH ({scheme}) written to {ENV_PATH.resolve()} (mode 0600).")
    print("Login at the dashboard with this password; the plaintext is gone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
