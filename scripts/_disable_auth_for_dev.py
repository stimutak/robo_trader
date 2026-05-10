"""Flip DASH_AUTH_ENABLED=false in .env with a TEMPORARY comment.

Used to bring the dashboard up locally without setting a password hash. This
must be reverted before any non-loopback exposure or live trading. See
DEV_SETUP.md.

Reads/writes .env without echoing values. Run from project root:
    .venv/bin/python scripts/_disable_auth_for_dev.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_PATH = Path(".env")

DEV_AUTH_NOTE = (
    "# TEMPORARY: dashboard auth disabled for local dev. Re-enable by setting\n"
    "# DASH_AUTH_ENABLED=true and running scripts/_set_dashboard_password.py\n"
    "# BEFORE binding to a non-loopback host or running live trading.\n"
)


def main() -> int:
    if not ENV_PATH.exists():
        print(f"ERROR: {ENV_PATH} not found.", file=sys.stderr)
        return 1

    lines = ENV_PATH.read_text().splitlines()
    new_lines: list[str] = []
    flipped = False
    note_emitted = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("DASH_AUTH_ENABLED="):
            # Drop any prior copy of the dev note that might already be above
            # (avoid duplicates on repeated runs).
            while new_lines and new_lines[-1].startswith("# TEMPORARY:"):
                new_lines.pop()
            while new_lines and new_lines[-1].startswith("# DASH_AUTH_ENABLED=true"):
                new_lines.pop()
            while new_lines and new_lines[-1].startswith("# BEFORE"):
                new_lines.pop()
            new_lines.extend(DEV_AUTH_NOTE.rstrip("\n").splitlines())
            new_lines.append("DASH_AUTH_ENABLED=false")
            flipped = True
            note_emitted = True
            continue
        new_lines.append(line)

    if not flipped:
        # Key wasn't present — append at end with note.
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.extend(DEV_AUTH_NOTE.rstrip("\n").splitlines())
        new_lines.append("DASH_AUTH_ENABLED=false")
        note_emitted = True

    ENV_PATH.write_text("\n".join(new_lines) + "\n")
    try:
        os.chmod(ENV_PATH, 0o600)
    except OSError:
        pass

    print("Done. .env now has DASH_AUTH_ENABLED=false with a TEMPORARY note.")
    print("Revert with:  .venv/bin/python scripts/_set_dashboard_password.py")
    print("              (or manually flip the flag back to true after setting the hash)")
    return 0 if note_emitted else 2


if __name__ == "__main__":
    sys.exit(main())
