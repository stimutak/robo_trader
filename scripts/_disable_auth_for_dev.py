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

# Bootstrap: allow `from _atomic_env import write_env_atomic` regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _atomic_env import write_env_atomic  # noqa: E402

ENV_PATH = Path(".env")

# Markers wrap the auto-emitted dev note so we can find and remove only OUR
# block on subsequent runs without eating unrelated comments.
_BEGIN = "# >>> ROBOTRADER DEV-AUTH NOTE BEGIN >>>"
_END = "# <<< ROBOTRADER DEV-AUTH NOTE END <<<"

DEV_AUTH_NOTE_LINES = [
    _BEGIN,
    "# TEMPORARY: dashboard auth disabled for local dev. Re-enable by setting",
    "# DASH_AUTH_ENABLED=true and running scripts/_set_dashboard_password.py",
    "# BEFORE binding to a non-loopback host or running live trading.",
    _END,
]


def _strip_existing_note(lines: list[str]) -> list[str]:
    """Remove any prior copy of our dev-auth note using a state machine.

    Only lines between (inclusive) `_BEGIN` and `_END` markers are dropped,
    so unrelated `# TEMPORARY:` comments elsewhere in .env are preserved.
    """
    out: list[str] = []
    inside_marker = False
    for line in lines:
        s = line.strip()
        if not inside_marker and s == _BEGIN:
            inside_marker = True
            continue
        if inside_marker:
            if s == _END:
                inside_marker = False
            # Inside marker block (or on the END line itself): drop.
            continue
        out.append(line)
    return out


def main() -> int:
    # Refuse sudo: .env would end up root-owned.
    try:
        if os.geteuid() == 0 and os.environ.get("SUDO_USER"):
            sys.stderr.write(
                "ERROR: do not run this script with sudo. The .env file will "
                "be owned by root.\nRun as your normal user.\n"
            )
            return 2
    except AttributeError:
        pass

    if not ENV_PATH.exists():
        print(f"ERROR: {ENV_PATH} not found.", file=sys.stderr)
        return 1

    raw_lines = ENV_PATH.read_text().splitlines()
    # Pre-clean any prior dev-auth note block so we don't accumulate copies.
    lines = _strip_existing_note(raw_lines)

    new_lines: list[str] = []
    flipped = False
    note_emitted = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("DASH_AUTH_ENABLED="):
            new_lines.extend(DEV_AUTH_NOTE_LINES)
            new_lines.append("DASH_AUTH_ENABLED=false")
            flipped = True
            note_emitted = True
            continue
        new_lines.append(line)

    if not flipped:
        # Key wasn't present — append at end with note.
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.extend(DEV_AUTH_NOTE_LINES)
        new_lines.append("DASH_AUTH_ENABLED=false")
        note_emitted = True

    write_env_atomic(ENV_PATH, "\n".join(new_lines) + "\n")

    print("Done. .env now has DASH_AUTH_ENABLED=false with a TEMPORARY note.")
    print("Revert with:  .venv/bin/python scripts/_set_dashboard_password.py")
    print("              (or manually flip the flag back to true after setting the hash)")
    return 0 if note_emitted else 2


if __name__ == "__main__":
    sys.exit(main())
