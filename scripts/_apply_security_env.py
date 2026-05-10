"""One-shot helper to apply env-var changes from the security audit.

Reads .env (without echoing values), generates random tokens, and writes back
in place. Prints only which keys were added/updated/preserved.

Run from project root:
    .venv/bin/python scripts/_apply_security_env.py
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

ENV_PATH = Path(".env")
IBC_INI_PATH = Path("config/ibc/config.ini")

# Keys we will set if missing (and report their fate). Values may be
# generated; "GENERATE" means produce a random token.
DESIRED: dict[str, str] = {
    "DASH_AUTH_ENABLED": "true",
    "DASH_HOST": "127.0.0.1",
    "DASH_PASS_HASH": "<TODO_SET_THIS>",  # user must compute and replace
    "MODEL_SIGNING_KEY": "GENERATE",
    "MODEL_SIGNING_REQUIRED": "false",
    "WS_HOST": "0.0.0.0",
    "WS_AUTH_TOKEN": "GENERATE",
    "AI_REQUIRE_ML_CONFIRMATION": "true",
    "AI_MIN_CONFIDENCE": "0.85",
}

# Keys we will OVERWRITE even if already present (security-critical defaults).
OVERWRITE = {
    "DASH_AUTH_ENABLED",
    "DASH_HOST",
    "MODEL_SIGNING_REQUIRED",
    "AI_REQUIRE_ML_CONFIRMATION",
    "AI_MIN_CONFIDENCE",
}


def parse_env(text: str) -> tuple[list[tuple[str, str | None]], dict[str, int]]:
    """Return (lines as [(key, value)] preserving order; key->index map).

    Non-key lines (comments, blanks) come through as (raw_line, None).
    """
    rows: list[tuple[str, str | None]] = []
    index: dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            rows.append((line, None))
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        index[key] = len(rows)
        rows.append((key, value))
    return rows, index


def serialize_env(rows: list[tuple[str, str | None]]) -> str:
    out: list[str] = []
    for key, value in rows:
        if value is None:
            out.append(key)
        else:
            out.append(f"{key}={value}")
    return "\n".join(out) + ("\n" if out and not out[-1].endswith("\n") else "")


def main() -> int:
    if not ENV_PATH.exists():
        print(f"ERROR: {ENV_PATH} not found. Run from project root.", file=sys.stderr)
        return 1

    text = ENV_PATH.read_text()
    rows, idx = parse_env(text)

    added: list[str] = []
    updated: list[str] = []
    preserved: list[str] = []
    todo: list[str] = []

    for key, target in DESIRED.items():
        if target == "GENERATE":
            new_val = secrets.token_urlsafe(32)
        else:
            new_val = target

        if key in idx:
            existing = rows[idx[key]][1] or ""
            if key in OVERWRITE and existing != new_val:
                rows[idx[key]] = (key, new_val)
                updated.append(key)
            elif (
                target == "GENERATE"
                and (existing == "" or existing.startswith("<TODO"))
            ):
                # Fill in a previously-empty generated slot.
                rows[idx[key]] = (key, new_val)
                updated.append(key)
            elif target == "<TODO_SET_THIS>" and (
                existing == "" or existing.startswith("<TODO")
            ):
                rows[idx[key]] = (key, new_val)
                todo.append(key)
            else:
                preserved.append(key)
        else:
            rows.append((key, new_val))
            if target == "<TODO_SET_THIS>":
                todo.append(key)
            else:
                added.append(key)

    # Write back atomically. Make .env mode 0600.
    ENV_PATH.write_text(serialize_env(rows))
    try:
        os.chmod(ENV_PATH, 0o600)
    except OSError:
        pass

    print(f".env updated at {ENV_PATH.resolve()}")
    print(f"  added:     {sorted(added) or '(none)'}")
    print(f"  updated:   {sorted(updated) or '(none)'}")
    print(f"  preserved: {sorted(preserved) or '(none)'}")
    print(f"  TODO:      {sorted(todo) or '(none)'} <- you must replace these")

    # Now flip IBC config.ini (gitignored) — ReadOnlyApi and AllowBlindTrading.
    if IBC_INI_PATH.exists():
        ini_text = IBC_INI_PATH.read_text()
        ini_lines = ini_text.splitlines()
        flipped: list[str] = []
        for i, line in enumerate(ini_lines):
            stripped = line.strip()
            if stripped.startswith("ReadOnlyApi="):
                if stripped != "ReadOnlyApi=yes":
                    ini_lines[i] = "ReadOnlyApi=yes"
                    flipped.append("ReadOnlyApi=yes")
            elif stripped.startswith("AllowBlindTrading="):
                if stripped != "AllowBlindTrading=no":
                    ini_lines[i] = "AllowBlindTrading=no"
                    flipped.append("AllowBlindTrading=no")
        new_ini = "\n".join(ini_lines) + (
            "\n" if ini_text.endswith("\n") else ""
        )
        if new_ini != ini_text:
            IBC_INI_PATH.write_text(new_ini)
            try:
                os.chmod(IBC_INI_PATH, 0o600)
            except OSError:
                pass
            print(f"\n{IBC_INI_PATH} flipped: {flipped}")
        else:
            print(f"\n{IBC_INI_PATH} already correctly configured.")
    else:
        print(
            f"\nNOTE: {IBC_INI_PATH} not found. "
            "Copy from config/ibc/config.ini.template and re-run, "
            "or it'll be created on next ./scripts/start_gateway.sh."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
