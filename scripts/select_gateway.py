#!/usr/bin/env python3
"""Emit read-only Gateway selection fields; never launch an application."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import dotenv_values  # noqa: E402

from robo_trader.gateway_selection import (  # noqa: E402
    GatewaySelectionError,
    assert_running_gateway_matches,
    select_gateway,
)


def main() -> int:
    try:
        file_env = dotenv_values(PROJECT_ROOT / ".env")
        keys = ("GATEWAY_VERSION", "ROBOTRADER_IBC_PATH")
        if any(key in file_env and file_env[key] is None for key in keys):
            raise GatewaySelectionError("Gateway selection configuration has a missing value")
        env = {key: file_env[key] for key in keys if key in file_env}
        env.update({key: os.environ[key] for key in keys if key in os.environ})
        selection = select_gateway(PROJECT_ROOT, Path.home() / "Applications", env)
        assert_running_gateway_matches(selection)
        print(f"{selection.version}\t{selection.ibc_path}\t{selection.ibc_version}")
        return 0
    except (GatewaySelectionError, OSError) as exc:
        print(f"Gateway selection blocked: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
