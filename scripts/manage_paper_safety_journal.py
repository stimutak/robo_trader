#!/usr/bin/env python3
"""Generate paper safety identity and explicitly provision its empty journal.

This command never edits ``.env``.  Journal creation is a one-time, separately
confirmed operator action; normal trader startup only replays an existing
journal and will never create or repair one.
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_trader.config import (  # noqa: E402
    RuntimeContract,
    _derive_safety_account_scope,
    load_runtime_contract_from_env,
)
from robo_trader.safety import (  # noqa: E402
    PaperExecutionIdentity,
    SafetyJournal,
    SafetyRuntimeCoordinator,
)

CREATE_CONFIRMATION = "CREATE-EMPTY-PAPER-SAFETY-JOURNAL"


def generate_account_scope(account: str) -> tuple[str, str]:
    """Return one fresh secret key and its exact account-bound scope."""

    normalized_account = str(account).strip()
    if not normalized_account:
        raise ValueError("IBKR_ACCOUNT is required to generate a bound scope")
    key = secrets.token_hex(32)
    return key, _derive_safety_account_scope(key, normalized_account)


def _resolved_environment() -> dict[str, str]:
    values = dotenv_values(PROJECT_ROOT / ".env")
    malformed = sorted(key for key, value in values.items() if value is None)
    if malformed:
        raise ValueError("malformed .env entries: " + ", ".join(malformed))
    import os

    resolved = {key: value for key, value in values.items() if value is not None}
    resolved.update(os.environ)
    return resolved


def _paper_contract(environ: Mapping[str, str]) -> RuntimeContract:
    contract = load_runtime_contract_from_env(environ)
    if contract.execution_mode != "paper":
        raise ValueError("paper safety journal management requires EXECUTION_MODE=paper")
    if not contract.safety_journal_path:
        raise ValueError("SAFETY_JOURNAL_PATH is required")
    return contract


def verify_journal(environ: Mapping[str, str]) -> RuntimeContract:
    """Read-only identity-bound replay with startup quarantine enforcement."""

    contract = _paper_contract(environ)
    identity = PaperExecutionIdentity(
        contract.safety_execution_domain_scope,
        contract.safety_account_scope,
    )
    SafetyRuntimeCoordinator(
        identity,
        SafetyJournal(contract.safety_journal_path),
    ).start()
    return contract


def initialize_journal(
    environ: Mapping[str, str],
    *,
    confirmation: str,
) -> RuntimeContract:
    """Create one new empty journal after exact typed confirmation."""

    if confirmation != CREATE_CONFIRMATION:
        raise ValueError(f"confirmation must be exactly {CREATE_CONFIRMATION}")
    contract = _paper_contract(environ)
    path = Path(contract.safety_journal_path).expanduser()
    if path.exists() or path.is_symlink():
        raise FileExistsError("configured safety journal already exists; refusing to modify it")
    if not path.parent.is_dir():
        raise FileNotFoundError(
            "safety journal parent directory does not exist; create and permission it explicitly"
        )
    journal = SafetyJournal(path)
    journal.initialize(
        execution_domain_scope=contract.safety_execution_domain_scope,
        account_scope=contract.safety_account_scope,
    )
    SafetyRuntimeCoordinator(
        PaperExecutionIdentity(
            contract.safety_execution_domain_scope,
            contract.safety_account_scope,
        ),
        journal,
    ).start()
    return contract


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "generate-scope",
        help=(
            "Print a new SAFETY_ACCOUNT_SCOPE_KEY and matching account-bound "
            "SAFETY_ACCOUNT_SCOPE; never edits .env."
        ),
    )
    subparsers.add_parser("verify", help="Read-only replay of the configured journal.")
    initialize = subparsers.add_parser(
        "initialize",
        help="Create a new empty configured journal; refuses any existing path.",
    )
    initialize.add_argument("--confirm", required=True, metavar=CREATE_CONFIRMATION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate-scope":
            environ = _resolved_environment()
            key, scope = generate_account_scope(environ.get("IBKR_ACCOUNT", ""))
            print(f"SAFETY_ACCOUNT_SCOPE_KEY={key}")
            print(f"SAFETY_ACCOUNT_SCOPE={scope}")
            return 0
        environ = _resolved_environment()
        if args.command == "verify":
            contract = verify_journal(environ)
            print("Paper safety journal verified: " f"{contract.safety_journal_identity}")
            return 0
        contract = initialize_journal(environ, confirmation=args.confirm)
        print(
            "Created and verified empty paper safety journal: "
            f"{contract.safety_journal_identity}"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
