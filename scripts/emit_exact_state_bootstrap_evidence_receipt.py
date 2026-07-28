#!/usr/bin/env python3
"""Emit one producer-owned Ed25519 receipt for bootstrap evidence.

Run this in the evidence producer's isolated environment.  The bootstrap
consumer must receive only the matching public key and refuses private-key
capability in its own environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_trader.bootstrap_evidence_auth import (  # noqa: E402
    BootstrapEvidenceAuthenticationError,
    emit_broker_snapshot_receipt,
    emit_protective_mark_receipt,
    emit_reconciliation_report_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind",
        choices=("broker_snapshot", "reconciliation_report", "protective_mark"),
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--runtime-fingerprint", required=True)
    parser.add_argument("--account-scope", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        producer = {
            "broker_snapshot": emit_broker_snapshot_receipt,
            "reconciliation_report": emit_reconciliation_report_receipt,
            "protective_mark": emit_protective_mark_receipt,
        }[args.kind]
        receipt = producer(
            artifact_path=args.artifact,
            private_key_path=args.private_key,
            runtime_fingerprint=args.runtime_fingerprint,
            account_scope=args.account_scope,
        )
        print(
            json.dumps(
                {
                    "artifact_kind": args.kind,
                    "authorizes_startup": False,
                    "receipt_path": str(receipt),
                    "schema_version": 1,
                    "status": "PRODUCER_RECEIPT_EMITTED",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except BootstrapEvidenceAuthenticationError as exc:
        print(
            json.dumps(
                {
                    "authorizes_startup": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "schema_version": 1,
                    "status": "BLOCKED",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
