#!/usr/bin/env python3
"""Dormant SQLite backup, verification, and clean-room restore CLI.

Every command is local-file-only, emits ``authorizes_startup=false``, and
refuses to replace an existing output path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_trader.maintenance import SQLiteMaintenanceError, SQLiteMaintenanceService  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    backup = subparsers.add_parser("backup", help="create a verified online backup")
    backup.add_argument("--source", type=Path, required=True)
    backup.add_argument("--target", type=Path, required=True)
    backup.add_argument("--manifest", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify a database and optional manifest")
    verify.add_argument("--database", type=Path, required=True)
    verify.add_argument("--manifest", type=Path)

    restore = subparsers.add_parser(
        "restore-clean-room", help="restore only into a new clean-room database"
    )
    restore.add_argument("--backup", type=Path, required=True)
    restore.add_argument("--backup-manifest", type=Path, required=True)
    restore.add_argument("--target", type=Path, required=True)
    restore.add_argument("--restore-manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    service = SQLiteMaintenanceService()
    try:
        if args.command == "backup":
            service.assert_report_paths_disjoint(
                database_paths=(args.source, args.target),
                report_paths=(args.manifest,),
            )
            reservation = service.reserve_manifest(args.manifest)
            try:
                backup_manifest = service.backup(args.source, args.target)
                service.write_reserved_manifest(backup_manifest, reservation)
            finally:
                reservation.close()
            payload = backup_manifest.to_dict()
        elif args.command == "verify":
            if args.manifest:
                service.assert_report_paths_disjoint(
                    database_paths=(args.database,),
                    report_paths=(args.manifest,),
                )
            expected_manifest = service.load_manifest(args.manifest) if args.manifest else None
            evidence = service.verify(args.database, expected_manifest)
            payload = {
                "operation": "verify",
                "verified": True,
                "evidence": evidence.to_dict(),
                "contains_secrets": False,
                "mutated_authoritative_state": False,
                "authorizes_startup": False,
            }
        else:
            service.assert_report_paths_disjoint(
                database_paths=(args.backup, args.target),
                report_paths=(args.backup_manifest, args.restore_manifest),
            )
            reservation = service.reserve_manifest(args.restore_manifest)
            try:
                expected_backup = service.load_manifest(args.backup_manifest)
                restore_manifest = service.restore_clean_room(
                    args.backup,
                    args.target,
                    expected_backup,
                )
                service.write_reserved_manifest(restore_manifest, reservation)
            finally:
                reservation.close()
            payload = restore_manifest.to_dict()
    except SQLiteMaintenanceError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "contains_secrets": False,
                    "mutated_authoritative_state": False,
                    "authorizes_startup": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
