"""Report-only orchestration for broker/ledger reconciliation."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .broker import (
    BrokerSnapshotProvider,
    BrokerSnapshotProviderFactory,
    assert_read_only_provider_surface,
)
from .engine import reconcile
from .errors import (
    BrokerEvidenceError,
    IntegrityViolation,
    LedgerSafetyError,
    ReconciliationError,
    RuntimeSafetyError,
)
from .ibkr_adapter import diagnostic_provider_factory
from .identity import resolve_environment, validate_runtime_safety
from .integrity import EvidenceIntegrityGuard, protected_evidence_paths
from .ledger import ImmutableLedgerReader, validate_portfolio_ids
from .models import ReconciliationReport

EXIT_CLEAN_QUANTITY_COST = 0
EXIT_DIFFERENCES = 1
EXIT_BLOCKED = 2
EXIT_INTEGRITY_VIOLATION = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Produce non-mutating IBKR paper-account versus local-ledger evidence. "
            "This command never repairs state or authorizes trader startup."
        )
    )
    parser.add_argument(
        "--portfolio-id",
        action="append",
        required=True,
        dest="portfolio_ids",
        help="Explicit local portfolio ID. Repeat to show aggregate evidence.",
    )
    parser.add_argument("--json", action="store_true", help="Emit stable JSON evidence.")
    return parser


async def _build_provider(
    factory: BrokerSnapshotProviderFactory, runtime
) -> BrokerSnapshotProvider:
    provider_or_awaitable = factory(runtime)
    if inspect.isawaitable(provider_or_awaitable):
        provider = await provider_or_awaitable
    else:
        provider = provider_or_awaitable
    return provider


async def run_reconciliation(
    portfolio_ids: Sequence[str],
    *,
    project_root: Path,
    process_environ: Optional[Mapping[str, str]] = None,
    provider_factory: BrokerSnapshotProviderFactory = diagnostic_provider_factory,
    now: Optional[datetime] = None,
) -> ReconciliationReport:
    """Run all local gates before constructing a broker provider."""
    selected = validate_portfolio_ids(portfolio_ids)
    resolved_env = resolve_environment(project_root, process_environ)
    evidence_paths = protected_evidence_paths(project_root, resolved_env)
    with EvidenceIntegrityGuard(evidence_paths):
        runtime = validate_runtime_safety(project_root, resolved_env)
        contract = runtime.runtime_contract
        reader = ImmutableLedgerReader(project_root, str(contract.database_path))
        ledger = reader.read(selected)

        provider: Optional[BrokerSnapshotProvider] = None
        try:
            provider = await _build_provider(provider_factory, runtime)
            assert_read_only_provider_surface(provider)
            snapshot = await asyncio.wait_for(
                provider.get_broker_snapshot(
                    runtime.expected_account_for_provider,
                    max_age_seconds=30.0,
                ),
                timeout=60.0,
            )
        except Exception as exc:
            raise BrokerEvidenceError("broker evidence collection failed") from exc
        finally:
            if provider is not None:
                try:
                    await asyncio.wait_for(provider.close(), timeout=10.0)
                except Exception as exc:
                    raise BrokerEvidenceError("broker diagnostic transport cleanup failed") from exc

        return reconcile(
            snapshot,
            ledger,
            runtime_fingerprint=str(contract.fingerprint),
            database_identity=str(contract.database_identity),
            expected_account_alias=runtime.account_alias,
            now=now or datetime.now(timezone.utc),
        )


def _human_output(report: ReconciliationReport) -> str:
    lines = [
        "READ-ONLY BROKER/LEDGER RECONCILIATION",
        f"status={report.status}",
        f"account_alias={report.account_alias}",
        f"portfolios={','.join(report.selected_portfolio_ids)}",
        "mutated_state=false",
        "authorizes_startup=false",
        "This evidence cannot clear a kill switch, repair the ledger, bypass preflight, "
        "or authorize trader startup.",
    ]
    for comparison in report.position_comparisons:
        reasons = ",".join(comparison.reasons) or "none"
        lines.append(f"position {comparison.symbol}: {comparison.status}; reasons={reasons}")
    for blocker in report.blockers:
        lines.append(f"BLOCKER: {blocker}")
    for caveat in report.caveats:
        lines.append(f"CAVEAT: {caveat}")
    return "\n".join(lines)


def _safe_error_payload(error: ReconciliationError) -> dict[str, object]:
    if isinstance(error, IntegrityViolation):
        code = "INTEGRITY_VIOLATION"
    elif isinstance(error, RuntimeSafetyError):
        code = "RUNTIME_SAFETY_BLOCK"
    elif isinstance(error, LedgerSafetyError):
        code = "LEDGER_SAFETY_BLOCK"
    elif isinstance(error, BrokerEvidenceError):
        code = "BROKER_EVIDENCE_BLOCK"
    else:
        code = "RECONCILIATION_BLOCK"
    return {
        "schema_version": 1,
        "status": "BLOCKED",
        "error_code": code,
        "message": str(error),
        "mutated_state": False,
        "authorizes_startup": False,
    }


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    project_root: Optional[Path] = None,
    process_environ: Optional[Mapping[str, str]] = None,
    provider_factory: BrokerSnapshotProviderFactory = diagnostic_provider_factory,
    now: Optional[datetime] = None,
) -> int:
    args = build_parser().parse_args(argv)
    root = project_root or Path(__file__).resolve().parents[2]
    try:
        report = asyncio.run(
            run_reconciliation(
                args.portfolio_ids,
                project_root=root,
                process_environ=process_environ,
                provider_factory=provider_factory,
                now=now,
            )
        )
    except ReconciliationError as exc:
        payload = _safe_error_payload(exc)
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(
                f"BLOCKED: {payload['message']}\n"
                "mutated_state=false\n"
                "authorizes_startup=false",
                file=sys.stderr,
            )
        return EXIT_INTEGRITY_VIOLATION if isinstance(exc, IntegrityViolation) else EXIT_BLOCKED
    except Exception:
        payload = {
            "schema_version": 1,
            "status": "BLOCKED",
            "error_code": "UNEXPECTED_FAILURE",
            "message": "unexpected reconciliation failure",
            "mutated_state": False,
            "authorizes_startup": False,
        }
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(
                "BLOCKED: unexpected reconciliation failure\n"
                "mutated_state=false\nauthorizes_startup=false",
                file=sys.stderr,
            )
        return EXIT_BLOCKED

    if args.json:
        print(json.dumps(report.public_dict(), sort_keys=True, separators=(",", ":")))
    else:
        print(_human_output(report))
    if report.status == "QUANTITY_COST_COMPARABLE_ONLY":
        return EXIT_CLEAN_QUANTITY_COST
    if report.status == "MISMATCH":
        return EXIT_DIFFERENCES
    return EXIT_BLOCKED
