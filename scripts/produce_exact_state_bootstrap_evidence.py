#!/usr/bin/env python3
"""Produce one fresh, typed, authenticated exact-state evidence bundle.

The production command accepts no input artifacts, JSON payloads, prices, or
signing-key paths.  It owns fresh broker collection, immutable local-ledger
observation, reconciliation, and protective-mark production through typed
receivers.  Until the live protective-quote owner supplies its standalone
collector callback, the CLI fails closed before opening signing capabilities.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_trader.bootstrap_evidence_receivers import (  # noqa: E402
    BootstrapEvidenceReceiverSet,
    SealedBootstrapEvidenceArtifact,
    create_bootstrap_evidence_receivers,
)
from robo_trader.config import RuntimeContract  # noqa: E402
from robo_trader.reconciliation.bootstrap_producer import (  # noqa: E402
    produce_bootstrap_reconciliation,
)
from robo_trader.reconciliation.ibkr_adapter import (  # noqa: E402
    IBKRDiagnosticSnapshotProvider,
    await_cleanup_required,
    build_diagnostic_provider,
)
from robo_trader.reconciliation.identity import (  # noqa: E402
    RuntimeSafetyContext,
    validate_runtime_safety,
)


class BootstrapEvidencePipelineError(ValueError):
    """The fresh evidence pipeline cannot complete safely."""


class ProtectiveMarkCollector(Protocol):
    async def __call__(
        self,
        *,
        runtime_contract: RuntimeContract,
        mark_identities: tuple[tuple[str, str], ...],
        receiver: object,
    ) -> tuple[SealedBootstrapEvidenceArtifact, ...]:
        """Collect current authoritative quotes and invoke the typed receiver."""


async def produce_bootstrap_evidence_bundle(
    *,
    runtime_contract: RuntimeContract,
    snapshot_provider: IBKRDiagnosticSnapshotProvider,
    receivers: BootstrapEvidenceReceiverSet,
    protective_mark_collector: ProtectiveMarkCollector,
) -> dict[str, object]:
    """Run fresh producers in dependency order and require complete mark coverage."""

    try:
        broker_result = await snapshot_provider.produce_normalized_snapshot(max_age_seconds=30.0)
    finally:
        cleanup_cancelled = await await_cleanup_required(snapshot_provider.close())
    if cleanup_cancelled:
        raise asyncio.CancelledError
    broker_envelope = receivers.broker_snapshot.receive_broker_snapshot_producer_result(
        broker_result
    )
    broker_artifact = receivers.broker_artifact
    reconciliation_delivery = produce_bootstrap_reconciliation(
        broker_envelope,
        runtime_contract,
        receivers.reconciliation_report,
    )
    mark_artifacts = await protective_mark_collector(
        runtime_contract=runtime_contract,
        mark_identities=reconciliation_delivery.local_position_identities,
        receiver=receivers.protective_mark,
    )
    if (
        type(mark_artifacts) is not tuple
        or any(type(item) is not SealedBootstrapEvidenceArtifact for item in mark_artifacts)
        or len(mark_artifacts) != len(reconciliation_delivery.local_position_identities)
    ):
        raise BootstrapEvidencePipelineError(
            "protective mark collector returned incomplete evidence"
        )
    receivers.assert_complete(set(reconciliation_delivery.local_position_identities))
    reconciliation_artifact = reconciliation_delivery.receiver_result
    if type(reconciliation_artifact) is not SealedBootstrapEvidenceArtifact:
        raise BootstrapEvidencePipelineError("reconciliation receiver returned invalid evidence")
    return {
        "authorizes_startup": False,
        "broker_snapshot": str(broker_artifact.artifact_path),
        "protective_marks": [str(item.artifact_path) for item in mark_artifacts],
        "reconciliation_report": str(reconciliation_artifact.artifact_path),
        "schema_version": 1,
        "status": "EVIDENCE_COMPLETE_GATE_A_STILL_CLOSED",
    }


_PRODUCTION_MARK_COLLECTOR: ProtectiveMarkCollector | None = None


async def _run(args: argparse.Namespace) -> dict[str, object]:
    context: RuntimeSafetyContext = validate_runtime_safety(PROJECT_ROOT, os.environ)
    runtime = context.runtime_contract
    if type(runtime) is not RuntimeContract:
        raise BootstrapEvidencePipelineError("validated runtime contract is unavailable")
    # Fail before opening the broker connection or signing capabilities until
    # production wiring can supply the typed live-quote collectors and their
    # portfolio-owned StopLossMonitor instances.
    if _PRODUCTION_MARK_COLLECTOR is None:
        raise BootstrapEvidencePipelineError(
            "standalone authoritative protective-quote collection is not integrated"
        )
    provider = await build_diagnostic_provider(context)
    receivers = create_bootstrap_evidence_receivers(
        runtime_contract=runtime,
        capability_directory=args.capability_directory,
        output_directory=args.output_directory,
    )
    return await produce_bootstrap_evidence_bundle(
        runtime_contract=runtime,
        snapshot_provider=provider,
        receivers=receivers,
        protective_mark_collector=_PRODUCTION_MARK_COLLECTOR,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        report = asyncio.run(_run(_parser().parse_args(argv)))
    except (BootstrapEvidencePipelineError, ValueError, OSError) as exc:
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
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
