#!/usr/bin/env python3
"""Produce one fresh, typed, authenticated exact-state evidence bundle.

The production command accepts no input artifacts, JSON payloads, prices, or
signing-key paths.  It owns fresh broker collection, immutable local-ledger
observation, reconciliation, and protective-mark production through typed
receivers.  The diagnostic provider remains open through protective-mark
collection and is reaped before all signing capabilities are released.
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
from robo_trader.bootstrap_mark_producer import (  # noqa: E402
    collect_and_produce_bootstrap_protective_mark,
    create_runtime_bound_mark_only_producer,
)
from robo_trader.config import RuntimeContract  # noqa: E402
from robo_trader.reconciliation.bootstrap_producer import (  # noqa: E402
    produce_bootstrap_reconciliation,
)
from robo_trader.reconciliation.ibkr_adapter import (  # noqa: E402
    IBKRDiagnosticSnapshotProvider,
    ProtectiveQuoteSourceCapability,
    assert_factory_owned_protective_quote_source,
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
        quote_source: ProtectiveQuoteSourceCapability,
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

    provider_closed = False
    try:
        broker_envelope = await snapshot_provider.produce_normalized_snapshot(
            receiver=receivers.broker_snapshot,
            max_age_seconds=30.0,
        )
        broker_artifact = receivers.broker_artifact
        reconciliation_delivery = produce_bootstrap_reconciliation(
            broker_envelope,
            runtime_contract,
            receivers.reconciliation_report,
        )
        quote_source = snapshot_provider.issue_protective_quote_source(
            runtime_contract=runtime_contract,
        )
        mark_artifacts = await protective_mark_collector(
            runtime_contract=runtime_contract,
            mark_identities=reconciliation_delivery.local_position_identities,
            quote_source=quote_source,
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
            raise BootstrapEvidencePipelineError(
                "reconciliation receiver returned invalid evidence"
            )
        try:
            cleanup_cancelled = await await_cleanup_required(snapshot_provider.close())
        finally:
            provider_closed = True
        if cleanup_cancelled:
            raise asyncio.CancelledError
        receivers.publish_complete_bundle(set(reconciliation_delivery.local_position_identities))
        return {
            "authorizes_startup": False,
            "broker_snapshot": str(receivers.published_artifact_path(broker_artifact)),
            "protective_marks": [
                str(receivers.published_artifact_path(item)) for item in mark_artifacts
            ],
            "reconciliation_report": str(
                receivers.published_artifact_path(reconciliation_artifact)
            ),
            "schema_version": 1,
            "status": "EVIDENCE_COMPLETE_GATE_A_STILL_CLOSED",
        }
    except BaseException:
        receivers.discard_unpublished_bundle()
        raise
    finally:
        try:
            cleanup_cancelled = False
            if not provider_closed:
                cleanup_cancelled = await await_cleanup_required(snapshot_provider.close())
        finally:
            receivers.close()
        if cleanup_cancelled:
            raise asyncio.CancelledError


async def _collect_production_marks(
    *,
    runtime_contract: RuntimeContract,
    mark_identities: tuple[tuple[str, str], ...],
    quote_source: ProtectiveQuoteSourceCapability,
    receiver: object,
) -> tuple[SealedBootstrapEvidenceArtifact, ...]:
    """Collect and seal one current live mark for every reconciled position."""

    identity = assert_factory_owned_protective_quote_source(
        quote_source,
        runtime_contract=runtime_contract,
    )
    artifacts: list[SealedBootstrapEvidenceArtifact] = []
    for portfolio_id, symbol in mark_identities:
        discovery = await quote_source.get_protective_quotes(
            (symbol,),
            active_symbols=(symbol,),
        )
        if type(discovery) is not tuple or len(discovery) != 1:
            raise BootstrapEvidencePipelineError(
                "protective quote discovery returned incomplete evidence"
            )
        quote = discovery[0]
        if quote.symbol != symbol or quote.transport_generation != identity.transport_generation:
            raise BootstrapEvidencePipelineError(
                "protective quote discovery changed runtime generation or symbol"
            )
        producer = create_runtime_bound_mark_only_producer(
            runtime_contract,
            portfolio_id=portfolio_id,
        )
        artifact = await collect_and_produce_bootstrap_protective_mark(
            quote_source,
            producer,
            runtime_contract,
            receiver,
            expected_portfolio_id=portfolio_id,
            expected_symbol=symbol,
            expected_con_id=quote.con_id,
            expected_transport_generation=identity.transport_generation,
        )
        if type(artifact) is not SealedBootstrapEvidenceArtifact:
            raise BootstrapEvidencePipelineError(
                "protective mark receiver returned invalid evidence"
            )
        artifacts.append(artifact)
    if (
        assert_factory_owned_protective_quote_source(
            quote_source,
            runtime_contract=runtime_contract,
        )
        != identity
    ):
        raise BootstrapEvidencePipelineError("protective quote source changed during collection")
    return tuple(artifacts)


_PRODUCTION_MARK_COLLECTOR: ProtectiveMarkCollector = _collect_production_marks


async def _run(args: argparse.Namespace) -> dict[str, object]:
    context: RuntimeSafetyContext = validate_runtime_safety(PROJECT_ROOT, os.environ)
    runtime = context.runtime_contract
    if type(runtime) is not RuntimeContract:
        raise BootstrapEvidencePipelineError("validated runtime contract is unavailable")
    provider = await build_diagnostic_provider(context)
    bundle_owns_provider_cleanup = False
    try:
        receivers = create_bootstrap_evidence_receivers(
            runtime_contract=runtime,
            capability_directory=args.capability_directory,
            output_directory=args.output_directory,
        )
        bundle_owns_provider_cleanup = True
        return await produce_bootstrap_evidence_bundle(
            runtime_contract=runtime,
            snapshot_provider=provider,
            receivers=receivers,
            protective_mark_collector=_PRODUCTION_MARK_COLLECTOR,
        )
    finally:
        if not bundle_owns_provider_cleanup:
            cleanup_cancelled = await await_cleanup_required(provider.close())
            if cleanup_cancelled:
                raise asyncio.CancelledError


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
