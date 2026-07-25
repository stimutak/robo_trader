"""Dormant typed assembly boundary for PR 2B.1 paper safety evidence.

The pure safety package deliberately has no broker or database imports.  This
module is the only bridge that binds independently produced broker contract,
account, open-order, reconciliation, and allocation snapshots without
discarding their producer-owned identities or observation times.

The broker snapshot wrappers in this PR are integration-test scaffolding, not
an operational trust source.  They are intentionally unreachable from the
runner.  PR 2B.2 must replace them with account-bound broker-produced snapshots
and a live transport assertion before production authorization can be wired.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Tuple

from .clients.subprocess_ibkr_client import QualifiedStockContractLineage
from .database_async import SafetyAllocationSnapshot
from .safety import (
    AccountPosition,
    AuthoritativeContract,
    EvidenceStatus,
    OpenOrderSnapshot,
    PaperExecutionIdentity,
    PortfolioAllocation,
    ReconciliationStatus,
    TransportState,
)
from .safety.runtime import (
    CoherentSafetySnapshot,
    _assemble_coherent_safety_snapshot,
)


class TrustedEvidenceAssemblyError(ValueError):
    """Independent producer snapshots cannot form one coherent boundary."""


def _utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise TrustedEvidenceAssemblyError(f"{field_name} must be UTC")
    return value.astimezone(timezone.utc)


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 128:
        raise TrustedEvidenceAssemblyError(f"{field_name} is malformed")
    return value


@dataclass(frozen=True, slots=True)
class BrokerAccountPositionSnapshot:
    observed_at: datetime
    snapshot_id: str
    source: str
    transport_generation: str
    positions: Tuple[AccountPosition, ...]
    complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        _text(self.snapshot_id, "snapshot_id")
        _text(self.source, "source")
        _text(self.transport_generation, "transport_generation")
        if not isinstance(self.positions, tuple) or any(
            type(position) is not AccountPosition for position in self.positions
        ):
            raise TrustedEvidenceAssemblyError(
                "positions must be an immutable tuple of AccountPosition"
            )
        if type(self.complete) is not bool:
            raise TrustedEvidenceAssemblyError("complete must be bool")


@dataclass(frozen=True, slots=True)
class BrokerOpenOrderEvidence:
    observed_at: datetime
    snapshot_id: str
    source: str
    transport_generation: str
    active_con_ids: Tuple[int, ...]
    complete: bool
    all_clients: bool
    stable: bool
    unknown_order_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        _text(self.snapshot_id, "snapshot_id")
        _text(self.source, "source")
        _text(self.transport_generation, "transport_generation")
        if not isinstance(self.active_con_ids, tuple) or any(
            type(con_id) is not int or con_id <= 0 for con_id in self.active_con_ids
        ):
            raise TrustedEvidenceAssemblyError(
                "active_con_ids must be an immutable tuple of positive integers"
            )
        if len(set(self.active_con_ids)) != len(self.active_con_ids):
            raise TrustedEvidenceAssemblyError("active_con_ids contains duplicates")
        for field_name in ("complete", "all_clients", "stable"):
            if type(getattr(self, field_name)) is not bool:
                raise TrustedEvidenceAssemblyError(f"{field_name} must be bool")
        if type(self.unknown_order_count) is not int or self.unknown_order_count < 0:
            raise TrustedEvidenceAssemblyError("unknown_order_count must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class BrokerReconciliationSnapshot:
    observed_at: datetime
    snapshot_id: str
    source: str
    transport_generation: str
    status: ReconciliationStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        _text(self.snapshot_id, "snapshot_id")
        _text(self.source, "source")
        _text(self.transport_generation, "transport_generation")
        if type(self.status) is not ReconciliationStatus:
            raise TrustedEvidenceAssemblyError("status must be ReconciliationStatus")


def assemble_trusted_safety_evidence(
    identity: PaperExecutionIdentity,
    contract_lineage: QualifiedStockContractLineage,
    allocation_snapshot: SafetyAllocationSnapshot,
    account_snapshot: BrokerAccountPositionSnapshot,
    open_orders: BrokerOpenOrderEvidence,
    reconciliation: BrokerReconciliationSnapshot,
    *,
    max_collection_window_seconds: int = 30,
) -> tuple[AuthoritativeContract, CoherentSafetySnapshot]:
    """Bind dormant typed models while preserving every lineage boundary."""

    if type(identity) is not PaperExecutionIdentity:
        raise TypeError("identity must be PaperExecutionIdentity")
    if type(contract_lineage) is not QualifiedStockContractLineage:
        raise TypeError("contract_lineage must be QualifiedStockContractLineage")
    if type(allocation_snapshot) is not SafetyAllocationSnapshot:
        raise TypeError("allocation_snapshot must be SafetyAllocationSnapshot")
    if type(account_snapshot) is not BrokerAccountPositionSnapshot:
        raise TypeError("account_snapshot must be BrokerAccountPositionSnapshot")
    if type(open_orders) is not BrokerOpenOrderEvidence:
        raise TypeError("open_orders must be BrokerOpenOrderEvidence")
    if type(reconciliation) is not BrokerReconciliationSnapshot:
        raise TypeError("reconciliation must be BrokerReconciliationSnapshot")
    if type(max_collection_window_seconds) is not int or not (
        1 <= max_collection_window_seconds <= 30
    ):
        raise TrustedEvidenceAssemblyError("max_collection_window_seconds must be in [1, 30]")
    if not allocation_snapshot.complete:
        raise TrustedEvidenceAssemblyError("allocation snapshot is incomplete")
    if allocation_snapshot.symbol != contract_lineage.symbol:
        raise TrustedEvidenceAssemblyError("allocation symbol does not match qualified contract")
    if any(
        allocation.symbol != allocation_snapshot.symbol
        for allocation in allocation_snapshot.allocations
    ):
        raise TrustedEvidenceAssemblyError("allocation rows do not match their producer snapshot")

    generations = {
        contract_lineage.transport_generation,
        account_snapshot.transport_generation,
        open_orders.transport_generation,
        reconciliation.transport_generation,
    }
    if len(generations) != 1:
        raise TrustedEvidenceAssemblyError("broker evidence spans multiple transport generations")

    observed_times = (
        contract_lineage.retrieval_timestamp,
        allocation_snapshot.observed_at,
        account_snapshot.observed_at,
        open_orders.observed_at,
        reconciliation.observed_at,
    )
    if max(observed_times) - min(observed_times) > timedelta(seconds=max_collection_window_seconds):
        raise TrustedEvidenceAssemblyError(
            "producer snapshots exceed the coherent collection window"
        )

    contract_payload = (
        f"{contract_lineage.transport_generation}|{contract_lineage.con_id}|"
        f"{contract_lineage.retrieval_timestamp.isoformat()}"
    )
    contract_snapshot_id = (
        "contract-v1-" + hashlib.sha256(contract_payload.encode("utf-8")).hexdigest()
    )
    contract = AuthoritativeContract(
        con_id=contract_lineage.con_id,
        symbol=contract_lineage.symbol,
        local_symbol=contract_lineage.local_symbol,
        security_type=contract_lineage.security_type,
        currency=contract_lineage.currency,
        exchange=contract_lineage.exchange,
        primary_exchange=contract_lineage.primary_exchange,
        trading_class=contract_lineage.trading_class,
        observed_at=contract_lineage.retrieval_timestamp,
        snapshot_id=contract_snapshot_id,
        source="ibkr-qualified-contract-lineage",
        broker_timestamp=contract_lineage.broker_timestamp,
        retrieval_timestamp=contract_lineage.retrieval_timestamp,
        transport_generation=contract_lineage.transport_generation,
        status=EvidenceStatus.AUTHORITATIVE,
    )
    allocations = tuple(
        PortfolioAllocation(
            portfolio_id=row.portfolio_id,
            con_id=contract.con_id,
            symbol=allocation_snapshot.symbol,
            quantity=row.quantity,
        )
        for row in allocation_snapshot.allocations
    )
    snapshot = _assemble_coherent_safety_snapshot(
        execution_domain_scope=identity.execution_domain_scope,
        account_scope=identity.account_scope,
        observed_at=account_snapshot.observed_at,
        snapshot_id=account_snapshot.snapshot_id,
        source=account_snapshot.source,
        allocation_observed_at=allocation_snapshot.observed_at,
        allocation_snapshot_id=allocation_snapshot.snapshot_id,
        allocation_source="paper-allocation-database",
        reconciliation_observed_at=reconciliation.observed_at,
        reconciliation_snapshot_id=reconciliation.snapshot_id,
        transport_generation=contract_lineage.transport_generation,
        account_positions=account_snapshot.positions,
        portfolio_allocations=allocations,
        open_orders=OpenOrderSnapshot(
            observed_at=open_orders.observed_at,
            snapshot_id=open_orders.snapshot_id,
            transport_generation=open_orders.transport_generation,
            active_con_ids=open_orders.active_con_ids,
            complete=open_orders.complete,
            all_clients=open_orders.all_clients,
            stable=open_orders.stable,
            unknown_order_count=open_orders.unknown_order_count,
        ),
        transport_state=TransportState.CONNECTED,
        reconciliation_status=reconciliation.status,
        positions_complete=account_snapshot.complete,
        allocations_complete=allocation_snapshot.complete,
        contracts_complete=True,
    )
    return contract, snapshot
