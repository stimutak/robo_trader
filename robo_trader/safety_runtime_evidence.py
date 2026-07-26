"""Trusted evidence assembly for local paper-simulator reductions.

The pure safety package deliberately has no broker or database imports. This
module is the only active bridge that binds a producer-owned IBKR
contract/read-only transport proof to the producer-owned cross-portfolio local
simulator ledger. IBKR account positions and orders are diagnostic-only while
``PaperExecutor`` owns fills; they never authorize local simulator exposure.

The former PR 2B.1 caller-assembled broker-position and reconciliation adapters
remain as explicit rejection shims so stale integrations fail closed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Tuple

from .broker_safety_evidence import (
    BrokerContractSafetySnapshot,
    BrokerSafetySnapshot,
    assert_producer_owned_broker_contract_safety_snapshot,
    assert_producer_owned_broker_safety_snapshot,
)
from .clients.subprocess_ibkr_client import QualifiedStockContractLineage
from .config import RuntimeContract
from .database_async import (
    SafetyAllocationSnapshot,
    assert_producer_owned_safety_allocation_snapshot,
)
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


def _runtime_identity(
    identity: PaperExecutionIdentity,
    runtime_contract: RuntimeContract,
) -> None:
    """Require one exact paper/runtime identity without trusting aliases."""

    if type(identity) is not PaperExecutionIdentity:
        raise TypeError("identity must be PaperExecutionIdentity")
    if type(runtime_contract) is not RuntimeContract:
        raise TypeError("runtime_contract must be RuntimeContract")
    if (
        runtime_contract.execution_mode != "paper"
        or runtime_contract.execution_source != "paper_simulator"
        or runtime_contract.ibkr_port != 4002
        or runtime_contract.ibkr_readonly is not True
        or runtime_contract.state_namespace != "paper"
    ):
        raise TrustedEvidenceAssemblyError(
            "runtime contract is not the fixed paper/read-only simulator"
        )
    if (
        runtime_contract.safety_execution_domain_scope != identity.execution_domain_scope
        or runtime_contract.safety_account_scope != identity.account_scope
    ):
        raise TrustedEvidenceAssemblyError(
            "runtime contract does not match the paper execution identity"
        )


def assemble_local_paper_safety_evidence(
    identity: PaperExecutionIdentity,
    runtime_contract: RuntimeContract,
    broker_contract_snapshot: BrokerContractSafetySnapshot,
    allocation_snapshot: SafetyAllocationSnapshot,
    *,
    max_collection_window_seconds: int = 30,
) -> tuple[AuthoritativeContract, CoherentSafetySnapshot]:
    """Bind broker-qualified identity to the authoritative local paper ledger.

    The sanctioned runtime executes through ``PaperExecutor`` while IBKR stays
    read-only. Consequently, IBKR account positions and orders are diagnostic
    data, not simulator exposure. The cross-portfolio paper ledger is the
    complete simulator account/allocation source; the broker contributes only
    contract, transport, account-binding, and IBC read-only provenance.
    """

    _runtime_identity(identity, runtime_contract)
    if type(broker_contract_snapshot) is not BrokerContractSafetySnapshot:
        raise TypeError("broker_contract_snapshot must be BrokerContractSafetySnapshot")
    if type(allocation_snapshot) is not SafetyAllocationSnapshot:
        raise TypeError("allocation_snapshot must be SafetyAllocationSnapshot")
    if type(max_collection_window_seconds) is not int or not (
        1 <= max_collection_window_seconds <= 30
    ):
        raise TrustedEvidenceAssemblyError("max_collection_window_seconds must be in [1, 30]")
    assert_producer_owned_broker_contract_safety_snapshot(broker_contract_snapshot)
    assert_producer_owned_safety_allocation_snapshot(allocation_snapshot)

    if broker_contract_snapshot.account_scope != identity.account_scope:
        raise TrustedEvidenceAssemblyError(
            "broker contract proof does not match the paper account scope"
        )
    if broker_contract_snapshot.runtime_fingerprint != runtime_contract.fingerprint:
        raise TrustedEvidenceAssemblyError(
            "broker contract proof does not match the runtime contract"
        )
    if allocation_snapshot.database_path != runtime_contract.database_path:
        raise TrustedEvidenceAssemblyError(
            "allocation snapshot does not match the configured ledger path"
        )
    if allocation_snapshot.database_identity != runtime_contract.database_identity:
        raise TrustedEvidenceAssemblyError(
            "allocation snapshot does not match the configured ledger identity"
        )
    if not allocation_snapshot.complete:
        raise TrustedEvidenceAssemblyError("allocation snapshot is incomplete")
    if broker_contract_snapshot.requested_symbol != allocation_snapshot.symbol:
        raise TrustedEvidenceAssemblyError(
            "broker contract and allocation snapshots identify different symbols"
        )
    evidence_times = (
        broker_contract_snapshot.broker_time_before,
        broker_contract_snapshot.broker_time_after,
        broker_contract_snapshot.retrieved_at,
        allocation_snapshot.observed_at,
    )
    if max(evidence_times) - min(evidence_times) > timedelta(seconds=max_collection_window_seconds):
        raise TrustedEvidenceAssemblyError(
            "producer snapshots exceed the coherent collection window"
        )

    broker_contract = broker_contract_snapshot.qualified_contract
    contract = AuthoritativeContract(
        con_id=broker_contract.con_id,
        symbol=broker_contract.symbol,
        local_symbol=broker_contract.local_symbol,
        security_type=broker_contract.security_type,
        currency=broker_contract.currency,
        exchange=broker_contract.exchange,
        primary_exchange=broker_contract.primary_exchange,
        trading_class=broker_contract.trading_class,
        observed_at=broker_contract_snapshot.retrieved_at,
        snapshot_id=broker_contract_snapshot.snapshot_id,
        source=broker_contract_snapshot.source,
        broker_timestamp=broker_contract_snapshot.broker_time_after,
        retrieval_timestamp=broker_contract_snapshot.retrieved_at,
        transport_generation=broker_contract_snapshot.transport_generation,
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
    reconciliation_status = (
        ReconciliationStatus.PASSED
        if (allocation_snapshot.complete and not allocation_snapshot.has_offsetting_allocations)
        else ReconciliationStatus.FAILED
    )
    reconciliation_payload = (
        f"{broker_contract_snapshot.snapshot_id}|"
        f"{allocation_snapshot.snapshot_id}|"
        f"{allocation_snapshot.aggregate_allocated_quantity}|"
        f"{allocation_snapshot.has_offsetting_allocations}"
    )
    snapshot = _assemble_coherent_safety_snapshot(
        execution_domain_scope=identity.execution_domain_scope,
        account_scope=identity.account_scope,
        observed_at=allocation_snapshot.observed_at,
        snapshot_id=allocation_snapshot.snapshot_id,
        source="paper-simulator-ledger",
        allocation_observed_at=allocation_snapshot.observed_at,
        allocation_snapshot_id=allocation_snapshot.snapshot_id,
        allocation_source="paper-simulator-ledger",
        allocation_database_path=allocation_snapshot.database_path,
        allocation_database_identity=allocation_snapshot.database_identity,
        allocation_database_device=allocation_snapshot.database_device,
        allocation_database_inode=allocation_snapshot.database_inode,
        runtime_fingerprint=runtime_contract.fingerprint,
        ibc_proof_id=broker_contract_snapshot.ibc_proof_id,
        reconciliation_observed_at=max(evidence_times),
        reconciliation_snapshot_id=(
            "paper-ledger-reconciliation-v1-"
            + hashlib.sha256(reconciliation_payload.encode("utf-8")).hexdigest()
        ),
        transport_generation=broker_contract_snapshot.transport_generation,
        account_positions=(
            AccountPosition(
                con_id=contract.con_id,
                symbol=contract.symbol,
                quantity=allocation_snapshot.aggregate_allocated_quantity,
            ),
        ),
        portfolio_allocations=allocations,
        open_orders=OpenOrderSnapshot(
            observed_at=allocation_snapshot.observed_at,
            snapshot_id=allocation_snapshot.snapshot_id,
            transport_generation=broker_contract_snapshot.transport_generation,
            active_con_ids=(),
            complete=True,
            all_clients=True,
            stable=True,
            unknown_order_count=0,
        ),
        transport_state=TransportState.CONNECTED,
        reconciliation_status=reconciliation_status,
        positions_complete=allocation_snapshot.complete,
        allocations_complete=allocation_snapshot.complete,
        contracts_complete=True,
    )
    assert_producer_owned_broker_contract_safety_snapshot(broker_contract_snapshot)
    assert_producer_owned_safety_allocation_snapshot(allocation_snapshot)
    return contract, snapshot


def assemble_broker_bound_safety_evidence(
    identity: PaperExecutionIdentity,
    runtime_contract: RuntimeContract,
    broker_snapshot: BrokerSafetySnapshot,
    allocation_snapshot: SafetyAllocationSnapshot,
    *,
    max_collection_window_seconds: int = 30,
) -> tuple[AuthoritativeContract, CoherentSafetySnapshot]:
    """Reject the invalid broker-position/local-simulator authority mix.

    IBKR remains read-only while ``PaperExecutor`` owns simulated fills, so
    broker account positions cannot authorize or settle local simulator orders.
    """

    raise TrustedEvidenceAssemblyError(
        "IBKR account positions are diagnostic-only for local paper execution"
    )

    _runtime_identity(identity, runtime_contract)
    if type(broker_snapshot) is not BrokerSafetySnapshot:
        raise TypeError("broker_snapshot must be BrokerSafetySnapshot")
    if type(allocation_snapshot) is not SafetyAllocationSnapshot:
        raise TypeError("allocation_snapshot must be SafetyAllocationSnapshot")
    if type(max_collection_window_seconds) is not int or not (
        1 <= max_collection_window_seconds <= 30
    ):
        raise TrustedEvidenceAssemblyError("max_collection_window_seconds must be in [1, 30]")

    # These assertions reject dataclass copies and reconstructed objects before
    # any caller-controlled field can contribute to authorization.
    assert_producer_owned_broker_safety_snapshot(broker_snapshot)
    assert_producer_owned_safety_allocation_snapshot(allocation_snapshot)

    if broker_snapshot.account_scope != identity.account_scope:
        raise TrustedEvidenceAssemblyError("broker snapshot does not match the paper account scope")
    if broker_snapshot.runtime_fingerprint != runtime_contract.fingerprint:
        raise TrustedEvidenceAssemblyError("broker snapshot does not match the runtime contract")
    if allocation_snapshot.database_path != runtime_contract.database_path:
        raise TrustedEvidenceAssemblyError(
            "allocation snapshot does not match the configured ledger path"
        )
    if allocation_snapshot.database_identity != runtime_contract.database_identity:
        raise TrustedEvidenceAssemblyError(
            "allocation snapshot does not match the configured ledger identity"
        )
    if not allocation_snapshot.complete:
        raise TrustedEvidenceAssemblyError("allocation snapshot is incomplete")
    if broker_snapshot.requested_symbol != allocation_snapshot.symbol:
        raise TrustedEvidenceAssemblyError(
            "broker and allocation snapshots identify different symbols"
        )
    if broker_snapshot.broker_time_after - broker_snapshot.broker_time_before > timedelta(
        seconds=max_collection_window_seconds
    ):
        raise TrustedEvidenceAssemblyError("broker snapshot exceeds the coherent collection window")
    if max(broker_snapshot.observed_at, allocation_snapshot.observed_at) - min(
        broker_snapshot.observed_at, allocation_snapshot.observed_at
    ) > timedelta(seconds=max_collection_window_seconds):
        raise TrustedEvidenceAssemblyError(
            "producer snapshots exceed the coherent collection window"
        )

    broker_contract = broker_snapshot.requested_contract
    matching_positions = tuple(
        position for position in broker_snapshot.positions if position.contract == broker_contract
    )
    if len(matching_positions) != 1:
        raise TrustedEvidenceAssemblyError("broker snapshot lacks one exact requested position")
    account_quantity = matching_positions[0].quantity
    reconciliation_status = (
        ReconciliationStatus.PASSED
        if (
            allocation_snapshot.aggregate_allocated_quantity == account_quantity
            and not allocation_snapshot.has_offsetting_allocations
        )
        else ReconciliationStatus.FAILED
    )

    contract_snapshot_payload = (
        f"{broker_snapshot.snapshot_id}|{broker_snapshot.transport_generation}|"
        f"{broker_contract.con_id}|{broker_snapshot.observed_at.isoformat()}"
    )
    contract = AuthoritativeContract(
        con_id=broker_contract.con_id,
        symbol=broker_contract.symbol,
        local_symbol=broker_contract.local_symbol,
        security_type=broker_contract.security_type,
        currency=broker_contract.currency,
        exchange=broker_contract.exchange,
        primary_exchange=broker_contract.primary_exchange,
        trading_class=broker_contract.trading_class,
        observed_at=broker_snapshot.observed_at,
        snapshot_id=(
            "broker-contract-v1-"
            + hashlib.sha256(contract_snapshot_payload.encode("utf-8")).hexdigest()
        ),
        source=broker_snapshot.source,
        broker_timestamp=broker_snapshot.broker_time_after,
        retrieval_timestamp=broker_snapshot.observed_at,
        transport_generation=broker_snapshot.transport_generation,
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
    account_positions = tuple(
        AccountPosition(
            con_id=position.contract.con_id,
            symbol=position.contract.symbol,
            quantity=position.quantity,
        )
        for position in broker_snapshot.positions
    )
    reconciliation_payload = (
        f"{broker_snapshot.snapshot_id}|{allocation_snapshot.snapshot_id}|"
        f"{account_quantity}|{allocation_snapshot.aggregate_allocated_quantity}|"
        f"{allocation_snapshot.has_offsetting_allocations}"
    )
    snapshot = _assemble_coherent_safety_snapshot(
        execution_domain_scope=identity.execution_domain_scope,
        account_scope=identity.account_scope,
        observed_at=broker_snapshot.observed_at,
        snapshot_id=broker_snapshot.snapshot_id,
        source=broker_snapshot.source,
        allocation_observed_at=allocation_snapshot.observed_at,
        allocation_snapshot_id=allocation_snapshot.snapshot_id,
        allocation_source="paper-allocation-database",
        allocation_database_path=allocation_snapshot.database_path,
        allocation_database_identity=allocation_snapshot.database_identity,
        allocation_database_device=allocation_snapshot.database_device,
        allocation_database_inode=allocation_snapshot.database_inode,
        runtime_fingerprint=runtime_contract.fingerprint,
        ibc_proof_id=broker_snapshot.ibc_proof_id,
        reconciliation_observed_at=max(
            broker_snapshot.observed_at,
            allocation_snapshot.observed_at,
        ),
        reconciliation_snapshot_id=(
            "broker-ledger-reconciliation-v1-"
            + hashlib.sha256(reconciliation_payload.encode("utf-8")).hexdigest()
        ),
        transport_generation=broker_snapshot.transport_generation,
        account_positions=account_positions,
        portfolio_allocations=allocations,
        open_orders=OpenOrderSnapshot(
            observed_at=broker_snapshot.observed_at,
            snapshot_id=broker_snapshot.snapshot_id,
            transport_generation=broker_snapshot.transport_generation,
            active_con_ids=broker_snapshot.active_con_ids,
            complete=broker_snapshot.open_orders_complete,
            all_clients=broker_snapshot.open_orders_all_clients,
            stable=broker_snapshot.open_orders_stable,
            unknown_order_count=broker_snapshot.unknown_order_count,
        ),
        transport_state=TransportState.CONNECTED,
        reconciliation_status=reconciliation_status,
        positions_complete=broker_snapshot.positions_complete,
        allocations_complete=allocation_snapshot.complete,
        contracts_complete=True,
    )
    assert_producer_owned_broker_safety_snapshot(broker_snapshot)
    assert_producer_owned_safety_allocation_snapshot(allocation_snapshot)
    return contract, snapshot


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
    """Reject the former caller-labelable PR 2B.1 scaffolding."""

    raise TrustedEvidenceAssemblyError(
        "caller-supplied broker reconciliation cannot authorize paper execution"
    )

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
