"""Dormant paper-only runtime coordinator for the order-safety core.

The coordinator translates one coherent, caller-supplied snapshot into the
strict models owned by :mod:`robo_trader.safety`.  It deliberately has no
configuration, broker, database, or production-runner imports.  The only
submission sink in this module is an in-memory fake used by integration tests;
production broker routing remains out of scope.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable, Optional, Tuple

from ..runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
from .journal import SafetyJournal, StateTransitionError
from .models import (
    SAFETY_MAX_EVIDENCE_AGE_SECONDS,
    DecisionOutcome,
    EvidenceStatus,
    ExposureEvidence,
    GateContext,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioAllocationEvidence,
    ReconciliationStatus,
    Reservation,
    SafetyDecision,
    SubmissionClaim,
    SubmissionDescriptor,
    SubmissionPermit,
    TimeInForce,
    TransportState,
    ValidationError,
    _exact_decimal_add,
    canonical_json,
)
from .policy import evaluate_reduce_only

_ACCOUNT_SCOPE_RE = re.compile(r"^acct_v1_[0-9a-f]{64}$")
_DOMAIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
PAPER_EXECUTION_DOMAIN_SCOPE = PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
_TEXT_SCOPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
_TRUSTED_ASSEMBLY_MARKER = object()


class RuntimeSafetyError(RuntimeError):
    """Base class for fail-closed runtime coordinator errors."""


class RuntimeNotStarted(RuntimeSafetyError):
    """Authorization was attempted before successful startup replay."""


class RuntimeStartupBlocked(RuntimeSafetyError):
    """Journal replay found unresolved submission authority."""

    def __init__(self, reason_codes: Tuple[str, ...]) -> None:
        self.reason_codes = reason_codes
        super().__init__(", ".join(reason_codes))


class RuntimeAuthorizationBlocked(RuntimeSafetyError):
    """The coherent evidence set did not authorize a submission."""

    def __init__(self, decision: SafetyDecision) -> None:
        self.decision = decision
        self.reason_codes = decision.reason_codes
        super().__init__(", ".join(decision.reason_codes))


class FakeSubmissionOnlyError(RuntimeSafetyError):
    """A non-test submission sink was offered to the dormant coordinator."""


def _utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValidationError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValidationError(f"{field_name} must be UTC")
    return value.astimezone(timezone.utc)


def _scope(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _TEXT_SCOPE_RE.fullmatch(value):
        raise ValidationError(f"{field_name} has an invalid format")
    return value


def _symbol(value: object) -> str:
    if not isinstance(value, str) or not _SYMBOL_RE.fullmatch(value):
        raise ValidationError("symbol has an invalid format")
    return value


def _decimal(value: object, field_name: str) -> Decimal:
    # Constructing a boundary model is the canonical exact-Decimal validator.
    try:
        probe = ExposureEvidence(
            execution_domain_scope="paper-validation",
            account_scope="acct_v1_" + ("0" * 64),
            con_id=1,
            symbol="PROBE",
            position_quantity=value,  # type: ignore[arg-type]
            observed_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
            status=EvidenceStatus.AUTHORITATIVE,
            source="runtime-validation",
            snapshot_id="runtime-validation",
        )
    except ValidationError as exc:
        raise ValidationError(f"{field_name} must be an exact Decimal") from exc
    return probe.position_quantity


@dataclass(frozen=True)
class PaperExecutionIdentity:
    """Stable safety identity supplied independently of broker account text."""

    execution_domain_scope: str
    account_scope: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.execution_domain_scope, str)
            or not _DOMAIN_RE.fullmatch(self.execution_domain_scope)
            or self.execution_domain_scope != PAPER_EXECUTION_DOMAIN_SCOPE
        ):
            raise ValidationError(
                f"execution_domain_scope must be exactly {PAPER_EXECUTION_DOMAIN_SCOPE}"
            )
        if not isinstance(self.account_scope, str) or not _ACCOUNT_SCOPE_RE.fullmatch(
            self.account_scope
        ):
            raise ValidationError(
                "account_scope must be a supplied opaque acct_v1_<64 lowercase hex> value"
            )
        if len(set(self.account_scope.removeprefix("acct_v1_"))) == 1:
            raise ValidationError("account_scope must not use a placeholder digest")


@dataclass(frozen=True)
class AuthoritativeContract:
    """Qualified contract lineage retained through authorization."""

    con_id: int
    symbol: str
    local_symbol: str
    security_type: str
    currency: str
    exchange: str
    primary_exchange: str
    trading_class: str
    observed_at: datetime
    snapshot_id: str
    source: str
    broker_timestamp: datetime
    retrieval_timestamp: datetime
    transport_generation: str
    status: EvidenceStatus = EvidenceStatus.AUTHORITATIVE

    def __post_init__(self) -> None:
        if type(self.con_id) is not int or self.con_id <= 0:
            raise ValidationError("con_id must be a positive integer")
        _symbol(self.symbol)
        _symbol(self.local_symbol)
        if self.local_symbol != self.symbol:
            raise ValidationError("local_symbol must exactly match symbol")
        if self.security_type != "STK":
            raise ValidationError("security_type must be STK")
        if self.currency != "USD":
            raise ValidationError("currency must be USD")
        if self.exchange != "SMART":
            raise ValidationError("exchange must be SMART")
        _scope(self.primary_exchange, "primary_exchange")
        _scope(self.trading_class, "trading_class")
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        object.__setattr__(
            self,
            "broker_timestamp",
            _utc(self.broker_timestamp, "broker_timestamp"),
        )
        object.__setattr__(
            self,
            "retrieval_timestamp",
            _utc(self.retrieval_timestamp, "retrieval_timestamp"),
        )
        if self.observed_at != self.retrieval_timestamp:
            raise ValidationError("observed_at must equal retrieval_timestamp")
        if abs(self.broker_timestamp - self.retrieval_timestamp) > timedelta(seconds=120):
            raise ValidationError("broker_timestamp exceeds retrieval clock-skew allowance")
        _scope(self.snapshot_id, "snapshot_id")
        _scope(self.source, "source")
        _scope(self.transport_generation, "transport_generation")
        if type(self.status) is not EvidenceStatus:
            raise ValidationError("status must be EvidenceStatus")


@dataclass(frozen=True)
class AccountPosition:
    con_id: int
    symbol: str
    quantity: Decimal

    def __post_init__(self) -> None:
        if type(self.con_id) is not int or self.con_id <= 0:
            raise ValidationError("con_id must be a positive integer")
        _symbol(self.symbol)
        object.__setattr__(self, "quantity", _decimal(self.quantity, "quantity"))


@dataclass(frozen=True)
class PortfolioAllocation:
    portfolio_id: str
    con_id: int
    symbol: str
    quantity: Decimal

    def __post_init__(self) -> None:
        _scope(self.portfolio_id, "portfolio_id")
        if type(self.con_id) is not int or self.con_id <= 0:
            raise ValidationError("con_id must be a positive integer")
        _symbol(self.symbol)
        object.__setattr__(self, "quantity", _decimal(self.quantity, "quantity"))


@dataclass(frozen=True)
class OpenOrderSnapshot:
    observed_at: datetime
    snapshot_id: str
    transport_generation: str
    active_con_ids: Tuple[int, ...] = ()
    complete: bool = True
    all_clients: bool = True
    stable: bool = True
    unknown_order_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        _scope(self.snapshot_id, "snapshot_id")
        _scope(self.transport_generation, "transport_generation")
        if not isinstance(self.active_con_ids, tuple) or any(
            type(con_id) is not int or con_id <= 0 for con_id in self.active_con_ids
        ):
            raise ValidationError("active_con_ids must be a tuple of positive integers")
        if len(set(self.active_con_ids)) != len(self.active_con_ids):
            raise ValidationError("active_con_ids must not contain duplicates")
        for field_name in ("complete", "all_clients", "stable"):
            if type(getattr(self, field_name)) is not bool:
                raise ValidationError(f"{field_name} must be a bool")
        if type(self.unknown_order_count) is not int or self.unknown_order_count < 0:
            raise ValidationError("unknown_order_count must be nonnegative")


@dataclass(frozen=True)
class CoherentSafetySnapshot:
    """One account/allocation/open-order snapshot with explicit completeness."""

    execution_domain_scope: str
    account_scope: str
    observed_at: datetime
    snapshot_id: str
    source: str
    allocation_observed_at: datetime
    allocation_snapshot_id: str
    allocation_source: str
    reconciliation_observed_at: datetime
    reconciliation_snapshot_id: str
    transport_generation: str
    account_positions: Tuple[AccountPosition, ...]
    portfolio_allocations: Tuple[PortfolioAllocation, ...]
    open_orders: OpenOrderSnapshot
    transport_state: TransportState
    reconciliation_status: ReconciliationStatus
    _assembly_marker: object = field(repr=False, compare=False)
    positions_complete: bool = True
    allocations_complete: bool = True
    contracts_complete: bool = True

    def __post_init__(self) -> None:
        if self._assembly_marker is not _TRUSTED_ASSEMBLY_MARKER:
            raise ValidationError(
                "CoherentSafetySnapshot requires the trusted evidence assembly boundary"
            )
        _scope(self.execution_domain_scope, "execution_domain_scope")
        if not isinstance(self.account_scope, str) or not _ACCOUNT_SCOPE_RE.fullmatch(
            self.account_scope
        ):
            raise ValidationError("account_scope must be an opaque account scope")
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        object.__setattr__(
            self,
            "allocation_observed_at",
            _utc(self.allocation_observed_at, "allocation_observed_at"),
        )
        object.__setattr__(
            self,
            "reconciliation_observed_at",
            _utc(self.reconciliation_observed_at, "reconciliation_observed_at"),
        )
        _scope(self.snapshot_id, "snapshot_id")
        _scope(self.source, "source")
        _scope(self.allocation_snapshot_id, "allocation_snapshot_id")
        _scope(self.allocation_source, "allocation_source")
        _scope(self.reconciliation_snapshot_id, "reconciliation_snapshot_id")
        _scope(self.transport_generation, "transport_generation")
        if not isinstance(self.account_positions, tuple) or any(
            type(position) is not AccountPosition for position in self.account_positions
        ):
            raise ValidationError("account_positions must be a tuple of AccountPosition")
        if not isinstance(self.portfolio_allocations, tuple) or any(
            type(allocation) is not PortfolioAllocation for allocation in self.portfolio_allocations
        ):
            raise ValidationError("portfolio_allocations must be a tuple of PortfolioAllocation")
        if type(self.open_orders) is not OpenOrderSnapshot:
            raise ValidationError("open_orders must be OpenOrderSnapshot")
        if type(self.transport_state) is not TransportState:
            raise ValidationError("transport_state must be TransportState")
        if type(self.reconciliation_status) is not ReconciliationStatus:
            raise ValidationError("reconciliation_status must be ReconciliationStatus")
        for field_name in (
            "positions_complete",
            "allocations_complete",
            "contracts_complete",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValidationError(f"{field_name} must be a bool")


_TRUSTED_SNAPSHOT_REGISTRY_LOCK = threading.RLock()
_TRUSTED_SNAPSHOT_REGISTRY: dict[int, tuple[weakref.ReferenceType[CoherentSafetySnapshot], str]] = (
    {}
)


def _trusted_snapshot_digest(snapshot: CoherentSafetySnapshot) -> str:
    payload = canonical_json(
        {
            "account_scope": snapshot.account_scope,
            "execution_domain_scope": snapshot.execution_domain_scope,
            "observed_at": snapshot.observed_at,
            "snapshot_id": snapshot.snapshot_id,
            "source": snapshot.source,
            "allocation_observed_at": snapshot.allocation_observed_at,
            "allocation_snapshot_id": snapshot.allocation_snapshot_id,
            "allocation_source": snapshot.allocation_source,
            "reconciliation_observed_at": snapshot.reconciliation_observed_at,
            "reconciliation_snapshot_id": snapshot.reconciliation_snapshot_id,
            "transport_generation": snapshot.transport_generation,
            "account_positions": tuple(
                {
                    "con_id": position.con_id,
                    "quantity": position.quantity,
                    "symbol": position.symbol,
                }
                for position in snapshot.account_positions
            ),
            "portfolio_allocations": tuple(
                {
                    "con_id": allocation.con_id,
                    "portfolio_id": allocation.portfolio_id,
                    "quantity": allocation.quantity,
                    "symbol": allocation.symbol,
                }
                for allocation in snapshot.portfolio_allocations
            ),
            "open_orders": {
                "active_con_ids": snapshot.open_orders.active_con_ids,
                "all_clients": snapshot.open_orders.all_clients,
                "complete": snapshot.open_orders.complete,
                "observed_at": snapshot.open_orders.observed_at,
                "snapshot_id": snapshot.open_orders.snapshot_id,
                "stable": snapshot.open_orders.stable,
                "transport_generation": snapshot.open_orders.transport_generation,
                "unknown_order_count": snapshot.open_orders.unknown_order_count,
            },
            "transport_state": snapshot.transport_state,
            "reconciliation_status": snapshot.reconciliation_status,
            "positions_complete": snapshot.positions_complete,
            "allocations_complete": snapshot.allocations_complete,
            "contracts_complete": snapshot.contracts_complete,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _discard_trusted_snapshot(
    object_id: int,
    reference: weakref.ReferenceType[CoherentSafetySnapshot],
) -> None:
    with _TRUSTED_SNAPSHOT_REGISTRY_LOCK:
        registration = _TRUSTED_SNAPSHOT_REGISTRY.get(object_id)
        if registration is not None and registration[0] is reference:
            _TRUSTED_SNAPSHOT_REGISTRY.pop(object_id, None)


def _register_trusted_snapshot(snapshot: CoherentSafetySnapshot) -> None:
    """Register the exact assembled object, not merely a copyable marker."""

    object_id = id(snapshot)
    reference = weakref.ref(
        snapshot,
        lambda current, object_id=object_id: _discard_trusted_snapshot(
            object_id,
            current,
        ),
    )
    with _TRUSTED_SNAPSHOT_REGISTRY_LOCK:
        _TRUSTED_SNAPSHOT_REGISTRY[object_id] = (
            reference,
            _trusted_snapshot_digest(snapshot),
        )


def _assert_trusted_snapshot(snapshot: CoherentSafetySnapshot) -> None:
    """Reject dataclass copies, deserialized objects, and caller-forged clones."""

    with _TRUSTED_SNAPSHOT_REGISTRY_LOCK:
        registration = _TRUSTED_SNAPSHOT_REGISTRY.get(id(snapshot))
        if registration is None or registration[0]() is not snapshot:
            raise ValidationError(
                "CoherentSafetySnapshot is not the exact trusted assembled object"
            )
        if not hmac.compare_digest(
            registration[1],
            _trusted_snapshot_digest(snapshot),
        ):
            raise ValidationError("CoherentSafetySnapshot changed after trusted assembly")


def _assemble_coherent_safety_snapshot(**values: object) -> CoherentSafetySnapshot:
    """Internal constructor used only by the typed integration boundary."""

    snapshot = CoherentSafetySnapshot(
        **values,
        _assembly_marker=_TRUSTED_ASSEMBLY_MARKER,
    )
    _register_trusted_snapshot(snapshot)
    return snapshot


@dataclass(frozen=True)
class RuntimeOrderRequest:
    portfolio_id: str
    contract: AuthoritativeContract
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    time_in_force: TimeInForce
    order_ref: str
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    outside_regular_hours: bool = False
    reason: str = ""
    strategy: str = ""

    def __post_init__(self) -> None:
        _scope(self.portfolio_id, "portfolio_id")
        if type(self.contract) is not AuthoritativeContract:
            raise ValidationError("contract must be AuthoritativeContract")
        if type(self.side) is not OrderSide:
            raise ValidationError("side must be OrderSide")
        object.__setattr__(self, "quantity", _decimal(self.quantity, "quantity"))
        if self.quantity <= 0:
            raise ValidationError("quantity must be positive")
        if type(self.order_type) is not OrderType:
            raise ValidationError("order_type must be OrderType")
        if type(self.time_in_force) is not TimeInForce:
            raise ValidationError("time_in_force must be TimeInForce")
        if not isinstance(self.order_ref, str) or not self.order_ref:
            raise ValidationError("order_ref must be non-empty")
        if self.limit_price is not None:
            object.__setattr__(self, "limit_price", _decimal(self.limit_price, "limit_price"))
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", _decimal(self.stop_price, "stop_price"))
        if type(self.outside_regular_hours) is not bool:
            raise ValidationError("outside_regular_hours must be a bool")
        if not isinstance(self.reason, str) or not isinstance(self.strategy, str):
            raise ValidationError("reason and strategy must be strings")


@dataclass(frozen=True)
class RuntimeAuthorization:
    reservation: Reservation
    claim: SubmissionClaim
    decision: SafetyDecision
    contract: AuthoritativeContract
    evidence_snapshot_id: str
    allocation_snapshot_id: str
    contract_snapshot_id: str
    reconciliation_snapshot_id: str
    descriptor_fingerprint: str
    expires_at: datetime
    _permit: SubmissionPermit = field(repr=False)
    _coordinator_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "expires_at", _utc(self.expires_at, "expires_at"))


@dataclass(frozen=True)
class FakeSubmissionReceipt:
    descriptor_fingerprint: str
    submission_number: int
    status: str = "SIMULATED_ACCEPTED"


class FakeOrderSubmitter:
    """In-memory sink; it has no callback or broker-extension surface."""

    def __init__(self, *, fail: bool = False) -> None:
        if type(fail) is not bool:
            raise TypeError("fail must be a bool")
        self._fail = fail
        self._lock = threading.Lock()
        self._descriptors: list[SubmissionDescriptor] = []

    @property
    def descriptors(self) -> Tuple[SubmissionDescriptor, ...]:
        with self._lock:
            return tuple(self._descriptors)

    def _submit(self, descriptor: SubmissionDescriptor) -> FakeSubmissionReceipt:
        if self._fail:
            raise RuntimeError("injected fake submission failure")
        with self._lock:
            self._descriptors.append(descriptor)
            return FakeSubmissionReceipt(
                descriptor_fingerprint=descriptor.fingerprint(),
                submission_number=len(self._descriptors),
            )


class SafetyRuntimeCoordinator:
    """Fail-closed bridge from coherent paper evidence to the durable journal."""

    def __init__(
        self,
        identity: PaperExecutionIdentity,
        journal: SafetyJournal,
        *,
        clock: Optional[Callable[[], datetime]] = None,
        max_evidence_age_seconds: int = SAFETY_MAX_EVIDENCE_AGE_SECONDS,
    ) -> None:
        if type(identity) is not PaperExecutionIdentity:
            raise TypeError("identity must be PaperExecutionIdentity")
        if not isinstance(journal, SafetyJournal):
            raise TypeError("journal must be supplied separately as SafetyJournal")
        if type(max_evidence_age_seconds) is not int or not (
            0 <= max_evidence_age_seconds <= SAFETY_MAX_EVIDENCE_AGE_SECONDS
        ):
            raise ValidationError("max_evidence_age_seconds is outside the safety ceiling")
        self._identity = identity
        self._journal = journal
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._max_evidence_age_seconds = max_evidence_age_seconds
        self._startup_lock = threading.RLock()
        self._started = False
        self._coordinator_token = object()

    @property
    def started(self) -> bool:
        with self._startup_lock:
            return self._started

    def start(self) -> None:
        """Replay the supplied journal and refuse unresolved prior authority."""

        with self._startup_lock:
            if self._started:
                return
            self._started = False
            state = self._journal.replay_and_bind_runtime_path(
                expected_execution_domain_scope=self._identity.execution_domain_scope,
                expected_account_scope=self._identity.account_scope,
            )
            reasons = []
            if state.active_reservations:
                reasons.append("ACTIVE_RESERVATION_AT_STARTUP")
            if state.quarantined_reservations:
                reasons.append("QUARANTINED_RESERVATION_AT_STARTUP")
            if reasons:
                raise RuntimeStartupBlocked(tuple(reasons))
            self._started = True

    def authorize(
        self,
        idempotency_key: str,
        request: RuntimeOrderRequest,
        snapshot: CoherentSafetySnapshot,
    ) -> RuntimeAuthorization:
        """Build exact evidence and atomically acquire one journal permit."""

        with self._startup_lock:
            if not self._started:
                raise RuntimeNotStarted("successful journal replay is required")
            if type(request) is not RuntimeOrderRequest:
                raise TypeError("request must be RuntimeOrderRequest")
            if type(snapshot) is not CoherentSafetySnapshot:
                raise TypeError("snapshot must be CoherentSafetySnapshot")
            _assert_trusted_snapshot(snapshot)

            evaluated_at = _utc(self._clock(), "clock result")
            intent, exposure, allocation, gates, descriptor = self._build_boundary_models(
                request, snapshot, evaluated_at
            )
            _assert_trusted_snapshot(snapshot)
            decision = evaluate_reduce_only(intent, exposure, allocation, gates)
            if decision.outcome is not DecisionOutcome.ALLOW:
                self._journal.record_rejection(
                    idempotency_key,
                    intent,
                    exposure,
                    allocation,
                    gates,
                )
                raise RuntimeAuthorizationBlocked(decision)

            reservation, claim, permit = self._journal.authorize_submission(
                idempotency_key,
                intent,
                exposure,
                allocation,
                gates,
                descriptor,
            )
            if permit is None:
                raise RuntimeAuthorizationBlocked(
                    SafetyDecision(
                        outcome=DecisionOutcome.DENY,
                        risk_effect=decision.risk_effect,
                        reason_codes=("REPLAY_HAS_NO_SUBMISSION_AUTHORITY",),
                        current_quantity=decision.current_quantity,
                        computed_target_quantity=decision.computed_target_quantity,
                        intent_fingerprint=decision.intent_fingerprint,
                    )
                )
            evidence_deadline = min(
                request.contract.retrieval_timestamp,
                snapshot.observed_at,
                snapshot.allocation_observed_at,
                snapshot.open_orders.observed_at,
                snapshot.reconciliation_observed_at,
            ) + timedelta(seconds=self._max_evidence_age_seconds)
            return RuntimeAuthorization(
                reservation=reservation,
                claim=claim,
                decision=decision,
                contract=request.contract,
                evidence_snapshot_id=snapshot.snapshot_id,
                allocation_snapshot_id=snapshot.allocation_snapshot_id,
                contract_snapshot_id=request.contract.snapshot_id,
                reconciliation_snapshot_id=snapshot.reconciliation_snapshot_id,
                descriptor_fingerprint=descriptor.fingerprint(),
                expires_at=evidence_deadline,
                _permit=permit,
                _coordinator_token=self._coordinator_token,
            )

    def submit_fake(
        self,
        authorization: RuntimeAuthorization,
        submitter: FakeOrderSubmitter,
    ) -> FakeSubmissionReceipt:
        """Consume one permit and dispatch only to the sealed in-memory fake."""

        if type(authorization) is not RuntimeAuthorization:
            raise TypeError("authorization must be RuntimeAuthorization")
        if type(submitter) is not FakeOrderSubmitter:
            raise FakeSubmissionOnlyError("only the in-memory FakeOrderSubmitter is accepted")
        if authorization._coordinator_token is not self._coordinator_token:
            raise StateTransitionError("authorization belongs to another coordinator")
        if _utc(self._clock(), "clock result") > authorization.expires_at:
            self._journal.invalidate_unsubmitted_permit(authorization._permit)
            raise RuntimeSafetyError(
                "authorization expired before dispatch; fresh broker evidence is required"
            )
        descriptor = self._journal.consume_submission_permit(authorization._permit)
        receipt = submitter._submit(descriptor)
        if receipt.descriptor_fingerprint != authorization.descriptor_fingerprint:
            raise RuntimeSafetyError("fake receipt does not match authorized descriptor")
        return receipt

    def _build_boundary_models(
        self,
        request: RuntimeOrderRequest,
        snapshot: CoherentSafetySnapshot,
        evaluated_at: datetime,
    ) -> Tuple[
        OrderIntent,
        ExposureEvidence,
        PortfolioAllocationEvidence,
        GateContext,
        SubmissionDescriptor,
    ]:
        contract = request.contract
        hard_blocks = []
        if contract.status is not EvidenceStatus.AUTHORITATIVE:
            hard_blocks.append("CONTRACT_NOT_AUTHORITATIVE")
        if (
            contract.transport_generation != snapshot.transport_generation
            or snapshot.open_orders.transport_generation != snapshot.transport_generation
        ):
            hard_blocks.append("TRANSPORT_GENERATION_LINEAGE_MISMATCH")
        if contract.retrieval_timestamp > evaluated_at:
            hard_blocks.append("FUTURE_CONTRACT_EVIDENCE")
        elif evaluated_at - contract.retrieval_timestamp > timedelta(
            seconds=self._max_evidence_age_seconds
        ):
            hard_blocks.append("STALE_CONTRACT_EVIDENCE")
        if snapshot.reconciliation_observed_at > evaluated_at:
            hard_blocks.append("FUTURE_RECONCILIATION_EVIDENCE")
        elif evaluated_at - snapshot.reconciliation_observed_at > timedelta(
            seconds=self._max_evidence_age_seconds
        ):
            hard_blocks.append("STALE_RECONCILIATION_EVIDENCE")
        collection_times = (
            contract.retrieval_timestamp,
            snapshot.observed_at,
            snapshot.allocation_observed_at,
            snapshot.open_orders.observed_at,
            snapshot.reconciliation_observed_at,
        )
        if max(collection_times) - min(collection_times) > timedelta(
            seconds=self._max_evidence_age_seconds
        ):
            hard_blocks.append("SNAPSHOT_COLLECTION_WINDOW_EXCEEDED")
        if not snapshot.positions_complete:
            hard_blocks.append("ACCOUNT_POSITION_SNAPSHOT_INCOMPLETE")
        if not snapshot.allocations_complete:
            hard_blocks.append("ALLOCATION_SNAPSHOT_INCOMPLETE")
        if not snapshot.contracts_complete:
            hard_blocks.append("CONTRACT_SNAPSHOT_INCOMPLETE")
        if snapshot.open_orders.unknown_order_count:
            hard_blocks.append("UNKNOWN_BROKER_ORDER_EXISTS")

        account_matches = [
            position
            for position in snapshot.account_positions
            if position.con_id == contract.con_id and position.symbol == contract.symbol
        ]
        con_id_symbol_conflict = any(
            position.con_id == contract.con_id and position.symbol != contract.symbol
            for position in snapshot.account_positions
        )
        if con_id_symbol_conflict:
            hard_blocks.append("AUTHORITATIVE_CON_ID_SYMBOL_MISMATCH")
        symbol_con_id_conflict = any(
            position.symbol == contract.symbol and position.con_id != contract.con_id
            for position in snapshot.account_positions
        )
        if symbol_con_id_conflict:
            hard_blocks.append("AUTHORITATIVE_SYMBOL_CON_ID_AMBIGUOUS")
        if len(account_matches) != 1:
            hard_blocks.append(
                "ACCOUNT_POSITION_MISSING" if not account_matches else "DUPLICATE_ACCOUNT_POSITION"
            )
        account_quantity = (
            account_matches[0].quantity if len(account_matches) == 1 else Decimal("0")
        )

        matching_allocations = [
            allocation
            for allocation in snapshot.portfolio_allocations
            if allocation.con_id == contract.con_id and allocation.symbol == contract.symbol
        ]
        allocation_symbol_conflict = any(
            allocation.con_id == contract.con_id and allocation.symbol != contract.symbol
            for allocation in snapshot.portfolio_allocations
        )
        if allocation_symbol_conflict:
            hard_blocks.append("ALLOCATION_CON_ID_SYMBOL_MISMATCH")
        portfolio_ids = [allocation.portfolio_id for allocation in matching_allocations]
        if len(portfolio_ids) != len(set(portfolio_ids)):
            hard_blocks.append("DUPLICATE_PORTFOLIO_ALLOCATION")
        requested_matches = [
            allocation
            for allocation in matching_allocations
            if allocation.portfolio_id == request.portfolio_id
        ]
        if len(requested_matches) != 1:
            hard_blocks.append(
                "PORTFOLIO_ALLOCATION_MISSING"
                if not requested_matches
                else "DUPLICATE_REQUESTED_PORTFOLIO_ALLOCATION"
            )
        portfolio_quantity = (
            requested_matches[0].quantity if len(requested_matches) == 1 else Decimal("0")
        )
        aggregate_quantity = Decimal("0")
        try:
            for row in matching_allocations:
                aggregate_quantity = _exact_decimal_add(
                    aggregate_quantity,
                    row.quantity,
                    "aggregate allocated quantity",
                )
        except ValidationError:
            aggregate_quantity = Decimal("0")
            hard_blocks.append("AGGREGATE_ALLOCATION_ARITHMETIC_INVALID")
        has_positive = any(row.quantity > 0 for row in matching_allocations)
        has_negative = any(row.quantity < 0 for row in matching_allocations)
        has_offsetting = has_positive and has_negative

        signed_delta = (
            request.quantity
            if request.side in {OrderSide.BUY, OrderSide.BUY_TO_COVER}
            else request.quantity.copy_negate()
        )
        try:
            account_target = _exact_decimal_add(account_quantity, signed_delta, "account target")
            portfolio_target = _exact_decimal_add(
                portfolio_quantity, signed_delta, "portfolio target"
            )
        except ValidationError:
            account_target = account_quantity
            portfolio_target = portfolio_quantity
            hard_blocks.append("TARGET_ARITHMETIC_INVALID")

        evidence_status = (
            EvidenceStatus.AUTHORITATIVE
            if len(account_matches) == 1 and snapshot.positions_complete
            else EvidenceStatus.FAILED
        )
        allocation_status = (
            EvidenceStatus.AUTHORITATIVE
            if len(requested_matches) == 1
            and len(portfolio_ids) == len(set(portfolio_ids))
            and snapshot.allocations_complete
            else EvidenceStatus.FAILED
        )
        intent = OrderIntent(
            execution_domain_scope=self._identity.execution_domain_scope,
            account_scope=self._identity.account_scope,
            portfolio_id=request.portfolio_id,
            con_id=contract.con_id,
            symbol=contract.symbol,
            side=request.side,
            quantity=request.quantity,
            account_current_quantity=account_quantity,
            target_quantity=account_target,
            portfolio_current_quantity=portfolio_quantity,
            portfolio_target_quantity=portfolio_target,
            created_at=evaluated_at,
            reduce_only=True,
            reason=request.reason,
            strategy=request.strategy,
        )
        exposure = ExposureEvidence(
            execution_domain_scope=snapshot.execution_domain_scope,
            account_scope=snapshot.account_scope,
            con_id=contract.con_id,
            symbol=contract.symbol,
            position_quantity=account_quantity,
            observed_at=snapshot.observed_at,
            status=evidence_status,
            source=snapshot.source,
            snapshot_id=snapshot.snapshot_id,
        )
        allocation = PortfolioAllocationEvidence(
            execution_domain_scope=snapshot.execution_domain_scope,
            account_scope=snapshot.account_scope,
            portfolio_id=request.portfolio_id,
            con_id=contract.con_id,
            symbol=contract.symbol,
            position_quantity=portfolio_quantity,
            aggregate_allocated_quantity=aggregate_quantity,
            has_offsetting_allocations=has_offsetting,
            observed_at=snapshot.allocation_observed_at,
            status=allocation_status,
            source=snapshot.allocation_source,
            snapshot_id=snapshot.allocation_snapshot_id,
        )
        active_order_count = len(snapshot.open_orders.active_con_ids)
        gates = GateContext(
            execution_domain_scope=snapshot.execution_domain_scope,
            account_scope=snapshot.account_scope,
            con_id=contract.con_id,
            evaluated_at=evaluated_at,
            max_evidence_age_seconds=self._max_evidence_age_seconds,
            transport_state=snapshot.transport_state,
            reconciliation_status=snapshot.reconciliation_status,
            open_orders_complete=snapshot.open_orders.complete,
            open_orders_all_clients=snapshot.open_orders.all_clients,
            open_orders_snapshot_stable=snapshot.open_orders.stable,
            open_orders_observed_at=snapshot.open_orders.observed_at,
            open_orders_snapshot_id=snapshot.open_orders.snapshot_id,
            active_order_count=active_order_count,
            soft_entry_allowed=False,
            hard_block_reasons=tuple(hard_blocks),
        )
        descriptor = SubmissionDescriptor(
            execution_domain_scope=self._identity.execution_domain_scope,
            account_scope=self._identity.account_scope,
            con_id=contract.con_id,
            side=request.side,
            quantity=request.quantity,
            order_type=request.order_type,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            time_in_force=request.time_in_force,
            outside_regular_hours=request.outside_regular_hours,
            order_ref=request.order_ref,
        )
        return intent, exposure, allocation, gates, descriptor
