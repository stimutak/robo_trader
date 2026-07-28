"""Pure fail-closed policy for dormant broker-reconciliation evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable

from .domain import (
    DOMAIN_SCHEMA_VERSION,
    ExecutionDomainScope,
    NormalizedBrokerSnapshot,
    ReconciliationDomainError,
    _account_scope,
    _schema_version,
    _timestamp,
    canonical_json,
    canonical_timestamp,
    fingerprint,
)

_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")
_SUBJECT = re.compile(r"^(?:[A-Z][A-Z0-9._-]{0,31}|broker_snapshot|ibkr_account|local_ledger)$")
_EVIDENCE_ID = re.compile(
    r"^(?:(?:broker-reconciliation|broker-event|ledger-event|timing-proof)-v1-" r"[0-9a-f]{64})$"
)
_SNAPSHOT_ID = re.compile(r"^broker-reconciliation-v1-[0-9a-f]{64}$")
_BROKER_EVENT_ID = re.compile(r"^broker-event-v1-[0-9a-f]{64}$")
_TIMING_PROOF_ID = re.compile(r"^timing-proof-v1-[0-9a-f]{64}$")
_ACCOUNT_FRAGMENT = re.compile(r"(?:DU|U)\d{4,}", re.IGNORECASE)
MAX_EXPECTED_TIMING_LAG_SECONDS = 120
_ELIGIBLE_TIMING_LAG_REASONS = frozenset(
    {
        "BROKER_ORDER_EVENT_PENDING",
        "BROKER_EXECUTION_EVENT_PENDING",
        "BROKER_COMMISSION_REPORT_PENDING",
    }
)


class DifferenceKind(str, Enum):
    """Exact PR 5 mismatch taxonomy from the remediation plan."""

    EXPECTED_TIMING_LAG = "expected_timing_lag"
    RECOVERABLE_MISSING_EVENT = "recoverable_missing_event"
    DUPLICATE_EVENT = "duplicate_event"
    ACCOUNT_MISMATCH = "account_mismatch"
    QUANTITY_MISMATCH = "quantity_mismatch"
    CASH_MISMATCH = "cash_mismatch"
    UNKNOWN = "unknown"


class DifferenceMateriality(str, Enum):
    INFORMATIONAL = "informational"
    MATERIAL = "material"
    UNKNOWN = "unknown"


class ReconciliationStatus(str, Enum):
    PASSED = "passed"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"


_REQUIRED_MATERIALITY = {
    DifferenceKind.EXPECTED_TIMING_LAG: DifferenceMateriality.INFORMATIONAL,
    DifferenceKind.RECOVERABLE_MISSING_EVENT: DifferenceMateriality.MATERIAL,
    DifferenceKind.DUPLICATE_EVENT: DifferenceMateriality.MATERIAL,
    DifferenceKind.ACCOUNT_MISMATCH: DifferenceMateriality.MATERIAL,
    DifferenceKind.QUANTITY_MISMATCH: DifferenceMateriality.MATERIAL,
    DifferenceKind.CASH_MISMATCH: DifferenceMateriality.MATERIAL,
    DifferenceKind.UNKNOWN: DifferenceMateriality.UNKNOWN,
}


def _safe_identifier(value: object, field_name: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or value != value.strip() or not pattern.fullmatch(value):
        raise ReconciliationDomainError(f"{field_name} is malformed")
    if _ACCOUNT_FRAGMENT.search(value):
        raise ReconciliationDomainError(f"{field_name} contains raw account identity")
    return value


def _timing_proof_payload(
    *,
    broker_snapshot_id: str,
    difference_kind: DifferenceKind,
    reason_code: str,
    subject: str,
    broker_event_id: str,
    started_at: datetime,
    expires_at: datetime,
    schema_version: int,
) -> dict[str, object]:
    return {
        "broker_event_id": broker_event_id,
        "broker_snapshot_id": broker_snapshot_id,
        "difference_kind": difference_kind.value,
        "expires_at": canonical_timestamp(expires_at),
        "reason_code": reason_code,
        "schema_version": schema_version,
        "started_at": canonical_timestamp(started_at),
        "subject": subject,
    }


@dataclass(frozen=True, slots=True)
class ExpectedTimingLagProof:
    """Trusted producer evidence bound to one exact snapshot difference."""

    broker_snapshot_id: str
    difference_kind: DifferenceKind
    reason_code: str
    subject: str
    broker_event_id: str
    started_at: datetime
    expires_at: datetime
    proof_id: str
    schema_version: int = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "timing lag proof")
        object.__setattr__(
            self,
            "broker_snapshot_id",
            _safe_identifier(self.broker_snapshot_id, "broker_snapshot_id", _SNAPSHOT_ID),
        )
        if self.difference_kind is not DifferenceKind.EXPECTED_TIMING_LAG:
            raise ReconciliationDomainError("timing proof difference kind is ineligible")
        object.__setattr__(
            self,
            "reason_code",
            _safe_identifier(self.reason_code, "timing proof reason_code", _REASON_CODE),
        )
        if self.reason_code not in _ELIGIBLE_TIMING_LAG_REASONS:
            raise ReconciliationDomainError("timing proof reason is not eligible")
        object.__setattr__(
            self,
            "subject",
            _safe_identifier(self.subject, "timing proof subject", _SUBJECT),
        )
        object.__setattr__(
            self,
            "broker_event_id",
            _safe_identifier(
                self.broker_event_id,
                "timing proof broker_event_id",
                _BROKER_EVENT_ID,
            ),
        )
        started_at = _timestamp(self.started_at, "timing proof started_at")
        expires_at = _timestamp(self.expires_at, "timing proof expires_at")
        if expires_at <= started_at:
            raise ReconciliationDomainError("timing proof bound is reversed")
        if (expires_at - started_at).total_seconds() > MAX_EXPECTED_TIMING_LAG_SECONDS:
            raise ReconciliationDomainError("timing proof bound exceeds policy maximum")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(
            self,
            "proof_id",
            _safe_identifier(self.proof_id, "timing proof_id", _TIMING_PROOF_ID),
        )
        if self.proof_id != fingerprint("timing-proof-v1", self.binding_dict()):
            raise ReconciliationDomainError("timing proof fingerprint is not bound")

    @classmethod
    def from_trusted_producer(
        cls,
        *,
        broker_snapshot_id: str,
        reason_code: str,
        subject: str,
        broker_event_id: str,
        started_at: datetime,
        expires_at: datetime,
    ) -> ExpectedTimingLagProof:
        normalized_snapshot_id = _safe_identifier(
            broker_snapshot_id,
            "broker_snapshot_id",
            _SNAPSHOT_ID,
        )
        normalized_reason = _safe_identifier(
            reason_code,
            "timing proof reason_code",
            _REASON_CODE,
        )
        normalized_subject = _safe_identifier(subject, "timing proof subject", _SUBJECT)
        normalized_event = _safe_identifier(
            broker_event_id,
            "timing proof broker_event_id",
            _BROKER_EVENT_ID,
        )
        normalized_started = _timestamp(started_at, "timing proof started_at")
        normalized_expires = _timestamp(expires_at, "timing proof expires_at")
        payload = _timing_proof_payload(
            broker_snapshot_id=normalized_snapshot_id,
            difference_kind=DifferenceKind.EXPECTED_TIMING_LAG,
            reason_code=normalized_reason,
            subject=normalized_subject,
            broker_event_id=normalized_event,
            started_at=normalized_started,
            expires_at=normalized_expires,
            schema_version=DOMAIN_SCHEMA_VERSION,
        )
        return cls(
            broker_snapshot_id=normalized_snapshot_id,
            difference_kind=DifferenceKind.EXPECTED_TIMING_LAG,
            reason_code=normalized_reason,
            subject=normalized_subject,
            broker_event_id=normalized_event,
            started_at=normalized_started,
            expires_at=normalized_expires,
            proof_id=fingerprint("timing-proof-v1", payload),
        )

    def binding_dict(self) -> dict[str, object]:
        return _timing_proof_payload(
            broker_snapshot_id=self.broker_snapshot_id,
            difference_kind=self.difference_kind,
            reason_code=self.reason_code,
            subject=self.subject,
            broker_event_id=self.broker_event_id,
            started_at=self.started_at,
            expires_at=self.expires_at,
            schema_version=self.schema_version,
        )

    @property
    def binding_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.broker_snapshot_id,
            self.difference_kind.value,
            self.reason_code,
            self.subject,
            self.broker_event_id,
        )

    def canonical_dict(self) -> dict[str, object]:
        return {**self.binding_dict(), "proof_id": self.proof_id}


@dataclass(frozen=True, slots=True)
class ReconciliationDifference:
    """One deterministic, public-safe reconciliation difference."""

    kind: DifferenceKind
    materiality: DifferenceMateriality
    reason_code: str
    subject: str
    evidence_ids: tuple[str, ...] = ()
    schema_version: int = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "difference")
        if type(self.kind) is not DifferenceKind:
            raise ReconciliationDomainError("difference kind is invalid")
        if type(self.materiality) is not DifferenceMateriality:
            raise ReconciliationDomainError("difference materiality is invalid")
        if self.materiality is not _REQUIRED_MATERIALITY[self.kind]:
            raise ReconciliationDomainError("difference materiality contradicts its kind")
        object.__setattr__(
            self,
            "reason_code",
            _safe_identifier(self.reason_code, "difference reason_code", _REASON_CODE),
        )
        object.__setattr__(
            self,
            "subject",
            _safe_identifier(self.subject, "difference subject", _SUBJECT),
        )
        evidence_ids = tuple(self.evidence_ids)
        if any(
            _safe_identifier(value, "difference evidence_id", _EVIDENCE_ID) != value
            for value in evidence_ids
        ):
            raise ReconciliationDomainError("difference evidence IDs are not canonical")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ReconciliationDomainError("difference evidence IDs contain duplicates")
        object.__setattr__(self, "evidence_ids", tuple(sorted(evidence_ids)))
        if self.kind is DifferenceKind.EXPECTED_TIMING_LAG:
            if self.reason_code not in _ELIGIBLE_TIMING_LAG_REASONS:
                raise ReconciliationDomainError("timing lag reason is not eligible")
            if len(evidence_ids) != 1 or not _BROKER_EVENT_ID.fullmatch(evidence_ids[0]):
                raise ReconciliationDomainError("timing lag requires one bound broker event")

    @property
    def identity(self) -> tuple[str, str, str, tuple[str, ...]]:
        return (self.kind.value, self.reason_code, self.subject, self.evidence_ids)

    def canonical_dict(self) -> dict[str, object]:
        return {
            "evidence_ids": list(self.evidence_ids),
            "kind": self.kind.value,
            "materiality": self.materiality.value,
            "reason_code": self.reason_code,
            "schema_version": self.schema_version,
            "subject": self.subject,
        }


@dataclass(frozen=True, slots=True)
class ReconciliationCoverage:
    """Explicit comparison coverage; absence can never be interpreted as clean."""

    broker_account: bool
    broker_positions: bool
    broker_open_orders: bool
    broker_completed_orders: bool
    broker_executions: bool
    broker_commissions: bool
    ledger_positions: bool
    ledger_orders: bool
    ledger_executions: bool
    ledger_cash: bool

    def __post_init__(self) -> None:
        if any(type(getattr(self, field_name)) is not bool for field_name in self.__slots__):
            raise ReconciliationDomainError("reconciliation coverage flags must be exact booleans")

    @property
    def complete(self) -> bool:
        return all(getattr(self, field_name) for field_name in self.__slots__)

    def canonical_dict(self) -> dict[str, bool]:
        return {field_name: getattr(self, field_name) for field_name in self.__slots__}


@dataclass(frozen=True, slots=True)
class ReconciliationVerdict:
    """A non-authorizing result produced only by the pure policy evaluator."""

    execution_domain_scope: ExecutionDomainScope
    broker_snapshot_id: str
    expected_account_scope: str
    checked_at: datetime
    fresh_until: datetime
    evidence_fresh: bool
    comparison_complete: bool
    coverage: ReconciliationCoverage
    status: ReconciliationStatus
    differences: tuple[ReconciliationDifference, ...]
    schema_version: int = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "verdict")
        if self.execution_domain_scope is not ExecutionDomainScope.PAPER_SIMULATOR:
            raise ReconciliationDomainError(
                "this dormant policy supports only the local paper-simulator domain"
            )
        object.__setattr__(
            self,
            "broker_snapshot_id",
            _safe_identifier(self.broker_snapshot_id, "broker_snapshot_id", _SNAPSHOT_ID),
        )
        object.__setattr__(
            self,
            "expected_account_scope",
            _account_scope(self.expected_account_scope),
        )
        object.__setattr__(self, "checked_at", _timestamp(self.checked_at, "checked_at"))
        object.__setattr__(
            self,
            "fresh_until",
            _timestamp(self.fresh_until, "fresh_until"),
        )
        if self.fresh_until < self.checked_at and self.evidence_fresh:
            raise ReconciliationDomainError("fresh verdict expires before it was checked")
        if type(self.evidence_fresh) is not bool or type(self.comparison_complete) is not bool:
            raise ReconciliationDomainError("verdict evidence flags must be exact booleans")
        if type(self.coverage) is not ReconciliationCoverage:
            raise ReconciliationDomainError("verdict coverage is malformed")
        if type(self.status) is not ReconciliationStatus:
            raise ReconciliationDomainError("verdict status is malformed")
        differences = tuple(self.differences)
        if any(type(value) is not ReconciliationDifference for value in differences):
            raise ReconciliationDomainError("verdict differences are not normalized")
        identities = tuple(value.identity for value in differences)
        if len(identities) != len(set(identities)):
            raise ReconciliationDomainError("verdict contains duplicate differences")
        ordered = tuple(sorted(differences, key=lambda value: value.identity))
        object.__setattr__(self, "differences", ordered)
        expected_status = _status_for(ordered)
        if self.status is not expected_status:
            raise ReconciliationDomainError("verdict status contradicts its differences")
        if self.comparison_complete is not self.coverage.complete:
            raise ReconciliationDomainError("verdict completeness contradicts its coverage")
        reason_codes = {difference.reason_code for difference in ordered}
        if not self.evidence_fresh and "BROKER_EVIDENCE_STALE" not in reason_codes:
            raise ReconciliationDomainError("stale verdict lacks fail-closed difference")
        if not self.comparison_complete and "LOCAL_COMPARISON_INCOMPLETE" not in reason_codes:
            raise ReconciliationDomainError("partial verdict lacks fail-closed difference")

    @property
    def quarantine_required(self) -> bool:
        return self.status is ReconciliationStatus.QUARANTINED

    @property
    def mutated_state(self) -> bool:
        return False

    @property
    def authorizes_startup(self) -> bool:
        return False

    def canonical_dict(self) -> dict[str, object]:
        return {
            "authorizes_startup": False,
            "broker_snapshot_id": self.broker_snapshot_id,
            "checked_at": canonical_timestamp(self.checked_at),
            "comparison_complete": self.comparison_complete,
            "coverage": self.coverage.canonical_dict(),
            "differences": [difference.canonical_dict() for difference in self.differences],
            "evidence_fresh": self.evidence_fresh,
            "execution_domain_scope": self.execution_domain_scope.value,
            "expected_account_scope": self.expected_account_scope,
            "fresh_until": canonical_timestamp(self.fresh_until),
            "mutated_state": False,
            "quarantine_required": self.quarantine_required,
            "schema_version": self.schema_version,
            "status": self.status.value,
        }

    def canonical_payload(self) -> str:
        return canonical_json(self.canonical_dict())

    @property
    def verdict_id(self) -> str:
        return fingerprint("reconciliation-verdict-v1", self.canonical_dict())


def _difference(
    kind: DifferenceKind,
    reason_code: str,
    subject: str,
    *evidence_ids: str,
) -> ReconciliationDifference:
    return ReconciliationDifference(
        kind=kind,
        materiality=_REQUIRED_MATERIALITY[kind],
        reason_code=reason_code,
        subject=subject,
        evidence_ids=tuple(evidence_ids),
    )


def _status_for(differences: Iterable[ReconciliationDifference]) -> ReconciliationStatus:
    materialities = {difference.materiality for difference in differences}
    if materialities & {DifferenceMateriality.MATERIAL, DifferenceMateriality.UNKNOWN}:
        return ReconciliationStatus.QUARANTINED
    if DifferenceMateriality.INFORMATIONAL in materialities:
        return ReconciliationStatus.DEGRADED
    return ReconciliationStatus.PASSED


def evaluate_paper_simulator_reconciliation(
    snapshot: NormalizedBrokerSnapshot,
    coverage: ReconciliationCoverage,
    differences: Iterable[ReconciliationDifference] = (),
    timing_lag_proofs: Iterable[ExpectedTimingLagProof] = (),
    *,
    expected_account_scope: str,
    now: datetime,
    max_age_seconds: float,
) -> ReconciliationVerdict:
    """Evaluate containment without equating IBKR and simulator positions.

    The current execution authority is the local paper simulator.  Therefore a
    nonzero IBKR position or open IBKR order is unexpected external exposure,
    not a simulator-ledger row to copy, replace, or automatically reconcile.
    """

    if type(snapshot) is not NormalizedBrokerSnapshot:
        raise ReconciliationDomainError("policy requires one normalized broker snapshot")
    if type(coverage) is not ReconciliationCoverage:
        raise ReconciliationDomainError("policy requires explicit reconciliation coverage")
    expected_scope = _account_scope(expected_account_scope)
    checked_at = _timestamp(now, "policy clock")
    evidence_fresh = snapshot.is_fresh(now=checked_at, max_age_seconds=max_age_seconds)
    fresh_until = snapshot.retrieved_at + timedelta(seconds=float(max_age_seconds))

    normalized = tuple(differences)
    if any(type(value) is not ReconciliationDifference for value in normalized):
        raise ReconciliationDomainError("policy differences are not normalized")
    collected = {difference.identity: difference for difference in normalized}
    normalized_proofs = tuple(timing_lag_proofs)
    if any(type(value) is not ExpectedTimingLagProof for value in normalized_proofs):
        raise ReconciliationDomainError("policy timing-lag proofs are not normalized")
    proof_keys = tuple(proof.binding_key for proof in normalized_proofs)
    if len(proof_keys) != len(set(proof_keys)):
        raise ReconciliationDomainError("policy contains duplicate timing-lag proofs")
    proofs_by_key = {proof.binding_key: proof for proof in normalized_proofs}

    def add(value: ReconciliationDifference) -> None:
        collected[value.identity] = value

    for difference in normalized:
        if difference.kind is not DifferenceKind.EXPECTED_TIMING_LAG:
            continue
        broker_event_id = difference.evidence_ids[0]
        proof = proofs_by_key.get(
            (
                snapshot.snapshot_id,
                difference.kind.value,
                difference.reason_code,
                difference.subject,
                broker_event_id,
            )
        )
        if proof is None or not proof.started_at <= checked_at <= proof.expires_at:
            evidence_ids = difference.evidence_ids
            if proof is not None:
                evidence_ids += (proof.proof_id,)
            add(
                _difference(
                    DifferenceKind.UNKNOWN,
                    "EXPECTED_TIMING_LAG_UNPROVEN_OR_EXPIRED",
                    difference.subject,
                    *evidence_ids,
                )
            )

    if snapshot.account.account_scope != expected_scope:
        add(
            _difference(
                DifferenceKind.ACCOUNT_MISMATCH,
                "ACCOUNT_SCOPE_MISMATCH",
                "ibkr_account",
                snapshot.snapshot_id,
            )
        )
    if not evidence_fresh:
        add(
            _difference(
                DifferenceKind.UNKNOWN,
                "BROKER_EVIDENCE_STALE",
                "broker_snapshot",
                snapshot.snapshot_id,
            )
        )
    if not snapshot.completeness.complete:
        add(
            _difference(
                DifferenceKind.UNKNOWN,
                "BROKER_EVIDENCE_INCOMPLETE",
                "broker_snapshot",
                snapshot.snapshot_id,
            )
        )
    if not coverage.complete:
        add(
            _difference(
                DifferenceKind.UNKNOWN,
                "LOCAL_COMPARISON_INCOMPLETE",
                "local_ledger",
                snapshot.snapshot_id,
            )
        )
    for position in snapshot.positions:
        add(
            _difference(
                DifferenceKind.QUANTITY_MISMATCH,
                "UNEXPECTED_IBKR_POSITION_IN_PAPER_SIMULATOR",
                position.symbol,
                snapshot.snapshot_id,
            )
        )
    for order in snapshot.open_orders:
        add(
            _difference(
                DifferenceKind.UNKNOWN,
                "UNEXPECTED_IBKR_OPEN_ORDER_IN_PAPER_SIMULATOR",
                order.symbol,
                snapshot.snapshot_id,
            )
        )

    ordered = tuple(sorted(collected.values(), key=lambda value: value.identity))
    return ReconciliationVerdict(
        execution_domain_scope=ExecutionDomainScope.PAPER_SIMULATOR,
        broker_snapshot_id=snapshot.snapshot_id,
        expected_account_scope=expected_scope,
        checked_at=checked_at,
        fresh_until=fresh_until,
        evidence_fresh=evidence_fresh,
        comparison_complete=coverage.complete,
        coverage=coverage,
        status=_status_for(ordered),
        differences=ordered,
    )
