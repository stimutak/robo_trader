"""Dormant trigger-aware reconciliation service with fail-closed eligibility."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Protocol

from robo_trader.safety.sqlite_identity import SQLitePathBinding

from .domain import ReconciliationDomainError, _timestamp
from .ibkr_adapter import await_cleanup_required
from .persistence import PersistedReconciliation, ReconciliationPersistence
from .policy import (
    ReconciliationStatus,
    ReconciliationVerdict,
    evaluate_paper_simulator_reconciliation,
)
from .runtime_evidence import (
    VerifiedRuntimeReconciliationEvidence,
    assert_and_consume_verified_runtime_reconciliation_evidence,
    assert_runtime_reconciliation_evidence_sources_current,
)


class ReconciliationServiceBlocked(ReconciliationDomainError):
    """Reconciliation could not produce and persist trusted entry evidence."""


class ReconciliationTrigger(str, Enum):
    STARTUP = "startup"
    RECONNECT = "reconnect"
    PERIODIC = "periodic"
    BEFORE_LIVE = "before_live"
    AMBIGUOUS_ORDER = "ambiguous_order"


class ReconciliationServiceState(str, Enum):
    UNINITIALIZED = "uninitialized"
    IDLE = "idle"
    RUNNING = "running"
    READY = "ready"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ReconciliationServiceOutcome:
    trigger: ReconciliationTrigger
    verdict: ReconciliationVerdict
    persisted: PersistedReconciliation
    state: ReconciliationServiceState
    entry_eligible: bool
    eligible_until: datetime


class VerifiedEvidenceSource(Protocol):
    """Narrow source implemented by the core-authenticated provider pipeline.

    A structurally compatible source is not trusted: every result is consumed
    through the exact one-shot runtime evidence assertion before any field is
    inspected.
    """

    async def collect_verified_evidence(
        self, *, max_age_seconds: float
    ) -> VerifiedRuntimeReconciliationEvidence:
        """Collect one exact core-bound read-only broker generation."""

    async def close(self) -> None:
        """Close only the diagnostic read-only transport."""


def _system_clock() -> datetime:
    return datetime.now(timezone.utc)


class ReconciliationService:
    """Serialize startup, reconnect, and periodic reconciliation generations.

    The service never repairs broker or local state. It only collects evidence,
    evaluates the merged pure policy, appends durable records, and reports
    whether new entry eligibility remains quarantined.
    """

    def __init__(
        self,
        *,
        evidence_source: VerifiedEvidenceSource,
        persistence: ReconciliationPersistence,
        expected_account_scope: str,
        max_age_seconds: float = 30.0,
        periodic_interval_seconds: float = 15.0,
        clock: Callable[[], datetime] = _system_clock,
    ) -> None:
        if not isinstance(max_age_seconds, (int, float)) or isinstance(max_age_seconds, bool):
            raise ReconciliationServiceBlocked("maximum evidence age must be numeric")
        if not isinstance(periodic_interval_seconds, (int, float)) or isinstance(
            periodic_interval_seconds, bool
        ):
            raise ReconciliationServiceBlocked("periodic interval must be numeric")
        if (
            not math.isfinite(float(max_age_seconds))
            or float(max_age_seconds) <= 0
            or not math.isfinite(float(periodic_interval_seconds))
            or float(periodic_interval_seconds) <= 0
            or float(periodic_interval_seconds) > float(max_age_seconds)
        ):
            raise ReconciliationServiceBlocked(
                "periodic interval must be positive and no longer than evidence age"
            )
        if not callable(getattr(evidence_source, "collect_verified_evidence", None)):
            raise ReconciliationServiceBlocked("verified evidence source is unavailable")
        if not callable(getattr(evidence_source, "close", None)):
            raise ReconciliationServiceBlocked("evidence source cleanup is unavailable")
        self._evidence_source = evidence_source
        self._persistence = persistence
        self._expected_account_scope = expected_account_scope
        self._max_age_seconds = float(max_age_seconds)
        self._periodic_interval = timedelta(seconds=float(periodic_interval_seconds))
        self._clock = clock
        self._lock = asyncio.Lock()
        self._initialized = False
        self._state = ReconciliationServiceState.UNINITIALIZED
        self._latest_outcome: ReconciliationServiceOutcome | None = None
        self._last_completed_at: datetime | None = None
        self._latest_database_binding: tuple[str, int, int] | None = None

    @property
    def state(self) -> ReconciliationServiceState:
        return self._state

    @property
    def latest_outcome(self) -> ReconciliationServiceOutcome | None:
        return self._latest_outcome

    def entry_eligible(self, *, at: datetime | None = None) -> bool:
        """Return false once durable evidence is quarantined, missing, or stale."""

        outcome = self._latest_outcome
        if (
            outcome is None
            or not outcome.entry_eligible
            or self._state
            not in {ReconciliationServiceState.READY, ReconciliationServiceState.DEGRADED}
        ):
            return False
        try:
            checked_at = _timestamp(at if at is not None else self._clock(), "eligibility clock")
        except Exception:
            self._quarantine()
            return False
        if (
            checked_at < outcome.verdict.checked_at
            or self._last_completed_at is None
            or checked_at < self._last_completed_at
        ):
            self._quarantine()
            return False
        if checked_at > outcome.eligible_until:
            self._quarantine()
            return False
        database_binding = self._latest_database_binding
        if database_binding is None:
            self._quarantine()
            return False
        path, expected_device, expected_inode = database_binding
        binding: SQLitePathBinding | None = None
        try:
            binding = SQLitePathBinding.open_readonly(Path(path))
            binding.assert_path_identity()
            if (binding.device, binding.inode) != (expected_device, expected_inode):
                raise ReconciliationServiceBlocked("runtime database identity changed")
        except Exception:
            self._quarantine()
            return False
        finally:
            if binding is not None:
                try:
                    binding.close()
                except Exception:
                    self._quarantine()
                    return False
        return True

    async def initialize(self) -> None:
        async with self._lock:
            self._assert_not_closed()
            await self._initialize_locked()

    async def reconcile_startup(self) -> ReconciliationServiceOutcome:
        return await self._run(ReconciliationTrigger.STARTUP)

    async def reconcile_reconnect(self) -> ReconciliationServiceOutcome:
        return await self._run(ReconciliationTrigger.RECONNECT)

    async def reconcile_before_live(self) -> ReconciliationServiceOutcome:
        return await self._run(ReconciliationTrigger.BEFORE_LIVE)

    async def reconcile_ambiguous_order(self) -> ReconciliationServiceOutcome:
        return await self._run(ReconciliationTrigger.AMBIGUOUS_ORDER)

    async def reconcile_periodic_if_due(self) -> ReconciliationServiceOutcome | None:
        async with self._lock:
            self._assert_not_closed()
            await self._initialize_locked()
            now = self._clock_value("periodic clock")
            if self._last_completed_at is not None:
                if now < self._last_completed_at:
                    self._quarantine()
                    raise ReconciliationServiceBlocked("periodic clock moved backwards")
                if now - self._last_completed_at < self._periodic_interval:
                    return None
            return await self._run_locked(ReconciliationTrigger.PERIODIC, started_at=now)

    async def close(self) -> None:
        async with self._lock:
            if self._state is ReconciliationServiceState.CLOSED:
                return
            self._latest_outcome = None
            self._last_completed_at = None
            self._latest_database_binding = None
            self._state = ReconciliationServiceState.CLOSING
            try:
                cancellation_received = await await_cleanup_required(self._evidence_source.close())
            except BaseException:
                self._state = ReconciliationServiceState.CLOSING
                raise
            self._state = ReconciliationServiceState.CLOSED
            if cancellation_received:
                raise asyncio.CancelledError

    async def _run(self, trigger: ReconciliationTrigger) -> ReconciliationServiceOutcome:
        async with self._lock:
            self._assert_not_closed()
            await self._initialize_locked()
            return await self._run_locked(trigger, started_at=self._clock_value("run clock"))

    async def _initialize_locked(self) -> None:
        if self._initialized:
            return
        try:
            await self._persistence.initialize()
        except BaseException as exc:
            self._quarantine()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise ReconciliationServiceBlocked(
                "reconciliation persistence initialization failed"
            ) from exc
        self._initialized = True
        self._state = ReconciliationServiceState.IDLE

    async def _run_locked(
        self,
        trigger: ReconciliationTrigger,
        *,
        started_at: datetime,
    ) -> ReconciliationServiceOutcome:
        self._state = ReconciliationServiceState.RUNNING
        try:
            produced = await self._evidence_source.collect_verified_evidence(
                max_age_seconds=self._max_age_seconds
            )
            runtime_evidence = assert_and_consume_verified_runtime_reconciliation_evidence(produced)
            snapshot = runtime_evidence.snapshot
            checked_at = self._clock_value("policy clock")
            if checked_at < started_at:
                raise ReconciliationServiceBlocked("policy clock moved backwards")
            verdict = evaluate_paper_simulator_reconciliation(
                snapshot,
                runtime_evidence.comparison_coverage,
                runtime_evidence.differences,
                runtime_evidence.timing_lag_proofs,
                expected_account_scope=self._expected_account_scope,
                now=checked_at,
                max_age_seconds=self._max_age_seconds,
            )
            completed_at = self._clock_value("completion clock")
            if completed_at < checked_at:
                raise ReconciliationServiceBlocked("completion clock moved backwards")
            relied_on_proof_expiries = []
            proofs_by_key = {
                proof.binding_key: proof for proof in runtime_evidence.timing_lag_proofs
            }
            for difference in verdict.differences:
                if difference.kind.value != "expected_timing_lag":
                    continue
                proof = proofs_by_key.get(
                    (
                        snapshot.snapshot_id,
                        difference.kind.value,
                        difference.reason_code,
                        difference.subject,
                        difference.evidence_ids[0],
                    )
                )
                if proof is not None:
                    relied_on_proof_expiries.append(proof.expires_at)
            eligible_until = min(
                verdict.fresh_until,
                runtime_evidence.expires_at,
                *relied_on_proof_expiries,
            )
            persisted = await self._persistence.append_reconciliation(
                trigger_type=trigger.value,
                runtime_evidence=runtime_evidence,
                verdict=verdict,
                started_at=started_at,
                completed_at=completed_at,
                eligible_until=eligible_until,
            )
            assert_runtime_reconciliation_evidence_sources_current(runtime_evidence)
        except BaseException as exc:
            self._quarantine()
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, ReconciliationServiceBlocked):
                raise
            raise ReconciliationServiceBlocked("reconciliation generation failed closed") from exc

        state = {
            ReconciliationStatus.PASSED: ReconciliationServiceState.READY,
            ReconciliationStatus.DEGRADED: ReconciliationServiceState.DEGRADED,
            ReconciliationStatus.QUARANTINED: ReconciliationServiceState.QUARANTINED,
        }[verdict.status]
        outcome = ReconciliationServiceOutcome(
            trigger=trigger,
            verdict=verdict,
            persisted=persisted,
            state=state,
            entry_eligible=persisted.entry_eligible,
            eligible_until=eligible_until,
        )
        self._state = state
        self._latest_outcome = outcome
        self._last_completed_at = completed_at
        self._latest_database_binding = (
            runtime_evidence.database_path,
            runtime_evidence.database_device,
            runtime_evidence.database_inode,
        )
        return outcome

    def _clock_value(self, label: str) -> datetime:
        try:
            return _timestamp(self._clock(), label)
        except Exception as exc:
            raise ReconciliationServiceBlocked(f"{label} is unavailable") from exc

    def _quarantine(self) -> None:
        self._latest_outcome = None
        self._last_completed_at = None
        self._latest_database_binding = None
        self._state = ReconciliationServiceState.QUARANTINED

    def _assert_not_closed(self) -> None:
        if self._state is ReconciliationServiceState.CLOSED:
            raise ReconciliationServiceBlocked("reconciliation service is closed")
        if self._state is ReconciliationServiceState.CLOSING:
            raise ReconciliationServiceBlocked("reconciliation source cleanup is incomplete")
