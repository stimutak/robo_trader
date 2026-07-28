"""Dormant trigger-aware reconciliation service with fail-closed eligibility."""

from __future__ import annotations

import asyncio
import inspect
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Awaitable, Callable, Protocol

from .domain import NormalizedBrokerSnapshot, ReconciliationDomainError, _timestamp
from .persistence import PersistedReconciliation, ReconciliationPersistence
from .policy import (
    ExpectedTimingLagProof,
    ReconciliationCoverage,
    ReconciliationDifference,
    ReconciliationStatus,
    ReconciliationVerdict,
    evaluate_paper_simulator_reconciliation,
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
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ReconciliationComparison:
    coverage: ReconciliationCoverage
    differences: tuple[ReconciliationDifference, ...] = ()
    timing_lag_proofs: tuple[ExpectedTimingLagProof, ...] = ()

    def __post_init__(self) -> None:
        if type(self.coverage) is not ReconciliationCoverage:
            raise ReconciliationServiceBlocked("comparison coverage is not normalized")
        if any(type(value) is not ReconciliationDifference for value in self.differences):
            raise ReconciliationServiceBlocked("comparison differences are not normalized")
        if any(type(value) is not ExpectedTimingLagProof for value in self.timing_lag_proofs):
            raise ReconciliationServiceBlocked("comparison timing proofs are not normalized")


@dataclass(frozen=True, slots=True)
class ReconciliationServiceOutcome:
    trigger: ReconciliationTrigger
    verdict: ReconciliationVerdict
    persisted: PersistedReconciliation
    state: ReconciliationServiceState
    entry_eligible: bool


class NormalizedSnapshotSource(Protocol):
    """Narrow source implemented by the existing read-only provider pipeline.

    Runtime composition must supply the provider's authenticated normalized
    evidence handoff. The service intentionally receives no transport or order
    capability and cannot reach around that handoff.
    """

    async def collect_normalized_snapshot(
        self, *, max_age_seconds: float
    ) -> NormalizedBrokerSnapshot:
        """Collect one bounded read-only broker snapshot."""

    async def close(self) -> None:
        """Close only the diagnostic read-only transport."""


class ReconciliationComparisonSource(Protocol):
    def __call__(
        self,
        snapshot: NormalizedBrokerSnapshot,
        trigger: ReconciliationTrigger,
    ) -> ReconciliationComparison | Awaitable[ReconciliationComparison]:
        """Compare immutable local evidence without repairing or replacing it."""


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
        snapshot_source: NormalizedSnapshotSource,
        comparison_source: ReconciliationComparisonSource,
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
        if not callable(getattr(snapshot_source, "collect_normalized_snapshot", None)):
            raise ReconciliationServiceBlocked("normalized snapshot source is unavailable")
        if not callable(getattr(snapshot_source, "close", None)):
            raise ReconciliationServiceBlocked("snapshot source cleanup is unavailable")
        if not callable(comparison_source):
            raise ReconciliationServiceBlocked("comparison source is unavailable")
        self._snapshot_source = snapshot_source
        self._comparison_source = comparison_source
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
            return False
        return checked_at <= outcome.verdict.fresh_until

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
            self._quarantine()
            try:
                await self._snapshot_source.close()
            finally:
                self._state = ReconciliationServiceState.CLOSED

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
            snapshot = await self._snapshot_source.collect_normalized_snapshot(
                max_age_seconds=self._max_age_seconds
            )
            if type(snapshot) is not NormalizedBrokerSnapshot:
                raise ReconciliationServiceBlocked("snapshot source returned invalid evidence")
            comparison_result = self._comparison_source(snapshot, trigger)
            if inspect.isawaitable(comparison_result):
                comparison_result = await comparison_result
            if type(comparison_result) is not ReconciliationComparison:
                raise ReconciliationServiceBlocked("comparison source returned invalid evidence")
            checked_at = self._clock_value("policy clock")
            if checked_at < started_at:
                raise ReconciliationServiceBlocked("policy clock moved backwards")
            verdict = evaluate_paper_simulator_reconciliation(
                snapshot,
                comparison_result.coverage,
                comparison_result.differences,
                comparison_result.timing_lag_proofs,
                expected_account_scope=self._expected_account_scope,
                now=checked_at,
                max_age_seconds=self._max_age_seconds,
            )
            completed_at = self._clock_value("completion clock")
            if completed_at < checked_at:
                raise ReconciliationServiceBlocked("completion clock moved backwards")
            persisted = await self._persistence.append_reconciliation(
                trigger_type=trigger.value,
                snapshot=snapshot,
                verdict=verdict,
                started_at=started_at,
                completed_at=completed_at,
            )
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
        )
        self._state = state
        self._latest_outcome = outcome
        self._last_completed_at = completed_at
        return outcome

    def _clock_value(self, label: str) -> datetime:
        try:
            return _timestamp(self._clock(), label)
        except Exception as exc:
            raise ReconciliationServiceBlocked(f"{label} is unavailable") from exc

    def _quarantine(self) -> None:
        self._latest_outcome = None
        self._last_completed_at = None
        self._state = ReconciliationServiceState.QUARANTINED

    def _assert_not_closed(self) -> None:
        if self._state is ReconciliationServiceState.CLOSED:
            raise ReconciliationServiceBlocked("reconciliation service is closed")
