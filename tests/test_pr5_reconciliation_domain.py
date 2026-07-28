from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, Decimal, Overflow, localcontext

import pytest

from robo_trader.reconciliation.domain import (
    BrokerCollectionEvidence,
    BrokerCollectionKind,
    BrokerEvidenceCompleteness,
    BrokerOrderCollection,
    BrokerOrderSide,
    ExecutionDomainScope,
    NormalizedBrokerAccount,
    NormalizedBrokerExecution,
    NormalizedBrokerOrder,
    NormalizedBrokerPosition,
    NormalizedBrokerSnapshot,
    ReconciliationDomainError,
)
from robo_trader.reconciliation.policy import (
    DifferenceKind,
    DifferenceMateriality,
    ExpectedTimingLagProof,
    ReconciliationCoverage,
    ReconciliationDifference,
    ReconciliationStatus,
    evaluate_paper_simulator_reconciliation,
)

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
ACCOUNT_SCOPE = "acct_v1_" + "0123456789abcdef" * 4
OTHER_ACCOUNT_SCOPE = "acct_v1_" + "fedcba9876543210" * 4


def _account(*, scope: str = ACCOUNT_SCOPE) -> NormalizedBrokerAccount:
    return NormalizedBrokerAccount(
        account_scope=scope,
        account_alias="***1234",
        account_type="paper",
        base_currency="USD",
        total_cash=Decimal("25000.00"),
        buying_power=Decimal("100000.00"),
        observed_at=NOW - timedelta(seconds=1),
    )


def _position(*, scope: str = ACCOUNT_SCOPE) -> NormalizedBrokerPosition:
    return NormalizedBrokerPosition(
        account_scope=scope,
        con_id=265598,
        symbol="AAPL",
        currency="USD",
        signed_quantity=Decimal("-2"),
        average_cost=Decimal("210.125"),
        observed_at=NOW - timedelta(seconds=1),
    )


def _order(
    *,
    collection: BrokerOrderCollection = BrokerOrderCollection.OPEN,
    broker_order_id: int = 7,
    permanent_id: int = 70,
    symbol: str = "AAPL",
) -> NormalizedBrokerOrder:
    completed = collection is BrokerOrderCollection.COMPLETED
    return NormalizedBrokerOrder(
        account_scope=ACCOUNT_SCOPE,
        collection=collection,
        broker_order_id=broker_order_id,
        client_id=4,
        con_id=265598 if symbol == "AAPL" else 272093,
        symbol=symbol,
        side=BrokerOrderSide.SELL,
        total_quantity=Decimal("2"),
        filled_quantity=Decimal("2") if completed else Decimal("1"),
        remaining_quantity=Decimal("0") if completed else Decimal("1"),
        status="Filled" if completed else "Submitted",
        observed_at=NOW - timedelta(seconds=1),
        permanent_id=permanent_id,
    )


def _execution(*, execution_id: str = "0001.abc") -> NormalizedBrokerExecution:
    return NormalizedBrokerExecution(
        account_scope=ACCOUNT_SCOPE,
        execution_id=execution_id,
        con_id=265598,
        symbol="AAPL",
        side=BrokerOrderSide.SELL,
        quantity=Decimal("2"),
        price=Decimal("215.50"),
        executed_at=NOW - timedelta(seconds=1),
        broker_order_id=7,
        permanent_id=70,
        commission=Decimal("1.23"),
        commission_currency="USD",
    )


def _completeness(**overrides: bool) -> BrokerEvidenceCompleteness:
    values = {
        "account": True,
        "positions": True,
        "open_orders": True,
        "completed_orders": True,
        "executions": True,
        "commissions": True,
    }
    values.update(overrides)
    return BrokerEvidenceCompleteness(**values)


def _collection_evidence(
    collection: BrokerCollectionKind,
    result_count: int,
    observed_at: datetime,
) -> BrokerCollectionEvidence:
    digest = hashlib.sha256(
        f"{collection.value}:{result_count}:{observed_at.isoformat()}".encode()
    ).hexdigest()
    return BrokerCollectionEvidence(
        account_scope=ACCOUNT_SCOPE,
        collection=collection,
        evidence_id=f"broker-collection-v1-{digest}",
        result_count=result_count,
        observed_at=observed_at,
    )


def _coverage(**overrides: bool) -> ReconciliationCoverage:
    values = {
        "broker_account": True,
        "broker_positions": True,
        "broker_open_orders": True,
        "broker_completed_orders": True,
        "broker_executions": True,
        "broker_commissions": True,
        "ledger_positions": True,
        "ledger_orders": True,
        "ledger_executions": True,
        "ledger_cash": True,
    }
    values.update(overrides)
    return ReconciliationCoverage(**values)


def _snapshot(
    *,
    account: NormalizedBrokerAccount | None = None,
    positions: tuple[NormalizedBrokerPosition, ...] = (),
    orders: tuple[NormalizedBrokerOrder, ...] = (),
    executions: tuple[NormalizedBrokerExecution, ...] = (),
    completeness: BrokerEvidenceCompleteness | None = None,
    collection_evidence: tuple[BrokerCollectionEvidence, ...] | None = None,
    retrieved_at: datetime | None = None,
) -> NormalizedBrokerSnapshot:
    retrieval = retrieved_at or NOW - timedelta(seconds=1)
    claims = completeness or _completeness()
    open_order_count = sum(order.collection is BrokerOrderCollection.OPEN for order in orders)
    completed_order_count = sum(
        order.collection is BrokerOrderCollection.COMPLETED for order in orders
    )
    counts = {
        BrokerCollectionKind.POSITIONS: len(positions),
        BrokerCollectionKind.OPEN_ORDERS: open_order_count,
        BrokerCollectionKind.COMPLETED_ORDERS: completed_order_count,
        BrokerCollectionKind.EXECUTIONS: len(executions),
        BrokerCollectionKind.COMMISSIONS: len(executions),
    }
    completeness_by_kind = {
        BrokerCollectionKind.POSITIONS: claims.positions,
        BrokerCollectionKind.OPEN_ORDERS: claims.open_orders,
        BrokerCollectionKind.COMPLETED_ORDERS: claims.completed_orders,
        BrokerCollectionKind.EXECUTIONS: claims.executions,
        BrokerCollectionKind.COMMISSIONS: claims.commissions,
    }
    evidence = collection_evidence
    if evidence is None:
        evidence = tuple(
            _collection_evidence(kind, counts[kind], retrieval)
            for kind in BrokerCollectionKind
            if completeness_by_kind[kind]
        )
    return NormalizedBrokerSnapshot(
        account=account or replace(_account(), observed_at=retrieval),
        observed_from=retrieval - timedelta(seconds=1),
        observed_through=retrieval,
        retrieved_at=retrieval,
        completeness=claims,
        collection_evidence=evidence,
        positions=positions,
        orders=orders,
        executions=executions,
    )


def _difference(kind: DifferenceKind) -> ReconciliationDifference:
    materiality = {
        DifferenceKind.EXPECTED_TIMING_LAG: DifferenceMateriality.INFORMATIONAL,
        DifferenceKind.UNKNOWN: DifferenceMateriality.UNKNOWN,
    }.get(kind, DifferenceMateriality.MATERIAL)
    reason_code = f"TEST_{kind.value.upper()}"
    evidence_ids: tuple[str, ...] = ()
    if kind is DifferenceKind.EXPECTED_TIMING_LAG:
        reason_code = "BROKER_ORDER_EVENT_PENDING"
        evidence_ids = ("broker-event-v1-" + "b2" * 32,)
    return ReconciliationDifference(
        kind=kind,
        materiality=materiality,
        reason_code=reason_code,
        subject="AAPL",
        evidence_ids=evidence_ids,
    )


def _timing_proof(
    snapshot: NormalizedBrokerSnapshot,
    difference: ReconciliationDifference,
    *,
    started_at: datetime = NOW - timedelta(seconds=5),
    expires_at: datetime = NOW + timedelta(seconds=5),
) -> ExpectedTimingLagProof:
    return ExpectedTimingLagProof.from_trusted_producer(
        broker_snapshot_id=snapshot.snapshot_id,
        reason_code=difference.reason_code,
        subject=difference.subject,
        broker_event_id=difference.evidence_ids[0],
        started_at=started_at,
        expires_at=expires_at,
    )


def _evaluate(
    snapshot: NormalizedBrokerSnapshot,
    *,
    coverage: ReconciliationCoverage | None = None,
    differences: tuple[ReconciliationDifference, ...] = (),
    timing_lag_proofs: tuple[ExpectedTimingLagProof, ...] = (),
    expected_account_scope: str = ACCOUNT_SCOPE,
    now: datetime = NOW,
):
    return evaluate_paper_simulator_reconciliation(
        snapshot,
        coverage or _coverage(),
        differences,
        timing_lag_proofs,
        expected_account_scope=expected_account_scope,
        now=now,
        max_age_seconds=30,
    )


def test_exact_difference_kind_taxonomy_is_stable() -> None:
    assert {kind.value for kind in DifferenceKind} == {
        "expected_timing_lag",
        "recoverable_missing_event",
        "duplicate_event",
        "account_mismatch",
        "quantity_mismatch",
        "cash_mismatch",
        "unknown",
    }


def test_normalized_records_are_immutable_exact_and_versioned() -> None:
    account = _account()
    position = _position()
    order = _order(collection=BrokerOrderCollection.COMPLETED)
    execution = _execution()

    assert account.schema_version == 1
    assert position.signed_quantity == Decimal("-2")
    assert order.collection is BrokerOrderCollection.COMPLETED
    assert execution.commission == Decimal("1.23")
    assert account.source_scope is ExecutionDomainScope.IBKR_READ_ONLY
    with pytest.raises(FrozenInstanceError):
        account.total_cash = Decimal("0")  # type: ignore[misc]


def test_decimal_canonicalization_is_independent_of_ambient_context() -> None:
    snapshot = _snapshot(
        account=replace(
            _account(),
            total_cash=Decimal("123456789.123450000"),
            observed_at=NOW - timedelta(seconds=1),
        )
    )
    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_DOWN
        context.Emax = 3
        context.Emin = -3
        context.capitals = 0
        context.clamp = 1
        context.traps[Overflow] = False
        adversarial_payload = snapshot.canonical_payload()
        adversarial_id = snapshot.snapshot_id
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_DOWN
        context.Emax = 999999
        context.Emin = -999999
        context.capitals = 1
        context.clamp = 0
        context.traps[Overflow] = True
        high_precision_payload = snapshot.canonical_payload()
        high_precision_id = snapshot.snapshot_id

    assert adversarial_payload == high_precision_payload
    assert adversarial_id == high_precision_id
    assert '"total_cash":"123456789.12345"' in adversarial_payload
    assert "Infinity" not in adversarial_payload


def test_order_quantity_arithmetic_is_exact_under_low_ambient_precision() -> None:
    with localcontext() as context:
        context.prec = 3
        with pytest.raises(ReconciliationDomainError, match="quantities are inconsistent"):
            replace(
                _order(),
                total_quantity=Decimal("1.23"),
                filled_quantity=Decimal("1.23"),
                remaining_quantity=Decimal("0.0001"),
            )
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_DOWN
        context.Emax = 3
        context.Emin = -3
        context.capitals = 0
        context.clamp = 1
        context.traps[Overflow] = False
        valid = replace(
            _order(),
            total_quantity=Decimal("10000"),
            filled_quantity=Decimal("10000"),
            remaining_quantity=Decimal("0"),
        )
    assert valid.total_quantity == Decimal("10000")


@pytest.mark.parametrize(
    "record",
    [
        lambda: replace(_account(), schema_version=True),
        lambda: replace(_position(), schema_version=True),
        lambda: replace(_order(), schema_version=True),
        lambda: replace(_execution(), schema_version=True),
        lambda: replace(
            _collection_evidence(
                BrokerCollectionKind.POSITIONS,
                0,
                NOW - timedelta(seconds=1),
            ),
            schema_version=True,
        ),
        lambda: replace(_snapshot(), schema_version=True),
        lambda: replace(_difference(DifferenceKind.UNKNOWN), schema_version=True),
        lambda: replace(
            _timing_proof(
                _snapshot(),
                _difference(DifferenceKind.EXPECTED_TIMING_LAG),
            ),
            schema_version=True,
        ),
        lambda: replace(_evaluate(_snapshot()), schema_version=True),
    ],
)
def test_schema_version_rejects_boolean_true_everywhere(record) -> None:
    with pytest.raises(ReconciliationDomainError, match="schema version is unsupported"):
        record()


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            lambda: replace(_account(), account_alias="DU123456"),
            "account_alias",
        ),
        (
            lambda: replace(_account(), account_scope="acct_v1_" + "a" * 64),
            "placeholder",
        ),
        (
            lambda: replace(_account(), account_type="live"),
            "not paper",
        ),
        (
            lambda: replace(_account(), total_cash=12.5),
            "binary float",
        ),
        (
            lambda: replace(_position(), signed_quantity=Decimal("NaN")),
            "finite decimal",
        ),
        (
            lambda: replace(_order(), remaining_quantity=Decimal("0")),
            "quantities are inconsistent",
        ),
        (
            lambda: replace(_execution(), commission=None),
            "unavailable reason",
        ),
    ],
)
def test_normalized_records_fail_closed(record, message: str) -> None:
    with pytest.raises(ReconciliationDomainError, match=message):
        record()


def test_unavailable_commission_requires_explicit_reason_code() -> None:
    execution = replace(
        _execution(),
        commission=None,
        commission_currency=None,
        commission_unavailable_reason="COMMISSION_REPORT_UNAVAILABLE",
    )
    assert execution.commission is None
    assert execution.commission_unavailable_reason == "COMMISSION_REPORT_UNAVAILABLE"


def test_snapshot_rejects_cross_account_and_duplicate_identity() -> None:
    with pytest.raises(ReconciliationDomainError, match="multiple account scopes"):
        _snapshot(positions=(replace(_position(), account_scope=OTHER_ACCOUNT_SCOPE),))

    with pytest.raises(ReconciliationDomainError, match="duplicate order identity"):
        _snapshot(orders=(_order(), replace(_order(), permanent_id=71)))

    with pytest.raises(ReconciliationDomainError, match="duplicate execution identity"):
        _snapshot(executions=(_execution(), _execution()))


def test_snapshot_enforces_retrieval_and_record_chronology() -> None:
    snapshot = _snapshot()
    with pytest.raises(ReconciliationDomainError, match="retrieval predates"):
        replace(
            snapshot,
            retrieved_at=snapshot.observed_through - timedelta(microseconds=1),
        )
    with pytest.raises(ReconciliationDomainError, match="later than its snapshot"):
        _snapshot(executions=(replace(_execution(), executed_at=NOW + timedelta(microseconds=1)),))


def test_empty_complete_snapshot_requires_bound_collection_evidence() -> None:
    with pytest.raises(ReconciliationDomainError, match="lacks explicit evidence"):
        _snapshot(collection_evidence=())

    snapshot = _snapshot()
    assert snapshot.completeness.complete is True
    assert {evidence.collection for evidence in snapshot.collection_evidence} == set(
        BrokerCollectionKind
    )


def test_complete_commissions_rejects_execution_without_commission() -> None:
    unavailable = replace(
        _execution(),
        commission=None,
        commission_currency=None,
        commission_unavailable_reason="COMMISSION_REPORT_UNAVAILABLE",
    )
    with pytest.raises(ReconciliationDomainError, match="unavailable commission"):
        _snapshot(executions=(unavailable,))

    incomplete = _snapshot(
        executions=(unavailable,),
        completeness=_completeness(commissions=False),
    )
    assert incomplete.completeness.commissions is False


def test_order_collection_is_observation_source_not_terminal_fill_claim() -> None:
    cancelled = replace(
        _order(
            collection=BrokerOrderCollection.COMPLETED,
            broker_order_id=8,
            permanent_id=80,
            symbol="MSFT",
        ),
        filled_quantity=Decimal("0"),
        remaining_quantity=Decimal("2"),
        status="Cancelled",
    )
    overlapping = replace(_order(), collection=BrokerOrderCollection.COMPLETED)
    snapshot = _snapshot(orders=(_order(), overlapping, cancelled))

    assert len(snapshot.open_orders) == 1
    assert len(snapshot.completed_orders) == 2


def test_snapshot_payload_and_fingerprint_are_order_independent() -> None:
    first_order = _order(
        collection=BrokerOrderCollection.COMPLETED,
        broker_order_id=8,
        permanent_id=80,
        symbol="MSFT",
    )
    second_order = _order()
    first = _snapshot(orders=(first_order, second_order))
    second = _snapshot(orders=(second_order, first_order))

    assert first.canonical_payload() == second.canonical_payload()
    assert first.snapshot_id == second.snapshot_id
    assert first.snapshot_id.startswith("broker-reconciliation-v1-")


def test_snapshot_freshness_rejects_bad_clock_and_bound() -> None:
    snapshot = _snapshot()
    assert snapshot.is_fresh(now=NOW, max_age_seconds=30) is True
    assert snapshot.is_fresh(now=NOW + timedelta(minutes=1), max_age_seconds=30) is False
    assert snapshot.is_fresh(now=NOW - timedelta(minutes=1), max_age_seconds=30) is False
    with pytest.raises(ReconciliationDomainError, match="finite and positive"):
        snapshot.is_fresh(now=NOW, max_age_seconds=float("inf"))


def test_complete_fresh_zero_broker_exposure_is_non_authorizing_pass() -> None:
    verdict = _evaluate(_snapshot())

    assert verdict.status is ReconciliationStatus.PASSED
    assert verdict.quarantine_required is False
    assert verdict.evidence_fresh is True
    assert verdict.comparison_complete is True
    assert verdict.mutated_state is False
    assert verdict.authorizes_startup is False
    assert verdict.execution_domain_scope is ExecutionDomainScope.PAPER_SIMULATOR
    assert verdict.verdict_id.startswith("reconciliation-verdict-v1-")
    assert '"authorizes_startup":false' in verdict.canonical_payload()


def test_expected_timing_lag_is_degraded_but_not_quarantined() -> None:
    snapshot = _snapshot()
    difference = _difference(DifferenceKind.EXPECTED_TIMING_LAG)
    verdict = _evaluate(
        snapshot,
        differences=(difference,),
        timing_lag_proofs=(_timing_proof(snapshot, difference),),
    )

    assert verdict.status is ReconciliationStatus.DEGRADED
    assert verdict.quarantine_required is False


def test_expected_timing_lag_requires_eligible_bounded_proof() -> None:
    with pytest.raises(ReconciliationDomainError, match="reason is not eligible"):
        replace(
            _difference(DifferenceKind.EXPECTED_TIMING_LAG),
            reason_code="TEST_EXPECTED_TIMING_LAG",
        )
    with pytest.raises(ReconciliationDomainError, match="bound broker event"):
        replace(
            _difference(DifferenceKind.EXPECTED_TIMING_LAG),
            evidence_ids=(),
        )
    snapshot = _snapshot()
    difference = _difference(DifferenceKind.EXPECTED_TIMING_LAG)
    with pytest.raises(ReconciliationDomainError, match="exceeds policy maximum"):
        _timing_proof(
            snapshot,
            difference,
            expires_at=NOW + timedelta(seconds=121),
        )


def test_timing_proof_rejects_forged_hashes_and_wrong_kind() -> None:
    snapshot = _snapshot()
    difference = _difference(DifferenceKind.EXPECTED_TIMING_LAG)
    proof = _timing_proof(snapshot, difference)

    for forged_hash in ("a" * 64, "b" * 64):
        with pytest.raises(ReconciliationDomainError, match="fingerprint is not bound"):
            replace(proof, proof_id=f"timing-proof-v1-{forged_hash}")
    with pytest.raises(ReconciliationDomainError, match="difference kind is ineligible"):
        replace(proof, difference_kind=DifferenceKind.QUANTITY_MISMATCH)


def test_unproven_expired_future_or_wrong_snapshot_lag_quarantines() -> None:
    snapshot = _snapshot()
    lag = _difference(DifferenceKind.EXPECTED_TIMING_LAG)
    proof = _timing_proof(snapshot, lag)
    unproven = _evaluate(snapshot, differences=(lag,))
    expired = _evaluate(
        snapshot,
        differences=(lag,),
        timing_lag_proofs=(proof,),
        now=NOW + timedelta(seconds=6),
    )
    future = _evaluate(
        snapshot,
        differences=(lag,),
        timing_lag_proofs=(proof,),
        now=NOW - timedelta(seconds=6),
    )
    other_snapshot = replace(
        snapshot,
        retrieved_at=snapshot.retrieved_at + timedelta(microseconds=1),
    )
    wrong_snapshot = _evaluate(
        other_snapshot,
        differences=(lag,),
        timing_lag_proofs=(proof,),
    )

    for verdict in (unproven, expired, future, wrong_snapshot):
        assert verdict.quarantine_required is True
        assert "EXPECTED_TIMING_LAG_UNPROVEN_OR_EXPIRED" in {
            difference.reason_code for difference in verdict.differences
        }


@pytest.mark.parametrize(
    "kind",
    [
        DifferenceKind.RECOVERABLE_MISSING_EVENT,
        DifferenceKind.DUPLICATE_EVENT,
        DifferenceKind.ACCOUNT_MISMATCH,
        DifferenceKind.QUANTITY_MISMATCH,
        DifferenceKind.CASH_MISMATCH,
        DifferenceKind.UNKNOWN,
    ],
)
def test_material_or_unknown_difference_requires_quarantine(kind: DifferenceKind) -> None:
    verdict = _evaluate(_snapshot(), differences=(_difference(kind),))
    assert verdict.status is ReconciliationStatus.QUARANTINED
    assert verdict.quarantine_required is True


def test_difference_rejects_downgraded_materiality() -> None:
    with pytest.raises(ReconciliationDomainError, match="materiality contradicts"):
        ReconciliationDifference(
            kind=DifferenceKind.ACCOUNT_MISMATCH,
            materiality=DifferenceMateriality.INFORMATIONAL,
            reason_code="ACCOUNT_SCOPE_MISMATCH",
            subject="ibkr_account",
        )


def test_difference_rejects_raw_account_identity() -> None:
    with pytest.raises(ReconciliationDomainError, match="raw account identity"):
        ReconciliationDifference(
            kind=DifferenceKind.UNKNOWN,
            materiality=DifferenceMateriality.UNKNOWN,
            reason_code="UNEXPECTED_ACCOUNT",
            subject="DU123456",
        )


def test_difference_rejects_freeform_subjects_and_evidence_identifiers() -> None:
    with pytest.raises(ReconciliationDomainError, match="subject is malformed"):
        replace(_difference(DifferenceKind.UNKNOWN), subject="customer-account-1234")
    with pytest.raises(ReconciliationDomainError, match="evidence_id is malformed"):
        replace(_difference(DifferenceKind.UNKNOWN), evidence_ids=("freeform-id",))


def test_stale_or_incomplete_evidence_requires_quarantine() -> None:
    stale = _snapshot(retrieved_at=NOW - timedelta(minutes=5))
    stale_verdict = _evaluate(stale)
    assert stale_verdict.quarantine_required is True
    assert {difference.reason_code for difference in stale_verdict.differences} == {
        "BROKER_EVIDENCE_STALE"
    }

    incomplete = _snapshot(completeness=_completeness(completed_orders=False))
    incomplete_verdict = _evaluate(incomplete)
    assert incomplete_verdict.quarantine_required is True
    assert "BROKER_EVIDENCE_INCOMPLETE" in {
        difference.reason_code for difference in incomplete_verdict.differences
    }

    partial_comparison = _evaluate(
        _snapshot(),
        coverage=_coverage(ledger_executions=False),
    )
    assert partial_comparison.quarantine_required is True
    assert "LOCAL_COMPARISON_INCOMPLETE" in {
        difference.reason_code for difference in partial_comparison.differences
    }


def test_account_scope_mismatch_requires_quarantine_without_raw_identity() -> None:
    verdict = _evaluate(_snapshot(), expected_account_scope=OTHER_ACCOUNT_SCOPE)
    assert verdict.quarantine_required is True
    assert verdict.differences[0].kind is DifferenceKind.ACCOUNT_MISMATCH
    assert "DU" not in verdict.canonical_payload()


def test_simulator_policy_does_not_equate_local_and_broker_positions() -> None:
    verdict = _evaluate(_snapshot(positions=(_position(),)))

    assert verdict.quarantine_required is True
    assert any(
        difference.kind is DifferenceKind.QUANTITY_MISMATCH
        and difference.reason_code == "UNEXPECTED_IBKR_POSITION_IN_PAPER_SIMULATOR"
        for difference in verdict.differences
    )
    assert all("local" not in difference.subject for difference in verdict.differences)


def test_simulator_policy_quarantines_open_broker_orders_but_not_completed_history() -> None:
    open_verdict = _evaluate(_snapshot(orders=(_order(),)))
    assert open_verdict.quarantine_required is True
    assert "UNEXPECTED_IBKR_OPEN_ORDER_IN_PAPER_SIMULATOR" in {
        difference.reason_code for difference in open_verdict.differences
    }

    completed = _order(collection=BrokerOrderCollection.COMPLETED)
    completed_verdict = _evaluate(_snapshot(orders=(completed,), executions=(_execution(),)))
    assert completed_verdict.status is ReconciliationStatus.PASSED


def test_direct_verdict_construction_cannot_claim_a_stale_pass() -> None:
    passed = _evaluate(_snapshot())
    with pytest.raises(ReconciliationDomainError, match="stale verdict lacks"):
        replace(passed, evidence_fresh=False)


def test_verdict_fingerprint_is_deterministic_across_difference_order() -> None:
    missing = _difference(DifferenceKind.RECOVERABLE_MISSING_EVENT)
    duplicate = _difference(DifferenceKind.DUPLICATE_EVENT)
    first = _evaluate(_snapshot(), differences=(missing, duplicate))
    second = _evaluate(_snapshot(), differences=(duplicate, missing))

    assert first.canonical_payload() == second.canonical_payload()
    assert first.verdict_id == second.verdict_id
