import copy
import json
import os
import pickle
import sqlite3
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest

from robo_trader.safety import (
    IdempotencyConflict,
    JournalIntegrityError,
    ReconciliationEvidence,
    ReconciliationStatus,
    ReservationConflict,
    SafetyJournal,
    StateTransitionError,
    TerminalOrderStatus,
    TransportState,
    ValidationError,
)

from .conftest import ACCOUNT_B, make_case


def terminal_evidence(now, reservation, claim, intent, **changes):
    values = {
        "execution_domain_scope": intent.execution_domain_scope,
        "reservation_id": reservation.reservation_id,
        "claim_id": claim.claim_id,
        "claim_sequence": claim.sequence,
        "submission_descriptor_fingerprint": claim.submission_descriptor_fingerprint,
        "order_ref": claim.order_ref,
        "account_scope": intent.account_scope,
        "portfolio_id": intent.portfolio_id,
        "con_id": intent.con_id,
        "symbol": intent.symbol,
        "observed_at": now + timedelta(seconds=1),
        "position_observed_at": now + timedelta(seconds=1),
        "max_evidence_age_seconds": 30,
        "account_position_quantity": intent.target_quantity,
        "portfolio_position_quantity": intent.portfolio_target_quantity,
        "aggregate_allocated_quantity": intent.target_quantity,
        "has_offsetting_allocations": False,
        "status": ReconciliationStatus.PASSED,
        "transport_state": TransportState.CONNECTED,
        "open_orders_complete": True,
        "open_orders_all_clients": True,
        "open_orders_snapshot_stable": True,
        "active_order_count": 0,
        "terminal_order_status": TerminalOrderStatus.FILLED,
        "filled_quantity": intent.quantity,
        "remaining_quantity": Decimal("0"),
        "source": "independent-terminal-reconciliation",
    }
    values.update(changes)
    return ReconciliationEvidence(**values)


def authorize(journal, now, key="intent-key", **case_changes):
    intent, exposure, allocation, gates, _, descriptor = make_case(now, **case_changes)
    return (
        intent,
        descriptor,
        *journal.authorize_submission(key, intent, exposure, allocation, gates, descriptor),
    )


def test_explicit_initialization_and_atomic_claim_replay_never_renews_permit(tmp_path, now):
    path = tmp_path / "dedicated-safety.db"
    journal = SafetyJournal(path, clock=lambda: now)
    assert not path.exists()
    journal.initialize()
    intent, descriptor, reservation, claim, permit = authorize(journal, now)
    assert permit is not None
    assert journal.consume_submission_permit(permit) == descriptor
    with pytest.raises(StateTransitionError):
        journal.consume_submission_permit(permit)

    same_reservation, historical_claim, replay_permit = journal.authorize_submission(
        "intent-key",
        intent,
        make_case(now)[1],
        make_case(now)[2],
        make_case(now)[3],
        descriptor,
    )
    assert same_reservation.reservation_id == reservation.reservation_id
    assert not same_reservation.newly_acquired
    assert historical_claim.claim_id == claim.claim_id
    assert not historical_claim.granted
    assert replay_permit is None
    state = SafetyJournal(path).replay()
    assert state.last_sequence == 4
    assert state.quarantined_reservations[0].claim_id == claim.claim_id


def test_exact_idempotent_replay_survives_restart_and_later_clock(tmp_path, now):
    path = tmp_path / "dedicated-safety.db"
    clock = [now]
    journal = SafetyJournal(path, clock=lambda: clock[0])
    journal.initialize()
    case = make_case(now)
    reservation, claim, permit = journal.authorize_submission(
        "intent-key",
        case[0],
        case[1],
        case[2],
        case[3],
        case[5],
    )
    assert permit is not None

    clock[0] = now + timedelta(seconds=31)
    reopened = SafetyJournal(path, clock=lambda: clock[0])
    replayed_reservation, replayed_claim, replayed_permit = reopened.authorize_submission(
        "intent-key",
        case[0],
        case[1],
        case[2],
        case[3],
        case[5],
    )
    assert replayed_reservation.reservation_id == reservation.reservation_id
    assert not replayed_reservation.newly_acquired
    assert replayed_claim.claim_id == claim.claim_id
    assert not replayed_claim.granted
    assert replayed_permit is None


def test_generated_internal_ids_never_run_caller_account_fragment_scan(tmp_path, now, monkeypatch):
    import robo_trader.safety.journal as journal_module

    values = iter(f"f{'1' * 30}{suffix}" for suffix in "12345")
    monkeypatch.setattr(
        journal_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex=next(values)),
    )
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: now)
    journal.initialize()
    _, _, reservation, claim, permit = authorize(journal, now)
    assert reservation.reservation_id.startswith("res-f")
    assert claim.claim_id.startswith("claim-f")
    assert permit is not None


def test_permit_is_noncopyable_nonserializable_and_not_in_replay(tmp_path, now):
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: now)
    journal.initialize()
    _, _, _, _, permit = authorize(journal, now)
    with pytest.raises(TypeError):
        copy.copy(permit)
    with pytest.raises(TypeError):
        copy.deepcopy(permit)
    with pytest.raises(TypeError):
        pickle.dumps(permit)
    replay = journal.replay()
    assert not any(type(value).__name__ == "SubmissionPermit" for value in vars(replay).values())


def test_permit_is_one_shot_under_thread_race(tmp_path, now):
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: now)
    journal.initialize()
    _, _, _, _, permit = authorize(journal, now)
    barrier = threading.Barrier(2)

    def consume():
        barrier.wait()
        try:
            journal.consume_submission_permit(permit)
            return True
        except StateTransitionError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: consume(), range(2)))
    assert sorted(results) == [False, True]


def test_reconstructed_permit_from_replay_has_no_submission_authority(tmp_path, now):
    from robo_trader.safety import SubmissionPermit

    path = tmp_path / "safety.db"
    journal = SafetyJournal(path, clock=lambda: now)
    journal.initialize()
    _, _, _, _, original = authorize(journal, now)
    journal.consume_submission_permit(original)
    historical_claim_id = journal.replay().reservations[0].claim_id
    reconstructed = SubmissionPermit._issue(historical_claim_id)
    with pytest.raises(StateTransitionError, match="not issued by this live"):
        journal.consume_submission_permit(reconstructed)
    with pytest.raises(StateTransitionError, match="not issued by this live"):
        SafetyJournal(path).consume_submission_permit(original)
    with pytest.raises(RuntimeError, match="issuing SafetyJournal"):
        reconstructed.consume()


def test_same_key_changed_intent_or_submission_terms_is_hard_conflict(tmp_path, now):
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: now)
    journal.initialize()
    intent, descriptor, _, _, _ = authorize(journal, now)
    changed_intent = replace(
        intent,
        reason="different authorization field",
    )
    with pytest.raises(IdempotencyConflict):
        journal.authorize_submission(
            "intent-key",
            changed_intent,
            make_case(now)[1],
            make_case(now)[2],
            make_case(now)[3],
            descriptor,
        )
    with pytest.raises(IdempotencyConflict):
        journal.authorize_submission(
            "intent-key",
            intent,
            make_case(now)[1],
            make_case(now)[2],
            make_case(now)[3],
            replace(descriptor, outside_regular_hours=True),
        )


def test_same_key_changed_authoritative_context_is_hard_conflict(tmp_path, now):
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: now)
    journal.initialize()
    case = make_case(now)
    journal.authorize_submission("intent-key", case[0], case[1], case[2], case[3], case[5])
    with pytest.raises(IdempotencyConflict, match="authorization evidence"):
        journal.authorize_submission(
            "intent-key",
            case[0],
            replace(case[1], snapshot_id="different-account-snapshot"),
            case[2],
            case[3],
            case[5],
        )


@pytest.mark.parametrize(
    ("kind", "reason_code", "exception"),
    [
        ("descriptor", "DESCRIPTOR_MISMATCH", StateTransitionError),
        ("reservation", "RESERVATION_CONFLICT", ReservationConflict),
    ],
)
def test_rejected_authorization_attempts_are_durably_audited(
    tmp_path, now, kind, reason_code, exception
):
    journal = SafetyJournal(tmp_path / f"{kind}.db", clock=lambda: now)
    journal.initialize()
    case = make_case(now)
    if kind == "reservation":
        journal.authorize_submission("first-key", case[0], case[1], case[2], case[3], case[5])
        key = "rejected-key"
        descriptor = case[5]
    else:
        key = "rejected-key"
        descriptor = replace(case[5], con_id=case[5].con_id + 1)
    before = journal.replay().last_sequence
    with pytest.raises(exception):
        journal.authorize_submission(key, case[0], case[1], case[2], case[3], descriptor)
    state = journal.replay()
    assert state.last_sequence == before + 1
    event = state.events[-1]
    assert event.idempotency_key == key
    payload = json.loads(event.payload_json)
    assert payload["decision"]["outcome"] == "DENY"
    assert payload["decision"]["reason_codes"] == [reason_code]
    assert not any(item.idempotency_key == key for item in state.reservations)


def test_descriptor_is_snapshotted_and_subclasses_are_rejected(tmp_path, now):
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: now)
    journal.initialize()
    case = make_case(now)
    _, _, permit = journal.authorize_submission(
        "intent-key", case[0], case[1], case[2], case[3], case[5]
    )
    original_con_id = case[5].con_id
    object.__setattr__(case[5], "con_id", original_con_id + 1)
    consumed = journal.consume_submission_permit(permit)
    assert consumed.con_id == original_con_id
    assert consumed is not case[5]

    class DescriptorSubclass(type(consumed)):
        pass

    subclass = DescriptorSubclass(
        **{field: getattr(consumed, field) for field in consumed.__dataclass_fields__}
    )
    other = make_case(now)
    with pytest.raises(TypeError, match="SubmissionDescriptor"):
        journal.authorize_submission(
            "subclass-key",
            other[0],
            other[1],
            other[2],
            other[3],
            subclass,
        )


def test_independent_connection_duplicate_race_grants_exactly_one_permit(tmp_path, now):
    path = tmp_path / "safety.db"
    SafetyJournal(path).initialize()
    intent, exposure, allocation, gates, _, descriptor = make_case(now)

    def compete():
        return SafetyJournal(path, clock=lambda: now).authorize_submission(
            "same-key", intent, exposure, allocation, gates, descriptor
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: compete(), range(2)))
    assert sum(result[2] is not None for result in results) == 1
    assert {result[1].claim_id for result in results} == {results[0][1].claim_id}
    assert SafetyJournal(path).replay().last_sequence == 3


def test_dual_scope_collision_and_different_account_isolation(tmp_path, now):
    path = tmp_path / "safety.db"
    journal = SafetyJournal(path, clock=lambda: now)
    journal.initialize()
    authorize(journal, now, key="first", portfolio_id="portfolio-a")
    intent, exposure, allocation, gates, _, descriptor = make_case(now, portfolio_id="portfolio-b")
    with pytest.raises(ReservationConflict):
        journal.authorize_submission(
            "cross-portfolio", intent, exposure, allocation, gates, descriptor
        )

    other = make_case(now, account_scope=ACCOUNT_B, portfolio_id="portfolio-b")
    result = journal.authorize_submission(
        "other-account", other[0], other[1], other[2], other[3], other[5]
    )
    assert result[2] is not None
    assert len(journal.replay().active_reservations) == 2

    other_domain = make_case(
        now,
        execution_domain_scope="other-paper-domain",
        portfolio_id="portfolio-c",
    )
    with pytest.raises(ReservationConflict):
        journal.authorize_submission(
            "other-domain",
            other_domain[0],
            other_domain[1],
            other_domain[2],
            other_domain[3],
            other_domain[5],
        )


def test_different_key_concurrency_cannot_cumulatively_over_close(tmp_path, now):
    path = tmp_path / "safety.db"
    SafetyJournal(path).initialize()
    first = make_case(now, order_quantity=Decimal("4"))
    second = make_case(now, order_quantity=Decimal("4"))
    barrier = threading.Barrier(2)

    def compete(key, case):
        barrier.wait()
        try:
            result = SafetyJournal(path, clock=lambda: now).authorize_submission(
                key, case[0], case[1], case[2], case[3], case[5]
            )
            return result[2] is not None
        except ReservationConflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(compete, "first-key", first),
            executor.submit(compete, "second-key", second),
        )
        results = [future.result() for future in futures]
    assert sorted(results) == [False, True]
    assert len(SafetyJournal(path).replay().active_reservations) == 1


@pytest.mark.parametrize(
    ("fault_stage", "fault_sequence"),
    [
        ("AFTER_APPEND", 1),
        ("AFTER_APPEND", 2),
        ("AFTER_APPEND", 3),
        ("BEFORE_COMMIT", 3),
    ],
)
def test_fault_after_each_append_or_before_commit_rolls_back_everything(
    tmp_path, now, fault_stage, fault_sequence
):
    class InjectedCancellation(BaseException):
        pass

    path = tmp_path / f"fault-{fault_stage}-{fault_sequence}.db"
    SafetyJournal(path).initialize()

    def fail(stage, event):
        if stage == fault_stage and event.sequence == fault_sequence:
            raise InjectedCancellation()

    journal = SafetyJournal(path, clock=lambda: now, fault_hook=fail)
    intent, exposure, allocation, gates, _, descriptor = make_case(now)
    with pytest.raises(InjectedCancellation):
        journal.authorize_submission("key", intent, exposure, allocation, gates, descriptor)
    assert SafetyJournal(path).replay().last_sequence == 0


def test_unknown_outcome_and_restart_retain_quarantine_indefinitely(tmp_path, now):
    path = tmp_path / "safety.db"
    journal = SafetyJournal(path, clock=lambda: now)
    journal.initialize()
    intent, _, _, claim, _ = authorize(journal, now)
    journal.mark_outcome_unknown("intent-key", intent.fingerprint())
    for _ in range(3):
        state = SafetyJournal(path).replay()
        assert state.quarantined_reservations[0].claim_id == claim.claim_id
        assert state.active_reservations[0].outcome_unknown


def test_unknown_outcome_revokes_unconsumed_live_permit(tmp_path, now):
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: now)
    journal.initialize()
    intent, _, _, _, permit = authorize(journal, now)
    journal.mark_outcome_unknown("intent-key", intent.fingerprint())
    with pytest.raises(StateTransitionError, match="not issued by this live"):
        journal.consume_submission_permit(permit)


def test_release_and_permit_consumption_are_mutually_safe(tmp_path, now):
    clock = [now]
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: clock[0])
    journal.initialize()
    intent, _, reservation, claim, permit = authorize(journal, now)
    evidence = terminal_evidence(
        now,
        reservation,
        claim,
        intent,
        terminal_order_status=TerminalOrderStatus.NO_SUBMISSION_CONFIRMED,
        filled_quantity=Decimal("0"),
        remaining_quantity=intent.quantity,
        account_position_quantity=intent.account_current_quantity,
        portfolio_position_quantity=intent.portfolio_current_quantity,
        aggregate_allocated_quantity=intent.account_current_quantity,
    )
    clock[0] = now + timedelta(seconds=1)
    barrier = threading.Barrier(2)

    def consume():
        barrier.wait()
        try:
            journal.consume_submission_permit(permit)
            return "consumed"
        except StateTransitionError:
            return "consume-blocked"

    def release():
        barrier.wait()
        try:
            journal.release_after_reconciliation(
                "intent-key",
                intent.fingerprint(),
                evidence,
            )
            return "released"
        except StateTransitionError:
            return "release-blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(consume), executor.submit(release))
        outcomes = {future.result() for future in futures}
    assert outcomes in (
        {"consumed", "release-blocked"},
        {"consume-blocked", "released"},
    )
    state = journal.replay()
    if "consumed" in outcomes:
        assert state.active_reservations
    else:
        assert not state.active_reservations


def test_release_revokes_unconsumed_live_permit(tmp_path, now):
    clock = [now]
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: clock[0])
    journal.initialize()
    intent, _, reservation, claim, permit = authorize(journal, now)
    evidence = terminal_evidence(
        now,
        reservation,
        claim,
        intent,
        terminal_order_status=TerminalOrderStatus.NO_SUBMISSION_CONFIRMED,
        filled_quantity=Decimal("0"),
        remaining_quantity=intent.quantity,
        account_position_quantity=intent.account_current_quantity,
        portfolio_position_quantity=intent.portfolio_current_quantity,
        aggregate_allocated_quantity=intent.account_current_quantity,
    )
    clock[0] = now + timedelta(seconds=1)
    journal.release_after_reconciliation(
        "intent-key",
        intent.fingerprint(),
        evidence,
    )
    with pytest.raises(StateTransitionError, match="not issued by this live"):
        journal.consume_submission_permit(permit)


def test_cross_instance_release_and_consume_serialize_durably(tmp_path, now):
    path = tmp_path / "safety.db"
    clock = [now]
    issuer = SafetyJournal(path, clock=lambda: clock[0])
    issuer.initialize()
    intent, _, reservation, claim, permit = authorize(issuer, now)
    evidence = terminal_evidence(
        now,
        reservation,
        claim,
        intent,
        terminal_order_status=TerminalOrderStatus.NO_SUBMISSION_CONFIRMED,
        filled_quantity=Decimal("0"),
        remaining_quantity=intent.quantity,
        account_position_quantity=intent.account_current_quantity,
        portfolio_position_quantity=intent.portfolio_current_quantity,
        aggregate_allocated_quantity=intent.account_current_quantity,
    )
    clock[0] = now + timedelta(seconds=1)
    reconciler = SafetyJournal(path, clock=lambda: clock[0])
    reconciler.release_after_reconciliation("intent-key", intent.fingerprint(), evidence)
    with pytest.raises(StateTransitionError, match="already terminal"):
        issuer.consume_submission_permit(permit)


def test_cross_instance_consume_blocks_no_submission_release(tmp_path, now):
    path = tmp_path / "safety.db"
    clock = [now]
    issuer = SafetyJournal(path, clock=lambda: clock[0])
    issuer.initialize()
    intent, descriptor, reservation, claim, permit = authorize(issuer, now)
    assert issuer.consume_submission_permit(permit) == descriptor
    evidence = terminal_evidence(
        now,
        reservation,
        claim,
        intent,
        terminal_order_status=TerminalOrderStatus.NO_SUBMISSION_CONFIRMED,
        filled_quantity=Decimal("0"),
        remaining_quantity=intent.quantity,
        account_position_quantity=intent.account_current_quantity,
        portfolio_position_quantity=intent.portfolio_current_quantity,
        aggregate_allocated_quantity=intent.account_current_quantity,
    )
    clock[0] = now + timedelta(seconds=1)
    reconciler = SafetyJournal(path, clock=lambda: clock[0])
    with pytest.raises(StateTransitionError, match="dispatched authority"):
        reconciler.release_after_reconciliation("intent-key", intent.fingerprint(), evidence)
    state = reconciler.replay()
    assert state.active_reservations[0].outcome_unknown


def test_cross_instance_release_consume_race_has_exactly_one_winner(tmp_path, now):
    path = tmp_path / "safety.db"
    issuer = SafetyJournal(path, clock=lambda: now)
    issuer.initialize()
    intent, _, reservation, claim, permit = authorize(issuer, now)
    evidence = terminal_evidence(
        now,
        reservation,
        claim,
        intent,
        terminal_order_status=TerminalOrderStatus.NO_SUBMISSION_CONFIRMED,
        filled_quantity=Decimal("0"),
        remaining_quantity=intent.quantity,
        account_position_quantity=intent.account_current_quantity,
        portfolio_position_quantity=intent.portfolio_current_quantity,
        aggregate_allocated_quantity=intent.account_current_quantity,
    )
    reconciler = SafetyJournal(
        path,
        clock=lambda: now + timedelta(seconds=1),
    )
    barrier = threading.Barrier(2)

    def consume():
        barrier.wait()
        try:
            issuer.consume_submission_permit(permit)
            return "consumed"
        except StateTransitionError:
            return "consume-blocked"

    def release():
        barrier.wait()
        try:
            reconciler.release_after_reconciliation("intent-key", intent.fingerprint(), evidence)
            return "released"
        except StateTransitionError:
            return "release-blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(consume), executor.submit(release))
        outcomes = {future.result() for future in futures}
    assert outcomes in (
        {"consumed", "release-blocked"},
        {"consume-blocked", "released"},
    )


def test_cross_instance_unknown_outcome_blocks_permit_consume(tmp_path, now):
    path = tmp_path / "safety.db"
    issuer = SafetyJournal(path, clock=lambda: now)
    issuer.initialize()
    intent, _, _, _, permit = authorize(issuer, now)
    SafetyJournal(path, clock=lambda: now).mark_outcome_unknown("intent-key", intent.fingerprint())
    with pytest.raises(StateTransitionError, match="outcome-unknown"):
        issuer.consume_submission_permit(permit)


def test_release_requires_exact_fresh_claim_bound_terminal_truth(tmp_path, now):
    path = tmp_path / "safety.db"
    clock = [now]
    journal = SafetyJournal(path, clock=lambda: clock[0])
    journal.initialize()
    intent, _, reservation, claim, permit = authorize(journal, now)
    journal.consume_submission_permit(permit)
    evidence = terminal_evidence(now, reservation, claim, intent)
    clock[0] = now + timedelta(seconds=1)
    bad_evidence = (
        replace(evidence, claim_id="claim-" + ("f" * 32)),
        replace(evidence, claim_sequence=claim.sequence + 1),
        replace(evidence, reservation_id="res-" + ("f" * 32)),
        replace(evidence, observed_at=claim.claimed_at),
        replace(evidence, position_observed_at=claim.claimed_at),
        replace(evidence, observed_at=now - timedelta(microseconds=1)),
        replace(evidence, position_observed_at=now - timedelta(microseconds=1)),
        replace(evidence, open_orders_all_clients=False),
        replace(evidence, open_orders_snapshot_stable=False),
        replace(evidence, active_order_count=1),
        replace(evidence, account_position_quantity=Decimal("9")),
    )
    for bad in bad_evidence:
        with pytest.raises(StateTransitionError):
            journal.release_after_reconciliation(
                "intent-key",
                intent.fingerprint(),
                bad,
            )
    released = journal.release_after_reconciliation("intent-key", intent.fingerprint(), evidence)
    assert released.released
    assert not journal.replay().active_reservations


def test_partial_terminal_release_uses_exact_filled_quantity(tmp_path, now):
    clock = [now]
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: clock[0])
    journal.initialize()
    intent, _, reservation, claim, permit = authorize(journal, now)
    journal.consume_submission_permit(permit)
    evidence = terminal_evidence(
        now,
        reservation,
        claim,
        intent,
        terminal_order_status=TerminalOrderStatus.CANCELLED,
        filled_quantity=Decimal("1"),
        remaining_quantity=Decimal("1"),
        account_position_quantity=Decimal("9"),
        portfolio_position_quantity=Decimal("4"),
        aggregate_allocated_quantity=Decimal("9"),
    )
    clock[0] = now + timedelta(seconds=1)
    assert journal.release_after_reconciliation(
        "intent-key", intent.fingerprint(), evidence
    ).released


def test_reconciliation_arithmetic_ignores_low_ambient_decimal_precision(
    tmp_path,
    now,
):
    clock = [now]
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: clock[0])
    journal.initialize()
    intent, _, reservation, claim, permit = authorize(
        journal,
        now,
        account_quantity=Decimal("1.123456789012"),
        portfolio_quantity=Decimal("1.123456789012"),
        order_quantity=Decimal("0.123456789012"),
    )
    journal.consume_submission_permit(permit)
    evidence = terminal_evidence(
        now,
        reservation,
        claim,
        intent,
        terminal_order_status=TerminalOrderStatus.CANCELLED,
        filled_quantity=Decimal("0.123456789011"),
        remaining_quantity=Decimal("0.000000000001"),
        account_position_quantity=Decimal("1.000000000001"),
        portfolio_position_quantity=Decimal("1.000000000001"),
        aggregate_allocated_quantity=Decimal("1.000000000001"),
    )
    clock[0] = now + timedelta(seconds=1)
    with localcontext() as context:
        context.prec = 3
        released = journal.release_after_reconciliation(
            "intent-key",
            intent.fingerprint(),
            evidence,
        )
    assert released.released
    assert journal.replay().reservations[0].released


def test_authoritative_no_submission_reconciliation_releases_crash_quarantine(tmp_path, now):
    clock = [now]
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: clock[0])
    journal.initialize()
    intent, _, reservation, claim, _ = authorize(journal, now)
    evidence = terminal_evidence(
        now,
        reservation,
        claim,
        intent,
        terminal_order_status=TerminalOrderStatus.NO_SUBMISSION_CONFIRMED,
        filled_quantity=Decimal("0"),
        remaining_quantity=intent.quantity,
        account_position_quantity=intent.account_current_quantity,
        portfolio_position_quantity=intent.portfolio_current_quantity,
        aggregate_allocated_quantity=intent.account_current_quantity,
    )
    clock[0] = now + timedelta(seconds=1)
    released = journal.release_after_reconciliation("intent-key", intent.fingerprint(), evidence)
    assert released.released
    assert not journal.replay().quarantined_reservations


def test_released_claim_cannot_reuse_stale_authoritative_snapshot(tmp_path, now):
    clock = [now]
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: clock[0])
    journal.initialize()
    case = make_case(now)
    reservation, claim, permit = journal.authorize_submission(
        "first", case[0], case[1], case[2], case[3], case[5]
    )
    journal.consume_submission_permit(permit)
    evidence = terminal_evidence(now, reservation, claim, case[0])
    clock[0] = now + timedelta(seconds=1)
    journal.release_after_reconciliation(
        "first",
        case[0].fingerprint(),
        evidence,
    )
    clock[0] = now + timedelta(seconds=2)
    with pytest.raises(StateTransitionError, match="SNAPSHOTS_NOT_NEWER"):
        journal.authorize_submission(
            "second",
            case[0],
            case[1],
            case[2],
            case[3],
            case[5],
        )
    renamed = (
        case[0],
        replace(case[1], snapshot_id="account-snapshot-renamed"),
        replace(case[2], snapshot_id="allocation-snapshot-renamed"),
        replace(case[3], open_orders_snapshot_id="orders-snapshot-renamed"),
        case[4],
        case[5],
    )
    with pytest.raises(StateTransitionError, match="SNAPSHOTS_NOT_NEWER"):
        journal.authorize_submission(
            "third",
            renamed[0],
            renamed[1],
            renamed[2],
            renamed[3],
            renamed[5],
        )


def test_update_delete_denied_and_payload_tamper_detected(tmp_path, now):
    path = tmp_path / "safety.db"
    journal = SafetyJournal(path, clock=lambda: now)
    journal.initialize()
    authorize(journal, now)
    connection = sqlite3.connect(path)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE safety_journal_events SET payload_json = '{}' WHERE sequence = 1"
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("DELETE FROM safety_journal_events WHERE sequence = 1")
    connection.execute("DROP TRIGGER safety_journal_no_update")
    connection.execute("UPDATE safety_journal_events SET payload_json = '{}' WHERE sequence = 1")
    connection.commit()
    connection.close()
    with pytest.raises(JournalIntegrityError):
        journal.replay()


def test_schema_tamper_and_malicious_bound_strings(tmp_path, now):
    path = tmp_path / "safety.db"
    journal = SafetyJournal(path, clock=lambda: now)
    journal.initialize()
    case = make_case(now)
    malicious = replace(case[0], reason="x'); DROP TABLE safety_journal_events;--")
    from robo_trader.safety import evaluate_reduce_only

    decision = evaluate_reduce_only(malicious, case[1], case[2], case[3])
    journal.authorize_submission(
        "key'; DROP TABLE safety_schema_version;--",
        malicious,
        case[1],
        case[2],
        case[3],
        case[5],
    )
    assert journal.replay().last_sequence == 3
    connection = sqlite3.connect(path)
    connection.execute("DROP TRIGGER safety_schema_version_no_update")
    connection.execute("UPDATE safety_schema_version SET version = 999")
    connection.commit()
    connection.close()
    with pytest.raises(JournalIntegrityError):
        journal.replay()


def test_initialize_refuses_to_reuse_an_unrelated_database(tmp_path):
    path = tmp_path / "trading-data.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE trades(id INTEGER PRIMARY KEY)")
    connection.commit()
    original_journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
    connection.close()
    path.chmod(0o644)
    original_bytes = path.read_bytes()
    original_stat = path.stat()
    with pytest.raises(JournalIntegrityError, match="unrelated tables"):
        SafetyJournal(path).initialize()
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == original_journal_mode
    finally:
        connection.close()
    assert not type(path)(f"{path}-wal").exists()
    assert not type(path)(f"{path}-shm").exists()


def test_initialize_rejects_existing_symlink_without_mutating_target(tmp_path, now):
    target = tmp_path / "target.db"
    SafetyJournal(target, clock=lambda: now).initialize()
    target.chmod(0o644)
    original_bytes = target.read_bytes()
    original_stat = target.stat()
    link = tmp_path / "safety.db"
    link.symlink_to(target)

    with pytest.raises(JournalIntegrityError, match="non-symlink regular file"):
        SafetyJournal(link).initialize()

    assert link.is_symlink()
    assert target.read_bytes() == original_bytes
    assert target.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert not type(target)(f"{target}-wal").exists()
    assert not type(target)(f"{target}-shm").exists()


def test_replay_rejects_existing_symlink(tmp_path, now):
    target = tmp_path / "target.db"
    SafetyJournal(target, clock=lambda: now).initialize()
    link = tmp_path / "safety.db"
    link.symlink_to(target)

    with pytest.raises(JournalIntegrityError, match="non-symlink regular file"):
        SafetyJournal(link).replay()


def _swap_open_and_restore(monkeypatch, path, replacement, parked):
    import robo_trader.safety.journal as journal_module

    real_connect = journal_module.sqlite3.connect

    def connect_to_replacement(database, *args, **kwargs):
        database_text = str(database)
        if database_text != str(path) and not database_text.startswith(path.as_uri()):
            return real_connect(database, *args, **kwargs)
        path.replace(parked)
        replacement.replace(path)
        try:
            connection = real_connect(database, *args, **kwargs)
            connection.execute("SELECT COUNT(*) FROM safety_journal_events").fetchone()
        finally:
            path.replace(replacement)
            parked.replace(path)
        return connection

    monkeypatch.setattr(journal_module.sqlite3, "connect", connect_to_replacement)
    return real_connect


def test_replay_rejects_swap_open_and_restore_of_same_schema(
    tmp_path,
    now,
    monkeypatch,
):
    path = tmp_path / "safety.db"
    replacement = tmp_path / "replacement.db"
    SafetyJournal(path, clock=lambda: now).initialize()
    replacement_journal = SafetyJournal(replacement, clock=lambda: now)
    replacement_journal.initialize()
    authorize(replacement_journal, now)
    parked = tmp_path / "parked.db"
    real_connect = _swap_open_and_restore(monkeypatch, path, replacement, parked)

    with pytest.raises(JournalIntegrityError, match="identity changed"):
        SafetyJournal(path).replay()

    original = real_connect(f"{path.as_uri()}?mode=ro", uri=True)
    substituted = real_connect(f"{replacement.as_uri()}?mode=ro", uri=True)
    try:
        assert original.execute("SELECT COUNT(*) FROM safety_journal_events").fetchone() == (0,)
        assert substituted.execute("SELECT COUNT(*) FROM safety_journal_events").fetchone() == (3,)
    finally:
        original.close()
        substituted.close()


def test_replay_rejects_substitution_even_with_expected_inode_decoy(
    tmp_path,
    now,
    monkeypatch,
):
    import robo_trader.safety.journal as journal_module

    path = tmp_path / "safety.db"
    replacement = tmp_path / "replacement.db"
    parked = tmp_path / "parked.db"
    SafetyJournal(path, clock=lambda: now).initialize()
    replacement_journal = SafetyJournal(replacement, clock=lambda: now)
    replacement_journal.initialize()
    authorize(replacement_journal, now)
    real_connect = journal_module.sqlite3.connect
    held_descriptors = []

    def connect_to_replacement_with_decoy(database, *args, **kwargs):
        if not str(database).startswith(path.as_uri()):
            return real_connect(database, *args, **kwargs)
        path.replace(parked)
        replacement.replace(path)
        try:
            connection = real_connect(database, *args, **kwargs)
            connection.execute("SELECT COUNT(*) FROM safety_journal_events").fetchone()
            held_descriptors.append(os.open(parked, os.O_RDONLY))
        finally:
            path.replace(replacement)
            parked.replace(path)
        return connection

    monkeypatch.setattr(
        journal_module.sqlite3,
        "connect",
        connect_to_replacement_with_decoy,
    )
    try:
        with pytest.raises(JournalIntegrityError, match="identity changed"):
            SafetyJournal(path).replay()
    finally:
        for file_descriptor in held_descriptors:
            os.close(file_descriptor)


def test_write_rejects_swap_open_and_restore_of_same_schema(
    tmp_path,
    now,
    monkeypatch,
):
    path = tmp_path / "safety.db"
    replacement = tmp_path / "replacement.db"
    journal = SafetyJournal(path, clock=lambda: now)
    journal.initialize()
    replacement_journal = SafetyJournal(replacement, clock=lambda: now)
    replacement_journal.initialize()
    authorize(replacement_journal, now)
    parked = tmp_path / "parked.db"
    real_connect = _swap_open_and_restore(monkeypatch, path, replacement, parked)

    with pytest.raises(JournalIntegrityError, match="identity changed"):
        authorize(journal, now, key="must-not-write")

    original = real_connect(f"{path.as_uri()}?mode=ro", uri=True)
    substituted = real_connect(f"{replacement.as_uri()}?mode=ro", uri=True)
    try:
        assert original.execute("SELECT COUNT(*) FROM safety_journal_events").fetchone() == (0,)
        assert substituted.execute("SELECT COUNT(*) FROM safety_journal_events").fetchone() == (3,)
    finally:
        original.close()
        substituted.close()


def test_fault_hook_can_replay_without_self_deadlock(tmp_path, now):
    observations = []
    journal = None

    def inspect_uncommitted_state(stage, _event):
        observations.append((stage, journal.replay().last_sequence))

    journal = SafetyJournal(
        tmp_path / "safety.db",
        clock=lambda: now,
        fault_hook=inspect_uncommitted_state,
    )
    journal.initialize()
    authorize(journal, now)
    assert observations == [
        ("AFTER_APPEND", 0),
        ("AFTER_APPEND", 0),
        ("AFTER_APPEND", 0),
        ("BEFORE_COMMIT", 0),
    ]


@pytest.mark.parametrize("guard_name", ["Py_GIL_DISABLED", "Py_TRACE_REFS"])
def test_incompatible_cpython_abi_fails_before_pointer_access(
    tmp_path,
    monkeypatch,
    guard_name,
):
    import robo_trader.safety.journal as journal_module

    path = tmp_path / "plain.db"
    connection = sqlite3.connect(path)
    real_get_config_var = journal_module.sysconfig.get_config_var

    def guarded_config(name):
        if name == guard_name:
            return 1
        return real_get_config_var(name)

    monkeypatch.setattr(journal_module.sysconfig, "get_config_var", guarded_config)
    with pytest.raises(JournalIntegrityError, match="free-threaded or trace-reference"):
        journal_module._sqlite_connection_file_identity(connection)
    connection.close()


def test_non_cpython_runtime_fails_before_pointer_access(tmp_path, monkeypatch):
    import robo_trader.safety.journal as journal_module

    path = tmp_path / "plain.db"
    connection = sqlite3.connect(path)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            journal_module.sys,
            "implementation",
            SimpleNamespace(name="not-cpython"),
        )
        with pytest.raises(JournalIntegrityError, match="supported CPython"):
            journal_module._sqlite_connection_file_identity(connection)
    connection.close()


def test_initialize_revalidates_same_connection_after_path_swap(
    tmp_path,
    now,
    monkeypatch,
):
    path = tmp_path / "safety.db"
    SafetyJournal(path, clock=lambda: now).initialize()
    replacement = tmp_path / "replacement.db"
    connection = sqlite3.connect(replacement)
    connection.execute("CREATE TABLE trades(id INTEGER PRIMARY KEY)")
    connection.commit()
    original_journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
    connection.close()
    replacement.chmod(0o644)
    original_bytes = replacement.read_bytes()

    journal = SafetyJournal(path, clock=lambda: now)
    original_precheck = journal._assert_existing_path_is_dedicated

    def precheck_then_swap():
        original_precheck()
        replacement.replace(path)

    monkeypatch.setattr(
        journal,
        "_assert_existing_path_is_dedicated",
        precheck_then_swap,
    )
    with pytest.raises(JournalIntegrityError, match="unrelated tables"):
        journal.initialize()
    assert path.read_bytes() == original_bytes
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == original_journal_mode
    finally:
        connection.close()


def test_new_initialize_detects_swap_before_connection_binding(
    tmp_path,
    now,
    monkeypatch,
):
    path = tmp_path / "safety.db"
    replacement = tmp_path / "replacement.db"
    connection = sqlite3.connect(replacement)
    connection.execute("CREATE TABLE trades(id INTEGER PRIMARY KEY)")
    connection.commit()
    original_journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
    connection.close()
    replacement.chmod(0o644)
    original_bytes = replacement.read_bytes()

    journal = SafetyJournal(path, clock=lambda: now)
    original_bind = journal._bind_connection_to_path

    def swap_then_bind(
        connection,
        pre_connect_identity,
        guardian_file_descriptor,
    ):
        replacement.replace(path)
        original_bind(
            connection,
            pre_connect_identity,
            guardian_file_descriptor,
        )

    monkeypatch.setattr(
        journal,
        "_bind_connection_to_path",
        swap_then_bind,
    )
    with pytest.raises(
        JournalIntegrityError,
        match="identity changed|moved or replaced",
    ):
        journal.initialize()

    assert path.read_bytes() == original_bytes
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == original_journal_mode
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        } == {"trades"}
    finally:
        connection.close()


def test_open_cleanup_releases_binding_after_post_bind_exception(
    tmp_path,
    now,
    monkeypatch,
):
    path = tmp_path / "safety.db"
    SafetyJournal(path, clock=lambda: now).initialize()
    journal = SafetyJournal(path, clock=lambda: now)
    original_bind = journal._bind_connection_to_path
    captured = {}

    def bind_then_interrupt(
        connection,
        expected_identity,
        guardian_file_descriptor,
    ):
        original_bind(
            connection,
            expected_identity,
            guardian_file_descriptor,
        )
        binding = journal._path_bindings[connection]
        captured["connection"] = connection
        captured["guardian_file_descriptor"] = binding.file_descriptor
        raise KeyboardInterrupt

    monkeypatch.setattr(
        journal,
        "_bind_connection_to_path",
        bind_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        journal.replay()

    assert journal._path_bindings == {}
    with pytest.raises(OSError):
        os.fstat(captured["guardian_file_descriptor"])
    with pytest.raises(sqlite3.ProgrammingError):
        captured["connection"].execute("SELECT 1")
    assert SafetyJournal(path, clock=lambda: now).replay().last_sequence == 0


def test_write_revalidates_replaced_path_before_mutating_it(tmp_path, now):
    path = tmp_path / "safety.db"
    journal = SafetyJournal(path, clock=lambda: now)
    journal.initialize()
    replacement = tmp_path / "replacement.db"
    connection = sqlite3.connect(replacement)
    connection.execute("CREATE TABLE trades(id INTEGER PRIMARY KEY)")
    connection.commit()
    original_journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
    connection.close()
    replacement.chmod(0o644)
    original_bytes = replacement.read_bytes()
    replacement.replace(path)

    case = make_case(now)
    with pytest.raises(JournalIntegrityError, match="unrelated tables"):
        journal.authorize_submission(
            "intent-key",
            case[0],
            case[1],
            case[2],
            case[3],
            case[5],
        )
    assert path.read_bytes() == original_bytes
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == original_journal_mode
    finally:
        connection.close()


def test_permit_consume_fails_if_path_swaps_after_connection_validation(
    tmp_path,
    now,
    monkeypatch,
):
    path = tmp_path / "safety.db"
    journal = SafetyJournal(path, clock=lambda: now)
    journal.initialize()
    _, _, _, _, permit = authorize(journal, now)

    replacement = tmp_path / "replacement.db"
    connection = sqlite3.connect(replacement)
    connection.execute("CREATE TABLE trades(id INTEGER PRIMARY KEY)")
    connection.commit()
    original_journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
    connection.close()
    replacement.chmod(0o644)
    original_bytes = replacement.read_bytes()

    original_validation = journal._validate_connection_before_mutation

    def validate_then_swap(connection, *, allow_empty):
        original_validation(connection, allow_empty=allow_empty)
        replacement.replace(path)

    monkeypatch.setattr(
        journal,
        "_validate_connection_before_mutation",
        validate_then_swap,
    )
    with pytest.raises(JournalIntegrityError, match="identity"):
        journal.consume_submission_permit(permit)

    assert path.read_bytes() == original_bytes
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == original_journal_mode
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        } == {"trades"}
    finally:
        connection.close()


def test_special_character_path_replays_and_journal_is_owner_only(tmp_path, now):
    path = tmp_path / "safety?#%journal.db"
    journal = SafetyJournal(path, clock=lambda: now)
    journal.initialize()
    authorize(journal, now)
    assert journal.replay().last_sequence == 3
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for suffix in ("-wal", "-shm"):
        companion = type(path)(f"{path}{suffix}")
        if companion.exists():
            assert stat.S_IMODE(companion.stat().st_mode) == 0o600


def test_all_frozen_journal_models_reject_raw_account_scope(tmp_path, now):
    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: now)
    journal.initialize()
    _, _, reservation, claim, _ = authorize(journal, now)
    state = journal.replay()
    for model in (
        reservation,
        claim,
        state.events[0],
        state.reservations[0],
    ):
        with pytest.raises(ValidationError):
            replace(model, account_scope="DU1234567")
    with pytest.raises(ValidationError):
        replace(state, last_sequence=state.last_sequence + 1)


def test_extra_schema_object_is_detected_on_replay(tmp_path):
    path = tmp_path / "safety.db"
    journal = SafetyJournal(path)
    journal.initialize()
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE hidden_mutable_state(value TEXT)")
    connection.commit()
    connection.close()
    with pytest.raises(JournalIntegrityError, match="sqlite_master"):
        journal.replay()


def test_denied_decision_and_reasons_are_durable_but_create_no_authority(tmp_path, now):
    from robo_trader.safety import OrderSide

    journal = SafetyJournal(tmp_path / "safety.db", clock=lambda: now)
    journal.initialize()
    intent, exposure, allocation, gates, _, _ = make_case(now, side=OrderSide.BUY)
    event = journal.record_rejection("denied-key", intent, exposure, allocation, gates)
    assert event.event_type.value == "SAFETY_DECISION"
    payload = json.loads(event.payload_json)
    assert payload["decision"]["outcome"] == "DENY"
    assert payload["decision"]["reason_codes"]
    state = journal.replay()
    assert not state.reservations
    assert journal.record_rejection("denied-key", intent, exposure, allocation, gates).sequence == 1


def test_hours_old_context_cannot_authorize_under_any_caller_window(tmp_path, now):
    case = make_case(now)
    journal = SafetyJournal(
        tmp_path / "safety.db",
        clock=lambda: now + timedelta(hours=23),
    )
    journal.initialize()
    with pytest.raises(StateTransitionError, match="STALE_AUTHORIZATION_CONTEXT"):
        journal.authorize_submission(
            "stale-context",
            case[0],
            case[1],
            case[2],
            case[3],
            case[5],
        )
    state = journal.replay()
    assert state.last_sequence == 1
    assert not state.reservations
