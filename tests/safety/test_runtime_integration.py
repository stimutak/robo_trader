import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from datetime import timedelta
from decimal import Decimal, localcontext

import pytest

from robo_trader.safety import (
    PAPER_EXECUTION_DOMAIN_SCOPE,
    AccountPosition,
    AuthoritativeContract,
    EvidenceStatus,
    FakeOrderSubmitter,
    FakeSubmissionOnlyError,
    JournalError,
    JournalIntegrityError,
    OpenOrderSnapshot,
    OrderSide,
    OrderType,
    PaperExecutionIdentity,
    PortfolioAllocation,
    ReconciliationStatus,
    ReservationConflict,
    RuntimeAuthorizationBlocked,
    RuntimeNotStarted,
    RuntimeOrderRequest,
    RuntimeSafetyError,
    RuntimeStartupBlocked,
    SafetyJournal,
    SafetyRuntimeCoordinator,
    TimeInForce,
    TransportState,
    ValidationError,
)
from robo_trader.safety.runtime import _assemble_coherent_safety_snapshot

RUNTIME_ACCOUNT_SCOPE = "acct_v1_" + ("0123456789abcdef" * 4)


def replace_trusted_snapshot(snapshot, **changes):
    """Assemble a fresh trusted test fixture without trusting dataclass clones."""

    values = {
        definition.name: getattr(snapshot, definition.name)
        for definition in fields(snapshot)
        if definition.name != "_assembly_marker"
    }
    values.update(changes)
    return _assemble_coherent_safety_snapshot(**values)


def make_runtime_case(tmp_path, now, *, journal=None):
    identity = PaperExecutionIdentity(
        PAPER_EXECUTION_DOMAIN_SCOPE,
        RUNTIME_ACCOUNT_SCOPE,
    )
    journal = journal or SafetyJournal(tmp_path / "runtime-safety.db", clock=lambda: now)
    journal.initialize(
        execution_domain_scope=identity.execution_domain_scope,
        account_scope=identity.account_scope,
    )
    coordinator = SafetyRuntimeCoordinator(identity, journal, clock=lambda: now)
    contract = AuthoritativeContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        trading_class="NMS",
        observed_at=now,
        snapshot_id="contract-snapshot-1",
        source="qualified-contract-cache",
        broker_timestamp=now,
        retrieval_timestamp=now,
        transport_generation="worker-generation-1",
    )
    request = RuntimeOrderRequest(
        portfolio_id="portfolio-a",
        contract=contract,
        side=OrderSide.SELL,
        quantity=Decimal("2.125"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("198.25"),
        time_in_force=TimeInForce.DAY,
        outside_regular_hours=False,
        order_ref="runtime-close-aapl-1",
        reason="protective reduction",
        strategy="stop-loss",
    )
    open_orders = OpenOrderSnapshot(
        observed_at=now,
        snapshot_id="open-orders-1",
        transport_generation="worker-generation-1",
    )
    snapshot = _assemble_coherent_safety_snapshot(
        execution_domain_scope=identity.execution_domain_scope,
        account_scope=identity.account_scope,
        observed_at=now,
        snapshot_id="account-snapshot-1",
        source="broker-account-snapshot",
        allocation_observed_at=now,
        allocation_snapshot_id="allocation-snapshot-1",
        allocation_source="allocation-database",
        reconciliation_observed_at=now,
        reconciliation_snapshot_id="reconciliation-snapshot-1",
        transport_generation="worker-generation-1",
        account_positions=(AccountPosition(265598, "AAPL", Decimal("10.500")),),
        portfolio_allocations=(
            PortfolioAllocation("portfolio-a", 265598, "AAPL", Decimal("5.500")),
            PortfolioAllocation("portfolio-b", 265598, "AAPL", Decimal("5.000")),
        ),
        open_orders=open_orders,
        transport_state=TransportState.CONNECTED,
        reconciliation_status=ReconciliationStatus.PASSED,
    )
    return coordinator, journal, request, snapshot


@pytest.mark.parametrize(
    ("domain", "scope"),
    [
        ("live-domain", RUNTIME_ACCOUNT_SCOPE),
        ("production", RUNTIME_ACCOUNT_SCOPE),
        (PAPER_EXECUTION_DOMAIN_SCOPE, "DU1234567"),
        (PAPER_EXECUTION_DOMAIN_SCOPE, "acct_v1_" + ("A" * 64)),
        (PAPER_EXECUTION_DOMAIN_SCOPE, "1234567"),
        (PAPER_EXECUTION_DOMAIN_SCOPE, "acct_v1_" + ("0" * 64)),
    ],
)
def test_identity_is_paper_only_and_requires_supplied_opaque_scope(domain, scope):
    with pytest.raises(ValidationError):
        PaperExecutionIdentity(domain, scope)


def test_coordinator_requires_separately_supplied_journal(now):
    identity = PaperExecutionIdentity(
        PAPER_EXECUTION_DOMAIN_SCOPE,
        RUNTIME_ACCOUNT_SCOPE,
    )
    with pytest.raises(TypeError, match="supplied separately"):
        SafetyRuntimeCoordinator(identity, object(), clock=lambda: now)


@pytest.mark.parametrize("skew_seconds", (-121, 121))
def test_authoritative_contract_rejects_absolute_broker_clock_skew(
    now,
    skew_seconds,
):
    with pytest.raises(ValidationError, match="clock-skew"):
        AuthoritativeContract(
            con_id=265598,
            symbol="AAPL",
            local_symbol="AAPL",
            security_type="STK",
            currency="USD",
            exchange="SMART",
            primary_exchange="NASDAQ",
            trading_class="NMS",
            observed_at=now,
            snapshot_id="contract-snapshot-skew",
            source="qualified-contract-cache",
            broker_timestamp=now + timedelta(seconds=skew_seconds),
            retrieval_timestamp=now,
            transport_generation="worker-generation-1",
        )


def test_startup_does_not_initialize_a_missing_journal(tmp_path, now):
    coordinator = SafetyRuntimeCoordinator(
        PaperExecutionIdentity(
            PAPER_EXECUTION_DOMAIN_SCOPE,
            RUNTIME_ACCOUNT_SCOPE,
        ),
        SafetyJournal(tmp_path / "missing.db", clock=lambda: now),
        clock=lambda: now,
    )
    with pytest.raises(JournalError):
        coordinator.start()
    assert not coordinator.started
    assert not (tmp_path / "missing.db").exists()


def test_startup_replay_is_required_before_authorization(tmp_path, now):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    with pytest.raises(RuntimeNotStarted):
        coordinator.authorize("runtime-key", request, snapshot)


def test_coherent_snapshot_rejects_caller_supplied_assembly_marker(tmp_path, now):
    _, _, _, snapshot = make_runtime_case(tmp_path, now)

    with pytest.raises(ValidationError, match="trusted evidence assembly"):
        replace(snapshot, _assembly_marker=object())


def test_coordinator_rejects_replace_clone_with_copied_trusted_marker(tmp_path, now):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    forged = replace(
        snapshot,
        account_positions=(AccountPosition(265598, "AAPL", Decimal("1000.000")),),
        portfolio_allocations=(
            PortfolioAllocation(
                "portfolio-a",
                265598,
                "AAPL",
                Decimal("1000.000"),
            ),
        ),
    )

    with pytest.raises(ValidationError, match="exact trusted assembled object"):
        coordinator.authorize("forged-replace", request, forged)


def test_coordinator_rejects_same_object_mutated_after_trusted_assembly(tmp_path, now):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    object.__setattr__(snapshot, "transport_state", TransportState.DISCONNECTED)

    with pytest.raises(ValidationError, match="changed after trusted assembly"):
        coordinator.authorize("mutated-registered-object", request, snapshot)


def test_startup_rejects_journal_bound_to_another_account_scope(tmp_path, now):
    journal = SafetyJournal(tmp_path / "bound-journal.db", clock=lambda: now)
    journal.initialize(
        execution_domain_scope=PAPER_EXECUTION_DOMAIN_SCOPE,
        account_scope=RUNTIME_ACCOUNT_SCOPE,
    )
    another_scope = "acct_v1_" + ("fedcba9876543210" * 4)
    coordinator = SafetyRuntimeCoordinator(
        PaperExecutionIdentity(PAPER_EXECUTION_DOMAIN_SCOPE, another_scope),
        journal,
        clock=lambda: now,
    )

    with pytest.raises(JournalError, match="identity"):
        coordinator.start()

    assert not coordinator.started


def test_authorization_rejects_journal_inode_replaced_after_startup(tmp_path, now):
    coordinator, journal, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    replacement_path = tmp_path / "replacement-safety.db"
    replacement = SafetyJournal(replacement_path, clock=lambda: now)
    replacement.initialize(
        execution_domain_scope=PAPER_EXECUTION_DOMAIN_SCOPE,
        account_scope=RUNTIME_ACCOUNT_SCOPE,
    )
    os.replace(replacement_path, journal.database_path)

    with pytest.raises(JournalIntegrityError, match="inode replayed at startup"):
        coordinator.authorize("replaced-runtime-journal", request, snapshot)


def test_exact_decimal_cross_portfolio_aggregation_and_conid_are_journaled(tmp_path, now):
    coordinator, journal, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    with localcontext() as context:
        context.prec = 3
        authorization = coordinator.authorize("runtime-key", request, snapshot)

    assert authorization.decision.allowed
    assert authorization.decision.current_quantity == Decimal("10.500")
    assert authorization.decision.computed_target_quantity == Decimal("8.375")
    assert authorization.claim.con_id == 265598
    assert authorization.contract == request.contract
    assert authorization.contract.primary_exchange == "NASDAQ"
    assert authorization.evidence_snapshot_id == snapshot.snapshot_id
    assert authorization.allocation_snapshot_id == snapshot.allocation_snapshot_id
    assert authorization.contract_snapshot_id == request.contract.snapshot_id
    assert authorization.reconciliation_snapshot_id == snapshot.reconciliation_snapshot_id
    decision_event = journal.replay().events[0]
    payload = json.loads(decision_event.payload_json)
    context = payload["authorization_context"]
    assert context["exposure"]["position_quantity"] == "10.5"
    assert context["allocation"]["position_quantity"] == "5.5"
    assert context["allocation"]["aggregate_allocated_quantity"] == "10.5"
    assert context["allocation"]["con_id"] == 265598


@pytest.mark.parametrize("bad_quantity", [2, 2.0, "2", Decimal("NaN")])
def test_runtime_boundary_rejects_inexact_quantity(tmp_path, now, bad_quantity):
    _, _, request, _ = make_runtime_case(tmp_path, now)
    with pytest.raises(ValidationError, match="exact Decimal"):
        replace(request, quantity=bad_quantity)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("security_type", "OPT"),
        ("currency", "EUR"),
        ("exchange", "NYSE"),
        ("local_symbol", "aapl"),
        ("local_symbol", "MSFT"),
        ("primary_exchange", ""),
        ("trading_class", ""),
    ],
)
def test_contract_lineage_requires_qualified_smart_us_stock_identity(
    tmp_path, now, field_name, bad_value
):
    _, _, request, _ = make_runtime_case(tmp_path, now)
    with pytest.raises(ValidationError):
        replace(request.contract, **{field_name: bad_value})


def test_contract_transport_generation_must_match_coherent_snapshot(tmp_path, now):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    request = replace(
        request,
        contract=replace(request.contract, transport_generation="another-generation"),
    )
    with pytest.raises(RuntimeAuthorizationBlocked) as exc_info:
        coordinator.authorize("lineage-generation", request, snapshot)
    assert "HARD_GATE:TRANSPORT_GENERATION_LINEAGE_MISMATCH" in exc_info.value.reason_codes


def test_authoritative_contract_conid_cannot_be_substituted_by_symbol(tmp_path, now):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    snapshot = replace_trusted_snapshot(
        snapshot,
        account_positions=(AccountPosition(999, "AAPL", Decimal("10.500")),),
    )
    with pytest.raises(RuntimeAuthorizationBlocked) as exc_info:
        coordinator.authorize("wrong-conid", request, snapshot)
    assert "HARD_GATE:ACCOUNT_POSITION_MISSING" in exc_info.value.reason_codes


def test_same_symbol_under_another_conid_hard_blocks_symbol_only_allocation(
    tmp_path,
    now,
):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    snapshot = replace_trusted_snapshot(
        snapshot,
        account_positions=(
            *snapshot.account_positions,
            AccountPosition(999, "AAPL", Decimal("1000")),
        ),
    )

    with pytest.raises(RuntimeAuthorizationBlocked) as exc_info:
        coordinator.authorize("ambiguous-symbol-conid", request, snapshot)

    assert "HARD_GATE:AUTHORITATIVE_SYMBOL_CON_ID_AMBIGUOUS" in exc_info.value.reason_codes


@pytest.mark.parametrize(
    ("field_name", "reason"),
    [
        ("positions_complete", "HARD_GATE:ACCOUNT_POSITION_SNAPSHOT_INCOMPLETE"),
        ("allocations_complete", "HARD_GATE:ALLOCATION_SNAPSHOT_INCOMPLETE"),
        ("contracts_complete", "HARD_GATE:CONTRACT_SNAPSHOT_INCOMPLETE"),
    ],
)
def test_incomplete_coherent_snapshot_hard_blocks(tmp_path, now, field_name, reason):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    with pytest.raises(RuntimeAuthorizationBlocked) as exc_info:
        coordinator.authorize(
            f"incomplete-{field_name}",
            request,
            replace_trusted_snapshot(snapshot, **{field_name: False}),
        )
    assert reason in exc_info.value.reason_codes


@pytest.mark.parametrize(
    ("mutate_request", "mutate_snapshot", "reason"),
    [
        (
            lambda request, now: replace(
                request,
                contract=replace(
                    request.contract,
                    observed_at=now - timedelta(seconds=31),
                    retrieval_timestamp=now - timedelta(seconds=31),
                ),
            ),
            lambda snapshot, now: snapshot,
            "HARD_GATE:STALE_CONTRACT_EVIDENCE",
        ),
        (
            lambda request, now: request,
            lambda snapshot, now: replace_trusted_snapshot(
                snapshot,
                observed_at=now - timedelta(seconds=31),
            ),
            "STALE_EXPOSURE_EVIDENCE",
        ),
        (
            lambda request, now: request,
            lambda snapshot, now: replace_trusted_snapshot(
                snapshot,
                open_orders=replace(
                    snapshot.open_orders,
                    observed_at=now - timedelta(seconds=31),
                ),
            ),
            "STALE_OPEN_ORDER_EVIDENCE",
        ),
    ],
)
def test_stale_contract_position_and_order_evidence_hard_block(
    tmp_path, now, mutate_request, mutate_snapshot, reason
):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    request = mutate_request(request, now)
    snapshot = mutate_snapshot(snapshot, now)
    with pytest.raises(RuntimeAuthorizationBlocked) as exc_info:
        coordinator.authorize(f"stale-{reason}", request, snapshot)
    assert reason in exc_info.value.reason_codes


@pytest.mark.parametrize(
    ("open_orders", "reason"),
    [
        ({"active_con_ids": (265598,)}, "ACTIVE_BROKER_ORDER_EXISTS"),
        ({"unknown_order_count": 1}, "HARD_GATE:UNKNOWN_BROKER_ORDER_EXISTS"),
        ({"complete": False}, "OPEN_ORDER_EVIDENCE_INCOMPLETE"),
        ({"all_clients": False}, "OPEN_ORDER_EVIDENCE_NOT_ALL_CLIENTS"),
        ({"stable": False}, "OPEN_ORDER_SNAPSHOT_UNSTABLE"),
    ],
)
def test_open_and_unknown_order_states_hard_block(tmp_path, now, open_orders, reason):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    snapshot = replace_trusted_snapshot(
        snapshot,
        open_orders=replace(snapshot.open_orders, **open_orders),
    )
    with pytest.raises(RuntimeAuthorizationBlocked) as exc_info:
        coordinator.authorize(f"orders-{reason}", request, snapshot)
    assert reason in exc_info.value.reason_codes


def test_active_order_for_another_contract_still_hard_blocks_account(tmp_path, now):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    snapshot = replace_trusted_snapshot(
        snapshot,
        open_orders=replace(snapshot.open_orders, active_con_ids=(999999,)),
    )
    with pytest.raises(RuntimeAuthorizationBlocked) as exc_info:
        coordinator.authorize("other-contract-order", request, snapshot)
    assert "ACTIVE_BROKER_ORDER_EXISTS" in exc_info.value.reason_codes


@pytest.mark.parametrize(
    ("field_name", "value", "reason"),
    [
        (
            "transport_state",
            TransportState.DISCONNECTED,
            "TRANSPORT_NOT_CERTAINLY_CONNECTED",
        ),
        (
            "transport_state",
            TransportState.AMBIGUOUS,
            "TRANSPORT_NOT_CERTAINLY_CONNECTED",
        ),
        (
            "reconciliation_status",
            ReconciliationStatus.FAILED,
            "RECONCILIATION_NOT_PASSED",
        ),
        (
            "reconciliation_status",
            ReconciliationStatus.UNKNOWN,
            "RECONCILIATION_NOT_PASSED",
        ),
    ],
)
def test_transport_and_reconciliation_uncertainty_hard_block(
    tmp_path, now, field_name, value, reason
):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    with pytest.raises(RuntimeAuthorizationBlocked) as exc_info:
        coordinator.authorize(
            f"gate-{field_name}-{value.value}",
            request,
            replace_trusted_snapshot(snapshot, **{field_name: value}),
        )
    assert reason in exc_info.value.reason_codes


def test_offsetting_allocations_are_detected_before_net_aggregation(tmp_path, now):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    snapshot = replace_trusted_snapshot(
        snapshot,
        account_positions=(AccountPosition(265598, "AAPL", Decimal("3")),),
        portfolio_allocations=(
            PortfolioAllocation("portfolio-a", 265598, "AAPL", Decimal("5")),
            PortfolioAllocation("portfolio-b", 265598, "AAPL", Decimal("-2")),
        ),
    )
    with pytest.raises(RuntimeAuthorizationBlocked) as exc_info:
        coordinator.authorize("offsetting", request, snapshot)
    assert "OFFSETTING_ALLOCATIONS_EXIST" in exc_info.value.reason_codes


def test_aggregate_allocation_must_equal_authoritative_account_position(tmp_path, now):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    snapshot = replace_trusted_snapshot(
        snapshot,
        portfolio_allocations=(
            PortfolioAllocation("portfolio-a", 265598, "AAPL", Decimal("5.500")),
            PortfolioAllocation("portfolio-b", 265598, "AAPL", Decimal("4.999")),
        ),
    )
    with pytest.raises(RuntimeAuthorizationBlocked) as exc_info:
        coordinator.authorize("aggregate-mismatch", request, snapshot)
    assert "AGGREGATE_ALLOCATION_MISMATCH" in exc_info.value.reason_codes


def test_fake_submit_consumes_permit_once_and_leaves_unknown_quarantine(tmp_path, now):
    coordinator, journal, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    authorization = coordinator.authorize("fake-submit", request, snapshot)
    submitter = FakeOrderSubmitter()

    receipt = coordinator.submit_fake(authorization, submitter)

    assert receipt.status == "SIMULATED_ACCEPTED"
    assert receipt.descriptor_fingerprint == authorization.descriptor_fingerprint
    assert len(submitter.descriptors) == 1
    state = journal.replay()
    assert state.quarantined_reservations[0].outcome_unknown
    with pytest.raises(Exception, match="already|issued"):
        coordinator.submit_fake(authorization, submitter)


def test_nonfake_sink_is_rejected_before_permit_consumption(tmp_path, now):
    coordinator, journal, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    authorization = coordinator.authorize("nonfake", request, snapshot)

    with pytest.raises(FakeSubmissionOnlyError):
        coordinator.submit_fake(authorization, object())

    state = journal.replay()
    assert state.active_reservations[0].outcome_unknown is False
    coordinator.submit_fake(authorization, FakeOrderSubmitter())


def test_expired_authorization_cannot_dispatch_and_blocks_restart(tmp_path, now):
    coordinator, journal, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    authorization = coordinator.authorize("expired-before-submit", request, snapshot)
    coordinator._clock = lambda: now + timedelta(seconds=31)

    with pytest.raises(RuntimeSafetyError, match="expired before dispatch"):
        coordinator.submit_fake(authorization, FakeOrderSubmitter())

    assert authorization._permit.consumed
    state = journal.replay()
    assert len(state.active_reservations) == 1
    assert state.active_reservations[0].outcome_unknown is False
    with pytest.raises(Exception, match="already|issued"):
        coordinator.submit_fake(authorization, FakeOrderSubmitter())


def test_crash_before_permit_consumption_blocks_restart(tmp_path, now):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    coordinator.authorize("before-consume-crash", request, snapshot)

    restarted = SafetyRuntimeCoordinator(
        PaperExecutionIdentity(
            PAPER_EXECUTION_DOMAIN_SCOPE,
            RUNTIME_ACCOUNT_SCOPE,
        ),
        SafetyJournal(tmp_path / "runtime-safety.db", clock=lambda: now),
        clock=lambda: now,
    )
    with pytest.raises(RuntimeStartupBlocked) as exc_info:
        restarted.start()
    assert "ACTIVE_RESERVATION_AT_STARTUP" in exc_info.value.reason_codes


def test_failure_after_permit_consumption_blocks_restart_as_quarantined(tmp_path, now):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    authorization = coordinator.authorize("after-consume-failure", request, snapshot)

    with pytest.raises(RuntimeError, match="injected"):
        coordinator.submit_fake(authorization, FakeOrderSubmitter(fail=True))

    restarted = SafetyRuntimeCoordinator(
        PaperExecutionIdentity(
            PAPER_EXECUTION_DOMAIN_SCOPE,
            RUNTIME_ACCOUNT_SCOPE,
        ),
        SafetyJournal(tmp_path / "runtime-safety.db", clock=lambda: now),
        clock=lambda: now,
    )
    with pytest.raises(RuntimeStartupBlocked) as exc_info:
        restarted.start()
    assert exc_info.value.reason_codes == (
        "ACTIVE_RESERVATION_AT_STARTUP",
        "QUARANTINED_RESERVATION_AT_STARTUP",
    )


def test_exact_idempotent_replay_never_renews_submission_authority(tmp_path, now):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    coordinator.authorize("same-key", request, snapshot)

    with pytest.raises(RuntimeAuthorizationBlocked) as exc_info:
        coordinator.authorize("same-key", request, snapshot)
    assert exc_info.value.reason_codes == ("REPLAY_HAS_NO_SUBMISSION_AUTHORITY",)


def test_concurrent_authorizations_yield_exactly_one_permit(tmp_path, now):
    coordinator, journal, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()

    def authorize(index):
        try:
            coordinator.authorize(f"concurrent-{index}", request, snapshot)
            return "allowed"
        except (ReservationConflict, RuntimeAuthorizationBlocked):
            return "blocked"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(authorize, range(8)))

    assert outcomes.count("allowed") == 1
    assert outcomes.count("blocked") == 7
    assert len(journal.replay().active_reservations) == 1


def test_start_and_authorize_lifecycle_transitions_are_serialized(tmp_path, now, monkeypatch):
    coordinator, _, request, snapshot = make_runtime_case(tmp_path, now)
    coordinator.start()
    entered_authorization = threading.Event()
    release_authorization = threading.Event()
    restart_returned = threading.Event()
    original = coordinator._build_boundary_models

    def blocked_build(*args):
        entered_authorization.set()
        assert release_authorization.wait(timeout=2)
        return original(*args)

    monkeypatch.setattr(coordinator, "_build_boundary_models", blocked_build)
    authorization_thread = threading.Thread(
        target=coordinator.authorize,
        args=("lifecycle-race", request, snapshot),
    )
    authorization_thread.start()
    assert entered_authorization.wait(timeout=2)

    restart_thread = threading.Thread(target=lambda: (coordinator.start(), restart_returned.set()))
    restart_thread.start()
    assert not restart_returned.wait(timeout=0.05)

    release_authorization.set()
    authorization_thread.join(timeout=2)
    restart_thread.join(timeout=2)
    assert not authorization_thread.is_alive()
    assert not restart_thread.is_alive()
    assert restart_returned.is_set()
