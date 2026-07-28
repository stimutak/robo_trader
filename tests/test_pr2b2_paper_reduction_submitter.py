from __future__ import annotations

import ast
import copy
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

import robo_trader.paper_reduction_submitter as submitter_module
from robo_trader.execution import ExecutionResult, Order, PaperExecutor
from robo_trader.paper_execution_capability import (
    PaperExecutionCapabilityError,
    _issue_gateway_reduction_terminal_dispatch,
)
from robo_trader.paper_reduction_submitter import (
    LocalPaperOrderStatus,
    LocalPaperOutcomeProvenance,
    LocalPaperTerminalOutcome,
    PaperReductionSubmissionError,
    PaperReductionSubmitter,
    _bind_paper_reduction_submitter,
    _exact_limit_price,
    _terminal_outcome,
)
from robo_trader.safety.journal import SafetyJournal
from robo_trader.safety.models import (
    EvidenceStatus,
    OrderSide,
    OrderType,
    ReconciliationStatus,
    TimeInForce,
    TransportState,
)
from robo_trader.safety.runtime import (
    AccountPosition,
    AuthoritativeContract,
    CoherentSafetySnapshot,
    ConsumedPaperSubmissionEnvelope,
    FinalSubmissionEvidenceProof,
    OpenOrderSnapshot,
    PaperExecutionIdentity,
    PortfolioAllocation,
    RuntimeOrderRequest,
    RuntimeSafetyError,
    SafetyRuntimeCoordinator,
    _assemble_coherent_safety_snapshot,
)
from tests.paper_execution_test_support import bind_gateway_reduction_authority

ACCOUNT_SCOPE = "acct_v1_" + hashlib.sha256(b"paper-submit-account").hexdigest()
OTHER_ACCOUNT_SCOPE = "acct_v1_" + hashlib.sha256(b"other-account").hexdigest()
NOW = datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc)


@dataclass
class RuntimeCase:
    coordinator: SafetyRuntimeCoordinator
    executor: PaperExecutor
    submitter: PaperReductionSubmitter
    request: RuntimeOrderRequest
    contract: AuthoritativeContract
    snapshot: CoherentSafetySnapshot
    clock: list[datetime]


def _contract(
    *,
    observed_at: datetime = NOW,
    generation: str = "generation-1",
) -> AuthoritativeContract:
    return AuthoritativeContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        trading_class="NMS",
        observed_at=observed_at,
        snapshot_id=f"contract-{generation}",
        source="qualified-contract-cache",
        broker_timestamp=observed_at,
        retrieval_timestamp=observed_at,
        transport_generation=generation,
        status=EvidenceStatus.AUTHORITATIVE,
    )


def _snapshot(
    *,
    observed_at: datetime = NOW,
    account_scope: str = ACCOUNT_SCOPE,
    generation: str = "generation-1",
    account_quantity: Decimal = Decimal("10"),
    portfolio_quantity: Decimal = Decimal("10"),
    database_path: str = "/trusted/trading-data.db",
    database_identity: str = "ledger-identity-1",
    database_device: int = 101,
    database_inode: int = 202,
    runtime_fingerprint: str = "0123456789abcdef",
    ibc_proof_id: str = "ibc-proof-v1-" + ("a" * 64),
) -> CoherentSafetySnapshot:
    return _assemble_coherent_safety_snapshot(
        execution_domain_scope="paper-simulator-v1",
        account_scope=account_scope,
        observed_at=observed_at,
        snapshot_id=f"account-{generation}",
        source="broker-account-snapshot",
        allocation_observed_at=observed_at,
        allocation_snapshot_id=f"allocation-{generation}",
        allocation_source="allocation-database",
        allocation_database_path=database_path,
        allocation_database_identity=database_identity,
        allocation_database_device=database_device,
        allocation_database_inode=database_inode,
        runtime_fingerprint=runtime_fingerprint,
        ibc_proof_id=ibc_proof_id,
        reconciliation_observed_at=observed_at,
        reconciliation_snapshot_id=f"reconciliation-{generation}",
        transport_generation=generation,
        account_positions=(AccountPosition(265598, "AAPL", account_quantity),),
        portfolio_allocations=(
            PortfolioAllocation(
                "portfolio-a",
                265598,
                "AAPL",
                portfolio_quantity,
            ),
        ),
        open_orders=OpenOrderSnapshot(
            observed_at=observed_at,
            snapshot_id=f"open-orders-{generation}",
            transport_generation=generation,
        ),
        transport_state=TransportState.CONNECTED,
        reconciliation_status=ReconciliationStatus.PASSED,
    )


def _trusted_replace(
    snapshot: CoherentSafetySnapshot,
    **changes: object,
) -> CoherentSafetySnapshot:
    values = {
        definition.name: getattr(snapshot, definition.name)
        for definition in fields(snapshot)
        if definition.name != "_assembly_marker"
    }
    values.update(changes)
    return _assemble_coherent_safety_snapshot(**values)


def _make_case(
    tmp_path: Path,
    *,
    side: OrderSide = OrderSide.SELL,
    quantity: Decimal = Decimal("3"),
    order_type: OrderType = OrderType.LIMIT,
    limit_price: Decimal | None = Decimal("187.25"),
    account_scope: str = ACCOUNT_SCOPE,
    key_suffix: str = "one",
) -> RuntimeCase:
    clock = [NOW]
    identity = PaperExecutionIdentity("paper-simulator-v1", account_scope)
    journal = SafetyJournal(
        tmp_path / f"safety-{key_suffix}.db",
        clock=lambda: clock[0],
    )
    journal.initialize(
        execution_domain_scope=identity.execution_domain_scope,
        account_scope=identity.account_scope,
    )
    coordinator = SafetyRuntimeCoordinator(identity, journal, clock=lambda: clock[0])
    coordinator.start()
    contract = _contract()
    request = RuntimeOrderRequest(
        portfolio_id="portfolio-a",
        contract=contract,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        time_in_force=TimeInForce.DAY,
        outside_regular_hours=False,
        order_ref=f"paper-submit-{key_suffix}",
        reason="protective reduction",
        strategy="stop-loss",
    )
    starting_quantity = Decimal("-10") if side is OrderSide.BUY_TO_COVER else Decimal("10")
    snapshot = _snapshot(
        account_scope=account_scope,
        account_quantity=starting_quantity,
        portfolio_quantity=starting_quantity,
    )
    executor = PaperExecutor()
    authority, _ = bind_gateway_reduction_authority(
        executor,
        "portfolio-a",
        coordinator=coordinator,
    )
    submitter = _bind_paper_reduction_submitter(
        executor,
        coordinator,
        authority,
        "portfolio-a",
    )
    return RuntimeCase(
        coordinator,
        executor,
        submitter,
        request,
        contract,
        snapshot,
        clock,
    )


def _authorization(case: RuntimeCase, *, suffix: str = "auth"):
    return case.coordinator.authorize(
        f"paper-submit-{suffix}",
        case.request,
        case.snapshot,
    )


def _envelope(
    case: RuntimeCase,
    *,
    final_contract: AuthoritativeContract | None = None,
    final_snapshot: CoherentSafetySnapshot | None = None,
):
    authorization = _authorization(case)
    proof = case.coordinator._bind_fresh_final_evidence_proof(
        authorization,
        final_contract or case.contract,
        final_snapshot or case.snapshot,
    )
    return case.coordinator._consume_authorization_for_paper_submission(
        authorization,
        proof,
    )


def _submit(case: RuntimeCase, envelope):
    pre_position = Decimal("-10") if case.request.side is OrderSide.BUY_TO_COVER else Decimal("10")
    return case.submitter._submit_once(
        envelope,
        pre_position_quantity=pre_position,
    )


def test_binding_is_private_sealed_and_exact(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    with pytest.raises(PaperReductionSubmissionError, match="private bind"):
        PaperReductionSubmitter(case.executor, case.coordinator, object(), "portfolio-a")

    class PaperExecutorSubclass(PaperExecutor):
        pass

    with pytest.raises(PaperReductionSubmissionError, match="exactly PaperExecutor"):
        _bind_paper_reduction_submitter(
            PaperExecutorSubclass(),
            case.coordinator,
            object(),
            "portfolio-a",
        )
    with pytest.raises(PaperReductionSubmissionError, match="exactly SafetyRuntime"):
        _bind_paper_reduction_submitter(
            case.executor,
            object(),
            object(),
            "portfolio-a",
        )  # type: ignore[arg-type]

    assert not hasattr(case.submitter, "executor")
    with pytest.raises(AttributeError, match="sealed"):
        case.submitter.executor = case.executor  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("side", "order_type", "limit_price", "expected_price"),
    [
        (OrderSide.SELL, OrderType.MARKET, None, None),
        (OrderSide.BUY_TO_COVER, OrderType.MARKET, None, None),
        (OrderSide.SELL, OrderType.LIMIT, Decimal("123.45"), Decimal("123.45")),
        (
            OrderSide.BUY_TO_COVER,
            OrderType.LIMIT,
            Decimal("187.25"),
            Decimal("187.25"),
        ),
    ],
)
def test_exact_mapping_and_one_execution_call(
    tmp_path: Path,
    side: OrderSide,
    order_type: OrderType,
    limit_price: Decimal | None,
    expected_price: Decimal | None,
) -> None:
    case = _make_case(
        tmp_path,
        side=side,
        order_type=order_type,
        limit_price=limit_price,
    )
    if expected_price is None:
        case.executor._execution_cache["AAPL"] = 187.25
        case.executor._execution_cache_ts["AAPL"] = datetime.utcnow()
    outcome = _submit(case, _envelope(case))
    assert type(outcome) is LocalPaperTerminalOutcome
    assert outcome.order_ref == "paper-submit-one"
    assert outcome.status is LocalPaperOrderStatus.FILLED
    assert outcome.requested_quantity == Decimal("3")
    assert outcome.filled_quantity == Decimal("3")
    assert outcome.remaining_quantity == Decimal("0")
    assert outcome.exact_fill_price == (expected_price or Decimal("187.25"))
    assert outcome.provenance is LocalPaperOutcomeProvenance.LOCAL_PAPER_EXECUTOR
    assert outcome.terminal is True
    assert len(case.executor.fills) == 1
    filled_order = next(iter(case.executor.fills.values()))[1]
    assert filled_order == Order(
        symbol="AAPL",
        quantity=3,
        side=side.value,
        price=expected_price,
        order_ref="paper-submit-one",
    )


@pytest.mark.parametrize("outcome", ["success", "rejection", "exception"])
def test_replay_rejects_after_every_execution_outcome(
    tmp_path: Path,
    outcome: str,
) -> None:
    case = _make_case(
        tmp_path,
        order_type=OrderType.MARKET if outcome == "rejection" else OrderType.LIMIT,
        limit_price=None if outcome == "rejection" else Decimal("187.25"),
    )
    envelope = _envelope(case)

    class RaisingFills(dict):
        attempts = 0

        def __setitem__(self, _key, _value):
            self.attempts += 1
            raise LookupError("injected paper failure")

    if outcome == "exception":
        case.executor.fills = RaisingFills()
    if outcome == "exception":
        with pytest.raises(LookupError, match="injected"):
            _submit(case, envelope)
    else:
        _submit(case, envelope)
    with pytest.raises(PaperReductionSubmissionError, match="claim failed"):
        _submit(case, envelope)
    if outcome == "success":
        assert len(case.executor.fills) == 1
    elif outcome == "rejection":
        assert case.executor.fills == {}
    else:
        assert case.executor.fills.attempts == 1


def test_concurrent_envelope_claim_allows_exactly_one_call(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    envelope = _envelope(case)
    barrier = threading.Barrier(2)

    def submit() -> object:
        barrier.wait()
        try:
            return _submit(case, envelope)
        except PaperReductionSubmissionError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))

    assert sum(type(result) is LocalPaperTerminalOutcome for result in results) == 1
    assert sum(type(result) is PaperReductionSubmissionError for result in results) == 1
    assert len(case.executor.fills) == 1


def test_proof_and_envelope_copies_forgery_and_replay_reject(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    authorization = _authorization(case)
    proof = case.coordinator._bind_fresh_final_evidence_proof(
        authorization,
        case.contract,
        case.snapshot,
    )
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(proof)
    proof_clone = replace(proof)
    with pytest.raises(RuntimeSafetyError, match="forged.*replayed"):
        case.coordinator._consume_authorization_for_paper_submission(
            authorization,
            proof_clone,
        )

    envelope = case.coordinator._consume_authorization_for_paper_submission(
        authorization,
        proof,
    )
    with pytest.raises(RuntimeSafetyError, match="invalidated"):
        case.coordinator._consume_authorization_for_paper_submission(
            authorization,
            proof,
        )
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(envelope)
    envelope_clone = replace(envelope)
    with pytest.raises(PaperReductionSubmissionError, match="claim failed"):
        _submit(case, envelope_clone)

    values = {
        definition.name: getattr(envelope, definition.name) for definition in fields(envelope)
    }
    values["_producer_marker"] = object()
    with pytest.raises(RuntimeSafetyError, match="requires coordinator"):
        ConsumedPaperSubmissionEnvelope(**values)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("execution_domain_scope", "paper-other"),
        ("account_scope", OTHER_ACCOUNT_SCOPE),
        ("con_id", 999),
        ("descriptor_fingerprint", "0" * 64),
        ("final_evidence_fingerprint", "1" * 64),
    ],
)
def test_tampered_final_proof_rejects_before_permit_consumption(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    case = _make_case(tmp_path)
    authorization = _authorization(case)
    proof = case.coordinator._bind_fresh_final_evidence_proof(
        authorization,
        case.contract,
        case.snapshot,
    )
    object.__setattr__(proof, field_name, value)
    with pytest.raises(RuntimeSafetyError, match="does not match|forged"):
        case.coordinator._consume_authorization_for_paper_submission(
            authorization,
            proof,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("execution_domain_scope", "paper-other"),
        ("account_scope", OTHER_ACCOUNT_SCOPE),
        ("con_id", 999),
        ("descriptor_fingerprint", "0" * 64),
        ("final_evidence_fingerprint", "1" * 64),
    ],
)
def test_tampered_envelope_scope_identity_and_fingerprint_reject_before_fill(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    case = _make_case(tmp_path)
    envelope = _envelope(case)
    object.__setattr__(envelope, field_name, value)
    with pytest.raises(PaperReductionSubmissionError, match="claim failed"):
        _submit(case, envelope)
    assert case.executor.fills == {}


def test_cross_account_coordinator_cannot_claim_envelope(tmp_path: Path) -> None:
    first = _make_case(tmp_path, key_suffix="first")
    second = _make_case(
        tmp_path,
        account_scope=OTHER_ACCOUNT_SCOPE,
        key_suffix="second",
    )
    foreign_submitter = _bind_paper_reduction_submitter(
        second.executor,
        second.coordinator,
        bind_gateway_reduction_authority(
            second.executor,
            "portfolio-a",
            coordinator=second.coordinator,
        )[0],
        "portfolio-a",
    )
    with pytest.raises(PaperReductionSubmissionError, match="claim failed"):
        foreign_submitter._submit_once(
            _envelope(first),
            pre_position_quantity=Decimal("10"),
        )


def test_final_smaller_exposure_rejects_and_invalidates_authorization(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, quantity=Decimal("10"))
    authorization = _authorization(case)
    smaller = _snapshot(
        account_quantity=Decimal("5"),
        portfolio_quantity=Decimal("5"),
    )
    with pytest.raises(RuntimeSafetyError, match="no longer authorizes"):
        case.coordinator._bind_fresh_final_evidence_proof(
            authorization,
            case.contract,
            smaller,
        )
    with pytest.raises(RuntimeSafetyError, match="invalidated"):
        case.coordinator._bind_fresh_final_evidence_proof(
            authorization,
            case.contract,
            case.snapshot,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("portfolio_id", "portfolio-b"),
        ("side", OrderSide.BUY_TO_COVER),
        ("quantity", Decimal("2")),
        ("order_type", OrderType.MARKET),
        ("limit_price", Decimal("188")),
        ("outside_regular_hours", True),
    ],
)
def test_authorized_request_drift_invalidates_before_final_proof(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    case = _make_case(tmp_path)
    authorization = _authorization(case)
    object.__setattr__(authorization._request, field_name, value)
    with pytest.raises(RuntimeSafetyError, match="changed"):
        case.coordinator._bind_fresh_final_evidence_proof(
            authorization,
            case.contract,
            case.snapshot,
        )
    with pytest.raises(RuntimeSafetyError, match="invalidated"):
        case.coordinator._bind_fresh_final_evidence_proof(
            authorization,
            case.contract,
            case.snapshot,
        )


def test_transport_generation_change_rejects(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    authorization = _authorization(case)
    with pytest.raises(RuntimeSafetyError, match="transport lineage"):
        case.coordinator._bind_fresh_final_evidence_proof(
            authorization,
            _contract(generation="generation-2"),
            _snapshot(generation="generation-2"),
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("allocation_database_path", "/other/trading-data.db"),
        ("allocation_database_identity", "ledger-identity-2"),
        ("allocation_database_device", 303),
        ("allocation_database_inode", 404),
        ("runtime_fingerprint", "fedcba9876543210"),
        ("ibc_proof_id", "ibc-proof-v1-" + ("b" * 64)),
    ],
)
def test_final_ledger_and_runtime_provenance_change_rejects(
    tmp_path: Path,
    field_name: str,
    value: object,
) -> None:
    case = _make_case(tmp_path)
    authorization = _authorization(case)
    final_snapshot = _trusted_replace(case.snapshot, **{field_name: value})
    with pytest.raises(RuntimeSafetyError, match="provenance changed"):
        case.coordinator._bind_fresh_final_evidence_proof(
            authorization,
            case.contract,
            final_snapshot,
        )


def test_expiry_invalidates_before_proof_or_submission(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    authorization = _authorization(case)
    case.clock[0] += timedelta(seconds=121)
    with pytest.raises(RuntimeSafetyError, match="expired"):
        case.coordinator._bind_fresh_final_evidence_proof(
            authorization,
            case.contract,
            case.snapshot,
        )


def test_final_proof_expiry_invalidates_before_permit_consumption(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    authorization = _authorization(case)
    proof = case.coordinator._bind_fresh_final_evidence_proof(
        authorization,
        case.contract,
        case.snapshot,
    )
    case.clock[0] += timedelta(seconds=121)
    with pytest.raises(RuntimeSafetyError, match="expired"):
        case.coordinator._consume_authorization_for_paper_submission(
            authorization,
            proof,
        )


def test_narrow_explicit_invalidation_rejects_copies_foreign_and_replay(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, key_suffix="owner")
    authorization = _authorization(case)
    clone = replace(authorization)
    with pytest.raises(RuntimeSafetyError, match="forged"):
        case.coordinator._invalidate_unsubmitted_authorization(clone)

    foreign = _make_case(
        tmp_path,
        account_scope=OTHER_ACCOUNT_SCOPE,
        key_suffix="foreign",
    )
    with pytest.raises(RuntimeSafetyError, match="foreign"):
        foreign.coordinator._invalidate_unsubmitted_authorization(authorization)

    case.coordinator._invalidate_unsubmitted_authorization(authorization)
    with pytest.raises(RuntimeSafetyError, match="invalidated"):
        case.coordinator._invalidate_unsubmitted_authorization(authorization)


@pytest.mark.parametrize("slippage", [float("nan"), float("inf"), -0.01, 10_000.0])
def test_nonfinite_or_out_of_bounds_slippage_rejects_before_execution(
    tmp_path: Path,
    slippage: float,
) -> None:
    case = _make_case(tmp_path)
    envelope = _envelope(case)
    case.executor.slippage_bps = slippage
    with pytest.raises(PaperReductionSubmissionError, match="slippage"):
        _submit(case, envelope)
    with pytest.raises(PaperReductionSubmissionError, match="claim failed"):
        _submit(case, envelope)
    assert case.executor.fills == {}


@pytest.mark.parametrize(
    "failure_point",
    ["descriptor_snapshot", "contract_snapshot", "nan_slippage", "order_map"],
)
def test_preterminal_failure_retires_traceback_retained_final_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    case = _make_case(tmp_path, key_suffix=f"retire-{failure_point}")
    envelope = _envelope(case)
    original_map = submitter_module._map_order

    if failure_point == "descriptor_snapshot":
        monkeypatch.setattr(
            submitter_module,
            "_snapshot_descriptor",
            lambda _descriptor: (_ for _ in ()).throw(RuntimeError("snapshot descriptor")),
        )
    elif failure_point == "contract_snapshot":
        monkeypatch.setattr(
            submitter_module,
            "_snapshot_contract",
            lambda _contract: (_ for _ in ()).throw(RuntimeError("snapshot contract")),
        )
    elif failure_point == "nan_slippage":
        case.executor.slippage_bps = float("nan")
    else:
        monkeypatch.setattr(
            submitter_module,
            "_map_order",
            lambda _descriptor, _contract: (_ for _ in ()).throw(RuntimeError("order map")),
        )

    with pytest.raises((PaperReductionSubmissionError, RuntimeError)) as raised:
        _submit(case, envelope)

    retained: dict[str, object] = {}
    traceback = raised.value.__traceback__
    while traceback is not None:
        for name in ("final_allocation", "descriptor", "contract"):
            if name in traceback.tb_frame.f_locals:
                retained[name] = traceback.tb_frame.f_locals[name]
        traceback = traceback.tb_next
    assert set(retained) == {"final_allocation", "descriptor", "contract"}

    order = original_map(retained["descriptor"], retained["contract"])
    authority = getattr(case.submitter, "_PaperReductionSubmitter__authority")
    with pytest.raises(PaperExecutionCapabilityError, match="final allocation"):
        _issue_gateway_reduction_terminal_dispatch(
            authority,
            submitter=case.submitter,
            executor=case.executor,
            coordinator=case.coordinator,
            final_allocation=retained["final_allocation"],
            descriptor=retained["descriptor"],
            contract=retained["contract"],
            order=order,
            pre_position_quantity=Decimal("10"),
        )

    assert case.executor.fills == {}


@pytest.mark.parametrize(
    "bad_result",
    [
        object(),
        ExecutionResult(1, "accepted", 187.25),
        ExecutionResult(True, "accepted", None),
        ExecutionResult(True, "accepted", float("nan")),
        ExecutionResult(True, "accepted", float("inf")),
        ExecutionResult(True, "accepted", 0.0),
        ExecutionResult(False, "rejected", 187.25),
        ExecutionResult(False, "", None),
    ],
)
def test_malformed_execution_result_fails_closed(
    tmp_path: Path,
    bad_result: object,
) -> None:
    case = _make_case(tmp_path)
    with pytest.raises(PaperReductionSubmissionError):
        _terminal_outcome(bad_result, case.request)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("price", "message"),
    [
        (Decimal("1.00001"), "supported tick"),
        (Decimal("1000000.25"), "exceeds"),
    ],
)
def test_unsafe_decimal_price_rejects(
    tmp_path: Path,
    price: Decimal,
    message: str,
) -> None:
    case = _make_case(tmp_path, limit_price=price)
    with pytest.raises(PaperReductionSubmissionError, match=message):
        _submit(case, _envelope(case))


@pytest.mark.parametrize("price", [Decimal("NaN"), Decimal("Infinity")])
def test_nonfinite_decimal_price_rejects(price: Decimal) -> None:
    with pytest.raises(PaperReductionSubmissionError, match="finite positive"):
        _exact_limit_price(price)


def test_limit_price_reaches_executor_as_exact_decimal(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, limit_price=Decimal("123.45"))
    _submit(case, _envelope(case))
    assert len(case.executor.fills) == 1
    filled_order = next(iter(case.executor.fills.values()))[1]
    assert type(filled_order.price) is Decimal
    assert filled_order.price == Decimal("123.45")


def test_legacy_soft_gate_is_bypassed_only_after_consumed_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _make_case(tmp_path)

    def kill_switch(order: Order) -> ExecutionResult:
        return ExecutionResult(False, "Kill switch active")

    monkeypatch.setattr(case.executor, "validate_order", kill_switch)
    normal = case.executor.place_order(Order("AAPL", 3, "SELL", 187.25))
    assert normal.ok is False
    assert case.executor.fills == {}

    result = _submit(case, _envelope(case))
    assert result.ok is True
    assert result.fill_price == 187.25
    assert len(case.executor.fills) == 1


def test_arbitrary_finite_slippage_returns_one_normalized_exact_terminal_fill(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    case.executor.slippage_bps = 0.3333333333333333

    outcome = _submit(case, _envelope(case))

    assert outcome.ok is True
    assert outcome.exact_fill_price is not None
    assert outcome.exact_fill_price.as_tuple().exponent == -4
    assert Decimal(str(outcome.fill_price)) == outcome.exact_fill_price
    assert len(case.executor.fills) == 1


def test_rejection_returns_exact_terminal_zero_fill_metadata(
    tmp_path: Path,
) -> None:
    case = _make_case(
        tmp_path,
        order_type=OrderType.MARKET,
        limit_price=None,
    )

    outcome = _submit(case, _envelope(case))

    assert outcome.status is LocalPaperOrderStatus.REJECTED
    assert outcome.ok is False
    assert outcome.requested_quantity == Decimal("3")
    assert outcome.filled_quantity == Decimal("0")
    assert outcome.remaining_quantity == Decimal("3")
    assert outcome.exact_fill_price is None
    assert outcome.fill_price is None
    assert outcome.terminal is True


def test_exact_fill_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    mismatched = ExecutionResult(
        True,
        "accepted",
        187.25,
        exact_fill_price=Decimal("187.24"),
    )

    with pytest.raises(PaperReductionSubmissionError, match="matching exact Decimal"):
        _terminal_outcome(mismatched, case.request)  # type: ignore[arg-type]


def test_adapter_has_one_capability_call_and_no_forbidden_route() -> None:
    module_path = Path(__file__).parents[1] / "robo_trader" / "paper_reduction_submitter.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    named_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert sum(node.func.id == "_submit_gateway_reduction_once" for node in named_calls) == 1
    assert not any(node.func.attr == "_submit_reduction_once" for node in calls)
    assert not any(node.func.attr == "_place_simple_order" for node in calls)
    forbidden = {
        "place_order",
        "place_order_async",
        "validate_order",
        "_place_smart_order",
        "_execute_smart_order_async",
        "execute_order",
    }
    assert not {node.func.attr for node in calls}.intersection(forbidden)
