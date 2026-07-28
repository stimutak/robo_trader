"""Test-only construction of an exact gateway registration lifecycle."""

from __future__ import annotations

import hashlib
import math
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from robo_trader.execution import ExecutionResult, Order, PaperExecutor
from robo_trader.paper_execution_capability import (
    PaperReductionExecutionAuthority,
    _bind_gateway_reduction_execution,
    _issue_gateway_reduction_binding_capability,
)
from robo_trader.paper_reduction_gateway import (
    PaperReductionGateway,
    _PaperRuntimeBindingSession,
)
from robo_trader.paper_reduction_submitter import (
    PaperReductionSubmissionError,
    _bind_paper_reduction_submitter,
    _claim_and_issue_terminal_dispatch,
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
    OpenOrderSnapshot,
    PaperExecutionIdentity,
    PortfolioAllocation,
    RuntimeOrderRequest,
    SafetyRuntimeCoordinator,
    _assemble_coherent_safety_snapshot,
)

_ACCOUNT_SCOPE = "acct_v1_" + hashlib.sha256(b"paper-execution-test-support").hexdigest()


@dataclass
class GatewayBoundReductionHarness:
    executor: PaperExecutor
    coordinator: SafetyRuntimeCoordinator
    authority: PaperReductionExecutionAuthority
    submitter_identity: object
    portfolio_id: str
    clock: datetime
    _counter: int = 0

    def _envelope(self, order: Order, pre_position_quantity: Decimal):
        self._counter += 1
        con_id = 100_000 + self._counter
        generation = f"test-generation-{self._counter}"
        contract = AuthoritativeContract(
            con_id=con_id,
            symbol=order.symbol,
            local_symbol=order.symbol,
            security_type="STK",
            currency="USD",
            exchange="SMART",
            primary_exchange="NASDAQ",
            trading_class="NMS",
            observed_at=self.clock,
            snapshot_id=f"contract-{self._counter}",
            source="test-qualified-contract-cache",
            broker_timestamp=self.clock,
            retrieval_timestamp=self.clock,
            transport_generation=generation,
            status=EvidenceStatus.AUTHORITATIVE,
        )
        if order.side == "SELL":
            side = OrderSide.SELL
        elif order.side == "BUY_TO_COVER":
            side = OrderSide.BUY_TO_COVER
        else:
            raise PaperReductionSubmissionError("test harness admits only reductions")
        if order.order_ref is None:
            order.order_ref = f"paper-test-{self._counter}"
        if order.price is None:
            order_type = OrderType.MARKET
            limit_price = None
        else:
            if type(order.price) is Decimal:
                limit_price = order.price
            else:
                numeric = float(order.price)
                if not math.isfinite(numeric):
                    raise PaperReductionSubmissionError("paper test price is non-finite")
                limit_price = Decimal(str(numeric))
                order.price = limit_price
            order_type = OrderType.LIMIT
        request = RuntimeOrderRequest(
            portfolio_id=self.portfolio_id,
            contract=contract,
            side=side,
            quantity=Decimal(order.quantity),
            order_type=order_type,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
            outside_regular_hours=False,
            order_ref=order.order_ref,
            reason="test protective reduction",
            strategy="test-stop-loss",
        )
        snapshot = _assemble_coherent_safety_snapshot(
            execution_domain_scope="paper-simulator-v1",
            account_scope=_ACCOUNT_SCOPE,
            observed_at=self.clock,
            snapshot_id=f"account-{self._counter}",
            source="test-broker-account-snapshot",
            allocation_observed_at=self.clock,
            allocation_snapshot_id=f"allocation-{self._counter}",
            allocation_source="test-allocation-database",
            allocation_database_path="/tmp/isolated-paper-execution-test.db",
            allocation_database_identity="test-ledger-identity",
            allocation_database_device=101,
            allocation_database_inode=202,
            runtime_fingerprint="0123456789abcdef",
            ibc_proof_id="ibc-proof-v1-" + ("a" * 64),
            reconciliation_observed_at=self.clock,
            reconciliation_snapshot_id=f"reconciliation-{self._counter}",
            transport_generation=generation,
            account_positions=(AccountPosition(con_id, order.symbol, pre_position_quantity),),
            portfolio_allocations=(
                PortfolioAllocation(
                    self.portfolio_id,
                    con_id,
                    order.symbol,
                    pre_position_quantity,
                ),
            ),
            open_orders=OpenOrderSnapshot(
                observed_at=self.clock,
                snapshot_id=f"open-orders-{self._counter}",
                transport_generation=generation,
            ),
            transport_state=TransportState.CONNECTED,
            reconciliation_status=ReconciliationStatus.PASSED,
        )
        authorization = self.coordinator.authorize(f"paper-test-{self._counter}", request, snapshot)
        proof = self.coordinator._bind_fresh_final_evidence_proof(authorization, contract, snapshot)
        envelope = self.coordinator._consume_authorization_for_paper_submission(
            authorization, proof
        )
        return envelope

    def issue(
        self,
        order: Order,
        *,
        pre_position_quantity: Decimal,
    ) -> object:
        envelope = self._envelope(order, pre_position_quantity)
        _, mapped_order, dispatch = _claim_and_issue_terminal_dispatch(
            self.authority,
            submitter=self.submitter_identity,
            executor=self.executor,
            coordinator=self.coordinator,
            envelope=envelope,
            pre_position_quantity=pre_position_quantity,
        )
        assert mapped_order.symbol == order.symbol
        return dispatch

    def submit(
        self,
        order: Order,
        *,
        pre_position_quantity: Decimal,
    ) -> ExecutionResult:
        from robo_trader.paper_execution_capability import _submit_gateway_reduction_once

        try:
            dispatch = self.issue(
                order,
                pre_position_quantity=pre_position_quantity,
            )
            return _submit_gateway_reduction_once(
                self.authority,
                dispatch,
                submitter=self.submitter_identity,
                order=order,
                pre_position_quantity=pre_position_quantity,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return ExecutionResult(False, f"reduction exposure rejected: {exc}")


def bind_gateway_reduction_harness(
    executor: PaperExecutor,
    portfolio_id: str,
    *,
    coordinator: SafetyRuntimeCoordinator | None = None,
) -> GatewayBoundReductionHarness:
    """Exercise the exact one-shot registration functions without I/O."""
    now = datetime.now(timezone.utc)
    if coordinator is None:
        temp_root = Path(tempfile.mkdtemp(prefix="paper-execution-test-"))
        identity = PaperExecutionIdentity("paper-simulator-v1", _ACCOUNT_SCOPE)
        journal = SafetyJournal(temp_root / "safety.db", clock=lambda: now)
        journal.initialize(
            execution_domain_scope=identity.execution_domain_scope,
            account_scope=identity.account_scope,
        )
        coordinator = SafetyRuntimeCoordinator(identity, journal, clock=lambda: now)
        coordinator.start()
    authority, coordinator = bind_gateway_reduction_authority(
        executor, portfolio_id, coordinator=coordinator
    )
    submitter_identity = _bind_paper_reduction_submitter(
        executor,
        coordinator,
        authority,
        portfolio_id,
    )
    return GatewayBoundReductionHarness(
        executor=executor,
        coordinator=coordinator,
        authority=authority,
        submitter_identity=submitter_identity,
        portfolio_id=portfolio_id,
        clock=now,
    )


def bind_gateway_reduction_authority(
    executor: PaperExecutor,
    portfolio_id: str,
    *,
    coordinator: SafetyRuntimeCoordinator | None = None,
) -> tuple[PaperReductionExecutionAuthority, SafetyRuntimeCoordinator]:
    """Return an unattached authority from one exact registration session."""

    if coordinator is None:
        coordinator = SafetyRuntimeCoordinator.__new__(SafetyRuntimeCoordinator)
        coordinator._startup_lock = threading.RLock()
        coordinator._started = True
        coordinator._coordinator_token = object()

    gateway = PaperReductionGateway.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._runtime_context = object()
    gateway._coordinator = coordinator
    session = _PaperRuntimeBindingSession(
        gateway=gateway,
        runtime_context=gateway._runtime_context,
        executor=executor,
        portfolio_id=portfolio_id,
    )
    gateway._active_runtime_binding_session = session
    try:
        capability = _issue_gateway_reduction_binding_capability(
            gateway=gateway,
            runtime_context=gateway._runtime_context,
            binding_session=session,
            executor=executor,
            portfolio_id=portfolio_id,
            coordinator=coordinator,
        )
        authority = _bind_gateway_reduction_execution(
            gateway=gateway,
            runtime_context=gateway._runtime_context,
            binding_session=session,
            executor=executor,
            portfolio_id=portfolio_id,
            coordinator=coordinator,
            capability=capability,
        )
    finally:
        gateway._active_runtime_binding_session = None

    return authority, coordinator
