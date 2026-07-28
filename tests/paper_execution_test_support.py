"""Test-only construction of an exact gateway registration lifecycle."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from decimal import Decimal

from robo_trader.execution import ExecutionResult, Order, PaperExecutor
from robo_trader.paper_execution_capability import (
    _CAPABILITIES,
    _CAPABILITY_TOKEN,
    _REDUCTION_DISPATCHES,
    _REGISTRY_LOCK,
    _TERMINAL_DISPATCH_TOKEN,
    PaperReductionExecutionAuthority,
    _attach_gateway_reduction_submitter,
    _bind_gateway_reduction_execution,
    _CapabilityKind,
    _CapabilityRecord,
    _fingerprint_order,
    _issue_gateway_reduction_binding_capability,
    _PaperExecutionCapability,
    _ReductionTerminalDispatch,
    _TerminalDispatchRecord,
)
from robo_trader.paper_reduction_gateway import (
    PaperReductionGateway,
    _PaperRuntimeBindingSession,
)
from robo_trader.safety.runtime import SafetyRuntimeCoordinator


@dataclass(frozen=True)
class GatewayBoundReductionHarness:
    executor: PaperExecutor
    coordinator: SafetyRuntimeCoordinator
    authority: PaperReductionExecutionAuthority
    submitter_identity: object
    portfolio_id: str

    def issue(
        self,
        order: Order,
        *,
        pre_position_quantity: Decimal,
    ) -> object:
        fingerprint = _fingerprint_order(order)
        dispatch = _ReductionTerminalDispatch(_token=_TERMINAL_DISPATCH_TOKEN)
        with _REGISTRY_LOCK:
            _REDUCTION_DISPATCHES[dispatch] = _TerminalDispatchRecord(
                binding=self.authority,
                gateway=None,
                runtime_context=None,
                active_session=None,
                submitter=self.submitter_identity,
                coordinator=self.coordinator,
                executor=self.executor,
                portfolio_id=self.portfolio_id,
                fingerprint=fingerprint,
                pre_position_quantity=pre_position_quantity,
            )
        return dispatch

    def submit(
        self,
        order: Order,
        *,
        pre_position_quantity: Decimal,
    ) -> ExecutionResult:
        capability = _PaperExecutionCapability(_token=_CAPABILITY_TOKEN)
        with _REGISTRY_LOCK:
            _CAPABILITIES[capability] = _CapabilityRecord(
                authority=self.authority,
                executor=self.executor,
                order=order,
                portfolio_id=self.portfolio_id,
                kind=_CapabilityKind.REDUCTION,
                fingerprint=_fingerprint_order(order),
                pre_position_quantity=pre_position_quantity,
            )
        return self.executor._place_simple_order(order, _capability=capability)


def bind_gateway_reduction_harness(
    executor: PaperExecutor,
    portfolio_id: str,
    *,
    coordinator: SafetyRuntimeCoordinator | None = None,
) -> GatewayBoundReductionHarness:
    """Exercise the exact one-shot registration functions without I/O."""

    authority, coordinator = bind_gateway_reduction_authority(
        executor,
        portfolio_id,
        coordinator=coordinator,
    )
    submitter_identity = object()
    _attach_gateway_reduction_submitter(
        authority,
        submitter=submitter_identity,
        executor=executor,
        coordinator=coordinator,
        portfolio_id=portfolio_id,
    )
    return GatewayBoundReductionHarness(
        executor=executor,
        coordinator=coordinator,
        authority=authority,
        submitter_identity=submitter_identity,
        portfolio_id=portfolio_id,
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
