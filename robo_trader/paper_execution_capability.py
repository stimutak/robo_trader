"""One-shot integrity capabilities for exposure-reducing paper fills.

This staging module deliberately contains no baseline BUY issuer or terminal
entry path. Semantic reductions remain constrained by the coordinator's final
allocation and exact pre-position evidence. Python closure state is treated as
implementation integrity, never as isolation from hostile same-process code.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import threading
import weakref
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .execution import PaperExecutor


class PaperExecutionCapabilityError(RuntimeError):
    """A terminal paper submission lacked exact one-shot authority."""


_PAPER_FILL_PRICE_TICK = Decimal("0.0001")


class _CapabilityKind(Enum):
    REDUCTION = "reduction"


@dataclass(frozen=True, slots=True)
class _OrderFingerprint:
    symbol: str
    quantity: int
    side: str
    price: object
    order_ref: str | None
    take_profit: object


def _validate_gateway_binding_scope(
    *,
    gateway: object,
    runtime_context: object,
    binding_session: object,
    executor: object,
    portfolio_id: str,
) -> None:
    from .execution import PaperExecutor
    from .paper_reduction_gateway import PaperReductionGateway, _PaperRuntimeBindingSession

    if (
        type(gateway) is not PaperReductionGateway
        or gateway.started is not True
        or getattr(gateway, "_runtime_context", None) is not runtime_context
        or getattr(gateway, "_active_runtime_binding_session", None) is not binding_session
        or type(binding_session) is not _PaperRuntimeBindingSession
        or binding_session.gateway is not gateway
        or binding_session.runtime_context is not runtime_context
        or binding_session.executor is not executor
        or binding_session.portfolio_id != portfolio_id
        or type(executor) is not PaperExecutor
        or type(portfolio_id) is not str
        or not portfolio_id
        or portfolio_id.strip() != portfolio_id
    ):
        raise PaperExecutionCapabilityError("gateway execution binding scope is invalid")


def _fingerprint_order(order: object) -> _OrderFingerprint:
    from .execution import Order

    if type(order) is not Order:
        raise PaperExecutionCapabilityError("terminal order must be exactly Order")
    if (
        type(order.symbol) is not str
        or not order.symbol
        or order.symbol.strip() != order.symbol
        or type(order.quantity) is not int
        or order.quantity <= 0
        or type(order.side) is not str
        or order.side != order.side.upper()
    ):
        raise PaperExecutionCapabilityError("terminal order identity is malformed")
    if order.order_ref is not None and type(order.order_ref) is not str:
        raise PaperExecutionCapabilityError("terminal order reference is malformed")
    return _OrderFingerprint(
        symbol=order.symbol,
        quantity=order.quantity,
        side=order.side,
        price=order.price,
        order_ref=order.order_ref,
        take_profit=order.take_profit,
    )


def _validate_reduction_bounds(
    fingerprint: _OrderFingerprint,
    pre_position_quantity: Decimal,
) -> None:
    if (
        type(pre_position_quantity) is not Decimal
        or not pre_position_quantity.is_finite()
        or type(fingerprint.quantity) is not int
        or fingerprint.quantity <= 0
    ):
        raise PaperExecutionCapabilityError("reduction position evidence is malformed")
    quantity = Decimal(fingerprint.quantity)
    if fingerprint.side == "SELL":
        if pre_position_quantity <= 0 or quantity > pre_position_quantity:
            raise PaperExecutionCapabilityError(
                "SELL reduction would create or cross into short exposure"
            )
        final_quantity = pre_position_quantity - quantity
        if final_quantity < 0:
            raise PaperExecutionCapabilityError("SELL reduction crosses zero")
    elif fingerprint.side == "BUY_TO_COVER":
        if pre_position_quantity >= 0 or quantity > pre_position_quantity.copy_abs():
            raise PaperExecutionCapabilityError(
                "BUY_TO_COVER reduction would create or cross into long exposure"
            )
        final_quantity = pre_position_quantity + quantity
        if final_quantity > 0:
            raise PaperExecutionCapabilityError("BUY_TO_COVER reduction crosses zero")
    else:
        raise PaperExecutionCapabilityError("reduction capability admits only SELL or BUY_TO_COVER")


def _build_sealed_capability_runtime():
    """Build one-shot integrity state for exposure-reducing paper fills.

    Python code in one interpreter is one trust domain: closure cells are
    introspectable and are not a security-isolation boundary. This runtime
    therefore mints capabilities only for semantic reductions. Baseline BUY
    authority remains unavailable until admission is enforced outside that
    shared trust domain or by an independently verified integrated boundary.
    """

    fingerprint_order = _fingerprint_order
    validate_reduction_bounds = _validate_reduction_bounds
    validate_gateway_binding_scope = _validate_gateway_binding_scope
    replace_record = dataclass_replace
    reduction_bind_capability_token = object()
    reduction_authority_token = object()
    terminal_dispatch_token = object()
    capability_token = object()
    registry_lock = threading.Lock()

    @dataclass(frozen=True, slots=True)
    class CapabilityRecord:
        authority: object
        executor: object
        order: object
        portfolio_id: str
        kind: _CapabilityKind
        fingerprint: _OrderFingerprint
        pre_position_quantity: Decimal | None
        consumed: bool = False
        validated: bool = False
        fill_consumed: bool = False

    @dataclass(frozen=True, slots=True)
    class ReductionBindingRecord:
        gateway: object
        runtime_context: object
        binding_session: object
        executor: object
        portfolio_id: str
        consumed: bool = False
        coordinator: object = None

    @dataclass(frozen=True, slots=True)
    class ReductionAuthorityRecord:
        gateway: object
        runtime_context: object
        executor: object
        portfolio_id: str
        coordinator: object
        submitter: object | None = None

    @dataclass(frozen=True, slots=True)
    class TerminalDispatchRecord:
        binding: object | None
        gateway: object | None
        runtime_context: object | None
        active_session: object | None
        submitter: object | None
        coordinator: object | None
        executor: object
        portfolio_id: str
        fingerprint: _OrderFingerprint
        pre_position_quantity: Decimal | None
        consumed: bool = False

    capabilities: weakref.WeakKeyDictionary[object, CapabilityRecord] = weakref.WeakKeyDictionary()
    reduction_bindings: weakref.WeakKeyDictionary[object, ReductionBindingRecord] = (
        weakref.WeakKeyDictionary()
    )
    reduction_authorities: weakref.WeakKeyDictionary[object, ReductionAuthorityRecord] = (
        weakref.WeakKeyDictionary()
    )
    reduction_dispatches: weakref.WeakKeyDictionary[object, TerminalDispatchRecord] = (
        weakref.WeakKeyDictionary()
    )

    class PaperExecutionCapability:
        __slots__ = ("__weakref__",)

        def __new__(cls, *, _token: object | None = None):
            if _token is not capability_token:
                raise PaperExecutionCapabilityError(
                    "paper execution capabilities are minted only by a bound authority"
                )
            return super().__new__(cls)

        def __copy__(self):
            with registry_lock:
                record = capabilities.get(self)
                if type(record) is CapabilityRecord:
                    capabilities[self] = replace_record(record, consumed=True)
            raise PaperExecutionCapabilityError("paper execution capabilities cannot copy")

        def __deepcopy__(self, _memo):
            return self.__copy__()

        def __reduce__(self):
            with registry_lock:
                record = capabilities.get(self)
                if type(record) is CapabilityRecord:
                    capabilities[self] = replace_record(record, consumed=True)
            raise PaperExecutionCapabilityError("paper execution capabilities cannot serialize")

    class GatewayReductionBindingCapability:
        __slots__ = ("__weakref__",)

        def __new__(cls, *, _token: object | None = None):
            if _token is not reduction_bind_capability_token:
                raise PaperExecutionCapabilityError("reduction binding capability is issuer-only")
            return super().__new__(cls)

        def __copy__(self):
            with registry_lock:
                record = reduction_bindings.get(self)
                if type(record) is ReductionBindingRecord:
                    reduction_bindings[self] = replace_record(record, consumed=True)
            raise PaperExecutionCapabilityError("reduction binding capability cannot copy")

        def __deepcopy__(self, _memo):
            return self.__copy__()

        def __reduce__(self):
            raise PaperExecutionCapabilityError("reduction binding capability cannot serialize")

    class ReductionTerminalDispatch:
        __slots__ = ("__weakref__",)

        def __new__(cls, *, _token: object | None = None):
            if _token is not terminal_dispatch_token:
                raise PaperExecutionCapabilityError("reduction terminal dispatch is issuer-only")
            return super().__new__(cls)

        def __copy__(self):
            with registry_lock:
                record = reduction_dispatches.get(self)
                if type(record) is TerminalDispatchRecord:
                    reduction_dispatches[self] = replace_record(record, consumed=True)
            raise PaperExecutionCapabilityError("reduction terminal dispatch cannot copy")

        def __deepcopy__(self, _memo):
            return self.__copy__()

        def __reduce__(self):
            with registry_lock:
                record = reduction_dispatches.get(self)
                if type(record) is TerminalDispatchRecord:
                    reduction_dispatches[self] = replace_record(record, consumed=True)
            raise PaperExecutionCapabilityError("reduction terminal dispatch cannot serialize")

    class ReductionExecutionAuthority:
        __slots__ = ("__weakref__",)

        def __new__(cls, *, _token: object | None = None):
            if _token is not reduction_authority_token:
                raise PaperExecutionCapabilityError("paper reduction authority is gateway-only")
            return super().__new__(cls)

        def __copy__(self):
            raise PaperExecutionCapabilityError("paper reduction authority cannot copy")

        def __deepcopy__(self, _memo):
            return self.__copy__()

        def __reduce__(self):
            raise PaperExecutionCapabilityError("paper reduction authority cannot serialize")

    def issue_gateway_reduction_binding_capability(
        *, gateway, runtime_context, binding_session, executor, portfolio_id, coordinator
    ):
        validate_gateway_binding_scope(
            gateway=gateway,
            runtime_context=runtime_context,
            binding_session=binding_session,
            executor=executor,
            portfolio_id=portfolio_id,
        )
        from .safety import SafetyRuntimeCoordinator

        if (
            type(coordinator) is not SafetyRuntimeCoordinator
            or coordinator.started is not True
            or getattr(gateway, "_coordinator", None) is not coordinator
        ):
            raise PaperExecutionCapabilityError("reduction coordinator binding is invalid")
        if getattr(binding_session, "reduction_capability_issued", False):
            raise PaperExecutionCapabilityError(
                "gateway runtime binding session already issued reduction capability"
            )
        setattr(binding_session, "reduction_capability_issued", True)
        capability = GatewayReductionBindingCapability(_token=reduction_bind_capability_token)
        with registry_lock:
            reduction_bindings[capability] = ReductionBindingRecord(
                gateway,
                runtime_context,
                binding_session,
                executor,
                portfolio_id,
                False,
                coordinator,
            )
        return capability

    def bind_gateway_reduction_execution(
        *,
        gateway,
        runtime_context,
        binding_session,
        executor,
        portfolio_id,
        coordinator,
        capability,
    ):
        if type(capability) is not GatewayReductionBindingCapability:
            raise PaperExecutionCapabilityError("reduction binding capability is invalid")
        with registry_lock:
            record = reduction_bindings.get(capability)
            if type(record) is not ReductionBindingRecord or record.consumed:
                raise PaperExecutionCapabilityError(
                    "reduction binding capability is unknown or already consumed"
                )
            reduction_bindings[capability] = replace_record(record, consumed=True)
        validate_gateway_binding_scope(
            gateway=gateway,
            runtime_context=runtime_context,
            binding_session=binding_session,
            executor=executor,
            portfolio_id=portfolio_id,
        )
        if (
            record.gateway is not gateway
            or record.runtime_context is not runtime_context
            or record.binding_session is not binding_session
            or record.executor is not executor
            or record.portfolio_id != portfolio_id
            or record.coordinator is not coordinator
        ):
            raise PaperExecutionCapabilityError("reduction binding does not match runtime")
        authority = ReductionExecutionAuthority(_token=reduction_authority_token)
        with registry_lock:
            reduction_authorities[authority] = ReductionAuthorityRecord(
                gateway, runtime_context, executor, portfolio_id, coordinator
            )
        return authority

    def attach_gateway_reduction_submitter(
        authority, *, submitter, executor, coordinator, portfolio_id
    ):
        if type(authority) is not ReductionExecutionAuthority:
            raise PaperExecutionCapabilityError("reduction authority is invalid")
        with registry_lock:
            record = reduction_authorities.get(authority)
            if (
                type(record) is not ReductionAuthorityRecord
                or record.submitter is not None
                or record.executor is not executor
                or record.coordinator is not coordinator
                or record.portfolio_id != portfolio_id
            ):
                raise PaperExecutionCapabilityError("reduction submitter binding is invalid")
            reduction_authorities[authority] = replace_record(record, submitter=submitter)

    def reduction_authority_matches(authority, *, executor, coordinator, portfolio_id):
        with registry_lock:
            record = (
                reduction_authorities.get(authority)
                if type(authority) is ReductionExecutionAuthority
                else None
            )
            return bool(
                type(record) is ReductionAuthorityRecord
                and record.executor is executor
                and record.coordinator is coordinator
                and record.portfolio_id == portfolio_id
            )

    def consume_paper_execution_capability(executor, order, capability):
        if type(capability) is not PaperExecutionCapability:
            raise PaperExecutionCapabilityError(
                "terminal paper execution requires an exact submission capability"
            )
        with registry_lock:
            record = capabilities.get(capability)
            if type(record) is not CapabilityRecord or record.consumed:
                raise PaperExecutionCapabilityError(
                    "paper execution capability is unknown or already consumed"
                )
            record = replace_record(record, consumed=True)
            capabilities[capability] = record
        fingerprint = fingerprint_order(order)
        if record.executor is not executor or record.fingerprint != fingerprint:
            raise PaperExecutionCapabilityError(
                "paper execution capability does not match executor or order"
            )
        if record.kind is not _CapabilityKind.REDUCTION:
            raise PaperExecutionCapabilityError("paper execution capability kind is unsupported")
        if type(record.pre_position_quantity) is not Decimal:
            raise PaperExecutionCapabilityError(
                "reduction capability lacks exact position evidence"
            )
        validate_reduction_bounds(fingerprint, record.pre_position_quantity)
        with registry_lock:
            current = capabilities.get(capability)
            if current is not record or record.consumed is not True:
                raise PaperExecutionCapabilityError("paper execution capability state changed")
            record = replace_record(record, validated=True)
            capabilities[capability] = record

    def apply_consumed_paper_fill(executor, order, capability):
        if type(capability) is not PaperExecutionCapability:
            raise PaperExecutionCapabilityError(
                "paper fill requires an exact consumed submission capability"
            )
        with registry_lock:
            record = capabilities.get(capability)
            if (
                type(record) is not CapabilityRecord
                or record.consumed is not True
                or record.validated is not True
                or record.fill_consumed
                or record.executor is not executor
                or record.order is not order
            ):
                raise PaperExecutionCapabilityError(
                    "paper fill capability is unknown, unconsumed, mismatched, or already filled"
                )
            expected_fingerprint = record.fingerprint
            capabilities[capability] = replace_record(record, fill_consumed=True)
            del record
        fingerprint = fingerprint_order(order)
        if expected_fingerprint != fingerprint:
            raise PaperExecutionCapabilityError("paper fill capability order fingerprint changed")

        from .execution import (
            ExecutionResult,
            LocalPaperExecutionEvidence,
            Order,
            PaperExecutor,
        )

        if type(executor) is not PaperExecutor or type(order) is not Order:
            raise PaperExecutionCapabilityError(
                "sealed paper fill requires the exact executor and order types"
            )
        exact_base = None
        if order.price is not None:
            if type(order.price) is Decimal:
                if not order.price.is_finite() or order.price <= 0:
                    return ExecutionResult(False, "Invalid price for paper execution")
                exact_base = order.price
                base = float(exact_base)
            else:
                try:
                    base = float(order.price)
                except (TypeError, ValueError):
                    return ExecutionResult(False, "Invalid price for paper execution")
            if not math.isfinite(base):
                return ExecutionResult(False, "Non-finite price for paper execution")
            if base <= 0:
                return ExecutionResult(False, "Non-positive price for paper execution")
            executor._execution_cache[order.symbol] = base
            executor._execution_cache_ts[order.symbol] = dt.datetime.utcnow()
        else:
            base = executor._execution_cache.get(order.symbol)
            if base is None:
                return ExecutionResult(False, "No reference price for market order")
            timestamp = executor._execution_cache_ts.get(order.symbol)
            if (
                timestamp is None
                or (dt.datetime.utcnow() - timestamp).total_seconds()
                > executor._execution_cache_max_age_seconds
            ):
                return ExecutionResult(False, "Stale reference price for market order")
        if exact_base is not None:
            slip_decimal = (
                exact_base * Decimal(str(executor.slippage_bps)) / Decimal("10000")
                if executor.slippage_bps
                else Decimal("0")
            )
            unrounded_fill = (
                exact_base + slip_decimal
                if order.side.upper() in {"BUY", "BUY_TO_COVER"}
                else exact_base - slip_decimal
            )
            if not unrounded_fill.is_finite() or unrounded_fill <= 0:
                return ExecutionResult(False, "Invalid paper execution fill")
            fill_decimal = unrounded_fill.quantize(_PAPER_FILL_PRICE_TICK, rounding=ROUND_HALF_EVEN)
            fill = float(fill_decimal)
        else:
            slip = base * (executor.slippage_bps / 10_000.0) if executor.slippage_bps else 0.0
            fill = base + slip if order.side.upper() in {"BUY", "BUY_TO_COVER"} else base - slip
            fill_decimal = Decimal(str(fill))
        if not math.isfinite(fill) or fill <= 0:
            return ExecutionResult(False, "Invalid paper execution fill")
        occurred_at = dt.datetime.now(dt.timezone.utc)
        if type(order.order_ref) is not str or not order.order_ref:
            raise PaperExecutionCapabilityError(
                "sealed paper execution requires a durable order reference"
            )
        execution_material = "\x1f".join(
            (
                order.order_ref,
                order.symbol,
                order.side,
                str(order.quantity),
            )
        )
        execution_id = (
            "lpfill-" + hashlib.sha256(execution_material.encode("utf-8")).hexdigest()[:32]
        )
        evidence = LocalPaperExecutionEvidence(
            execution_id=execution_id,
            filled_quantity=Decimal(order.quantity),
            exact_fill_price=fill_decimal,
            # The current local-paper executor's explicit cost model has no
            # commission.  Zero is producer evidence here, not a database
            # default or a value inferred after execution.
            commission_minor=0,
            commission_currency="USD",
            commission_source="LOCAL_PAPER_EXECUTOR_EXACT_COMMISSION_V1",
            occurred_at=occurred_at,
        )
        executor.fills[f"{order.symbol}-{len(executor.fills)+1}"] = (
            occurred_at,
            order,
            fill,
        )
        return ExecutionResult(
            True,
            "Paper fill",
            fill,
            exact_fill_price=fill_decimal,
            local_paper_evidence=evidence,
        )

    def execute_sealed_paper_fill(executor, order, capability):
        consume_paper_execution_capability(executor, order, capability)
        return apply_consumed_paper_fill(executor, order, capability)

    terminal_sink = execute_sealed_paper_fill

    def expected_reduction_fingerprint(descriptor, contract):
        from .runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
        from .safety.models import OrderSide, OrderType, TimeInForce

        if (
            descriptor.execution_domain_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
            or descriptor.con_id != contract.con_id
            or descriptor.side not in {OrderSide.SELL, OrderSide.BUY_TO_COVER}
            or type(descriptor.quantity) is not Decimal
            or not descriptor.quantity.is_finite()
            or descriptor.quantity <= 0
            or descriptor.quantity > Decimal("2147483647")
            or descriptor.quantity != descriptor.quantity.to_integral_value()
            or descriptor.attempt_number != 1
            or type(descriptor.attempt_number) is not int
            or descriptor.slice_count != 1
            or type(descriptor.slice_count) is not int
            or descriptor.bracket is not False
            or descriptor.time_in_force is not TimeInForce.DAY
            or descriptor.outside_regular_hours is not False
        ):
            raise PaperExecutionCapabilityError(
                "reduction final allocation is not terminally representable"
            )
        if descriptor.order_type is OrderType.MARKET:
            if descriptor.limit_price is not None or descriptor.stop_price is not None:
                raise PaperExecutionCapabilityError(
                    "reduction final allocation is not terminally representable"
                )
            price = None
        elif descriptor.order_type is OrderType.LIMIT:
            if (
                type(descriptor.limit_price) is not Decimal
                or not descriptor.limit_price.is_finite()
                or descriptor.limit_price <= 0
                or descriptor.limit_price > Decimal("1000000")
                or descriptor.limit_price != descriptor.limit_price.quantize(Decimal("0.0001"))
                or descriptor.stop_price is not None
            ):
                raise PaperExecutionCapabilityError(
                    "reduction final allocation is not terminally representable"
                )
            price = descriptor.limit_price
        else:
            raise PaperExecutionCapabilityError(
                "reduction final allocation is not terminally representable"
            )
        return _OrderFingerprint(
            symbol=contract.symbol,
            quantity=int(descriptor.quantity),
            side=descriptor.side.value,
            price=price,
            order_ref=descriptor.order_ref,
            take_profit=None,
        )

    from .safety.runtime import (
        _consume_claimed_paper_submission_allocation as consume_claimed_allocation,
    )

    def issue_gateway_reduction_terminal_dispatch(
        authority,
        *,
        submitter,
        executor,
        coordinator,
        final_allocation,
        descriptor,
        contract,
        order,
        pre_position_quantity,
    ):
        try:
            claimed_pre_position_quantity = consume_claimed_allocation(
                final_allocation,
                coordinator=coordinator,
                descriptor=descriptor,
                contract=contract,
                pre_position_quantity=pre_position_quantity,
            )
        except (RuntimeError, TypeError) as exc:
            raise PaperExecutionCapabilityError(
                "reduction terminal dispatch lacks exact final allocation"
            ) from exc
        fingerprint = fingerprint_order(order)
        expected_fingerprint = expected_reduction_fingerprint(descriptor, contract)
        if fingerprint != expected_fingerprint:
            raise PaperExecutionCapabilityError(
                "reduction order does not match the claimed final allocation"
            )
        validate_reduction_bounds(fingerprint, claimed_pre_position_quantity)
        with registry_lock:
            authority_record = (
                reduction_authorities.get(authority)
                if type(authority) is ReductionExecutionAuthority
                else None
            )
            if (
                type(authority_record) is not ReductionAuthorityRecord
                or authority_record.submitter is not submitter
                or authority_record.executor is not executor
                or authority_record.coordinator is not coordinator
            ):
                raise PaperExecutionCapabilityError("reduction terminal issuer is not bound")
            dispatch = ReductionTerminalDispatch(_token=terminal_dispatch_token)
            reduction_dispatches[dispatch] = TerminalDispatchRecord(
                authority,
                authority_record.gateway,
                authority_record.runtime_context,
                None,
                submitter,
                coordinator,
                executor,
                authority_record.portfolio_id,
                fingerprint,
                claimed_pre_position_quantity,
            )
        return dispatch

    def _submit_gateway_reduction_once(
        authority, dispatch, *, submitter, order, pre_position_quantity
    ):
        if type(dispatch) is not ReductionTerminalDispatch:
            raise PaperExecutionCapabilityError("reduction terminal dispatch is invalid")
        with registry_lock:
            record = reduction_dispatches.get(dispatch)
            if type(record) is not TerminalDispatchRecord or record.consumed:
                raise PaperExecutionCapabilityError(
                    "reduction terminal dispatch is unknown or already consumed"
                )
            expected_authority = record.binding
            expected_submitter = record.submitter
            executor = cast("PaperExecutor", record.executor)
            portfolio_id = record.portfolio_id
            expected_fingerprint = record.fingerprint
            expected_pre_position = record.pre_position_quantity
            reduction_dispatches[dispatch] = replace_record(
                record,
                consumed=True,
                binding=None,
                gateway=None,
                runtime_context=None,
                submitter=None,
                coordinator=None,
            )
        fingerprint = fingerprint_order(order)
        validate_reduction_bounds(fingerprint, pre_position_quantity)
        if (
            type(authority) is not ReductionExecutionAuthority
            or expected_authority is not authority
            or expected_submitter is not submitter
            or fingerprint != expected_fingerprint
            or pre_position_quantity != expected_pre_position
        ):
            raise PaperExecutionCapabilityError(
                "reduction terminal dispatch does not match attempt"
            )
        capability = PaperExecutionCapability(_token=capability_token)
        with registry_lock:
            capabilities[capability] = CapabilityRecord(
                authority,
                executor,
                order,
                portfolio_id,
                _CapabilityKind.REDUCTION,
                fingerprint,
                pre_position_quantity,
            )
        return terminal_sink(executor, order, capability)

    return (
        PaperExecutionCapability,
        GatewayReductionBindingCapability,
        ReductionTerminalDispatch,
        ReductionExecutionAuthority,
        issue_gateway_reduction_binding_capability,
        bind_gateway_reduction_execution,
        attach_gateway_reduction_submitter,
        reduction_authority_matches,
        issue_gateway_reduction_terminal_dispatch,
        _submit_gateway_reduction_once,
        consume_paper_execution_capability,
        apply_consumed_paper_fill,
        execute_sealed_paper_fill,
    )


(
    _PaperExecutionCapability,
    _GatewayReductionBindingCapability,
    _ReductionTerminalDispatch,
    PaperReductionExecutionAuthority,
    _issue_gateway_reduction_binding_capability,
    _bind_gateway_reduction_execution,
    _attach_gateway_reduction_submitter,
    _reduction_authority_matches,
    _issue_gateway_reduction_terminal_dispatch,
    _submit_gateway_reduction_once,
    consume_paper_execution_capability,
    _apply_consumed_paper_fill,
    _execute_sealed_paper_fill,
) = _build_sealed_capability_runtime()

# Prevent callers from constructing a second authority universe.
del _build_sealed_capability_runtime
