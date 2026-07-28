"""One-shot capabilities for the terminal local-paper execution sink.

The account gateway consumes a one-shot runtime binding grant before it can
bind baseline execution. The resulting baseline binding has no submission
method; only an exact active gateway entry session can derive a terminal
capability. Reduction authority remains separately constrained by the sealed
coordinator submission adapter and exact pre-position evidence.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .execution import PaperExecutor


class PaperExecutionCapabilityError(RuntimeError):
    """A terminal paper submission lacked exact one-shot authority."""


class _CapabilityKind(Enum):
    BASELINE_ENTRY = "baseline-entry"
    REDUCTION = "reduction"


@dataclass(frozen=True, slots=True)
class _OrderFingerprint:
    symbol: str
    quantity: int
    side: str
    price: object
    order_ref: str | None
    take_profit: object


class _PaperExecutionCapability:
    """Immutable opaque identity; validity lives in the private registry."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *, _token: object | None = None):
        if _token is not _CAPABILITY_TOKEN:
            raise PaperExecutionCapabilityError(
                "paper execution capabilities are minted only by a bound authority"
            )
        return super().__new__(cls)


class _GatewayExecutionBindingCapability:
    """Opaque one-shot grant issued only during exact gateway registration."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *, _token: object | None = None):
        if _token is not _BIND_CAPABILITY_TOKEN:
            raise PaperExecutionCapabilityError(
                "gateway execution binding capabilities are issuer-only"
            )
        return super().__new__(cls)

    def __copy__(self):
        with _REGISTRY_LOCK:
            record = _GATEWAY_BINDINGS.get(self)
            if type(record) is _GatewayBindingRecord:
                record.consumed = True
        raise PaperExecutionCapabilityError("gateway execution binding capabilities cannot copy")

    def __deepcopy__(self, _memo):
        return self.__copy__()


class _GatewayReductionBindingCapability:
    """Opaque one-shot grant for the reduction half of registration."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *, _token: object | None = None):
        if _token is not _REDUCTION_BIND_CAPABILITY_TOKEN:
            raise PaperExecutionCapabilityError("reduction binding capability is issuer-only")
        return super().__new__(cls)

    def __copy__(self):
        with _REGISTRY_LOCK:
            record = _REDUCTION_BINDINGS.get(self)
            if type(record) is _ReductionBindingRecord:
                record.consumed = True
        raise PaperExecutionCapabilityError("reduction binding capability cannot copy")

    def __deepcopy__(self, _memo):
        return self.__copy__()

    def __reduce__(self):
        raise PaperExecutionCapabilityError("reduction binding capability cannot serialize")


class _BaselineTerminalDispatch:
    __slots__ = ("__weakref__",)

    def __new__(cls, *, _token: object | None = None):
        if _token is not _TERMINAL_DISPATCH_TOKEN:
            raise PaperExecutionCapabilityError("baseline terminal dispatch is issuer-only")
        return super().__new__(cls)


class _ReductionTerminalDispatch:
    __slots__ = ("__weakref__",)

    def __new__(cls, *, _token: object | None = None):
        if _token is not _TERMINAL_DISPATCH_TOKEN:
            raise PaperExecutionCapabilityError("reduction terminal dispatch is issuer-only")
        return super().__new__(cls)


@dataclass(slots=True)
class _CapabilityRecord:
    authority: object
    executor: object
    portfolio_id: str
    kind: _CapabilityKind
    fingerprint: _OrderFingerprint
    pre_position_quantity: Decimal | None
    consumed: bool = False


@dataclass(slots=True)
class _GatewayBindingRecord:
    gateway: object
    runtime_context: object
    binding_session: object
    executor: object
    portfolio_id: str
    consumed: bool = False


@dataclass(slots=True)
class _BaselineBindingRecord:
    gateway: object
    runtime_context: object
    executor: object
    portfolio_id: str


@dataclass(slots=True)
class _ReductionBindingRecord(_GatewayBindingRecord):
    coordinator: object = None


@dataclass(slots=True)
class _ReductionAuthorityRecord:
    gateway: object
    runtime_context: object
    executor: object
    portfolio_id: str
    coordinator: object
    submitter: object | None = None


@dataclass(slots=True)
class _TerminalDispatchRecord:
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


_BIND_CAPABILITY_TOKEN = object()
_REDUCTION_BIND_CAPABILITY_TOKEN = object()
_REDUCTION_AUTHORITY_TOKEN = object()
_BASELINE_BIND_TOKEN = object()
_TERMINAL_DISPATCH_TOKEN = object()
_CAPABILITY_TOKEN = object()
_REGISTRY_LOCK = threading.Lock()
_CAPABILITIES: weakref.WeakKeyDictionary[_PaperExecutionCapability, _CapabilityRecord] = (
    weakref.WeakKeyDictionary()
)
_GATEWAY_BINDINGS: weakref.WeakKeyDictionary[
    _GatewayExecutionBindingCapability, _GatewayBindingRecord
] = weakref.WeakKeyDictionary()
_BASELINE_BINDINGS: weakref.WeakKeyDictionary[
    _GatewayBaselineExecutionBinding, _BaselineBindingRecord
] = weakref.WeakKeyDictionary()
_REDUCTION_BINDINGS: weakref.WeakKeyDictionary[
    _GatewayReductionBindingCapability, _ReductionBindingRecord
] = weakref.WeakKeyDictionary()
_REDUCTION_AUTHORITIES: weakref.WeakKeyDictionary[
    PaperReductionExecutionAuthority, _ReductionAuthorityRecord
] = weakref.WeakKeyDictionary()
_BASELINE_DISPATCHES: weakref.WeakKeyDictionary[
    _BaselineTerminalDispatch, _TerminalDispatchRecord
] = weakref.WeakKeyDictionary()
_REDUCTION_DISPATCHES: weakref.WeakKeyDictionary[
    _ReductionTerminalDispatch, _TerminalDispatchRecord
] = weakref.WeakKeyDictionary()


class PaperReductionExecutionAuthority:
    """Opaque methodless identity for one gateway-bound reduction runtime."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *, _token: object | None = None):
        if _token is not _REDUCTION_AUTHORITY_TOKEN:
            raise PaperExecutionCapabilityError("paper reduction authority is gateway-only")
        return super().__new__(cls)

    def __copy__(self):
        raise PaperExecutionCapabilityError("paper reduction authority cannot copy")

    def __deepcopy__(self, _memo):
        return self.__copy__()

    def __reduce__(self):
        raise PaperExecutionCapabilityError("paper reduction authority cannot serialize")


class _GatewayBaselineExecutionBinding:
    """Opaque registry-backed identity for one baseline runtime binding."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *, _token: object | None = None):
        if _token is not _BASELINE_BIND_TOKEN:
            raise PaperExecutionCapabilityError("baseline execution binding is gateway-only")
        return super().__new__(cls)

    def __copy__(self):
        raise PaperExecutionCapabilityError("baseline execution binding cannot copy")

    def __deepcopy__(self, _memo):
        return self.__copy__()

    def __reduce__(self):
        raise PaperExecutionCapabilityError("baseline execution binding cannot serialize")


def _issue_gateway_execution_binding_capability(
    *,
    gateway: object,
    runtime_context: object,
    binding_session: object,
    executor: object,
    portfolio_id: str,
) -> _GatewayExecutionBindingCapability:
    """Issue one bind grant only inside an exact started gateway session."""

    _validate_gateway_binding_scope(
        gateway=gateway,
        runtime_context=runtime_context,
        binding_session=binding_session,
        executor=executor,
        portfolio_id=portfolio_id,
    )
    if getattr(binding_session, "capability_issued", False):
        raise PaperExecutionCapabilityError(
            "gateway runtime binding session already issued its capability"
        )
    setattr(binding_session, "capability_issued", True)
    capability = _GatewayExecutionBindingCapability(_token=_BIND_CAPABILITY_TOKEN)
    with _REGISTRY_LOCK:
        _GATEWAY_BINDINGS[capability] = _GatewayBindingRecord(
            gateway=gateway,
            runtime_context=runtime_context,
            binding_session=binding_session,
            executor=executor,
            portfolio_id=portfolio_id,
        )
    return capability


def _bind_gateway_baseline_execution(
    *,
    gateway: object,
    runtime_context: object,
    binding_session: object,
    executor: object,
    portfolio_id: str,
    capability: object,
) -> _GatewayBaselineExecutionBinding:
    """Consume one exact gateway grant and seal the baseline binding."""

    if type(capability) is not _GatewayExecutionBindingCapability:
        raise PaperExecutionCapabilityError("gateway execution binding capability is invalid")
    with _REGISTRY_LOCK:
        record = _GATEWAY_BINDINGS.get(capability)
        if type(record) is not _GatewayBindingRecord or record.consumed:
            raise PaperExecutionCapabilityError(
                "gateway execution binding capability is unknown or already consumed"
            )
        record.consumed = True
    _validate_gateway_binding_scope(
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
    ):
        raise PaperExecutionCapabilityError(
            "gateway execution binding capability does not match runtime"
        )
    binding = _GatewayBaselineExecutionBinding(_token=_BASELINE_BIND_TOKEN)
    with _REGISTRY_LOCK:
        _BASELINE_BINDINGS[binding] = _BaselineBindingRecord(
            gateway=gateway,
            runtime_context=runtime_context,
            executor=executor,
            portfolio_id=portfolio_id,
        )
    return binding


def _issue_gateway_reduction_binding_capability(
    *,
    gateway: object,
    runtime_context: object,
    binding_session: object,
    executor: object,
    portfolio_id: str,
    coordinator: object,
) -> _GatewayReductionBindingCapability:
    """Issue exactly one reduction binding grant in gateway registration."""

    _validate_gateway_binding_scope(
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
    capability = _GatewayReductionBindingCapability(_token=_REDUCTION_BIND_CAPABILITY_TOKEN)
    with _REGISTRY_LOCK:
        _REDUCTION_BINDINGS[capability] = _ReductionBindingRecord(
            gateway=gateway,
            runtime_context=runtime_context,
            binding_session=binding_session,
            executor=executor,
            portfolio_id=portfolio_id,
            coordinator=coordinator,
        )
    return capability


def _bind_gateway_reduction_execution(
    *,
    gateway: object,
    runtime_context: object,
    binding_session: object,
    executor: object,
    portfolio_id: str,
    coordinator: object,
    capability: object,
) -> PaperReductionExecutionAuthority:
    """Consume one gateway grant and register a methodless reduction authority."""

    if type(capability) is not _GatewayReductionBindingCapability:
        raise PaperExecutionCapabilityError("reduction binding capability is invalid")
    with _REGISTRY_LOCK:
        record = _REDUCTION_BINDINGS.get(capability)
        if type(record) is not _ReductionBindingRecord or record.consumed:
            raise PaperExecutionCapabilityError(
                "reduction binding capability is unknown or already consumed"
            )
        record.consumed = True
    _validate_gateway_binding_scope(
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
    authority = PaperReductionExecutionAuthority(_token=_REDUCTION_AUTHORITY_TOKEN)
    with _REGISTRY_LOCK:
        _REDUCTION_AUTHORITIES[authority] = _ReductionAuthorityRecord(
            gateway=gateway,
            runtime_context=runtime_context,
            executor=executor,
            portfolio_id=portfolio_id,
            coordinator=coordinator,
        )
    return authority


def _attach_gateway_reduction_submitter(
    authority: object,
    *,
    submitter: object,
    executor: object,
    coordinator: object,
    portfolio_id: str,
) -> None:
    """Bind the exact sealed submitter once, before the gateway publishes it."""

    if type(authority) is not PaperReductionExecutionAuthority:
        raise PaperExecutionCapabilityError("reduction authority is invalid")
    with _REGISTRY_LOCK:
        record = _REDUCTION_AUTHORITIES.get(authority)
        if (
            type(record) is not _ReductionAuthorityRecord
            or record.submitter is not None
            or record.executor is not executor
            or record.coordinator is not coordinator
            or record.portfolio_id != portfolio_id
        ):
            raise PaperExecutionCapabilityError("reduction submitter binding is invalid")
        record.submitter = submitter


def _reduction_authority_matches(
    authority: object,
    *,
    executor: object,
    coordinator: object,
    portfolio_id: str,
) -> bool:
    with _REGISTRY_LOCK:
        record = (
            _REDUCTION_AUTHORITIES.get(authority)
            if type(authority) is PaperReductionExecutionAuthority
            else None
        )
        return bool(
            type(record) is _ReductionAuthorityRecord
            and record.executor is executor
            and record.coordinator is coordinator
            and record.portfolio_id == portfolio_id
        )


def _issue_gateway_baseline_terminal_dispatch(
    binding: object,
    *,
    gateway: object,
    runtime_context: object,
    active_session: object,
    order: object,
) -> _BaselineTerminalDispatch:
    """Register one terminal dispatch for the exact active entry session."""

    from .paper_reduction_gateway import (
        PaperReductionGateway,
        _ActiveEntrySession,
    )

    task = asyncio.current_task()
    sessions = getattr(gateway, "_active_entry_sessions", None)
    fingerprint = _fingerprint_order(order)
    if (
        type(gateway) is not PaperReductionGateway
        or gateway.started is not True
        or getattr(gateway, "_runtime_context", None) is not runtime_context
        or type(active_session) is not _ActiveEntrySession
        or not isinstance(sessions, dict)
        or task is None
        or sessions.get(task) is not active_session
        or active_session.consumed is not True
    ):
        raise PaperExecutionCapabilityError("baseline gateway session does not match binding")
    if (
        fingerprint.side != "BUY"
        or fingerprint.take_profit is not None
        or fingerprint.symbol != active_session.symbol
        or fingerprint.price != active_session.quote.price
    ):
        raise PaperExecutionCapabilityError("baseline order does not match gateway session")
    with _REGISTRY_LOCK:
        binding_record = (
            _BASELINE_BINDINGS.get(binding)
            if type(binding) is _GatewayBaselineExecutionBinding
            else None
        )
        if (
            type(binding_record) is not _BaselineBindingRecord
            or binding_record.gateway is not gateway
            or binding_record.runtime_context is not runtime_context
            or active_session.portfolio_id != binding_record.portfolio_id
            or active_session.dispatch_issued is True
        ):
            raise PaperExecutionCapabilityError(
                "baseline gateway session does not match binding or already issued"
            )
        active_session.dispatch_issued = True
        dispatch = _BaselineTerminalDispatch(_token=_TERMINAL_DISPATCH_TOKEN)
        _BASELINE_DISPATCHES[dispatch] = _TerminalDispatchRecord(
            binding=binding,
            gateway=gateway,
            runtime_context=runtime_context,
            active_session=active_session,
            submitter=None,
            coordinator=None,
            executor=binding_record.executor,
            portfolio_id=binding_record.portfolio_id,
            fingerprint=fingerprint,
            pre_position_quantity=None,
        )
    return dispatch


def _submit_gateway_baseline_once(
    binding: object,
    dispatch: object,
    *,
    gateway: object,
    runtime_context: object,
    active_session: object,
    order: object,
):
    """Atomically burn one registered baseline dispatch before submission."""

    if type(dispatch) is not _BaselineTerminalDispatch:
        raise PaperExecutionCapabilityError("baseline terminal dispatch is invalid")
    with _REGISTRY_LOCK:
        record = _BASELINE_DISPATCHES.get(dispatch)
        if type(record) is not _TerminalDispatchRecord or record.consumed:
            raise PaperExecutionCapabilityError(
                "baseline terminal dispatch is unknown or already consumed"
            )
        record.consumed = True
        expected_binding = record.binding
        expected_gateway = record.gateway
        expected_context = record.runtime_context
        expected_session = record.active_session
        executor = cast("PaperExecutor", record.executor)
        portfolio_id = record.portfolio_id
        expected_fingerprint = record.fingerprint
        record.binding = None
        record.gateway = None
        record.runtime_context = None
        record.active_session = None
    fingerprint = _fingerprint_order(order)
    if (
        type(binding) is not _GatewayBaselineExecutionBinding
        or expected_binding is not binding
        or expected_gateway is not gateway
        or expected_context is not runtime_context
        or expected_session is not active_session
        or fingerprint != expected_fingerprint
    ):
        raise PaperExecutionCapabilityError("baseline terminal dispatch does not match attempt")
    capability = _PaperExecutionCapability(_token=_CAPABILITY_TOKEN)
    with _REGISTRY_LOCK:
        _CAPABILITIES[capability] = _CapabilityRecord(
            authority=binding,
            executor=executor,
            portfolio_id=portfolio_id,
            kind=_CapabilityKind.BASELINE_ENTRY,
            fingerprint=fingerprint,
            pre_position_quantity=None,
        )
    return executor._place_simple_order(order, _capability=capability)


def _issue_gateway_reduction_terminal_dispatch(
    authority: object,
    *,
    submitter: object,
    executor: object,
    coordinator: object,
    final_allocation: object,
    descriptor: object,
    contract: object,
    order: object,
    pre_position_quantity: Decimal,
) -> _ReductionTerminalDispatch:
    """Register one dispatch after the coordinator envelope was claimed."""

    from .safety.runtime import _consume_claimed_paper_submission_allocation

    try:
        _consume_claimed_paper_submission_allocation(
            final_allocation,
            coordinator=coordinator,
            descriptor=descriptor,
            contract=contract,
        )
    except (RuntimeError, TypeError) as exc:
        raise PaperExecutionCapabilityError(
            "reduction terminal dispatch lacks exact final allocation"
        ) from exc
    fingerprint = _fingerprint_order(order)
    try:
        from .paper_reduction_submitter import _map_order

        expected_fingerprint = _fingerprint_order(_map_order(descriptor, contract))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PaperExecutionCapabilityError(
            "reduction final allocation is not terminally representable"
        ) from exc
    if fingerprint != expected_fingerprint:
        raise PaperExecutionCapabilityError(
            "reduction order does not match the claimed final allocation"
        )
    _validate_reduction_bounds(fingerprint, pre_position_quantity)
    with _REGISTRY_LOCK:
        authority_record = (
            _REDUCTION_AUTHORITIES.get(authority)
            if type(authority) is PaperReductionExecutionAuthority
            else None
        )
        if (
            type(authority_record) is not _ReductionAuthorityRecord
            or authority_record.submitter is not submitter
            or authority_record.executor is not executor
            or authority_record.coordinator is not coordinator
        ):
            raise PaperExecutionCapabilityError("reduction terminal issuer is not bound")
        dispatch = _ReductionTerminalDispatch(_token=_TERMINAL_DISPATCH_TOKEN)
        _REDUCTION_DISPATCHES[dispatch] = _TerminalDispatchRecord(
            binding=authority,
            gateway=authority_record.gateway,
            runtime_context=authority_record.runtime_context,
            active_session=None,
            submitter=submitter,
            coordinator=coordinator,
            executor=executor,
            portfolio_id=authority_record.portfolio_id,
            fingerprint=fingerprint,
            pre_position_quantity=pre_position_quantity,
        )
    return dispatch


def _submit_gateway_reduction_once(
    authority: object,
    dispatch: object,
    *,
    submitter: object,
    order: object,
    pre_position_quantity: Decimal,
):
    """Atomically burn one submitter-issued reduction dispatch."""

    if type(dispatch) is not _ReductionTerminalDispatch:
        raise PaperExecutionCapabilityError("reduction terminal dispatch is invalid")
    with _REGISTRY_LOCK:
        record = _REDUCTION_DISPATCHES.get(dispatch)
        if type(record) is not _TerminalDispatchRecord or record.consumed:
            raise PaperExecutionCapabilityError(
                "reduction terminal dispatch is unknown or already consumed"
            )
        record.consumed = True
        expected_authority = record.binding
        expected_submitter = record.submitter
        executor = cast("PaperExecutor", record.executor)
        portfolio_id = record.portfolio_id
        expected_fingerprint = record.fingerprint
        expected_pre_position = record.pre_position_quantity
        record.binding = None
        record.gateway = None
        record.runtime_context = None
        record.submitter = None
        record.coordinator = None
    fingerprint = _fingerprint_order(order)
    _validate_reduction_bounds(fingerprint, pre_position_quantity)
    if (
        type(authority) is not PaperReductionExecutionAuthority
        or expected_authority is not authority
        or expected_submitter is not submitter
        or fingerprint != expected_fingerprint
        or pre_position_quantity != expected_pre_position
    ):
        raise PaperExecutionCapabilityError("reduction terminal dispatch does not match attempt")
    capability = _PaperExecutionCapability(_token=_CAPABILITY_TOKEN)
    with _REGISTRY_LOCK:
        _CAPABILITIES[capability] = _CapabilityRecord(
            authority=authority,
            executor=executor,
            portfolio_id=portfolio_id,
            kind=_CapabilityKind.REDUCTION,
            fingerprint=fingerprint,
            pre_position_quantity=pre_position_quantity,
        )
    return executor._place_simple_order(order, _capability=capability)


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


def consume_paper_execution_capability(
    executor: object,
    order: object,
    capability: object,
) -> None:
    """Consume and validate exact authority before terminal fill calculation."""

    if type(capability) is not _PaperExecutionCapability:
        raise PaperExecutionCapabilityError(
            "terminal paper execution requires an exact submission capability"
        )
    with _REGISTRY_LOCK:
        record = _CAPABILITIES.get(capability)
        if type(record) is not _CapabilityRecord or record.consumed:
            raise PaperExecutionCapabilityError(
                "paper execution capability is unknown or already consumed"
            )
        # Burn first: mismatch, substitution, and downstream failure are all
        # one-shot and cannot be retried with the same authority.
        record.consumed = True

    fingerprint = _fingerprint_order(order)
    if record.executor is not executor or record.fingerprint != fingerprint:
        raise PaperExecutionCapabilityError(
            "paper execution capability does not match executor or order"
        )
    if record.kind is _CapabilityKind.BASELINE_ENTRY:
        if fingerprint.side != "BUY" or fingerprint.take_profit is not None:
            raise PaperExecutionCapabilityError("baseline entry capability is malformed")
        return
    if record.kind is not _CapabilityKind.REDUCTION:
        raise PaperExecutionCapabilityError("paper execution capability kind is unsupported")
    if type(record.pre_position_quantity) is not Decimal:
        raise PaperExecutionCapabilityError("reduction capability lacks exact position evidence")
    _validate_reduction_bounds(fingerprint, record.pre_position_quantity)


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
