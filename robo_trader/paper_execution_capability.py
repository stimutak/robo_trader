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
from typing import Any


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


_REDUCTION_BIND_TOKEN = object()
_BIND_CAPABILITY_TOKEN = object()
_BASELINE_BIND_TOKEN = object()
_CAPABILITY_TOKEN = object()
_REGISTRY_LOCK = threading.Lock()
_CAPABILITIES: weakref.WeakKeyDictionary[_PaperExecutionCapability, _CapabilityRecord] = (
    weakref.WeakKeyDictionary()
)
_GATEWAY_BINDINGS: weakref.WeakKeyDictionary[
    _GatewayExecutionBindingCapability, _GatewayBindingRecord
] = weakref.WeakKeyDictionary()


class PaperReductionExecutionAuthority:
    """Sealed executor/portfolio binding for coordinator-authorized reductions."""

    __slots__ = ("__executor", "__portfolio_id", "__sealed")
    __executor: Any
    __portfolio_id: str
    __sealed: bool

    def __init__(
        self,
        executor: object,
        portfolio_id: str,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _REDUCTION_BIND_TOKEN:
            raise PaperExecutionCapabilityError(
                "paper reduction authority must be bound by its sealed submitter"
            )
        if (
            not isinstance(portfolio_id, str)
            or not portfolio_id
            or portfolio_id.strip() != portfolio_id
        ):
            raise PaperExecutionCapabilityError("paper execution portfolio scope is malformed")
        object.__setattr__(self, "_PaperReductionExecutionAuthority__executor", executor)
        object.__setattr__(self, "_PaperReductionExecutionAuthority__portfolio_id", portfolio_id)
        object.__setattr__(self, "_PaperReductionExecutionAuthority__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_PaperReductionExecutionAuthority__sealed", False):
            raise AttributeError("PaperReductionExecutionAuthority is sealed")
        object.__setattr__(self, name, value)

    def _is_bound_to(self, executor: object, portfolio_id: str) -> bool:
        return self.__executor is executor and self.__portfolio_id == portfolio_id

    def _submit_reduction_once(self, order: object, *, pre_position_quantity: Decimal):
        fingerprint = _fingerprint_order(order)
        _validate_reduction_bounds(fingerprint, pre_position_quantity)
        capability = _PaperExecutionCapability(_token=_CAPABILITY_TOKEN)
        with _REGISTRY_LOCK:
            _CAPABILITIES[capability] = _CapabilityRecord(
                authority=self,
                executor=self.__executor,
                portfolio_id=self.__portfolio_id,
                kind=_CapabilityKind.REDUCTION,
                fingerprint=fingerprint,
                pre_position_quantity=pre_position_quantity,
            )
        return self.__executor._place_simple_order(
            order,
            _capability=capability,
        )


class _GatewayBaselineExecutionBinding:
    """Sealed baseline binding with no submission method of its own."""

    __slots__ = (
        "_executor",
        "_gateway",
        "_portfolio_id",
        "_runtime_context",
        "__sealed",
    )
    _executor: Any
    _gateway: object
    _portfolio_id: str
    _runtime_context: object
    __sealed: bool

    def __init__(
        self,
        *,
        gateway: object,
        runtime_context: object,
        executor: object,
        portfolio_id: str,
        _token: object | None = None,
    ) -> None:
        if _token is not _BASELINE_BIND_TOKEN:
            raise PaperExecutionCapabilityError("baseline execution binding is gateway-only")
        object.__setattr__(self, "_gateway", gateway)
        object.__setattr__(self, "_runtime_context", runtime_context)
        object.__setattr__(self, "_executor", executor)
        object.__setattr__(self, "_portfolio_id", portfolio_id)
        object.__setattr__(self, "_GatewayBaselineExecutionBinding__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_GatewayBaselineExecutionBinding__sealed", False):
            raise AttributeError("GatewayBaselineExecutionBinding is sealed")
        object.__setattr__(self, name, value)


def _bind_paper_reduction_execution_authority(
    executor: object,
    portfolio_id: str,
) -> PaperReductionExecutionAuthority:
    """Private reduction-submitter binding hook."""

    return PaperReductionExecutionAuthority(
        executor,
        portfolio_id,
        _token=_REDUCTION_BIND_TOKEN,
    )


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
    return _GatewayBaselineExecutionBinding(
        gateway=gateway,
        runtime_context=runtime_context,
        executor=executor,
        portfolio_id=portfolio_id,
        _token=_BASELINE_BIND_TOKEN,
    )


def _submit_gateway_baseline_once(
    binding: object,
    *,
    gateway: object,
    runtime_context: object,
    active_session: object,
    order: object,
):
    """Submit a BUY only from the exact active gateway entry session."""

    from .paper_reduction_gateway import (
        PaperReductionGateway,
        _ActiveEntrySession,
    )

    if type(binding) is not _GatewayBaselineExecutionBinding:
        raise PaperExecutionCapabilityError("baseline execution binding is invalid")
    task = asyncio.current_task()
    sessions = getattr(gateway, "_active_entry_sessions", None)
    if (
        type(gateway) is not PaperReductionGateway
        or gateway.started is not True
        or binding._gateway is not gateway
        or binding._runtime_context is not runtime_context
        or getattr(gateway, "_runtime_context", None) is not runtime_context
        or type(active_session) is not _ActiveEntrySession
        or not isinstance(sessions, dict)
        or task is None
        or sessions.get(task) is not active_session
        or active_session.consumed is not True
        or active_session.portfolio_id != binding._portfolio_id
    ):
        raise PaperExecutionCapabilityError("baseline gateway session does not match binding")
    fingerprint = _fingerprint_order(order)
    if (
        fingerprint.side != "BUY"
        or fingerprint.take_profit is not None
        or fingerprint.symbol != active_session.symbol
        or fingerprint.price != active_session.quote.price
    ):
        raise PaperExecutionCapabilityError("baseline order does not match gateway session")
    capability = _PaperExecutionCapability(_token=_CAPABILITY_TOKEN)
    with _REGISTRY_LOCK:
        _CAPABILITIES[capability] = _CapabilityRecord(
            authority=binding,
            executor=binding._executor,
            portfolio_id=binding._portfolio_id,
            kind=_CapabilityKind.BASELINE_ENTRY,
            fingerprint=fingerprint,
            pre_position_quantity=None,
        )
    return binding._executor._place_simple_order(order, _capability=capability)


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
