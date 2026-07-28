"""One-shot capabilities for the terminal local-paper execution sink.

Only the account gateway binds an authority to a registered ``PaperExecutor``.
The authority never exposes a reusable execution method: it mints an immutable
capability for one exact order and the terminal sink consumes that capability
before performing any other work.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


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


@dataclass(slots=True)
class _CapabilityRecord:
    authority: "PaperExecutionAuthority"
    executor: object
    portfolio_id: str
    kind: _CapabilityKind
    fingerprint: _OrderFingerprint
    pre_position_quantity: Decimal | None
    consumed: bool = False


_BIND_TOKEN = object()
_CAPABILITY_TOKEN = object()
_REGISTRY_LOCK = threading.Lock()
_CAPABILITIES: weakref.WeakKeyDictionary[_PaperExecutionCapability, _CapabilityRecord] = (
    weakref.WeakKeyDictionary()
)


class PaperExecutionAuthority:
    """Sealed executor/portfolio binding owned by the account gateway."""

    __slots__ = ("__executor", "__portfolio_id", "__sealed")

    def __init__(
        self,
        executor: object,
        portfolio_id: str,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _BIND_TOKEN:
            raise PaperExecutionCapabilityError(
                "paper execution authority must be bound by the account gateway"
            )
        if (
            not isinstance(portfolio_id, str)
            or not portfolio_id
            or portfolio_id.strip() != portfolio_id
        ):
            raise PaperExecutionCapabilityError("paper execution portfolio scope is malformed")
        object.__setattr__(self, "_PaperExecutionAuthority__executor", executor)
        object.__setattr__(self, "_PaperExecutionAuthority__portfolio_id", portfolio_id)
        object.__setattr__(self, "_PaperExecutionAuthority__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_PaperExecutionAuthority__sealed", False):
            raise AttributeError("PaperExecutionAuthority is sealed")
        object.__setattr__(self, name, value)

    def _is_bound_to(self, executor: object, portfolio_id: str) -> bool:
        return self.__executor is executor and self.__portfolio_id == portfolio_id

    def _submit_baseline_once(self, order: object):
        fingerprint = _fingerprint_order(order)
        if fingerprint.side != "BUY" or fingerprint.take_profit is not None:
            raise PaperExecutionCapabilityError(
                "baseline authority admits only simple long BUY orders"
            )
        capability = self._mint(
            kind=_CapabilityKind.BASELINE_ENTRY,
            fingerprint=fingerprint,
            pre_position_quantity=None,
        )
        return self.__executor._place_simple_order(order, _capability=capability)

    def _submit_reduction_once(self, order: object, *, pre_position_quantity: Decimal):
        fingerprint = _fingerprint_order(order)
        _validate_reduction_bounds(fingerprint, pre_position_quantity)
        capability = self._mint(
            kind=_CapabilityKind.REDUCTION,
            fingerprint=fingerprint,
            pre_position_quantity=pre_position_quantity,
        )
        return self.__executor._place_simple_order(order, _capability=capability)

    def _mint(
        self,
        *,
        kind: _CapabilityKind,
        fingerprint: _OrderFingerprint,
        pre_position_quantity: Decimal | None,
    ) -> _PaperExecutionCapability:
        capability = _PaperExecutionCapability(_token=_CAPABILITY_TOKEN)
        record = _CapabilityRecord(
            authority=self,
            executor=self.__executor,
            portfolio_id=self.__portfolio_id,
            kind=kind,
            fingerprint=fingerprint,
            pre_position_quantity=pre_position_quantity,
        )
        with _REGISTRY_LOCK:
            _CAPABILITIES[capability] = record
        return capability


def _bind_paper_execution_authority(
    executor: object,
    portfolio_id: str,
) -> PaperExecutionAuthority:
    """Private account-gateway binding hook."""

    return PaperExecutionAuthority(executor, portfolio_id, _token=_BIND_TOKEN)


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
