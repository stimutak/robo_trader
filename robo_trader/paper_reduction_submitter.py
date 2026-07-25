"""One-shot bridge from consumed safety authority to paper execution.

The adapter accepts only an exact coordinator-issued envelope. Claiming that
envelope is the first operation, so exceptions and malformed executor results
cannot make the same authority reusable.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation

from .execution import ExecutionResult, Order, PaperExecutor
from .runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
from .safety.models import (
    EvidenceStatus,
    OrderSide,
    OrderType,
    SubmissionDescriptor,
    TimeInForce,
    ValidationError,
)
from .safety.runtime import (
    AuthoritativeContract,
    ConsumedPaperSubmissionEnvelope,
    SafetyRuntimeCoordinator,
)

_BIND_TOKEN = object()
_LIMIT_PRICE_TICK = Decimal("0.0001")
_MAX_LIMIT_PRICE = Decimal("1000000")
_MAX_QUANTITY = Decimal("2147483647")


class PaperReductionSubmissionError(RuntimeError):
    """The sealed paper submission boundary rejected unsafe input or output."""


class PaperReductionSubmitter:
    """Capability-narrow adapter for exactly one simple paper submission."""

    __slots__ = ("__coordinator", "__executor", "__sealed")

    def __init__(
        self,
        executor: PaperExecutor,
        coordinator: SafetyRuntimeCoordinator,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _BIND_TOKEN:
            raise PaperReductionSubmissionError(
                "PaperReductionSubmitter must be created by its private bind function"
            )
        if type(executor) is not PaperExecutor:
            raise PaperReductionSubmissionError("executor must be exactly PaperExecutor")
        if type(coordinator) is not SafetyRuntimeCoordinator:
            raise PaperReductionSubmissionError(
                "coordinator must be exactly SafetyRuntimeCoordinator"
            )
        if not coordinator.started:
            raise PaperReductionSubmissionError("coordinator must be started before binding")
        self.__executor = executor
        self.__coordinator = coordinator
        self.__sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_PaperReductionSubmitter__sealed", False):
            raise AttributeError("PaperReductionSubmitter is sealed")
        object.__setattr__(self, name, value)

    def _submit_once(
        self,
        envelope: ConsumedPaperSubmissionEnvelope,
    ) -> ExecutionResult:
        """Claim and submit one exact coordinator-consumed envelope."""

        executor = self.__executor
        coordinator = self.__coordinator
        if type(executor) is not PaperExecutor:
            raise PaperReductionSubmissionError("bound executor is no longer PaperExecutor")
        if type(coordinator) is not SafetyRuntimeCoordinator:
            raise PaperReductionSubmissionError(
                "bound coordinator is no longer SafetyRuntimeCoordinator"
            )

        # Claim first. Every subsequent failure remains one-shot and cannot
        # cause a second paper fill through replay or concurrent reuse.
        try:
            descriptor, contract = coordinator._claim_consumed_paper_submission(envelope)
        except (RuntimeError, TypeError, ValidationError) as exc:
            raise PaperReductionSubmissionError("paper submission envelope claim failed") from exc

        safe_descriptor = _snapshot_descriptor(descriptor)
        safe_contract = _snapshot_contract(contract)
        _validate_executor_configuration(executor)
        order = _map_order(safe_descriptor, safe_contract)

        # This is deliberately the adapter's sole execution call. It bypasses
        # legacy soft gates only after final safety revalidation and durable
        # permit consumption; there is no validation, smart, fallback, or retry.
        result = executor._place_simple_order(order)
        _validate_execution_result(result)
        return result


def _bind_paper_reduction_submitter(
    executor: PaperExecutor,
    coordinator: SafetyRuntimeCoordinator,
) -> PaperReductionSubmitter:
    """Bind the private one-shot adapter to exact executor and coordinator."""

    return PaperReductionSubmitter(executor, coordinator, _token=_BIND_TOKEN)


def _validate_executor_configuration(executor: PaperExecutor) -> None:
    slippage_bps = executor.slippage_bps
    if type(slippage_bps) is not float or not math.isfinite(slippage_bps):
        raise PaperReductionSubmissionError("paper slippage must be a finite float")
    if slippage_bps < 0.0 or slippage_bps >= 10_000.0:
        raise PaperReductionSubmissionError("paper slippage is outside safe bounds")


def _validate_execution_result(result: object) -> None:
    if type(result) is not ExecutionResult:
        raise PaperReductionSubmissionError("PaperExecutor returned an unexpected execution result")
    if type(result.ok) is not bool:
        raise PaperReductionSubmissionError("execution result ok must be exactly bool")
    if (
        type(result.message) is not str
        or not result.message
        or result.message != result.message.strip()
        or len(result.message) > 512
        or not result.message.isprintable()
    ):
        raise PaperReductionSubmissionError("execution result message is malformed")
    if result.ok:
        if (
            type(result.fill_price) is not float
            or not math.isfinite(result.fill_price)
            or result.fill_price <= 0.0
        ):
            raise PaperReductionSubmissionError(
                "successful execution result requires a finite positive float fill"
            )
    elif result.fill_price is not None:
        raise PaperReductionSubmissionError("rejected execution result must not carry a fill price")


def _snapshot_descriptor(descriptor: SubmissionDescriptor) -> SubmissionDescriptor:
    if type(descriptor) is not SubmissionDescriptor:
        raise PaperReductionSubmissionError("descriptor must be exactly SubmissionDescriptor")

    try:
        payload_before = descriptor.canonical_payload()
        snapshot = SubmissionDescriptor(
            execution_domain_scope=descriptor.execution_domain_scope,
            account_scope=descriptor.account_scope,
            con_id=descriptor.con_id,
            side=descriptor.side,
            quantity=descriptor.quantity,
            order_type=descriptor.order_type,
            limit_price=descriptor.limit_price,
            stop_price=descriptor.stop_price,
            time_in_force=descriptor.time_in_force,
            outside_regular_hours=descriptor.outside_regular_hours,
            order_ref=descriptor.order_ref,
            attempt_number=descriptor.attempt_number,
            slice_count=descriptor.slice_count,
            bracket=descriptor.bracket,
            schema_version=descriptor.schema_version,
        )
        payload_after = descriptor.canonical_payload()
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise PaperReductionSubmissionError("descriptor failed exact revalidation") from exc

    if payload_before != payload_after or snapshot.canonical_payload() != payload_before:
        raise PaperReductionSubmissionError("descriptor changed while being validated")
    return snapshot


def _contract_values(contract: AuthoritativeContract) -> tuple[object, ...]:
    return (
        contract.con_id,
        contract.symbol,
        contract.local_symbol,
        contract.security_type,
        contract.currency,
        contract.exchange,
        contract.primary_exchange,
        contract.trading_class,
        contract.observed_at,
        contract.snapshot_id,
        contract.source,
        contract.broker_timestamp,
        contract.retrieval_timestamp,
        contract.transport_generation,
        contract.status,
    )


def _snapshot_contract(contract: AuthoritativeContract) -> AuthoritativeContract:
    if type(contract) is not AuthoritativeContract:
        raise PaperReductionSubmissionError("contract must be exactly AuthoritativeContract")

    try:
        values_before = _contract_values(contract)
        snapshot = AuthoritativeContract(
            con_id=contract.con_id,
            symbol=contract.symbol,
            local_symbol=contract.local_symbol,
            security_type=contract.security_type,
            currency=contract.currency,
            exchange=contract.exchange,
            primary_exchange=contract.primary_exchange,
            trading_class=contract.trading_class,
            observed_at=contract.observed_at,
            snapshot_id=contract.snapshot_id,
            source=contract.source,
            broker_timestamp=contract.broker_timestamp,
            retrieval_timestamp=contract.retrieval_timestamp,
            transport_generation=contract.transport_generation,
            status=contract.status,
        )
        values_after = _contract_values(contract)
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise PaperReductionSubmissionError("contract failed exact revalidation") from exc

    if values_before != values_after or _contract_values(snapshot) != values_before:
        raise PaperReductionSubmissionError("contract changed while being validated")
    if snapshot.status is not EvidenceStatus.AUTHORITATIVE:
        raise PaperReductionSubmissionError("contract evidence is not authoritative")
    return snapshot


def _exact_limit_price(value: Decimal) -> Decimal:
    if type(value) is not Decimal or not value.is_finite() or value <= 0:
        raise PaperReductionSubmissionError("LIMIT price must be a finite positive Decimal")
    if value > _MAX_LIMIT_PRICE:
        raise PaperReductionSubmissionError("LIMIT price exceeds paper adapter bound")
    try:
        if value != value.quantize(_LIMIT_PRICE_TICK):
            raise PaperReductionSubmissionError(
                "LIMIT price is not quantized to the supported tick"
            )
    except InvalidOperation as exc:
        raise PaperReductionSubmissionError("LIMIT price cannot be quantized safely") from exc
    return value


def _map_order(
    descriptor: SubmissionDescriptor,
    contract: AuthoritativeContract,
) -> Order:
    if descriptor.execution_domain_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE:
        raise PaperReductionSubmissionError("descriptor is not bound to the paper domain")
    if descriptor.con_id != contract.con_id:
        raise PaperReductionSubmissionError("descriptor and contract con_id do not match")
    if descriptor.side not in {OrderSide.SELL, OrderSide.BUY_TO_COVER}:
        raise PaperReductionSubmissionError("only semantic SELL or BUY_TO_COVER is allowed")
    if type(descriptor.quantity) is not Decimal or not descriptor.quantity.is_finite():
        raise PaperReductionSubmissionError("quantity must be an exact finite Decimal")
    if (
        descriptor.quantity <= 0
        or descriptor.quantity > _MAX_QUANTITY
        or descriptor.quantity != descriptor.quantity.to_integral_value()
    ):
        raise PaperReductionSubmissionError("quantity must be a bounded positive integral Decimal")
    if descriptor.attempt_number != 1 or type(descriptor.attempt_number) is not int:
        raise PaperReductionSubmissionError("only submission attempt 1 is allowed")
    if descriptor.slice_count != 1 or type(descriptor.slice_count) is not int:
        raise PaperReductionSubmissionError("only one unsliced order is allowed")
    if descriptor.bracket is not False:
        raise PaperReductionSubmissionError("bracket orders are not allowed")
    if descriptor.time_in_force is not TimeInForce.DAY:
        raise PaperReductionSubmissionError("only DAY time-in-force is representable")
    if descriptor.outside_regular_hours is not False:
        raise PaperReductionSubmissionError("outside-regular-hours routing is not representable")

    price: Decimal | None
    if descriptor.order_type is OrderType.MARKET:
        if descriptor.limit_price is not None or descriptor.stop_price is not None:
            raise PaperReductionSubmissionError("MARKET descriptor carries a price")
        price = None
    elif descriptor.order_type is OrderType.LIMIT:
        if descriptor.limit_price is None or descriptor.stop_price is not None:
            raise PaperReductionSubmissionError("LIMIT descriptor has invalid price fields")
        price = _exact_limit_price(descriptor.limit_price)
    else:
        raise PaperReductionSubmissionError("only MARKET and LIMIT orders are supported")

    return Order(
        symbol=contract.symbol,
        quantity=int(descriptor.quantity),
        side=descriptor.side.value,
        price=price,
        order_ref=descriptor.order_ref,
    )
