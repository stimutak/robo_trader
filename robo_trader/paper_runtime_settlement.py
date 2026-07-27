"""Runner-owned projection boundary for committed local-paper settlements.

The authoritative SQLite transaction is committed before this boundary runs.
The participant then projects that exact receipt into the runner, Portfolio,
advanced-risk, and protective-stop views.  A reservation may be released only
after the participant returns an exact projection matching the receipt.

This module deliberately contains no broker code and makes no IBKR-truth claim.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Awaitable, Callable, Optional

from .paper_terminal_settlement import (
    PaperTerminalSettlementReceipt,
    assert_producer_owned_paper_terminal_settlement_receipt,
)
from .safety.models import _exact_decimal_subtract, _strict_decimal

_SETTLEMENT_ID_RE = re.compile(r"^pset-[0-9a-f]{32}$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_ABSOLUTE_TOLERANCE = Decimal("0.000000001")
_FLOAT_RELATIVE_TOLERANCE = Decimal("0.000000000001")


class PaperRuntimeSettlementError(RuntimeError):
    """A committed settlement could not be projected without ambiguity."""


def _exact_integral(value: object, field_name: str) -> Decimal:
    if type(value) is not Decimal or not value.is_finite() or value != value.to_integral_value():
        raise PaperRuntimeSettlementError(f"{field_name} must be an exact integral Decimal")
    return value


def bounded_float_projection_matches(actual: Decimal, expected: Decimal) -> bool:
    """Compare Decimal renderings of inherently-float runtime components."""

    if (
        type(actual) is not Decimal
        or not actual.is_finite()
        or type(expected) is not Decimal
        or not expected.is_finite()
    ):
        return False
    tolerance = max(
        _FLOAT_ABSOLUTE_TOLERANCE,
        abs(expected) * _FLOAT_RELATIVE_TOLERANCE,
    )
    return abs(actual - expected) <= tolerance


@dataclass(frozen=True, slots=True)
class PaperRuntimeProjection:
    """Post-commit state observed by the registered runner participant.

    Durable account values and signed quantities remain exact. Fields sourced
    from legacy float ledgers are finite Decimal renderings verified only by
    :func:`bounded_float_projection_matches`.
    """

    settlement_id: str
    settlement_receipt_fingerprint: str
    portfolio_id: str
    symbol: str
    runner_position_quantity: Decimal
    portfolio_position_quantity: Decimal
    account_cash: Decimal
    account_realized_pnl: Decimal
    risk_visible_daily_pnl_before: Decimal
    risk_visible_daily_pnl: Decimal
    protective_stop_quantity: Optional[Decimal]
    advanced_risk_position_quantity: Optional[Decimal]
    advanced_risk_position_avg_price: Optional[Decimal]
    advanced_risk_total_pnl: Optional[Decimal]
    advanced_risk_daily_pnl: Optional[Decimal]

    def __post_init__(self) -> None:
        if type(self.settlement_id) is not str or not _SETTLEMENT_ID_RE.fullmatch(
            self.settlement_id
        ):
            raise PaperRuntimeSettlementError("projection settlement_id is malformed")
        if type(self.settlement_receipt_fingerprint) is not str or not _FINGERPRINT_RE.fullmatch(
            self.settlement_receipt_fingerprint
        ):
            raise PaperRuntimeSettlementError(
                "projection settlement receipt fingerprint is malformed"
            )
        for field_name in ("portfolio_id", "symbol"):
            value = getattr(self, field_name)
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or not value.isprintable()
            ):
                raise PaperRuntimeSettlementError(f"projection {field_name} is malformed")
        _exact_integral(
            self.runner_position_quantity,
            "runner_position_quantity",
        )
        _exact_integral(
            self.portfolio_position_quantity,
            "portfolio_position_quantity",
        )
        for field_name in ("account_cash", "account_realized_pnl"):
            try:
                _strict_decimal(getattr(self, field_name), field_name)
            except Exception as exc:
                raise PaperRuntimeSettlementError(
                    f"{field_name} must be an exact finite Decimal"
                ) from exc
        for field_name in ("risk_visible_daily_pnl_before", "risk_visible_daily_pnl"):
            value = getattr(self, field_name)
            if type(value) is not Decimal or not value.is_finite():
                raise PaperRuntimeSettlementError(
                    f"{field_name} must be a finite Decimal float projection"
                )
        if self.protective_stop_quantity is not None:
            _exact_integral(
                self.protective_stop_quantity,
                "protective_stop_quantity",
            )
        if self.advanced_risk_position_quantity is not None:
            _exact_integral(
                self.advanced_risk_position_quantity,
                "advanced_risk_position_quantity",
            )
        for field_name in (
            "advanced_risk_position_avg_price",
            "advanced_risk_total_pnl",
            "advanced_risk_daily_pnl",
        ):
            value = getattr(self, field_name)
            if value is not None:
                if type(value) is not Decimal or not value.is_finite():
                    raise PaperRuntimeSettlementError(
                        f"{field_name} must be a finite Decimal float projection"
                    )


ProjectionCallback = Callable[
    [PaperTerminalSettlementReceipt],
    Awaitable[PaperRuntimeProjection],
]
QuarantineCallback = Callable[[str], None]


class PaperRuntimeSettlementParticipant:
    """Narrow adapter for one runner's post-commit projection and freeze latch."""

    __slots__ = ("_apply_callback", "_portfolio_id", "_quarantine_callback")

    def __init__(
        self,
        portfolio_id: str,
        *,
        apply_callback: ProjectionCallback,
        quarantine_callback: QuarantineCallback,
    ) -> None:
        if (
            type(portfolio_id) is not str
            or not portfolio_id
            or portfolio_id != portfolio_id.strip()
            or not portfolio_id.isprintable()
        ):
            raise PaperRuntimeSettlementError("participant portfolio_id is malformed")
        if not callable(apply_callback) or not callable(quarantine_callback):
            raise PaperRuntimeSettlementError(
                "participant requires projection and quarantine callbacks"
            )
        self._portfolio_id = portfolio_id
        self._apply_callback = apply_callback
        self._quarantine_callback = quarantine_callback

    @property
    def portfolio_id(self) -> str:
        return self._portfolio_id

    async def apply_and_verify(
        self,
        receipt: PaperTerminalSettlementReceipt,
    ) -> PaperRuntimeProjection:
        """Apply one producer-owned receipt and prove every required projection."""

        if type(receipt) is not PaperTerminalSettlementReceipt:
            raise PaperRuntimeSettlementError(
                "participant requires an exact paper settlement receipt"
            )
        try:
            assert_producer_owned_paper_terminal_settlement_receipt(receipt)
        except RuntimeError as exc:
            raise PaperRuntimeSettlementError("participant receipt is not producer-owned") from exc
        if receipt.request.portfolio_id != self._portfolio_id:
            raise PaperRuntimeSettlementError("participant portfolio does not match settlement")

        pending = self._apply_callback(receipt)
        if not inspect.isawaitable(pending):
            raise PaperRuntimeSettlementError("projection callback did not return an awaitable")
        projection = await pending
        if type(projection) is not PaperRuntimeProjection:
            raise PaperRuntimeSettlementError("projection callback returned an unexpected result")

        request = receipt.request
        expected_quantity = request.expected_post_position_quantity
        exact_pairs = (
            (projection.settlement_id, receipt.settlement_id, "settlement"),
            (
                projection.settlement_receipt_fingerprint,
                receipt.fingerprint(),
                "receipt fingerprint",
            ),
            (projection.portfolio_id, request.portfolio_id, "portfolio"),
            (projection.symbol, request.symbol, "symbol"),
            (
                projection.runner_position_quantity,
                expected_quantity,
                "runner position",
            ),
            (
                projection.portfolio_position_quantity,
                expected_quantity,
                "portfolio position",
            ),
            (
                projection.account_cash,
                request.expected_post_cash,
                "account cash",
            ),
            (
                projection.account_realized_pnl,
                request.expected_post_realized_pnl,
                "account realized P&L",
            ),
        )
        failures = [
            f"{label} projection mismatch"
            for actual, expected, label in exact_pairs
            if actual != expected
        ]
        expected_daily_delta = _exact_decimal_subtract(
            request.expected_post_daily_pnl,
            request.expected_pre_daily_pnl,
            "settlement daily P&L delta",
        )
        with localcontext() as context:
            context.prec = 64
            projected_daily_delta = (
                projection.risk_visible_daily_pnl - projection.risk_visible_daily_pnl_before
            )
        if not bounded_float_projection_matches(
            projected_daily_delta,
            expected_daily_delta,
        ):
            failures.append("risk-visible daily P&L omitted the exact settlement delta")
        if not bounded_float_projection_matches(
            projection.risk_visible_daily_pnl_before,
            request.expected_pre_daily_pnl,
        ):
            failures.append("risk-visible pre-daily P&L projection mismatch")
        if not bounded_float_projection_matches(
            projection.risk_visible_daily_pnl,
            request.expected_post_daily_pnl,
        ):
            failures.append("risk-visible post-daily P&L projection mismatch")
        if expected_quantity.is_zero():
            if projection.protective_stop_quantity is not None:
                failures.append("flat settlement retained a protective stop")
        elif projection.protective_stop_quantity != expected_quantity:
            failures.append("protective stop does not cover the exact remaining position")
        if (
            projection.advanced_risk_position_quantity is not None
            and projection.advanced_risk_position_quantity != expected_quantity
        ):
            failures.append("advanced-risk position projection mismatch")
        advanced_quantity = projection.advanced_risk_position_quantity
        advanced_avg_price = projection.advanced_risk_position_avg_price
        advanced_total_pnl = projection.advanced_risk_total_pnl
        advanced_daily_pnl = projection.advanced_risk_daily_pnl
        advanced_fields = (
            advanced_quantity,
            advanced_avg_price,
            advanced_total_pnl,
            advanced_daily_pnl,
        )
        if all(value is None for value in advanced_fields):
            pass
        elif advanced_quantity is None or advanced_total_pnl is None or advanced_daily_pnl is None:
            failures.append("advanced-risk projection fields are incomplete")
        else:
            if expected_quantity.is_zero():
                if advanced_avg_price is not None:
                    failures.append("flat settlement retained advanced-risk cost basis")
            elif (
                request.expected_position_cost_basis is None
                or advanced_avg_price is None
                or not bounded_float_projection_matches(
                    advanced_avg_price,
                    request.expected_position_cost_basis,
                )
            ):
                failures.append("advanced-risk remaining cost basis projection mismatch")
            if not bounded_float_projection_matches(
                advanced_total_pnl,
                request.expected_post_realized_pnl,
            ):
                failures.append("advanced-risk total P&L projection mismatch")
            if not bounded_float_projection_matches(
                advanced_daily_pnl,
                projection.risk_visible_daily_pnl,
            ):
                failures.append("advanced-risk daily P&L differs from runner daily P&L")
        if failures:
            raise PaperRuntimeSettlementError("; ".join(failures))
        return projection

    def latch_quarantine(self, reason: str) -> None:
        """Synchronously freeze entries without waiting on runner admission locks."""

        reason_text = str(reason or "paper settlement failure").strip()
        if not reason_text:
            reason_text = "paper settlement failure"
        result = self._quarantine_callback(reason_text)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise PaperRuntimeSettlementError("quarantine callback must latch synchronously")
