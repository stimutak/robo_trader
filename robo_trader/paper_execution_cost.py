"""Deterministic local-paper pricing; cost estimates grant no execution authority."""

from dataclasses import dataclass
from decimal import Decimal

LOCAL_PAPER_COMMISSION_MINOR = 0


def exact_paper_fill_price(reference: Decimal, slippage_bps: Decimal, side: str) -> Decimal:
    """Apply signed slippage and round once, half-even, to the 0.0001 USD tick."""
    if type(reference) is not Decimal or not reference.is_finite() or reference <= 0:
        raise ValueError("paper reference price must be an exact positive Decimal")
    if (
        type(slippage_bps) is not Decimal
        or not slippage_bps.is_finite()
        or not 0 <= slippage_bps < 10000
    ):
        raise ValueError("paper slippage must be exact and between zero and 10000 bps")
    if type(side) is not str or side not in {"BUY", "BUY_TO_COVER", "SELL"}:
        raise ValueError("unsupported paper pricing side")
    price_numerator, price_denominator = reference.as_integer_ratio()
    slip_numerator, slip_denominator = slippage_bps.as_integer_ratio()
    signed_slip = slip_numerator if side in {"BUY", "BUY_TO_COVER"} else -slip_numerator
    # Scaling by 10,000 ticks/USD cancels the basis-point denominator exactly.
    numerator = price_numerator * (10000 * slip_denominator + signed_slip)
    denominator = price_denominator * slip_denominator
    ticks, remainder = divmod(numerator, denominator)
    if remainder * 2 > denominator or (remainder * 2 == denominator and ticks % 2):
        ticks += 1
    if ticks <= 0:
        raise ValueError("paper fill is not positive after tick rounding")
    return Decimal((0, tuple(int(digit) for digit in str(ticks)), -4))


@dataclass(frozen=True, slots=True)
class PaperEntryCost:
    reference_price_usd: Decimal
    slippage_bps: Decimal
    modeled_fill_price_usd: Decimal
    price_ceiling_usd: Decimal
    commission_minor: int


def paper_entry_cost(reference: Decimal, slippage_bps: Decimal) -> PaperEntryCost:
    """Estimate the current explicit zero-commission paper model conservatively."""
    fill = exact_paper_fill_price(reference, slippage_bps, "BUY")
    return PaperEntryCost(
        reference, slippage_bps, fill, max(reference, fill), LOCAL_PAPER_COMMISSION_MINOR
    )
