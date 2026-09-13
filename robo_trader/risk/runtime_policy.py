"""Bind explicit runtime settings to the non-authorizing exact entry contract."""

from datetime import timedelta
from decimal import Decimal
from typing import Mapping

from robo_trader.config import ENTRY_POLICY_KEYS, Config
from robo_trader.safety.models import parse_fixed_decimal

from .entry_contract import EntryRiskLimits


def _configured_decimal(value, name):
    # These are already-parsed legacy configuration scalars, never prices,
    # balances or execution quantities. Preserve their public decimal spelling.
    if type(value) not in (int, float, Decimal):
        raise ValueError(f"{name} requires an explicit numeric policy value")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _setting(values, name, *, default=None):
    value = values.get(name, default)
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} requires an explicit fixed-decimal value")
    try:
        return parse_fixed_decimal(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite fixed-decimal value") from exc


def _age(values, name):
    seconds = _setting(values, name, default="5")
    if not 0 < seconds <= 5:
        raise ValueError(f"{name} must be positive and no greater than 5 seconds")
    numerator, denominator = seconds.as_integer_ratio()
    microseconds, remainder = divmod(numerator * 1_000_000, denominator)
    if remainder:
        raise ValueError(f"{name} must have exact microsecond precision")
    return timedelta(microseconds=microseconds)


def build_runtime_entry_risk_limits(config: Config, settings: Mapping[str, str]) -> EntryRiskLimits:
    """Build current per-portfolio limits; missing policy never means unlimited.

    The three additional exposure/liquidity settings have no inferred defaults.
    Freshness defaults to the gateway's conservative five-second envelope.
    Returning limits does not enable strategy, risk, or submission authority.
    """
    if type(config) is not Config:
        raise ValueError("exact Config is required for entry policy")
    config.validate_config_consistency()
    values = dict(settings)
    if set(values) - set(ENTRY_POLICY_KEYS):
        raise ValueError("unknown entry risk policy setting")
    portfolio = _setting(values, ENTRY_POLICY_KEYS[0])
    liquidity = _setting(values, ENTRY_POLICY_KEYS[1])
    participation = _setting(values, ENTRY_POLICY_KEYS[2])
    risk = config.risk
    if risk.max_daily_notional is None:
        raise ValueError("Gate-A daily notional limit cannot be disabled")
    return EntryRiskLimits(
        max_position_fraction=_configured_decimal(risk.max_position_pct, "max_position_pct"),
        max_sector_fraction=_configured_decimal(
            risk.max_sector_exposure_pct, "max_sector_exposure_pct"
        ),
        max_portfolio_gross_fraction=portfolio,
        max_absolute_correlation=min(
            _configured_decimal(risk.correlation_limit, "correlation_limit"),
            _configured_decimal(config.correlation.max_correlation, "max_correlation"),
        ),
        minimum_average_daily_dollar_volume_usd=liquidity,
        max_order_fraction_of_daily_dollar_volume=participation,
        max_daily_notional_usd=_configured_decimal(risk.max_daily_notional, "max_daily_notional"),
        max_order_notional_usd=(
            None
            if risk.max_order_notional is None
            else _configured_decimal(risk.max_order_notional, "max_order_notional")
        ),
        max_open_positions=risk.max_open_positions,
        max_account_leverage=_configured_decimal(risk.max_leverage, "max_leverage"),
        max_quote_age=_age(values, ENTRY_POLICY_KEYS[3]),
        max_account_evidence_age=_age(values, ENTRY_POLICY_KEYS[4]),
    )


def build_portfolio_entry_risk_limits(
    config: Config,
    portfolio_id: str,
    *,
    max_order_notional=None,
    max_daily_notional=None,
    max_correlation=None,
) -> EntryRiskLimits:
    """Resolve one active portfolio and current runner overrides, without caching."""
    if type(config) is not Config:
        raise ValueError("exact Config is required for entry policy")
    if type(portfolio_id) is not str or not portfolio_id or portfolio_id != portfolio_id.strip():
        raise ValueError("entry portfolio identifier is invalid")
    portfolios = config.portfolio_configs
    if type(portfolios) is not list or any(type(item) is not dict for item in portfolios):
        raise ValueError("entry portfolio configuration is invalid")
    matches = [item for item in portfolios if item.get("id") == portfolio_id]
    if len(matches) != 1 or matches[0].get("active", True) is not True:
        raise ValueError("entry portfolio must uniquely identify an active configuration")
    selected = matches[0]
    resolved = config.model_copy(deep=True)
    for field in ("max_position_pct", "max_open_positions"):
        value = selected.get(field)
        if value is not None:
            setattr(resolved.risk, field, value)
    for field, value in (
        ("max_order_notional", max_order_notional),
        ("max_daily_notional", max_daily_notional),
    ):
        if value is not None:
            setattr(resolved.risk, field, _configured_decimal(value, field))
    if max_correlation is not None:
        correlation = _configured_decimal(max_correlation, "runner max_correlation")
        if not 0 <= correlation <= 1:
            raise ValueError("runner max_correlation must be between zero and one")
        resolved.correlation.max_correlation = min(
            correlation,
            _configured_decimal(resolved.correlation.max_correlation, "max_correlation"),
        )
    return resolved.build_entry_risk_limits()
