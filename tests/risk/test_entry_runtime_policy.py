"""Exact entry policy is explicit and derived from the active configuration."""

from datetime import timedelta
from decimal import Decimal, localcontext

import pytest

from robo_trader.config import Config

POLICY = {
    "ENTRY_MAX_PORTFOLIO_GROSS_FRACTION": "0.75",
    "ENTRY_MINIMUM_AVERAGE_DAILY_DOLLAR_VOLUME_USD": "1000000",
    "ENTRY_MAX_ORDER_FRACTION_OF_DAILY_DOLLAR_VOLUME": "0.01",
}


def test_runtime_policy_maps_existing_limits_and_explicit_extra_units():
    config = Config(entry_risk_policy=POLICY)
    with localcontext() as context:
        context.prec = 2
        limits = config.build_entry_risk_limits()
    assert limits.max_position_fraction == Decimal("0.02")
    assert limits.max_sector_fraction == Decimal("0.3")
    assert limits.max_portfolio_gross_fraction == Decimal("0.75")
    assert limits.minimum_average_daily_dollar_volume_usd == Decimal("1000000")
    assert limits.max_daily_notional_usd == Decimal("100000")
    assert limits.max_order_notional_usd == Decimal("10000")
    assert limits.max_account_leverage == Decimal("2")
    assert limits.max_open_positions == 20
    assert limits.max_account_evidence_age == timedelta(seconds=5)


def test_runtime_policy_missing_values_never_mean_unlimited():
    with pytest.raises(ValueError, match="ENTRY_MAX_PORTFOLIO_GROSS_FRACTION"):
        Config().build_entry_risk_limits()


def test_runtime_policy_rejects_disabled_daily_limit_and_above_gate_a_position_cap():
    with pytest.raises(ValueError, match="daily"):
        Config(
            risk={"max_daily_notional": None}, entry_risk_policy=POLICY
        ).build_entry_risk_limits()
    with pytest.raises(ValueError, match="2%"):
        Config(risk={"max_position_pct": 0.03}, entry_risk_policy=POLICY).build_entry_risk_limits()


def test_runtime_policy_preserves_optional_order_cap_and_stricter_correlation():
    limits = Config(
        risk={"max_order_notional": None, "correlation_limit": 0.6},
        correlation={"max_correlation": 0.5},
        entry_risk_policy=POLICY,
    ).build_entry_risk_limits()
    assert limits.max_order_notional_usd is None
    assert limits.max_absolute_correlation == Decimal("0.5")


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e-2", "", "-1", "1.1"])
def test_runtime_policy_rejects_malformed_fraction(value):
    with pytest.raises(ValueError):
        Config(
            entry_risk_policy={**POLICY, "ENTRY_MAX_PORTFOLIO_GROSS_FRACTION": value}
        ).build_entry_risk_limits()


@pytest.mark.parametrize("value", ["0", "-1", "5.000001", "0.0000001"])
def test_runtime_policy_rejects_unusable_freshness(value):
    with pytest.raises(ValueError, match="ENTRY_MAX_ACCOUNT_EVIDENCE_AGE_SECONDS"):
        Config(
            entry_risk_policy={**POLICY, "ENTRY_MAX_ACCOUNT_EVIDENCE_AGE_SECONDS": value}
        ).build_entry_risk_limits()


def test_runtime_policy_preserves_exact_microsecond_age_and_rebuilds_changed_limits():
    config = Config(entry_risk_policy={**POLICY, "ENTRY_MAX_QUOTE_AGE_SECONDS": "0.123456"})
    assert config.build_entry_risk_limits().max_quote_age == timedelta(microseconds=123456)
    config.risk.max_order_notional = 12000
    assert config.build_entry_risk_limits().max_order_notional_usd == Decimal("12000")
    config.risk.max_open_positions = True
    with pytest.raises(ValueError, match="max_open_positions"):
        config.build_entry_risk_limits()


@pytest.mark.parametrize(
    "settings,valid",
    [(POLICY, True), ({}, True), ({"ENTRY_MAX_PORTFOLIO_GROSS_FRACTION": "0.75"}, False)],
)
def test_environment_loader_captures_and_validates_explicit_entry_policy(
    tmp_path, monkeypatch, settings, valid
):
    import robo_trader.config as module
    from tests.test_pr2b3_terminal_settlement_persistence import _runtime_contract

    runtime = _runtime_contract(tmp_path)
    monkeypatch.setattr(module, "load_dotenv", lambda: None)
    monkeypatch.setattr(module, "load_runtime_contract_from_env", lambda: runtime)
    monkeypatch.setattr(
        module.os,
        "environ",
        {"IBKR_CLIENT_ID": "7", "RISK_MAX_DAILY_NOTIONAL": "100000", **settings},
    )
    if not valid:
        with pytest.raises(
            module.ConfigValidationError, match="Invalid explicit entry risk policy"
        ):
            module.load_config_from_env()
        return
    config = module.load_config_from_env()
    assert config.entry_risk_policy == settings
    module.os.environ["ENTRY_MAX_PORTFOLIO_GROSS_FRACTION"] = "0.1"
    assert config.entry_risk_policy == settings  # captured, not a live env view
    if settings:
        assert config.build_entry_risk_limits().max_portfolio_gross_fraction == Decimal("0.75")
    else:
        with pytest.raises(ValueError, match="explicit"):
            config.build_entry_risk_limits()
