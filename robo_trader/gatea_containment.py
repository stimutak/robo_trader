"""Machine-enforced Gate-A strategy and order containment.

Gate A intentionally supports one narrow runtime profile: local paper execution,
simple long entries from the baseline SMA strategy, and semantic reductions.
Advanced entry producers stay unavailable until their full PR 7/11 contracts
and evidence gates are implemented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

GATE_A_PROFILE = "gate-a-simple-long-only-v1"
BASELINE_ENTRY_SOURCE = "baseline_sma"
ALLOWED_STRATEGIES = (BASELINE_ENTRY_SOURCE,)
REDUCTION_SIDES = frozenset({"SELL", "BUY_TO_COVER"})

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_DISABLED_ENV_FLAGS = (
    "AI_TRADING_ENABLED",
    "EXECUTION_SHORT_SELLING",
    "ML_ENHANCED_ENABLED",
    "ML_SELECTOR_ENABLED",
    "RISK_TAKE_PROFIT_ENABLED",
    "SMART_EXECUTION_ENABLED",
)


class GateAContainmentError(ValueError):
    """Configuration or intent would escape the Gate-A paper profile."""


def _strict_bool(env: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    normalized = str(raw).strip().casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise GateAContainmentError(f"{name} must be an explicit boolean")


def _strategy_list(raw: str, name: str) -> tuple[str, ...]:
    strategies = tuple(item.strip().casefold() for item in raw.split(",") if item.strip())
    if not strategies or len(strategies) != len(set(strategies)):
        raise GateAContainmentError(f"{name} must contain unique nonempty strategies")
    return strategies


def validate_gate_a_portfolio_strategies(
    value: object,
    *,
    name: str,
) -> tuple[str, ...]:
    """Validate an explicitly supplied per-portfolio strategy override."""

    if type(value) is str:
        strategies = _strategy_list(value, name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(type(item) is not str for item in value):
            raise GateAContainmentError(f"{name} must contain only strings")
        strategies = tuple(item.strip().casefold() for item in value)
        if (
            not strategies
            or any(not item for item in strategies)
            or len(strategies) != len(set(strategies))
        ):
            raise GateAContainmentError(f"{name} must contain unique nonempty strategies")
    else:
        raise GateAContainmentError(f"{name} must be a string or sequence")
    if strategies != ALLOWED_STRATEGIES:
        raise GateAContainmentError(f"{name} requires exactly baseline_sma")
    return strategies


@dataclass(frozen=True, slots=True)
class GateAEnvironment:
    enabled_strategies: tuple[str, ...]


def validate_gate_a_environment(env: Mapping[str, str]) -> GateAEnvironment:
    """Reject every legacy toggle that contradicts the Gate-A profile."""

    for name in _DISABLED_ENV_FLAGS:
        if _strict_bool(env, name):
            raise GateAContainmentError(f"{name}=true is disabled by {GATE_A_PROFILE}")

    primary = env.get("STRATEGY_ENABLED")
    legacy = env.get("STRATEGY_ENABLED_STRATEGIES")
    primary_values = _strategy_list(primary, "STRATEGY_ENABLED") if primary is not None else None
    legacy_values = (
        _strategy_list(legacy, "STRATEGY_ENABLED_STRATEGIES") if legacy is not None else None
    )
    if primary_values is not None and legacy_values is not None and primary_values != legacy_values:
        raise GateAContainmentError(
            "STRATEGY_ENABLED and STRATEGY_ENABLED_STRATEGIES contradict each other"
        )
    enabled = primary_values or legacy_values or ALLOWED_STRATEGIES
    if enabled != ALLOWED_STRATEGIES:
        raise GateAContainmentError(
            f"Gate A requires exactly {','.join(ALLOWED_STRATEGIES)}; configured={','.join(enabled)}"
        )
    return GateAEnvironment(enabled_strategies=enabled)


def assert_gate_a_config(config: object) -> None:
    """Validate direct or environment-built Config objects fail closed."""

    execution = getattr(config, "execution", None)
    strategy = getattr(config, "strategy", None)
    risk = getattr(config, "risk", None)
    mode = getattr(execution, "mode", None)
    mode_value = getattr(mode, "value", mode)
    enabled = tuple(
        str(item).strip().casefold() for item in getattr(strategy, "enabled_strategies", ())
    )
    contradictions: list[str] = []
    if mode_value not in {"paper", "backtest"}:
        contradictions.append("execution.mode must be paper or offline backtest")
    if getattr(execution, "enable_short_selling", None) is not False:
        contradictions.append("new shorts must be disabled")
    if getattr(execution, "use_smart_execution", None) is not False:
        contradictions.append("smart execution must be disabled")
    if enabled != ALLOWED_STRATEGIES:
        contradictions.append("only baseline_sma may be enabled")
    if getattr(strategy, "enable_ai_discovery", None) is not False:
        contradictions.append("AI discovery must be disabled")
    if getattr(strategy, "enable_ml_selectors", None) is not False:
        contradictions.append("ML selectors must be disabled")
    if getattr(risk, "enable_take_profit", None) is not False:
        contradictions.append("take-profit execution must be disabled")
    portfolio_configs = getattr(config, "portfolio_configs", ())
    for index, portfolio in enumerate(portfolio_configs):
        if not isinstance(portfolio, Mapping):
            contradictions.append(f"portfolio_configs[{index}] must be a mapping")
            continue
        if "enabled_strategies" not in portfolio or portfolio["enabled_strategies"] is None:
            continue
        try:
            validate_gate_a_portfolio_strategies(
                portfolio["enabled_strategies"],
                name=f"portfolio_configs[{index}].enabled_strategies",
            )
        except GateAContainmentError as exc:
            contradictions.append(str(exc))
    if contradictions:
        raise GateAContainmentError("; ".join(contradictions))


def assert_gate_a_runner_options(
    *,
    use_ml_strategy: bool,
    use_ml_enhanced: bool | None,
    use_smart_execution: bool | None,
    env: Mapping[str, str],
) -> None:
    """Reject constructor/legacy environment attempts to enable advanced entries."""

    validate_gate_a_environment(env)
    if use_ml_strategy is not False:
        raise GateAContainmentError("ML strategy selection is disabled by Gate A")
    if use_ml_enhanced is True:
        raise GateAContainmentError("ML-enhanced strategy selection is disabled by Gate A")
    if use_smart_execution is True:
        raise GateAContainmentError("smart execution is disabled by Gate A")


def validate_gate_a_order(
    *,
    side: object,
    take_profit: object,
) -> tuple[bool, str]:
    """Validate Gate-A order shape before capability-backed submission."""

    normalized_side = str(side or "").strip().upper()
    if take_profit is not None:
        return False, "Gate-A take-profit execution is quarantined"
    if normalized_side in REDUCTION_SIDES:
        return True, "Gate-A semantic reduction"
    if normalized_side == "SELL_SHORT":
        return False, "Gate-A profile blocks new short exposure"
    if normalized_side != "BUY":
        return False, "Gate-A profile rejects unsupported order side"
    return True, "Gate-A simple long candidate requires terminal capability"


def assert_quarantined_alternate_engine(component: str) -> None:
    """Prevent unused entry-capable engines from becoming a second runtime."""

    raise GateAContainmentError(
        f"{component} is quarantined by {GATE_A_PROFILE}; use robo_trader.runner_async.AsyncRunner"
    )
