"""
Enhanced configuration system with Pydantic validation and environment-based settings.

This module provides a comprehensive configuration system for the equity trading platform
with schema validation, environment-specific settings, and equity-specific constraints.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

from .logger import get_logger
from .utils.config_validator import ConfigValidator, EnhancedTradingConfig
from .utils.secure_config import ConfigValidationError, SecureConfig

logger = get_logger(__name__)


PAPER_PORTS = frozenset({4002})
LIVE_PORTS = frozenset({7496, 4001})
PAPER_ONLY_EXECUTION_SOURCE = "paper_simulator"


@dataclass(frozen=True)
class RuntimeContract:
    """Validated, non-secret runtime identity for the containment phase.

    Live order placement is intentionally unavailable.  This contract is
    shared by the runner and dashboard so mode/port/read-only status cannot
    drift between environment aliases or be presented as a hard-coded value.
    """

    environment: str
    execution_mode: str
    execution_source: str
    ibkr_host: str
    ibkr_port: int
    ibkr_readonly: bool
    database_path: str
    account_alias: Optional[str]

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.public_dict(include_fingerprint=False), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def public_dict(self, *, include_fingerprint: bool = True) -> Dict[str, object]:
        result: Dict[str, object] = {
            "environment": self.environment,
            "mode": self.execution_mode,
            "execution_source": self.execution_source,
            "ibkr_host": self.ibkr_host,
            "ibkr_port": self.ibkr_port,
            "ibkr_readonly": self.ibkr_readonly,
            "database_path": self.database_path,
            "account_alias": self.account_alias,
            "live_capability": "disabled",
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result


def _normalized_value(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default)
    return str(value).strip().lower()


def _masked_account_alias(account: Optional[str]) -> Optional[str]:
    if not account:
        return None
    clean = account.strip()
    if not clean:
        return None
    return f"***{clean[-4:]}" if len(clean) > 4 else "***"


def load_runtime_contract_from_env(
    environ: Optional[Mapping[str, str]] = None,
) -> RuntimeContract:
    """Load and fail-closed validate the current paper-only runtime contract.

    ``EXECUTION_MODE`` and ``ENVIRONMENT`` are canonical.  The legacy
    ``TRADING_MODE`` and ``TRADING_ENV`` aliases may remain during migration,
    but a conflict is fatal.  Live execution is rejected unconditionally until
    the dedicated live-adapter PR introduces a separately reviewed capability.
    """

    env = os.environ if environ is None else environ

    execution_mode = _normalized_value(env, "EXECUTION_MODE", "paper")
    legacy_mode = env.get("TRADING_MODE")
    if legacy_mode is not None and legacy_mode.strip().lower() != execution_mode:
        raise ConfigValidationError(
            "Conflicting execution modes: EXECUTION_MODE and TRADING_MODE must match."
        )
    if execution_mode != TradingMode.PAPER.value:
        raise ConfigValidationError(
            "Live trading capability is disabled during remediation. "
            "Set EXECUTION_MODE=paper and keep IBKR ReadOnlyApi=yes."
        )

    environment = _normalized_value(env, "ENVIRONMENT", "dev")
    legacy_environment = env.get("TRADING_ENV")
    if legacy_environment is not None:
        normalized_legacy = legacy_environment.strip().lower()
        aliases = {"development": "dev", "prod": "production"}
        normalized_legacy = aliases.get(normalized_legacy, normalized_legacy)
        normalized_environment = aliases.get(environment, environment)
        if normalized_legacy != normalized_environment:
            raise ConfigValidationError(
                "Conflicting environments: ENVIRONMENT and TRADING_ENV must match."
            )
        environment = normalized_environment

    port_raw = str(env.get("IBKR_PORT", "4002")).strip()
    try:
        ibkr_port = int(port_raw)
    except ValueError as exc:
        raise ConfigValidationError(f"IBKR_PORT must be an integer, got {port_raw!r}") from exc
    if ibkr_port not in PAPER_PORTS:
        raise ConfigValidationError(
            f"Paper-only remediation requires an IBKR paper port {sorted(PAPER_PORTS)}, "
            f"got {ibkr_port}."
        )

    readonly_raw = _normalized_value(env, "IBKR_READONLY", "true")
    ibkr_readonly = readonly_raw in {"true", "1", "yes", "on"}
    if not ibkr_readonly:
        raise ConfigValidationError(
            "Paper-only remediation requires IBKR_READONLY=true and IBC ReadOnlyApi=yes."
        )

    return RuntimeContract(
        environment=environment,
        execution_mode=execution_mode,
        execution_source=PAPER_ONLY_EXECUTION_SOURCE,
        ibkr_host=str(env.get("IBKR_HOST", "127.0.0.1")).strip(),
        ibkr_port=ibkr_port,
        ibkr_readonly=True,
        database_path=str(env.get("RT_DB_PATH", "trading_data.db")).strip(),
        account_alias=_masked_account_alias(env.get("IBKR_ACCOUNT")),
    )


class TradingMode(str, Enum):
    """Trading mode enumeration."""

    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class Environment(str, Enum):
    """Environment enumeration."""

    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class ExecutionConfig(BaseModel):
    """Execution configuration."""

    mode: TradingMode = Field(default=TradingMode.PAPER, description="Trading mode")
    account_type: str = Field(default="equity", description="Account type (equity only)")
    max_daily_trades: int = Field(default=100, ge=1, le=1000, description="Maximum trades per day")
    position_timeout_hours: int = Field(
        default=24, ge=1, le=168, description="Position timeout in hours"
    )
    order_timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Order timeout in seconds"
    )
    enable_short_selling: bool = Field(default=False, description="Enable short selling")

    # Smart Execution Parameters
    use_smart_execution: bool = Field(
        default=False, description="Enable smart execution algorithms"
    )
    default_execution_algorithm: str = Field(
        default="adaptive",
        description="Default execution algorithm: market, twap, vwap, iceberg, adaptive",
    )
    execution_duration_minutes: int = Field(
        default=10, ge=1, le=60, description="Default execution duration in minutes"
    )
    max_participation_rate: float = Field(
        default=0.15, gt=0, le=0.3, description="Maximum participation rate of volume"
    )
    execution_urgency: float = Field(
        default=0.5, ge=0, le=1, description="Execution urgency (0=patient, 1=aggressive)"
    )
    min_slice_size: int = Field(default=100, ge=1, description="Minimum execution slice size")
    max_slice_size: int = Field(default=10000, ge=100, description="Maximum execution slice size")
    iceberg_display_ratio: float = Field(
        default=0.2, gt=0, le=1, description="Iceberg order display ratio"
    )
    market_impact_factor: float = Field(
        default=0.1, ge=0, le=1, description="Market impact factor for cost estimation"
    )

    @field_validator("account_type")
    @classmethod
    def validate_account_type(cls, v: str) -> str:
        if v != "equity":
            raise ValueError("Only equity trading is supported in main branch")
        return v

    @field_validator("default_execution_algorithm")
    @classmethod
    def validate_execution_algorithm(cls, v: str) -> str:
        allowed = ["market", "twap", "vwap", "iceberg", "adaptive", "sniper"]
        if v not in allowed:
            raise ValueError(f"Execution algorithm must be one of {allowed}")
        return v


class RiskConfig(BaseModel):
    """Risk management configuration with enhanced validation."""

    max_position_pct: float = Field(
        default=0.02, gt=0, le=0.1, description="Max position size as % of portfolio"
    )
    max_daily_loss_pct: float = Field(
        default=0.005, gt=0, le=0.05, description="Max daily loss as % of portfolio"
    )
    max_position_loss_pct: float = Field(
        default=0.05,
        gt=0,
        le=0.10,
        description="Per-position unrealized loss that trips the kill switch",
    )

    @field_validator("max_position_loss_pct")
    @classmethod
    def validate_position_loss_pct(cls, v: float) -> float:
        if not 0 < v <= 0.10:
            raise ValueError(f"max_position_loss_pct must be between 0 and 0.10, got {v}")
        if v < 0.02:
            logger.warning(
                f"Very tight per-position loss limit: {v * 100:.1f}% — will trigger on normal volatility"
            )
        return v

    @field_validator("max_position_pct")
    @classmethod
    def validate_position_pct(cls, v: float) -> float:
        """Validate position percentage with safety checks."""
        if not 0 < v <= 0.1:
            raise ValueError(f"max_position_pct must be between 0 and 0.1, got {v}")
        if v > 0.05:
            logger.warning(
                f"High position size configured: {v * 100:.1f}% - ensure proper risk management"
            )
        return v

    @field_validator("max_daily_loss_pct")
    @classmethod
    def validate_daily_loss_pct(cls, v: float) -> float:
        """Validate daily loss percentage with safety checks."""
        if not 0 < v <= 0.05:
            raise ValueError(f"max_daily_loss_pct must be between 0 and 0.05, got {v}")
        if v > 0.02:
            logger.warning(f"High daily loss limit: {v * 100:.1f}% - be cautious")
        return v

    max_portfolio_beta: float = Field(
        default=1.2, gt=0, le=2.0, description="Maximum portfolio beta"
    )
    correlation_limit: float = Field(
        default=0.7, ge=0, le=1.0, description="Maximum correlation between positions"
    )
    min_volume: int = Field(
        default=1_000_000, ge=100_000, description="Minimum daily volume for stocks"
    )
    min_market_cap: float = Field(
        default=1_000_000_000, ge=100_000_000, description="Minimum market cap in USD"
    )
    max_sector_exposure_pct: float = Field(
        default=0.3, gt=0, le=0.5, description="Max sector exposure"
    )
    max_leverage: float = Field(default=2.0, ge=1.0, le=4.0, description="Maximum account leverage")

    @field_validator("max_leverage")
    @classmethod
    def validate_leverage(cls, v: float) -> float:
        """Validate leverage with safety warnings."""
        if not 1.0 <= v <= 4.0:
            raise ValueError(f"max_leverage must be between 1.0 and 4.0, got {v}")
        if v > 2.0:
            logger.warning(f"High leverage configured: {v}x - ensure you understand the risks")
        return v

    stop_loss_pct: float = Field(
        default=0.02, gt=0, le=0.1, description="Default stop loss percentage"
    )
    take_profit_pct: float = Field(
        default=0.05, gt=0, le=0.2, description="Default take profit percentage"
    )
    use_trailing_stop: bool = Field(
        default=True, description="Use trailing stops instead of fixed stops"
    )
    trailing_stop_pct: float = Field(
        default=0.05, gt=0, le=0.2, description="Trailing stop percentage (follows price up)"
    )

    # New risk parameters for enhanced control
    max_order_notional: Optional[float] = Field(
        default=10_000, ge=100, description="Max notional per order"
    )
    max_daily_notional: Optional[float] = Field(
        default=100_000, ge=1000, description="Max daily trading notional"
    )
    max_open_positions: int = Field(default=20, ge=1, le=100, description="Maximum open positions")

    @field_validator("max_order_notional")
    @classmethod
    def validate_order_notional(cls, v: Optional[float]) -> Optional[float]:
        """Validate order notional limit."""
        if v is not None:
            if v <= 0:
                raise ValueError(f"max_order_notional must be positive, got {v}")
            if v > 100_000:
                logger.warning(f"Very high order notional: ${v:,.2f}")
        return v

    @field_validator("max_daily_notional")
    @classmethod
    def validate_daily_notional(cls, v: Optional[float]) -> Optional[float]:
        """Validate daily notional limit."""
        if v is not None:
            if v <= 0:
                raise ValueError(f"max_daily_notional must be positive, got {v}")
            if v > 1_000_000:
                logger.warning(f"Very high daily notional: ${v:,.2f}")
        return v

    @field_validator("max_open_positions")
    @classmethod
    def validate_open_positions(cls, v: int) -> int:
        """Validate maximum open positions."""
        if v <= 0:
            raise ValueError(f"max_open_positions must be positive, got {v}")
        if v > 50:
            logger.warning(f"High position count: {v} - may impact monitoring and risk management")
        return v

    position_sizing_method: str = Field(
        default="fixed", description="Position sizing method: fixed, atr, kelly"
    )
    # NOTE: use_trailing_stop and trailing_stop_pct are defined above at lines 161-166
    # Do not duplicate them here!


class DataConfig(BaseModel):
    """Data management configuration."""

    provider: str = Field(default="IBKR", description="Data provider")
    storage: str = Field(default="sqlite", description="Storage backend")
    feature_window: int = Field(
        default=100, ge=20, le=500, description="Bars for indicator calculation"
    )
    tick_buffer: int = Field(
        default=10_000, ge=1000, description="Tick data buffer size"
    )  # Renamed for consistency
    cache_ttl: int = Field(
        default=300, ge=60, description="Cache TTL in seconds"
    )  # Renamed for consistency
    enable_realtime: bool = Field(
        default=True, description="Enable real-time data streaming"
    )  # Renamed for consistency
    historical_days: int = Field(
        default=30, ge=1, le=365, description="Days of historical data to fetch"
    )
    bar_size: str = Field(default="5 mins", description="Default bar size for historical data")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = ["IBKR", "ALPACA", "POLYGON"]
        if v not in allowed:
            raise ValueError(f"Provider must be one of {allowed}")
        return v


class IBKRConfig(BaseModel):
    """Interactive Brokers configuration."""

    host: str = Field(default="127.0.0.1", description="IBKR Gateway/TWS host")
    port: int = Field(default=4002, ge=1, le=65535, description="IBKR Gateway port")
    client_id: int = Field(default=123, ge=0, le=999, description="IBKR client ID")
    account: Optional[str] = Field(default=None, description="IBKR account number")
    readonly: bool = Field(default=True, description="Connect in read-only mode")
    timeout: float = Field(default=10.0, gt=0, description="Connection timeout in seconds")
    ssl_mode: Literal["auto", "require", "disabled"] = Field(
        default="auto",
        description=(
            "Socket transport strategy: auto (try TCP then TLS), require (TLS only), disabled (TCP only)"
        ),
    )

    @model_validator(mode="after")
    def validate_port_for_mode(self) -> "IBKRConfig":
        """Validate port matches trading mode."""
        # The supervised containment path manages IB Gateway paper on 4002.
        allowed_ssl_modes = {"auto", "require", "disabled"}
        if self.ssl_mode not in allowed_ssl_modes:
            raise ValueError(f"ssl_mode must be one of {sorted(allowed_ssl_modes)}")
        return self


class StrategyConfig(BaseModel):
    """Strategy configuration."""

    enabled_strategies: List[str] = Field(
        default=["momentum", "mean_reversion", "ml_enhanced", "microstructure", "pairs_trading"],
        description="List of enabled strategies",
    )
    combination_method: str = Field(
        default="weighted",
        description="Signal combination method: vote, weighted, priority",
    )
    min_confidence: float = Field(default=0.6, ge=0, le=1, description="Minimum signal confidence")
    rebalance_frequency: str = Field(default="daily", description="Rebalance frequency")

    # Strategy-specific parameters
    momentum_lookback: int = Field(default=20, ge=5, le=100, description="Momentum lookback period")
    mean_reversion_period: int = Field(default=14, ge=5, le=50, description="Mean reversion period")
    breakout_volume_factor: float = Field(
        default=1.5, gt=1, description="Volume factor for breakouts"
    )

    @field_validator("combination_method")
    @classmethod
    def validate_combination(cls, v: str) -> str:
        allowed = ["vote", "weighted", "priority"]
        if v not in allowed:
            raise ValueError(f"Combination method must be one of {allowed}")
        return v


class MLConfig(BaseModel):
    """Machine Learning configuration."""

    enable_ml_features: bool = Field(default=True, description="Enable ML feature generation")
    feature_store_path: str = Field(
        default="feature_store.db", description="Path to feature store database"
    )
    model_registry_path: str = Field(
        default="model_registry", description="Path to model registry directory"
    )
    auto_retrain: bool = Field(default=True, description="Enable automatic model retraining")
    retrain_threshold: float = Field(
        default=0.7, ge=0, le=1, description="Performance threshold for retraining"
    )
    retrain_frequency_hours: int = Field(
        default=24, ge=1, description="Hours between retrain checks"
    )
    feature_importance_threshold: float = Field(
        default=0.01, ge=0, le=1, description="Minimum feature importance to keep"
    )
    n_top_features: int = Field(
        default=50, ge=10, le=200, description="Number of top features to use"
    )
    validation_split: float = Field(
        default=0.2, gt=0, lt=1, description="Validation split for training"
    )
    enable_ensemble: bool = Field(default=True, description="Enable ensemble model training")
    hyperparameter_tuning: bool = Field(default=True, description="Enable hyperparameter tuning")
    cross_validation_folds: int = Field(default=5, ge=3, le=10, description="Number of CV folds")


class CorrelationConfig(BaseModel):
    """Correlation analysis configuration."""

    max_correlation: float = Field(
        default=0.7, ge=0, le=1, description="Maximum allowed correlation"
    )
    penalty_factor: float = Field(default=0.5, ge=0, le=1, description="Correlation penalty factor")
    update_interval: int = Field(
        default=300, ge=60, description="Correlation update interval in seconds"
    )
    lookback_days: int = Field(
        default=60, ge=20, le=252, description="Days for correlation calculation"
    )
    min_observations: int = Field(
        default=30, ge=10, description="Minimum observations for correlation"
    )
    enable_dynamic_sizing: bool = Field(
        default=True, description="Enable correlation-based position sizing"
    )
    concentration_limit: float = Field(
        default=0.3, gt=0, le=1, description="Portfolio concentration limit"
    )
    cluster_threshold: float = Field(
        default=0.8, ge=0, le=1, description="Threshold for correlation clustering"
    )


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration."""

    enable_alerts: bool = Field(default=True, description="Enable alerting")
    alert_email: Optional[str] = Field(default=None, description="Alert email address")
    alert_webhook: Optional[str] = Field(default=None, description="Alert webhook URL")
    metrics_port: int = Field(default=9090, ge=1024, le=65535, description="Metrics server port")
    health_check_interval: int = Field(
        default=60, ge=10, description="Health check interval in seconds"
    )
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: text or json")


class Config(BaseModel):
    """Main configuration class combining all sub-configurations."""

    environment: Environment = Field(default=Environment.DEVELOPMENT)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    ibkr: IBKRConfig = Field(default_factory=IBKRConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Runtime configuration
    symbols: List[str] = Field(default=["AAPL", "MSFT", "SPY"], description="Trading symbols")
    default_cash: float = Field(
        default=100_000, gt=0, description="Starting cash for paper trading"
    )

    # Multi-portfolio configuration (loaded from PORTFOLIOS env var or auto-created)
    portfolio_configs: List[dict] = Field(
        default_factory=list,
        description="List of portfolio configurations (populated at runtime)",
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v: List[str]) -> List[str]:
        """Validate and clean symbols."""
        cleaned = [s.strip().upper() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("At least one symbol must be specified")
        if len(cleaned) > 100:
            raise ValueError("Maximum 100 symbols allowed")
        return cleaned

    @model_validator(mode="after")
    def validate_config_consistency(self) -> "Config":
        """Validate configuration consistency across sections."""
        # PR-01 containment: live order placement is intentionally unavailable.
        # Reject direct Config(...) construction too, not just environment loads.
        if self.execution.mode == TradingMode.LIVE:
            raise ValueError("Live trading capability is disabled during remediation")
        if self.execution.mode == TradingMode.PAPER and self.ibkr.port not in PAPER_PORTS:
            raise ValueError(f"Paper mode requires a paper trading port: {sorted(PAPER_PORTS)}")
        if self.execution.mode == TradingMode.PAPER and not self.ibkr.readonly:
            raise ValueError("Paper mode requires IBKR read-only client permissions")

        # Ensure production has proper monitoring
        if self.environment == Environment.PRODUCTION:
            if not self.monitoring.enable_alerts:
                raise ValueError("Production environment requires alerts enabled")
            if self.monitoring.log_level not in ["INFO", "WARNING", "ERROR"]:
                raise ValueError("Production requires appropriate log level")

        return self


def load_config_from_env() -> Config:
    """
    Load configuration from environment variables with enhanced validation.

    Environment variables are prefixed with section names:
    - EXECUTION_MODE=paper
    - RISK_MAX_POSITION_PCT=0.02
    - DATA_PROVIDER=IBKR
    - IBKR_HOST=127.0.0.1
    - STRATEGY_ENABLED_STRATEGIES=momentum,mean_reversion
    - MONITORING_LOG_LEVEL=INFO

    Returns:
        Config: Validated configuration object with comprehensive risk checks

    Raises:
        ConfigValidationError: If configuration fails validation
    """
    load_dotenv()
    runtime_contract = load_runtime_contract_from_env()

    # Use enhanced validation for critical risk parameters
    validator = ConfigValidator()

    config_dict = {
        "environment": runtime_contract.environment,
        "execution": {
            "mode": runtime_contract.execution_mode,
            "account_type": os.getenv("EXECUTION_ACCOUNT_TYPE", "equity"),
            "max_daily_trades": int(os.getenv("EXECUTION_MAX_DAILY_TRADES", "100")),
            "position_timeout_hours": int(os.getenv("EXECUTION_POSITION_TIMEOUT", "24")),
            "order_timeout_seconds": int(os.getenv("EXECUTION_ORDER_TIMEOUT", "30")),
            "enable_short_selling": os.getenv("EXECUTION_SHORT_SELLING", "false").lower() == "true",
        },
        "risk": {
            # Use validated values for critical risk parameters
            "max_position_pct": validator.validate_percentage("RISK_MAX_POSITION_PCT", 0.02),
            "max_daily_loss_pct": validator.validate_percentage("RISK_MAX_DAILY_LOSS_PCT", 0.005),
            "max_position_loss_pct": validator.validate_percentage(
                "RISK_MAX_POSITION_LOSS_PCT", 0.05
            ),
            "max_portfolio_beta": validator.validate_range(
                "RISK_MAX_PORTFOLIO_BETA", 0.5, 2.0, 1.2
            ),
            "correlation_limit": validator.validate_range("RISK_CORRELATION_LIMIT", 0.0, 1.0, 0.7),
            "min_volume": max(int(os.getenv("RISK_MIN_VOLUME", "1000000")), 100000),
            "min_market_cap": max(float(os.getenv("RISK_MIN_MARKET_CAP", "1000000000")), 100000000),
            "max_sector_exposure_pct": validator.validate_percentage(
                "RISK_MAX_SECTOR_EXPOSURE", 0.3
            ),
            "max_leverage": validator.validate_leverage("RISK_MAX_LEVERAGE", 2.0),
            # Use STOP_LOSS_PERCENT from .env (value is in %, convert to decimal)
            "stop_loss_pct": float(os.getenv("STOP_LOSS_PERCENT", "2.0")) / 100,
            "take_profit_pct": validator.validate_percentage("RISK_TAKE_PROFIT_PCT", 0.05),
            # Validated notional limits
            "max_order_notional": (
                validator.validate_positive_float("RISK_MAX_ORDER_NOTIONAL", 10000, min_value=100)
                if os.getenv("RISK_MAX_ORDER_NOTIONAL")
                else None
            ),
            "max_daily_notional": (
                validator.validate_positive_float("RISK_MAX_DAILY_NOTIONAL", 100000, min_value=1000)
                if os.getenv("RISK_MAX_DAILY_NOTIONAL")
                else None
            ),
            "max_open_positions": validator.validate_position_limit("RISK_MAX_OPEN_POSITIONS", 20),
            "position_sizing_method": os.getenv("RISK_POSITION_SIZING", "fixed"),
            # Use same env var names as .env file (USE_TRAILING_STOP, TRAILING_STOP_PERCENT)
            "use_trailing_stop": os.getenv("USE_TRAILING_STOP", "true").lower()
            in ("true", "1", "yes", "on"),
            "trailing_stop_pct": float(os.getenv("TRAILING_STOP_PERCENT", "5.0"))
            / 100,  # Convert from % to decimal
        },
        "data": {
            "provider": os.getenv("DATA_PROVIDER", "IBKR"),
            "storage": os.getenv("DATA_STORAGE", "sqlite"),
            "feature_window": int(os.getenv("DATA_FEATURE_WINDOW", "100")),
            # Match DataConfig field names
            "tick_buffer": int(os.getenv("DATA_TICK_BUFFER", "10000")),
            "cache_ttl": int(os.getenv("DATA_CACHE_TTL", "300")),
            "enable_realtime": os.getenv("DATA_ENABLE_REALTIME", "true").lower() == "true",
            "historical_days": int(os.getenv("DATA_HISTORICAL_DAYS", "30")),
            "bar_size": os.getenv("DATA_BAR_SIZE", "5 mins"),
        },
        "ibkr": {
            "host": runtime_contract.ibkr_host,
            "port": runtime_contract.ibkr_port,
            "client_id": int(
                SecureConfig.get_secure_config("IBKR_CLIENT_ID", required=True, mask_in_logs=True)
            ),
            "account": SecureConfig.get_secure_config(
                "IBKR_ACCOUNT", required=False, mask_in_logs=True
            ),
            "readonly": runtime_contract.ibkr_readonly,
            "timeout": float(os.getenv("IBKR_TIMEOUT", "10.0")),
            "ssl_mode": os.getenv("IBKR_SSL_MODE", "auto").strip().lower(),
        },
        "strategy": {
            "enabled_strategies": os.getenv("STRATEGY_ENABLED", "momentum,mean_reversion").split(
                ","
            ),
            "combination_method": os.getenv("STRATEGY_COMBINATION", "weighted"),
            "min_confidence": float(os.getenv("STRATEGY_MIN_CONFIDENCE", "0.6")),
            "rebalance_frequency": os.getenv("STRATEGY_REBALANCE", "daily"),
            "momentum_lookback": int(os.getenv("STRATEGY_MOMENTUM_LOOKBACK", "20")),
            "mean_reversion_period": int(os.getenv("STRATEGY_MEAN_REVERSION_PERIOD", "14")),
            "breakout_volume_factor": float(os.getenv("STRATEGY_BREAKOUT_VOLUME", "1.5")),
        },
        "ml": {
            "enable_ml_features": os.getenv("ML_ENABLE_FEATURES", "true").lower() == "true",
            "feature_store_path": os.getenv("ML_FEATURE_STORE_PATH", "feature_store.db"),
            "model_registry_path": os.getenv("ML_MODEL_REGISTRY_PATH", "model_registry"),
            "auto_retrain": os.getenv("ML_AUTO_RETRAIN", "true").lower() == "true",
            "retrain_threshold": float(os.getenv("ML_RETRAIN_THRESHOLD", "0.7")),
            "retrain_frequency_hours": int(os.getenv("ML_RETRAIN_FREQUENCY", "24")),
            "feature_importance_threshold": float(
                os.getenv("ML_FEATURE_IMPORTANCE_THRESHOLD", "0.01")
            ),
            "n_top_features": int(os.getenv("ML_N_TOP_FEATURES", "50")),
            "validation_split": float(os.getenv("ML_VALIDATION_SPLIT", "0.2")),
            "enable_ensemble": os.getenv("ML_ENABLE_ENSEMBLE", "true").lower() == "true",
            "hyperparameter_tuning": os.getenv("ML_HYPERPARAMETER_TUNING", "true").lower()
            == "true",
            "cross_validation_folds": int(os.getenv("ML_CV_FOLDS", "5")),
        },
        "correlation": {
            "max_correlation": float(os.getenv("CORRELATION_MAX", "0.7")),
            "penalty_factor": float(os.getenv("CORRELATION_PENALTY_FACTOR", "0.5")),
            "update_interval": int(os.getenv("CORRELATION_UPDATE_INTERVAL", "300")),
            "lookback_days": int(os.getenv("CORRELATION_LOOKBACK_DAYS", "60")),
            "min_observations": int(os.getenv("CORRELATION_MIN_OBSERVATIONS", "30")),
            "enable_dynamic_sizing": os.getenv("CORRELATION_DYNAMIC_SIZING", "true").lower()
            == "true",
            "concentration_limit": float(os.getenv("CORRELATION_CONCENTRATION_LIMIT", "0.3")),
            "cluster_threshold": float(os.getenv("CORRELATION_CLUSTER_THRESHOLD", "0.8")),
        },
        "monitoring": {
            "enable_alerts": os.getenv("MONITORING_ENABLE_ALERTS", "true").lower() == "true",
            "alert_email": os.getenv("MONITORING_ALERT_EMAIL"),
            "alert_webhook": os.getenv("MONITORING_ALERT_WEBHOOK"),
            "metrics_port": int(os.getenv("MONITORING_METRICS_PORT", "9090")),
            "health_check_interval": int(os.getenv("MONITORING_HEALTH_CHECK_INTERVAL", "60")),
            "log_level": os.getenv("MONITORING_LOG_LEVEL", "INFO"),
            "log_format": os.getenv("MONITORING_LOG_FORMAT", "json"),
        },
        "symbols": [
            s.strip() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()
        ],
        "default_cash": float(os.getenv("DEFAULT_CASH", "100000")),
    }

    # Validation is now handled by SecureConfig.get_secure_config() above
    # Additional validation for port/mode consistency
    port = config_dict["ibkr"]["port"]
    paper_ports = PAPER_PORTS
    live_ports = LIVE_PORTS

    mode = config_dict["execution"]["mode"]
    if mode == "paper" and port in live_ports:
        raise ConfigValidationError(
            f"Paper mode configured but using live trading port {port}. "
            f"Use supervised IB Gateway paper port 4002."
        )
    elif mode == "live" and port in paper_ports:
        raise ConfigValidationError(
            f"Live mode configured but using paper trading port {port}. "
            f"Use port 7496 (TWS) or 4001 (Gateway) for live trading. BE CAREFUL!"
        )

    config = Config(**config_dict)

    # Load multi-portfolio configurations
    try:
        from .multiuser.portfolio_config import load_portfolio_configs

        portfolio_configs = load_portfolio_configs()
        config.portfolio_configs = [pc.to_dict() for pc in portfolio_configs]
    except Exception as e:
        logger.warning(f"Could not load portfolio configs: {e}")
        # Fall back to single default portfolio
        config.portfolio_configs = [
            {
                "id": "default",
                "name": "Default Portfolio",
                "starting_cash": config.default_cash,
                "symbols": ",".join(config.symbols),
                "active": True,
            }
        ]

    return config


def load_config() -> Config:
    """
    Load configuration with backward compatibility.

    This function maintains backward compatibility with the old configuration
    system while providing the new enhanced configuration.

    Returns:
        Config: Validated configuration object
    """
    return load_config_from_env()


def get_config_for_environment(env: Environment) -> Config:
    """
    Get environment-specific configuration presets.

    Args:
        env: Environment to get configuration for

    Returns:
        Config: Configuration with environment-specific defaults
    """
    base_config = load_config_from_env()
    base_config.environment = env

    if env == Environment.PRODUCTION:
        # PR-01 containment: production-like monitoring may be exercised in
        # paper mode, but no environment preset can enable broker writes.
        base_config.execution.mode = TradingMode.PAPER
        base_config.risk.max_position_pct = 0.01  # More conservative
        base_config.risk.max_daily_loss_pct = 0.003
        base_config.monitoring.enable_alerts = True
        base_config.monitoring.log_level = "INFO"
        base_config.ibkr.readonly = True

    elif env == Environment.STAGING:
        # Staging overrides
        base_config.execution.mode = TradingMode.PAPER
        base_config.monitoring.enable_alerts = True
        base_config.monitoring.log_level = "DEBUG"

    else:  # Development
        # Development overrides
        base_config.execution.mode = TradingMode.PAPER
        base_config.monitoring.enable_alerts = False
        base_config.monitoring.log_level = "DEBUG"
        base_config.risk.max_position_pct = 0.05  # Less conservative for testing

    return base_config


# Maintain backward compatibility
@dataclass
class LegacyConfig:
    """Legacy configuration for backward compatibility."""

    def __init__(self):
        new_config = load_config()
        self.ibkr_host = new_config.ibkr.host
        self.ibkr_port = new_config.ibkr.port
        self.ibkr_client_id = new_config.ibkr.client_id
        self.ibkr_account = new_config.ibkr.account
        self.trading_mode = new_config.execution.mode.value
        self.max_daily_loss = new_config.risk.max_daily_loss_pct * new_config.default_cash
        self.max_position_risk_pct = new_config.risk.max_position_pct
        self.max_symbol_exposure_pct = new_config.risk.max_sector_exposure_pct
        self.max_leverage = new_config.risk.max_leverage
        self.default_cash = new_config.default_cash
        self.symbols = new_config.symbols

    def __repr__(self) -> str:
        # Avoid leaking sensitive identifiers (client_id, account, etc.) into logs/tracebacks.
        return "LegacyConfig(<redacted>)"

    __str__ = __repr__


# Export main config loader
__all__ = [
    "Config",
    "load_config",
    "get_config_for_environment",
    "TradingMode",
    "Environment",
]
