"""
Production configuration management system.

Handles environment-specific configurations, secrets management,
and feature flags for safe production deployment.
"""

import json
import logging
import os
import secrets
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from dotenv import load_dotenv

from ..logger import get_logger

logger = get_logger(__name__)


class Environment(Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class SecretProvider(Enum):
    """Secret storage providers."""

    ENV_FILE = "env_file"
    AWS_SECRETS = "aws_secrets"
    HASHICORP_VAULT = "vault"
    ENCRYPTED_FILE = "encrypted_file"


@dataclass
class TradingLimits:
    """Production trading limits and safeguards."""

    max_position_size: float = 10000.0
    max_daily_loss: float = 1000.0
    max_daily_trades: int = 100
    max_open_positions: int = 10
    max_leverage: float = 1.0
    allowed_symbols: List[str] = field(default_factory=list)
    blocked_symbols: List[str] = field(default_factory=list)
    trading_hours_start: str = "09:30"
    trading_hours_end: str = "16:00"
    enable_short_selling: bool = False
    require_stop_loss: bool = True
    max_slippage_percent: float = 0.5


@dataclass
class AlertingConfig:
    """Alerting and notification configuration."""

    enable_alerts: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log"])

    # Slack configuration
    slack_webhook_url: Optional[str] = None
    slack_channel: str = "#trading-alerts"
    slack_username: str = "RoboTrader"

    # Email configuration
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_recipients: List[str] = field(default_factory=list)

    # Alert thresholds
    pnl_alert_threshold: float = 1000.0
    drawdown_alert_threshold: float = 0.05
    error_rate_threshold: float = 0.1
    latency_alert_ms: int = 1000


@dataclass
class FeatureFlags:
    """Feature flags for gradual rollout and testing."""

    enable_paper_trading: bool = True
    enable_live_trading: bool = False
    enable_ml_predictions: bool = False
    enable_news_sentiment: bool = False
    enable_options_trading: bool = False
    enable_crypto_trading: bool = False
    enable_international_markets: bool = False
    use_advanced_risk_models: bool = False
    log_all_orders: bool = True
    dry_run_mode: bool = False
    maintenance_mode: bool = False


@dataclass
class IBKRConfig:
    """Interactive Brokers configuration."""

    host: str = "127.0.0.1"
    port: int = 7497  # 7497 for paper, 7496 for live
    client_id: int = 1
    account: str = ""
    enable_delayed_data: bool = False
    connection_timeout: int = 30
    request_timeout: int = 10
    max_retry_attempts: int = 3


@dataclass
class DatabaseConfig:
    """Database configuration."""

    db_type: str = "sqlite"
    db_path: str = "trading.db"
    connection_pool_size: int = 10
    enable_ssl: bool = False

    # PostgreSQL specific
    pg_host: Optional[str] = None
    pg_port: int = 5432
    pg_database: Optional[str] = None
    pg_username: Optional[str] = None
    pg_password: Optional[str] = None


@dataclass
class ProductionConfig:
    """Complete production configuration."""

    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = False
    log_level: str = "INFO"

    # Component configurations
    trading_limits: TradingLimits = field(default_factory=TradingLimits)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Security
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    encryption_key: Optional[str] = None

    # Performance
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    request_rate_limit: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding sensitive data."""
        data = asdict(self)
        # Remove sensitive fields
        sensitive_fields = [
            "api_key",
            "secret_key",
            "encryption_key",
            "pg_password",
            "smtp_password",
            "slack_webhook_url",
        ]
        for field in sensitive_fields:
            if field in data:
                data[field] = "***REDACTED***"
        return data


class ConfigManager:
    """
    Manages production configurations across environments.

    Features:
    - Environment-specific configurations
    - Encrypted secrets management
    - Feature flag control
    - Configuration validation
    - Hot reloading support
    """

    def __init__(self, environment: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            environment: Environment name (dev/staging/prod)
        """
        self.environment = self._determine_environment(environment)
        self.config_dir = Path("config/environments")
        self.secrets_provider = self._setup_secrets_provider()
        self.config: ProductionConfig = self._load_config()
        self._validate_config()

    def _determine_environment(self, env: Optional[str]) -> Environment:
        """Determine deployment environment."""
        if env:
            return Environment(env.lower())

        # Check environment variable
        env_var = os.getenv("TRADING_ENV", "development").lower()

        try:
            return Environment(env_var)
        except ValueError:
            logger.warning(f"Invalid environment: {env_var}, defaulting to development")
            return Environment.DEVELOPMENT

    def _setup_secrets_provider(self) -> SecretProvider:
        """Set up appropriate secrets provider."""
        provider = os.getenv("SECRET_PROVIDER", "env_file")

        try:
            return SecretProvider(provider)
        except ValueError:
            return SecretProvider.ENV_FILE

    def _load_config(self) -> ProductionConfig:
        """Load configuration for current environment."""
        config = ProductionConfig(environment=self.environment)

        # Load base configuration
        base_config_path = self.config_dir / "base.json"
        if base_config_path.exists():
            with open(base_config_path) as f:
                base_data = json.load(f)
                config = self._merge_config(config, base_data)

        # Load environment-specific configuration
        env_config_path = self.config_dir / f"{self.environment.value}.json"
        if env_config_path.exists():
            with open(env_config_path) as f:
                env_data = json.load(f)
                config = self._merge_config(config, env_data)

        # Load secrets
        config = self._load_secrets(config)

        # Load environment variables (highest priority)
        config = self._load_env_vars(config)

        logger.info(f"Loaded configuration for {self.environment.value} environment")
        return config

    def _merge_config(
        self, config: ProductionConfig, data: Dict[str, Any]
    ) -> ProductionConfig:
        """Merge configuration data into config object."""
        # Update trading limits
        if "trading_limits" in data:
            for key, value in data["trading_limits"].items():
                setattr(config.trading_limits, key, value)

        # Update alerting
        if "alerting" in data:
            for key, value in data["alerting"].items():
                setattr(config.alerting, key, value)

        # Update feature flags
        if "feature_flags" in data:
            for key, value in data["feature_flags"].items():
                setattr(config.feature_flags, key, value)

        # Update IBKR config
        if "ibkr" in data:
            for key, value in data["ibkr"].items():
                setattr(config.ibkr, key, value)

        # Update database config
        if "database" in data:
            for key, value in data["database"].items():
                setattr(config.database, key, value)

        # Update root level configs
        root_fields = [
            "debug_mode",
            "log_level",
            "enable_caching",
            "cache_ttl_seconds",
            "request_rate_limit",
        ]
        for field in root_fields:
            if field in data:
                setattr(config, field, data[field])

        return config

    def _load_secrets(self, config: ProductionConfig) -> ProductionConfig:
        """Load secrets based on provider."""
        if self.secrets_provider == SecretProvider.ENV_FILE:
            # Load from .env file
            env_file = f".env.{self.environment.value}"
            if Path(env_file).exists():
                load_dotenv(env_file)
            else:
                load_dotenv()

        elif self.secrets_provider == SecretProvider.ENCRYPTED_FILE:
            # Load from encrypted file
            secrets_file = self.config_dir / f"secrets.{self.environment.value}.enc"
            if secrets_file.exists():
                config = self._decrypt_secrets(config, secrets_file)

        # AWS Secrets Manager and Vault would be implemented here

        return config

    def _load_env_vars(self, config: ProductionConfig) -> ProductionConfig:
        """Load configuration from environment variables."""
        # API keys
        config.api_key = os.getenv("TRADING_API_KEY", config.api_key)
        config.secret_key = os.getenv("TRADING_SECRET_KEY", config.secret_key)

        # IBKR settings
        config.ibkr.host = os.getenv("IBKR_HOST", config.ibkr.host)
        config.ibkr.port = int(os.getenv("IBKR_PORT", str(config.ibkr.port)))
        config.ibkr.account = os.getenv("IBKR_ACCOUNT", config.ibkr.account)

        # Database settings
        if os.getenv("DATABASE_URL"):
            # Parse database URL
            config = self._parse_database_url(config, os.getenv("DATABASE_URL"))

        # Alerting
        config.alerting.slack_webhook_url = os.getenv(
            "SLACK_WEBHOOK_URL", config.alerting.slack_webhook_url
        )

        # Feature flags from env
        if os.getenv("ENABLE_LIVE_TRADING"):
            config.feature_flags.enable_live_trading = (
                os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
            )

        return config

    def _decrypt_secrets(
        self, config: ProductionConfig, secrets_file: Path
    ) -> ProductionConfig:
        """Decrypt secrets from encrypted file."""
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            logger.warning("No encryption key found for secrets decryption")
            return config

        try:
            cipher = Fernet(encryption_key.encode())
            with open(secrets_file, "rb") as f:
                encrypted_data = f.read()
                decrypted_data = cipher.decrypt(encrypted_data)
                secrets_data = json.loads(decrypted_data)

            # Apply secrets to config
            config.api_key = secrets_data.get("api_key", config.api_key)
            config.secret_key = secrets_data.get("secret_key", config.secret_key)

            logger.info("Successfully decrypted secrets")

        except Exception as e:
            logger.error(f"Failed to decrypt secrets: {e}")

        return config

    def _parse_database_url(
        self, config: ProductionConfig, url: str
    ) -> ProductionConfig:
        """Parse database URL into configuration."""
        # Format: postgresql://user:pass@host:port/database
        if url.startswith("postgresql://"):
            parts = url.replace("postgresql://", "").split("@")
            if len(parts) == 2:
                user_pass = parts[0].split(":")
                host_port_db = parts[1].split("/")
                host_port = host_port_db[0].split(":")

                config.database.db_type = "postgresql"
                config.database.pg_username = user_pass[0] if user_pass else None
                config.database.pg_password = (
                    user_pass[1] if len(user_pass) > 1 else None
                )
                config.database.pg_host = host_port[0]
                config.database.pg_port = (
                    int(host_port[1]) if len(host_port) > 1 else 5432
                )
                config.database.pg_database = (
                    host_port_db[1] if len(host_port_db) > 1 else None
                )

        return config

    def _validate_config(self) -> None:
        """Validate configuration for production readiness."""
        errors = []
        warnings = []

        # Check production requirements
        if self.environment == Environment.PRODUCTION:
            if self.config.feature_flags.enable_live_trading:
                # Require authentication
                if not self.config.api_key:
                    errors.append("API key required for live trading")

                # Require stop losses
                if not self.config.trading_limits.require_stop_loss:
                    warnings.append("Stop losses should be required in production")

                # Check alerting
                if not self.config.alerting.enable_alerts:
                    warnings.append("Alerting should be enabled in production")

                # Ensure reasonable limits
                if self.config.trading_limits.max_leverage > 2.0:
                    warnings.append("High leverage detected for production")

        # Log validation results
        for error in errors:
            logger.error(f"Config validation error: {error}")

        for warning in warnings:
            logger.warning(f"Config validation warning: {warning}")

        if errors and self.environment == Environment.PRODUCTION:
            raise ValueError(f"Configuration validation failed: {errors}")

    def get_config(self) -> ProductionConfig:
        """Get current configuration."""
        return self.config

    def reload_config(self) -> None:
        """Reload configuration from files."""
        logger.info("Reloading configuration...")
        self.config = self._load_config()
        self._validate_config()

    def update_feature_flag(self, flag: str, value: bool) -> None:
        """
        Update a feature flag at runtime.

        Args:
            flag: Feature flag name
            value: New value
        """
        if hasattr(self.config.feature_flags, flag):
            setattr(self.config.feature_flags, flag, value)
            logger.info(f"Updated feature flag {flag} to {value}")
        else:
            logger.warning(f"Unknown feature flag: {flag}")

    def check_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        # Check maintenance mode
        if self.config.feature_flags.maintenance_mode:
            logger.warning("Trading blocked: maintenance mode")
            return False

        # Check environment
        if self.environment == Environment.PRODUCTION:
            if not self.config.feature_flags.enable_live_trading:
                logger.warning("Trading blocked: live trading disabled")
                return False
        else:
            if not self.config.feature_flags.enable_paper_trading:
                logger.warning("Trading blocked: paper trading disabled")
                return False

        return True

    def export_config(self, filepath: str, include_secrets: bool = False) -> None:
        """
        Export current configuration to file.

        Args:
            filepath: Path to export to
            include_secrets: Whether to include sensitive data
        """
        config_data = self.config.to_dict()

        if not include_secrets:
            # Already redacted in to_dict()
            pass

        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=2, default=str)

        logger.info(f"Exported configuration to {filepath}")


# Singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create config manager singleton."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> ProductionConfig:
    """Get current production configuration."""
    return get_config_manager().get_config()
