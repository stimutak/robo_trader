"""
Secure configuration utilities for handling sensitive data.

This module provides secure configuration management with:
- Validation of required sensitive values
- Masking of sensitive data in logs
- Secure retrieval of environment variables
- API key and secret management
"""

import hashlib
import os
import re
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Union

from ..logger import get_logger

logger = get_logger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class SecureConfig:
    """Secure configuration manager for sensitive values."""

    # Patterns to identify sensitive keys
    SENSITIVE_PATTERNS = [
        r".*_KEY$",
        r".*_SECRET$",
        r".*_PASSWORD$",
        r".*_TOKEN$",
        r".*_API_.*",
        r".*_CREDENTIALS$",
        r".*_AUTH$",
        r".*CLIENT_ID$",
        r".*ACCOUNT.*",
    ]

    # Minimum lengths for different types of secrets
    MIN_LENGTHS = {
        "API_KEY": 10,
        "SECRET": 10,
        "PASSWORD": 8,
        "TOKEN": 10,
        "CLIENT_ID": 1,
    }

    @classmethod
    def is_sensitive(cls, key: str) -> bool:
        """Check if a configuration key contains sensitive data."""
        key_upper = key.upper()
        for pattern in cls.SENSITIVE_PATTERNS:
            if re.match(pattern, key_upper):
                return True
        return False

    @classmethod
    def mask_value(cls, value: Any, reveal_length: int = 4) -> str:
        """
        Mask sensitive value for logging.

        Args:
            value: The sensitive value to mask
            reveal_length: Number of characters to reveal at start

        Returns:
            Masked string safe for logging
        """
        if value is None:
            return "None"

        str_value = str(value)
        if len(str_value) <= reveal_length:
            # Too short to reveal anything safely
            return "****"

        if len(str_value) <= 8:
            # Short values: show first 2 chars
            return f"{str_value[:2]}****"
        else:
            # Longer values: show first few chars and length
            return f"{str_value[:reveal_length]}****[{len(str_value)} chars]"

    @classmethod
    def hash_value(cls, value: str) -> str:
        """
        Create a hash of sensitive value for comparison without revealing it.

        Args:
            value: The sensitive value to hash

        Returns:
            SHA256 hash prefix (first 8 chars) for identification
        """
        if not value:
            return "empty"
        hash_obj = hashlib.sha256(str(value).encode())
        return hash_obj.hexdigest()[:8]

    @classmethod
    def get_secure_config(
        cls,
        key: str,
        required: bool = True,
        default: Optional[Any] = None,
        min_length: Optional[int] = None,
        validator: Optional[callable] = None,
        mask_in_logs: bool = True,
    ) -> Optional[Any]:
        """
        Securely retrieve configuration value from environment.

        Args:
            key: Environment variable name
            required: Whether the value is required
            default: Default value if not found and not required
            min_length: Minimum length requirement
            validator: Optional validation function
            mask_in_logs: Whether to mask value in logs

        Returns:
            Configuration value

        Raises:
            ConfigValidationError: If validation fails
        """
        value = os.getenv(key, default)

        # Check if required
        if required and not value:
            logger.error(f"Required configuration '{key}' not found in environment")
            raise ConfigValidationError(
                f"Required configuration '{key}' not found. "
                f"Please set the {key} environment variable."
            )

        if value:
            # Validate length
            if min_length and len(str(value)) < min_length:
                logger.error(
                    f"Configuration '{key}' too short "
                    f"(required: {min_length}, got: {len(str(value))})"
                )
                raise ConfigValidationError(
                    f"Configuration '{key}' must be at least {min_length} characters"
                )

            # Run custom validator
            if validator:
                try:
                    if not validator(value):
                        raise ConfigValidationError(f"Validation failed for '{key}'")
                except Exception as e:
                    logger.error(f"Configuration '{key}' validation failed: {e}")
                    raise ConfigValidationError(f"Invalid value for '{key}': {e}")

            # Log retrieval (with masking if sensitive)
            if cls.is_sensitive(key) and mask_in_logs:
                masked = cls.mask_value(value)
                hash_id = cls.hash_value(value)
                logger.info(f"Loaded secure config '{key}': {masked} (id: {hash_id})")
            else:
                logger.debug(f"Loaded config '{key}'")

        return value

    @classmethod
    def validate_ibkr_config(cls) -> dict:
        """
        Validate and retrieve IBKR configuration securely.

        Returns:
            Dictionary with validated IBKR configuration

        Raises:
            ConfigValidationError: If any required config is missing or invalid
        """
        config = {}

        # IBKR Host
        config["host"] = cls.get_secure_config(
            "IBKR_HOST",
            required=True,
            mask_in_logs=False,  # Host is not sensitive
            validator=lambda h: h and (h == "localhost" or h == "127.0.0.1" or "." in h),
        )

        # IBKR Port
        port_str = cls.get_secure_config(
            "IBKR_PORT",
            required=True,
            mask_in_logs=False,  # Port is not sensitive
            validator=lambda p: p and p.isdigit() and 1 <= int(p) <= 65535,
        )
        config["port"] = int(port_str)

        # Validate port for mode
        paper_ports = [7497, 4002]
        live_ports = [7496, 4001]
        if config["port"] in paper_ports:
            logger.info(f"Using paper trading port: {config['port']}")
        elif config["port"] in live_ports:
            logger.warning(f"Using LIVE trading port: {config['port']} - BE CAREFUL!")

        # IBKR Client ID
        client_id_str = cls.get_secure_config(
            "IBKR_CLIENT_ID",
            required=True,
            mask_in_logs=True,  # Client ID is somewhat sensitive
            validator=lambda c: c and c.isdigit() and 0 <= int(c) <= 999,
        )
        config["client_id"] = int(client_id_str)

        # IBKR Account (optional but sensitive)
        config["account"] = cls.get_secure_config(
            "IBKR_ACCOUNT",
            required=False,
            mask_in_logs=True,  # Account number is sensitive
            min_length=4,
        )

        # API Key/Token if exists
        config["api_key"] = cls.get_secure_config(
            "IBKR_API_KEY", required=False, mask_in_logs=True, min_length=10
        )

        return config

    @classmethod
    def validate_all_configs(cls) -> dict:
        """
        Validate all sensitive configurations.

        Returns:
            Dictionary with all validated configurations
        """
        all_configs = {}

        # IBKR Configuration
        try:
            all_configs["ibkr"] = cls.validate_ibkr_config()
            logger.info("IBKR configuration validated successfully")
        except ConfigValidationError as e:
            logger.error(f"IBKR configuration validation failed: {e}")
            raise

        # Database credentials if they exist
        db_password = cls.get_secure_config(
            "DB_PASSWORD", required=False, mask_in_logs=True, min_length=8
        )
        if db_password:
            all_configs["db_password"] = db_password

        # API Keys for external services
        for service in ["ALPHA_VANTAGE", "POLYGON", "FINNHUB", "IEX"]:
            key_name = f"{service}_API_KEY"
            api_key = cls.get_secure_config(
                key_name, required=False, mask_in_logs=True, min_length=10
            )
            if api_key:
                all_configs[key_name.lower()] = api_key

        # Webhook URLs
        webhook_url = cls.get_secure_config(
            "MONITORING_ALERT_WEBHOOK",
            required=False,
            mask_in_logs=True,
            validator=lambda u: u and (u.startswith("http://") or u.startswith("https://")),
        )
        if webhook_url:
            all_configs["webhook_url"] = webhook_url

        return all_configs


def secure_log_decorator(func):
    """
    Decorator to automatically mask sensitive data in function logs.

    Use this decorator on functions that might log sensitive data.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Mask any sensitive kwargs before logging
        safe_kwargs = {}
        for key, value in kwargs.items():
            if SecureConfig.is_sensitive(key):
                safe_kwargs[key] = SecureConfig.mask_value(value)
            else:
                safe_kwargs[key] = value

        logger.debug(f"Calling {func.__name__} with kwargs: {safe_kwargs}")

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # Log error without exposing sensitive data
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


def validate_env_file(env_path: Optional[str] = None) -> bool:
    """
    Validate .env file for required configurations.

    Args:
        env_path: Path to .env file (defaults to .env in project root)

    Returns:
        True if validation passes

    Raises:
        ConfigValidationError: If validation fails
    """
    if env_path is None:
        env_path = Path(__file__).parent.parent.parent / ".env"

    env_path = Path(env_path)

    if not env_path.exists():
        logger.warning(f".env file not found at {env_path}")
        return False

    # Check file permissions (should not be world-readable)
    stat_info = env_path.stat()
    mode = stat_info.st_mode
    if mode & 0o004:  # World readable
        logger.warning(f".env file at {env_path} is world-readable! " f"Run: chmod 600 {env_path}")

    # Read and validate contents
    required_vars = ["IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID"]
    found_vars = set()

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key = line.split("=")[0].strip()
                    if key in required_vars:
                        found_vars.add(key)

    missing = set(required_vars) - found_vars
    if missing:
        logger.error(f"Missing required variables in .env: {missing}")
        raise ConfigValidationError(f"Missing required environment variables: {', '.join(missing)}")

    logger.info(f"Environment file validated: {len(found_vars)} required vars found")
    return True


# Example usage function
def example_usage():
    """Example of how to use SecureConfig."""

    # Validate all configs at startup
    try:
        configs = SecureConfig.validate_all_configs()
        print("All configurations validated successfully")

        # Access IBKR config
        ibkr_config = configs["ibkr"]
        print(f"Connecting to IBKR at {ibkr_config['host']}:{ibkr_config['port']}")

        # Log with masking
        client_id_masked = SecureConfig.mask_value(ibkr_config["client_id"])
        print(f"Using client ID: {client_id_masked}")

    except ConfigValidationError as e:
        print(f"Configuration error: {e}")
        print("Please check your .env file and environment variables")
        raise


if __name__ == "__main__":
    # Run validation when module is executed directly
    validate_env_file()
    example_usage()
