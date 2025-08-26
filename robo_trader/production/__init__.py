"""
Production infrastructure for trading system.

This module provides production-ready components for:
- Configuration management
- Health monitoring
- Emergency stops
- Alerting and notifications
- Deployment utilities
"""

from .config_manager import (
    ConfigManager,
    ProductionConfig,
    Environment,
    TradingLimits,
    AlertingConfig,
    FeatureFlags,
    get_config_manager,
    get_config,
)

from .health import (
    HealthMonitor,
    HealthStatus,
    ComponentStatus,
    HealthMetrics,
    ComponentHealth,
    CircuitBreakerState,
    HealthEndpoint,
)

from .emergency_stop import (
    EmergencyStopManager,
    StopReason,
    StopScope,
    EmergencyStopEvent,
    TradingRestriction,
)

from .alerting import (
    AlertManager,
    Alert,
    AlertSeverity,
    AlertCategory,
    AlertRule,
    SlackNotifier,
    EmailNotifier,
)

__all__ = [
    # Config
    "ConfigManager",
    "ProductionConfig",
    "Environment",
    "TradingLimits",
    "AlertingConfig",
    "FeatureFlags",
    "get_config_manager",
    "get_config",
    # Health
    "HealthMonitor",
    "HealthStatus",
    "ComponentStatus",
    "HealthMetrics",
    "ComponentHealth",
    "CircuitBreakerState",
    "HealthEndpoint",
    # Emergency Stop
    "EmergencyStopManager",
    "StopReason",
    "StopScope",
    "EmergencyStopEvent",
    "TradingRestriction",
    # Alerting
    "AlertManager",
    "Alert",
    "AlertSeverity",
    "AlertCategory",
    "AlertRule",
    "SlackNotifier",
    "EmailNotifier",
]
