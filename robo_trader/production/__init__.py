"""
Production infrastructure for trading system.

This module provides production-ready components for:
- Configuration management
- Health monitoring
- Emergency stops
- Alerting and notifications
- Deployment utilities
"""

from .alerting import (Alert, AlertCategory, AlertManager, AlertRule,
                       AlertSeverity, EmailNotifier, SlackNotifier)
from .config_manager import (AlertingConfig, ConfigManager, Environment,
                             FeatureFlags, ProductionConfig, TradingLimits,
                             get_config, get_config_manager)
from .emergency_stop import (EmergencyStopEvent, EmergencyStopManager,
                             StopReason, StopScope, TradingRestriction)
from .health import (CircuitBreakerState, ComponentHealth, ComponentStatus,
                     HealthEndpoint, HealthMetrics, HealthMonitor,
                     HealthStatus)

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
