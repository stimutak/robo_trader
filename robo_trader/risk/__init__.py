"""
Risk management modules for RoboTrader
"""

from ..portfolio import PositionSnapshot as Position

# Import legacy classes from other modules for backward compatibility
from ..risk_manager import RiskManager
from .advanced_risk import (
    AdvancedRiskManager,
    CorrelationLimiter,
    KellySizer,
    KillSwitch,
    RiskLevel,
    RiskMetrics,
    risk_monitor_task,
)

# Kelly sizing is in advanced_risk module

__all__ = [
    "AdvancedRiskManager",
    "KellySizer",
    "CorrelationLimiter",
    "KillSwitch",
    "RiskLevel",
    "RiskMetrics",
    "risk_monitor_task",
    "RiskManager",
    "Position",
]
