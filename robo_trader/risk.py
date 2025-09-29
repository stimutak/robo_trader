"""
Risk management module - provides backward compatibility imports.

This module re-exports classes from risk_manager.py to maintain compatibility
with existing tests and imports that use `from robo_trader.risk import RiskManager`.
"""

from .risk_manager import (
    Position,
    RiskCalculations,
    RiskManager,
    RiskMetrics,
    RiskProfile,
    RiskViolation,
    RiskViolationType,
)

__all__ = [
    "RiskManager",
    "Position",
    "RiskViolationType",
    "RiskViolation",
    "RiskMetrics",
    "RiskProfile",
    "RiskCalculations",
]
