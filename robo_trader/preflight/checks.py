"""Registry of all preflight checks (spec §5.5).

This module is the single import point the runner reads. Adding a new
check is one line: import the class, instantiate it, append to
:data:`ALL_CHECKS`. Checks execute in list order for display, but they
are otherwise independent — the runner runs them all even if early ones
fail, so the operator sees every problem at once.

Display order rationale: kill-switch checks first (most likely cause of
the 2026-05-22-style livelock), then state-freshness, then config, then
network-state checks (the latter are most likely to be transient and
self-resolve on the next launcher attempt).
"""

from __future__ import annotations

from typing import List

from .equity_history_freshness_check import EquityHistoryFreshnessCheck
from .gateway_port_listening_check import GatewayPortListeningCheck
from .kill_switch_lock_check import KillSwitchLockCheck
from .kill_switch_state_check import KillSwitchStateCheck
from .protocol import Check
from .risk_threshold_check import RiskThresholdCheck
from .zombie_connections_check import ZombieConnectionsCheck

ALL_CHECKS: List[Check] = [
    KillSwitchStateCheck(),
    KillSwitchLockCheck(),
    EquityHistoryFreshnessCheck(),
    RiskThresholdCheck(),
    GatewayPortListeningCheck(),
    ZombieConnectionsCheck(),
]
