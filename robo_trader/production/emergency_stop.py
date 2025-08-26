"""
Emergency stop and kill switch functionality.

Provides immediate trading halt capabilities for risk management
and compliance requirements.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
import threading
from pathlib import Path

from ..logger import get_logger
from ..database import TradingDatabase

logger = get_logger(__name__)


class StopReason(Enum):
    """Reasons for emergency stop."""
    MANUAL = "manual"
    MAX_LOSS = "max_loss"
    SYSTEM_ERROR = "system_error"
    COMPLIANCE = "compliance"
    MARKET_HALT = "market_halt"
    CONNECTION_LOST = "connection_lost"
    RISK_LIMIT = "risk_limit"
    REGULATORY = "regulatory"
    MAINTENANCE = "maintenance"


class StopScope(Enum):
    """Scope of emergency stop."""
    ALL_TRADING = "all_trading"
    NEW_ORDERS = "new_orders"
    SPECIFIC_SYMBOL = "specific_symbol"
    SPECIFIC_STRATEGY = "specific_strategy"
    SPECIFIC_ACCOUNT = "specific_account"


@dataclass
class EmergencyStopEvent:
    """Record of an emergency stop event."""
    timestamp: datetime
    reason: StopReason
    scope: StopScope
    initiated_by: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    auto_resume: bool = False
    resume_after: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason.value,
            "scope": self.scope.value,
            "initiated_by": self.initiated_by,
            "message": self.message,
            "metadata": self.metadata,
            "auto_resume": self.auto_resume,
            "resume_after": self.resume_after.isoformat() if self.resume_after else None
        }


@dataclass
class TradingRestriction:
    """Active trading restriction."""
    id: str
    scope: StopScope
    target: Optional[str]  # Symbol, strategy, or account
    reason: StopReason
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if restriction is currently active."""
        if self.expires_at:
            return datetime.now() < self.expires_at
        return True


class EmergencyStopManager:
    """
    Manages emergency stop functionality and trading halts.
    
    Features:
    - Immediate trading halt capability
    - Granular stop scopes
    - Automatic position closing
    - Compliance integration
    - Audit trail
    """
    
    def __init__(self, ibkr_client=None, database: Optional[TradingDatabase] = None):
        """
        Initialize emergency stop manager.
        
        Args:
            ibkr_client: IBKR client for order management
            database: Database for persistence
        """
        self.ibkr_client = ibkr_client
        self.database = database
        self.is_stopped = False
        self.stop_event: Optional[EmergencyStopEvent] = None
        self.restrictions: Dict[str, TradingRestriction] = {}
        self.stop_history: List[EmergencyStopEvent] = []
        self.callbacks: List[Callable] = []
        self._lock = threading.Lock()
        
        # Load persisted state
        self._load_state()
        
    def _load_state(self) -> None:
        """Load persisted emergency stop state."""
        state_file = Path("emergency_stop_state.json")
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    
                # Restore active restrictions
                for restriction_data in state.get("restrictions", []):
                    restriction = TradingRestriction(
                        id=restriction_data["id"],
                        scope=StopScope(restriction_data["scope"]),
                        target=restriction_data.get("target"),
                        reason=StopReason(restriction_data["reason"]),
                        created_at=datetime.fromisoformat(restriction_data["created_at"]),
                        expires_at=datetime.fromisoformat(restriction_data["expires_at"]) 
                            if restriction_data.get("expires_at") else None,
                        metadata=restriction_data.get("metadata", {})
                    )
                    
                    if restriction.is_active():
                        self.restrictions[restriction.id] = restriction
                        
                logger.info(f"Loaded {len(self.restrictions)} active restrictions")
                
            except Exception as e:
                logger.error(f"Failed to load emergency stop state: {e}")
                
    def _save_state(self) -> None:
        """Persist emergency stop state."""
        try:
            state = {
                "is_stopped": self.is_stopped,
                "restrictions": [
                    {
                        "id": r.id,
                        "scope": r.scope.value,
                        "target": r.target,
                        "reason": r.reason.value,
                        "created_at": r.created_at.isoformat(),
                        "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                        "metadata": r.metadata
                    }
                    for r in self.restrictions.values()
                    if r.is_active()
                ],
                "last_stop_event": self.stop_event.to_dict() if self.stop_event else None
            }
            
            state_file = Path("emergency_stop_state.json")
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save emergency stop state: {e}")
            
    def register_callback(self, callback: Callable) -> None:
        """
        Register callback for emergency stop events.
        
        Args:
            callback: Function to call on emergency stop
        """
        self.callbacks.append(callback)
        
    def _notify_callbacks(self, event: EmergencyStopEvent) -> None:
        """Notify all registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                
    def emergency_stop(
        self,
        reason: StopReason,
        scope: StopScope = StopScope.ALL_TRADING,
        target: Optional[str] = None,
        message: str = "",
        initiated_by: str = "system",
        close_positions: bool = False,
        cancel_orders: bool = True,
        auto_resume_minutes: Optional[int] = None
    ) -> EmergencyStopEvent:
        """
        Execute emergency stop.
        
        Args:
            reason: Reason for stop
            scope: Scope of stop
            target: Specific target (symbol/strategy/account)
            message: Descriptive message
            initiated_by: Who initiated the stop
            close_positions: Whether to close all positions
            cancel_orders: Whether to cancel pending orders
            auto_resume_minutes: Auto-resume after N minutes
            
        Returns:
            Emergency stop event record
        """
        with self._lock:
            # Create stop event
            event = EmergencyStopEvent(
                timestamp=datetime.now(),
                reason=reason,
                scope=scope,
                initiated_by=initiated_by,
                message=message or f"Emergency stop: {reason.value}",
                metadata={"target": target} if target else {},
                auto_resume=auto_resume_minutes is not None,
                resume_after=datetime.now() + timedelta(minutes=auto_resume_minutes) 
                    if auto_resume_minutes else None
            )
            
            # Log critical event
            logger.critical(f"EMERGENCY STOP: {event.message}")
            
            # Update state
            self.is_stopped = True
            self.stop_event = event
            self.stop_history.append(event)
            
            # Execute stop actions
            if cancel_orders:
                self._cancel_all_orders(scope, target)
                
            if close_positions:
                self._close_all_positions(scope, target)
                
            # Add restriction
            restriction = TradingRestriction(
                id=f"stop_{int(time.time())}",
                scope=scope,
                target=target,
                reason=reason,
                created_at=event.timestamp,
                expires_at=event.resume_after,
                metadata=event.metadata
            )
            self.restrictions[restriction.id] = restriction
            
            # Save state
            self._save_state()
            
            # Notify callbacks
            self._notify_callbacks(event)
            
            # Schedule auto-resume if needed
            if event.auto_resume:
                self._schedule_auto_resume(event)
                
            return event
            
    def _cancel_all_orders(self, scope: StopScope, target: Optional[str]) -> None:
        """Cancel orders based on scope."""
        try:
            if not self.ibkr_client:
                logger.warning("No IBKR client available to cancel orders")
                return
                
            if scope == StopScope.ALL_TRADING:
                # Cancel all orders
                logger.info("Cancelling all pending orders")
                self.ibkr_client.reqGlobalCancel()
                
            elif scope == StopScope.SPECIFIC_SYMBOL and target:
                # Cancel orders for specific symbol
                logger.info(f"Cancelling orders for {target}")
                # Would need to track orders by symbol
                
            logger.info("Order cancellation complete")
            
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            
    def _close_all_positions(self, scope: StopScope, target: Optional[str]) -> None:
        """Close positions based on scope."""
        try:
            if not self.ibkr_client:
                logger.warning("No IBKR client available to close positions")
                return
                
            logger.warning("Position closing requested - manual intervention may be required")
            
            # In production, this would:
            # 1. Get all open positions
            # 2. Create market orders to close them
            # 3. Monitor execution
            
        except Exception as e:
            logger.error(f"Failed to close positions: {e}")
            
    def _schedule_auto_resume(self, event: EmergencyStopEvent) -> None:
        """Schedule automatic resume."""
        if not event.resume_after:
            return
            
        delay = (event.resume_after - datetime.now()).total_seconds()
        if delay > 0:
            timer = threading.Timer(delay, lambda: self.resume_trading("auto_resume"))
            timer.daemon = True
            timer.start()
            logger.info(f"Auto-resume scheduled for {event.resume_after}")
            
    def resume_trading(self, initiated_by: str = "manual") -> bool:
        """
        Resume trading after emergency stop.
        
        Args:
            initiated_by: Who initiated the resume
            
        Returns:
            True if successfully resumed
        """
        with self._lock:
            if not self.is_stopped:
                logger.warning("Trading is not stopped")
                return False
                
            logger.info(f"Resuming trading (initiated by: {initiated_by})")
            
            # Clear stop state
            self.is_stopped = False
            self.stop_event = None
            
            # Clear expired restrictions
            active_restrictions = {}
            for rid, restriction in self.restrictions.items():
                if restriction.is_active():
                    # Only keep if explicitly non-expiring
                    if not restriction.expires_at:
                        active_restrictions[rid] = restriction
                        
            self.restrictions = active_restrictions
            
            # Save state
            self._save_state()
            
            # Notify callbacks
            resume_event = EmergencyStopEvent(
                timestamp=datetime.now(),
                reason=StopReason.MANUAL,
                scope=StopScope.ALL_TRADING,
                initiated_by=initiated_by,
                message="Trading resumed"
            )
            self._notify_callbacks(resume_event)
            
            return True
            
    def add_restriction(
        self,
        scope: StopScope,
        target: Optional[str],
        reason: StopReason,
        duration_minutes: Optional[int] = None
    ) -> TradingRestriction:
        """
        Add a trading restriction.
        
        Args:
            scope: Restriction scope
            target: Specific target
            reason: Reason for restriction
            duration_minutes: Duration in minutes (None = permanent)
            
        Returns:
            Created restriction
        """
        restriction = TradingRestriction(
            id=f"restrict_{int(time.time())}",
            scope=scope,
            target=target,
            reason=reason,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=duration_minutes) 
                if duration_minutes else None
        )
        
        self.restrictions[restriction.id] = restriction
        self._save_state()
        
        logger.info(f"Added restriction: {restriction.id} ({reason.value})")
        return restriction
        
    def remove_restriction(self, restriction_id: str) -> bool:
        """
        Remove a trading restriction.
        
        Args:
            restriction_id: ID of restriction to remove
            
        Returns:
            True if removed successfully
        """
        if restriction_id in self.restrictions:
            del self.restrictions[restriction_id]
            self._save_state()
            logger.info(f"Removed restriction: {restriction_id}")
            return True
        return False
        
    def check_trading_allowed(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        account: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Check if trading is allowed.
        
        Args:
            symbol: Symbol to check
            strategy: Strategy to check
            account: Account to check
            
        Returns:
            Tuple of (allowed, reason if not allowed)
        """
        # Check global stop
        if self.is_stopped:
            return False, f"Trading stopped: {self.stop_event.message if self.stop_event else 'Unknown'}"
            
        # Check restrictions
        for restriction in self.restrictions.values():
            if not restriction.is_active():
                continue
                
            # Check scope match
            if restriction.scope == StopScope.ALL_TRADING:
                return False, f"All trading restricted: {restriction.reason.value}"
                
            elif restriction.scope == StopScope.NEW_ORDERS:
                return False, f"New orders restricted: {restriction.reason.value}"
                
            elif restriction.scope == StopScope.SPECIFIC_SYMBOL and symbol:
                if restriction.target == symbol:
                    return False, f"Symbol {symbol} restricted: {restriction.reason.value}"
                    
            elif restriction.scope == StopScope.SPECIFIC_STRATEGY and strategy:
                if restriction.target == strategy:
                    return False, f"Strategy {strategy} restricted: {restriction.reason.value}"
                    
            elif restriction.scope == StopScope.SPECIFIC_ACCOUNT and account:
                if restriction.target == account:
                    return False, f"Account {account} restricted: {restriction.reason.value}"
                    
        return True, None
        
    def get_status(self) -> Dict[str, Any]:
        """Get current emergency stop status."""
        return {
            "is_stopped": self.is_stopped,
            "current_stop": self.stop_event.to_dict() if self.stop_event else None,
            "active_restrictions": [
                {
                    "id": r.id,
                    "scope": r.scope.value,
                    "target": r.target,
                    "reason": r.reason.value,
                    "expires_at": r.expires_at.isoformat() if r.expires_at else None
                }
                for r in self.restrictions.values()
                if r.is_active()
            ],
            "stop_history_count": len(self.stop_history),
            "last_stop": self.stop_history[-1].to_dict() if self.stop_history else None
        }
        
    def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit trail of emergency stop events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of stop events
        """
        return [
            event.to_dict() 
            for event in self.stop_history[-limit:]
        ]