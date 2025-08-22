from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float


class RiskManager:
    """Advanced risk controls: ATR-based position sizing and multi-level exposure checks.

    Implements position sizing based on stop distance and risk per trade.
    """

    def __init__(
        self,
        max_daily_loss: float,
        max_position_risk_pct: float,
        max_symbol_exposure_pct: float,
        max_leverage: float,
        max_order_notional: float | None = None,
        max_daily_notional: float | None = None,
        per_trade_risk_bps: int = 50,  # Default 0.50% risk per trade
        max_weekly_loss_pct: float = 0.05,  # 5% weekly drawdown limit
    ) -> None:
        self.max_daily_loss = float(max_daily_loss)
        self.max_position_risk_pct = float(max_position_risk_pct)
        self.max_symbol_exposure_pct = float(max_symbol_exposure_pct)
        self.max_leverage = float(max_leverage)
        self.max_order_notional = float(max_order_notional) if max_order_notional is not None else None
        self.max_daily_notional = float(max_daily_notional) if max_daily_notional is not None else None
        self.per_trade_risk_bps = per_trade_risk_bps
        self.max_weekly_loss_pct = max_weekly_loss_pct

    def position_size(self, cash_available: float, entry_price: float) -> int:
        """Legacy position sizing - kept for compatibility.
        
        Use position_size_atr for ATR-based sizing.
        """
        if entry_price <= 0 or cash_available <= 0:
            return 0
        notional = cash_available * self.max_position_risk_pct
        return max(int(notional // entry_price), 0)
    
    def position_size_atr(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        atr: Optional[float] = None,
        risk_bps: Optional[int] = None,
        is_trend_following: bool = True
    ) -> int:
        """
        ATR-based position sizing that ties risk to stop distance.
        
        Formula: shares = (equity * risk_bps/10000) / stop_distance
        
        Args:
            equity: Total account equity
            entry_price: Planned entry price
            stop_price: Stop loss price level
            atr: Average True Range (optional, for dynamic stops)
            risk_bps: Risk in basis points (default to per_trade_risk_bps)
            is_trend_following: True for trend, False for mean-reversion
            
        Returns:
            Number of shares to trade
        """
        if equity <= 0 or entry_price <= 0:
            return 0
        
        # Use provided risk or default
        risk_bps = risk_bps or self.per_trade_risk_bps
        
        # Cap at maximum allowed risk (50 bps = 0.50%)
        risk_bps = min(risk_bps, 50)
        
        # Calculate stop distance
        if stop_price > 0:
            stop_distance = abs(entry_price - stop_price)
        elif atr and atr > 0:
            # Use ATR-based stop if no explicit stop provided
            atr_mult = 1.2 if is_trend_following else 0.8
            stop_distance = atr_mult * atr
        else:
            # Fallback to 2% stop if no stop or ATR provided
            stop_distance = entry_price * 0.02
        
        if stop_distance <= 0:
            return 0
        
        # Calculate position size
        risk_amount = equity * (risk_bps / 10000)
        shares = int(risk_amount / stop_distance)
        
        # Apply position limits
        max_shares_by_exposure = int((equity * self.max_symbol_exposure_pct) / entry_price)
        shares = min(shares, max_shares_by_exposure)
        
        # Ensure minimum viable position
        if shares * entry_price < 100:  # Minimum $100 position
            return 0
        
        return shares
    
    def validate_stop_loss(
        self,
        entry_price: float,
        stop_price: float,
        max_risk_pct: float = 0.005  # Max 0.50% risk
    ) -> Tuple[bool, str]:
        """
        Validate that stop loss is within acceptable risk parameters.
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            max_risk_pct: Maximum allowed risk as decimal
            
        Returns:
            Tuple of (is_valid, message)
        """
        if stop_price <= 0:
            return False, "Stop price must be positive"
        
        if entry_price <= 0:
            return False, "Entry price must be positive"
        
        # For long positions
        if stop_price < entry_price:
            risk_pct = (entry_price - stop_price) / entry_price
            if risk_pct > max_risk_pct * 2:  # Allow 2x single trade risk for wide stops
                return False, f"Stop too wide: {risk_pct:.2%} risk exceeds limit"
        # For short positions
        elif stop_price > entry_price:
            risk_pct = (stop_price - entry_price) / entry_price
            if risk_pct > max_risk_pct * 2:
                return False, f"Stop too wide: {risk_pct:.2%} risk exceeds limit"
        else:
            return False, "Stop price cannot equal entry price"
        
        return True, "Valid stop loss"

    def validate_order(
        self,
        symbol: str,
        order_qty: int,
        price: float,
        equity: float,
        daily_pnl: float,
        current_positions: Dict[str, Position],
        daily_executed_notional: float = 0.0,
        weekly_pnl: float = 0.0,
        stop_price: Optional[float] = None,
    ) -> Tuple[bool, str]:
        # Check daily drawdown
        if daily_pnl <= -abs(self.max_daily_loss):
            return False, "Daily loss limit reached (-2.0%)"
        
        # Check weekly drawdown
        if equity > 0 and weekly_pnl / equity <= -self.max_weekly_loss_pct:
            return False, f"Weekly loss limit reached (-{self.max_weekly_loss_pct:.1%})"
        
        if order_qty <= 0:
            return False, "Quantity must be positive"
        if price <= 0:
            return False, "Invalid price"
        
        # Require stop loss for all new positions
        if stop_price is None or stop_price <= 0:
            return False, "Stop loss required for all positions"

        symbol_exposure_notional = price * order_qty

        # Per-order notional ceiling
        if self.max_order_notional is not None and symbol_exposure_notional > self.max_order_notional:
            return False, "Order notional exceeds per-order limit"

        # Per-day notional ceiling
        if self.max_daily_notional is not None and (daily_executed_notional + symbol_exposure_notional) > self.max_daily_notional:
            return False, "Daily notional exceeds limit"
        max_symbol_notional = equity * self.max_symbol_exposure_pct
        if symbol_exposure_notional > max_symbol_notional:
            return False, "Symbol exposure exceeds limit"

        # Leverage check: sum of notionals / equity <= max_leverage
        existing_notional = sum(pos.quantity * pos.avg_price for pos in current_positions.values())
        total_after = existing_notional + symbol_exposure_notional
        if equity > 0 and (total_after / equity) > self.max_leverage:
            return False, "Account leverage exceeds limit"

        return True, "OK"


