"""
Trade Executor Module - Order execution and position management.

This module handles:
- Order placement with circuit breaker protection
- Kill switch enforcement
- Position size calculation (Kelly, correlation-adjusted)
- Stop-loss order management
- Trade recording and WebSocket notifications

Extracted from runner_async.py to improve modularity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol

from ..logger import get_logger

if TYPE_CHECKING:
    from ..circuit_breaker import CircuitManager
    from ..database_async import AsyncTradingDatabase
    from ..monitoring.performance import PerformanceMonitor
    from ..monitoring.production_monitor import ProductionMonitor
    from ..order_manager import OrderResult
    from ..portfolio import Portfolio
    from ..risk.advanced_risk import AdvancedRiskManager
    from ..risk.risk_manager import RiskManager
    from ..stop_loss import StopLossMonitor

logger = get_logger(__name__)


@dataclass
class Order:
    """Order to be executed."""

    symbol: str
    quantity: int
    side: str  # BUY, SELL, BUY_TO_COVER, SELL_SHORT
    price: Decimal


@dataclass
class ExecutionResult:
    """Result of trade execution."""

    symbol: str
    executed: bool
    quantity: int
    fill_price: float
    side: str
    message: str
    pnl: float = 0.0


class Position:
    """Position in a symbol."""

    def __init__(
        self,
        symbol: str,
        quantity: int,
        avg_price: float,
        entry_time: Optional[datetime] = None,
    ):
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.entry_time = entry_time or datetime.now()

    @property
    def avg_cost(self) -> float:
        return self.avg_price


class TradeExecutor:
    """
    Executes trades with risk management and monitoring.

    Features:
    - Circuit breaker protection
    - Kill switch enforcement
    - Kelly criterion position sizing
    - Correlation-adjusted sizing
    - Stop-loss management
    - WebSocket notifications

    Usage:
        executor = TradeExecutor(
            portfolio=portfolio,
            risk_manager=risk_manager,
            db=database,
        )
        result = await executor.execute_buy(symbol, price, signal_strength)
    """

    def __init__(
        self,
        portfolio: Portfolio,
        risk_manager: RiskManager,
        db: AsyncTradingDatabase,
        positions: Dict[str, Position],
        advanced_risk: Optional[AdvancedRiskManager] = None,
        circuit_manager: Optional[CircuitManager] = None,
        stop_loss_monitor: Optional[StopLossMonitor] = None,
        production_monitor: Optional[ProductionMonitor] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        portfolio_manager: Optional[Any] = None,
        correlation_manager: Optional[Any] = None,
        ws_client: Optional[Any] = None,
        place_order_func: Optional[Callable] = None,
        update_position_func: Optional[Callable] = None,
        enable_short_selling: bool = False,
        enable_stop_loss: bool = True,
        stop_loss_percent: float = 0.05,
        use_advanced_risk: bool = True,
        use_correlation_sizing: bool = False,
    ):
        self.portfolio = portfolio
        self.risk = risk_manager
        self.db = db
        self.positions = positions
        self.advanced_risk = advanced_risk
        self.circuit_manager = circuit_manager
        self.stop_loss_monitor = stop_loss_monitor
        self.production_monitor = production_monitor
        self.monitor = performance_monitor
        self.portfolio_manager = portfolio_manager
        self.correlation_manager = correlation_manager
        self.ws_client = ws_client
        self._place_order = place_order_func
        self._update_position_atomic = update_position_func

        self.enable_short_selling = enable_short_selling
        self.enable_stop_loss = enable_stop_loss
        self.stop_loss_percent = stop_loss_percent
        self.use_advanced_risk = use_advanced_risk
        self.use_correlation_sizing = use_correlation_sizing

        # Tracking
        self.daily_pnl = 0.0
        self.daily_executed_notional = 0.0
        self.active_strategy_name: Optional[str] = None

    def _check_kill_switch(self) -> bool:
        """Check if kill switch is active."""
        if self.advanced_risk and hasattr(self.advanced_risk, "kill_switch"):
            return self.advanced_risk.kill_switch.triggered
        return False

    async def execute_signal(
        self,
        symbol: str,
        signal: int,
        price: float,
        df: Any,
        signal_strength: float = 1.0,
    ) -> ExecutionResult:
        """
        Execute a trading signal.

        Args:
            symbol: Stock symbol
            signal: 1 for BUY, -1 for SELL, 0 for HOLD
            price: Current price
            df: Market data DataFrame (for ATR etc.)
            signal_strength: Signal confidence (0-1)

        Returns:
            ExecutionResult with execution details
        """
        if signal == 0:
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="HOLD",
                message="No action",
            )

        if signal == 1:
            return await self._execute_buy(symbol, price, df, signal_strength)
        else:
            return await self._execute_sell(symbol, price, df)

    async def _execute_buy(
        self,
        symbol: str,
        price: float,
        df: Any,
        signal_strength: float,
    ) -> ExecutionResult:
        """Execute a buy signal."""
        # Check if we need to cover a short first
        if symbol in self.positions and self.positions[symbol].quantity < 0:
            return await self._cover_short(symbol, price)

        # Check if we already have a long position
        if symbol in self.positions and self.positions[symbol].quantity > 0:
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="BUY",
                message="Buy signal: Already have long position",
            )

        # Check kill switch
        if self._check_kill_switch():
            logger.error(f"KILL SWITCH ACTIVE - Order blocked for {symbol} BUY")
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="BUY",
                message="Kill switch active - trading halted",
            )

        # Calculate position size
        equity = await self._get_equity(price)
        qty = await self._calculate_position_size(symbol, price, df, equity, signal_strength, "BUY")

        if qty <= 0:
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="BUY",
                message="Position size too small",
            )

        # Check minimum position value
        min_position_value = float(os.getenv("MIN_POSITION_VALUE", "500"))
        position_value = qty * price
        if position_value < min_position_value:
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="BUY",
                message=f"Position value ${position_value:.0f} below min ${min_position_value:.0f}",
            )

        # Validate order
        ok, msg = self.risk.validate_order(
            symbol,
            qty,
            Decimal(str(price)),
            equity,
            self.daily_pnl,
            self.positions,
            self.daily_executed_notional,
        )

        if not ok:
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="BUY",
                message=f"Buy signal rejected: {msg}",
            )

        # Place order
        order = Order(symbol=symbol, quantity=qty, side="BUY", price=Decimal(str(price)))
        result = await self._place_order_safe(order)

        if not result.ok:
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="BUY",
                message=f"Buy order failed: {result.msg}",
            )

        fill_price = result.fill_price if result.fill_price is not None else price

        # Update position
        if self._update_position_atomic:
            success = await self._update_position_atomic(symbol, qty, fill_price, "BUY")
            if not success:
                return ExecutionResult(
                    symbol=symbol,
                    executed=False,
                    quantity=0,
                    fill_price=fill_price,
                    side="BUY",
                    message="Buy order failed: atomic update error",
                )

        # Record trade
        await self._record_trade(symbol, "BUY", qty, fill_price, price)

        # Add stop-loss
        await self._add_stop_loss(symbol, qty, fill_price)

        # Send WebSocket update
        self._send_trade_update(symbol, "BUY", qty, fill_price)

        self.daily_executed_notional += price * qty

        return ExecutionResult(
            symbol=symbol,
            executed=True,
            quantity=qty,
            fill_price=fill_price,
            side="BUY",
            message=f"Opened long: Bought {qty} shares at ${fill_price:.2f}",
        )

    async def _execute_sell(
        self,
        symbol: str,
        price: float,
        df: Any,
    ) -> ExecutionResult:
        """Execute a sell signal."""
        logger.info(
            f"SELL signal for {symbol}: checking position "
            f"(have position: {symbol in self.positions})"
        )

        if symbol in self.positions:
            pos = self.positions[symbol]

            if pos.quantity > 0:
                # Close long position
                return await self._close_long(symbol, price, pos)
            # Already short, nothing to do
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="SELL",
                message="Already have short position",
            )

        # Open short if enabled
        if self.enable_short_selling:
            return await self._open_short(symbol, price, df)

        return ExecutionResult(
            symbol=symbol,
            executed=False,
            quantity=0,
            fill_price=price,
            side="SELL",
            message="No position to sell",
        )

    async def _cover_short(self, symbol: str, price: float) -> ExecutionResult:
        """Cover a short position."""
        pos = self.positions[symbol]
        qty_to_cover = abs(pos.quantity)

        if self._check_kill_switch():
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="BUY_TO_COVER",
                message="Kill switch active - trading halted",
            )

        order = Order(
            symbol=symbol, quantity=qty_to_cover, side="BUY_TO_COVER", price=Decimal(str(price))
        )
        result = await self._place_order_safe(order)

        if not result.ok:
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="BUY_TO_COVER",
                message=f"Cover order failed: {result.msg}",
            )

        fill_price = result.fill_price if result.fill_price is not None else price

        # Cancel stop-loss
        if self.stop_loss_monitor:
            self.stop_loss_monitor.cancel_stop(symbol)

        # Update position
        if self._update_position_atomic:
            success = await self._update_position_atomic(
                symbol, qty_to_cover, fill_price, "BUY_TO_COVER"
            )
            if success:
                self.daily_pnl = float(self.portfolio.realized_pnl)
                await self._record_trade(symbol, "BUY_TO_COVER", qty_to_cover, fill_price, price)
                await self.db.update_position(symbol, 0, 0, 0)

                # Calculate P&L for short cover
                pnl = (pos.avg_cost - fill_price) * qty_to_cover
                if self.production_monitor:
                    self.production_monitor.record_trade(symbol, pnl, pnl > 0)

                return ExecutionResult(
                    symbol=symbol,
                    executed=True,
                    quantity=qty_to_cover,
                    fill_price=fill_price,
                    side="BUY_TO_COVER",
                    message=f"Covered short: Bought {qty_to_cover} shares at ${fill_price:.2f}",
                    pnl=pnl,
                )

        return ExecutionResult(
            symbol=symbol,
            executed=False,
            quantity=0,
            fill_price=fill_price,
            side="BUY_TO_COVER",
            message="Cover order failed: atomic update error",
        )

    async def _close_long(self, symbol: str, price: float, pos: Position) -> ExecutionResult:
        """Close a long position."""
        # PRO TRADER RULE: Only sell on signal if position is profitable
        # If at a loss, let the stop-loss handle the exit (don't panic sell)
        entry_price = float(pos.avg_price)

        # Guard against division by zero (corrupt data edge case)
        if entry_price <= 0:
            logger.error(f"Invalid entry_price for {symbol}: {entry_price} - skipping sell")
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="SELL",
                message=f"Invalid entry price: {entry_price}",
            )

        pnl_percent = ((price - entry_price) / entry_price) * 100

        if price < entry_price:
            # Position is at a loss - don't sell on ML signal
            loss_amount = (entry_price - price) * pos.quantity
            message = (
                f"SELL signal IGNORED for {symbol}: position at loss "
                f"(entry ${entry_price:.2f}, current ${price:.2f}, "
                f"P&L {pnl_percent:.1f}% = -${loss_amount:.2f}). "
                f"Letting stop-loss handle exit."
            )
            logger.warning(message)
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="SELL",
                message=message,
            )

        logger.info(
            f"Attempting to SELL {pos.quantity} shares of {symbol} at ${price:.2f} "
            f"(profit: {pnl_percent:.1f}%)"
        )

        if self._check_kill_switch():
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="SELL",
                message="Kill switch active - trading halted",
            )

        order = Order(symbol=symbol, quantity=pos.quantity, side="SELL", price=Decimal(str(price)))
        result = await self._place_order_safe(order)

        if not result.ok:
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="SELL",
                message=f"Sell order failed: {result.msg}",
            )

        fill_price = result.fill_price if result.fill_price is not None else price

        # Cancel stop-loss
        if self.stop_loss_monitor:
            self.stop_loss_monitor.cancel_stop(symbol)

        # Update position
        if self._update_position_atomic:
            success = await self._update_position_atomic(symbol, pos.quantity, fill_price, "SELL")
            if success:
                self.daily_pnl = float(self.portfolio.realized_pnl)
                await self._record_trade(symbol, "SELL", pos.quantity, fill_price, price)
                await self.db.update_position(symbol, 0, 0, 0)

                # Calculate P&L
                pnl = (fill_price - pos.avg_price) * pos.quantity
                if self.production_monitor:
                    self.production_monitor.record_trade(symbol, pnl, pnl > 0)

                self._send_trade_update(symbol, "SELL", pos.quantity, fill_price)

                return ExecutionResult(
                    symbol=symbol,
                    executed=True,
                    quantity=pos.quantity,
                    fill_price=fill_price,
                    side="SELL",
                    message=f"Closed long: Sold {pos.quantity} shares at ${fill_price:.2f}",
                    pnl=pnl,
                )

        return ExecutionResult(
            symbol=symbol,
            executed=False,
            quantity=0,
            fill_price=fill_price,
            side="SELL",
            message="Sell order failed: atomic update error",
        )

    async def _open_short(self, symbol: str, price: float, df: Any) -> ExecutionResult:
        """Open a short position."""
        if self._check_kill_switch():
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="SELL_SHORT",
                message="Kill switch active - trading halted",
            )

        equity = await self._get_equity(price)
        qty = await self._calculate_position_size(symbol, price, df, equity, 1.0, "SELL_SHORT")

        if qty <= 0:
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="SELL_SHORT",
                message="Position size too small",
            )

        ok, msg = self.risk.validate_order(
            symbol,
            qty,
            Decimal(str(price)),
            equity,
            self.daily_pnl,
            self.positions,
            self.daily_executed_notional,
        )

        if not ok:
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="SELL_SHORT",
                message=f"Short rejected: {msg}",
            )

        order = Order(symbol=symbol, quantity=qty, side="SELL_SHORT", price=Decimal(str(price)))
        result = await self._place_order_safe(order)

        if not result.ok:
            return ExecutionResult(
                symbol=symbol,
                executed=False,
                quantity=0,
                fill_price=price,
                side="SELL_SHORT",
                message=f"Short order failed: {result.msg}",
            )

        fill_price = result.fill_price if result.fill_price is not None else price

        # Add stop-loss for short
        await self._add_stop_loss(symbol, -qty, fill_price)

        if self._update_position_atomic:
            success = await self._update_position_atomic(symbol, qty, fill_price, "SELL_SHORT")
            if success:
                self.daily_executed_notional += price * qty
                await self._record_trade(symbol, "SELL_SHORT", qty, fill_price, price)

                pos = self.positions.get(symbol)
                if pos:
                    await self.db.update_position(symbol, pos.quantity, pos.avg_price, price)

                return ExecutionResult(
                    symbol=symbol,
                    executed=True,
                    quantity=qty,
                    fill_price=fill_price,
                    side="SELL_SHORT",
                    message=f"Opened short: Sold {qty} shares at ${fill_price:.2f}",
                )

        return ExecutionResult(
            symbol=symbol,
            executed=False,
            quantity=0,
            fill_price=fill_price,
            side="SELL_SHORT",
            message="Short order failed: atomic update error",
        )

    async def _get_equity(self, current_price: float) -> Decimal:
        """Get current portfolio equity."""
        from ..utils.precise_pricing import PrecisePricing

        equity_prices = {}
        for sym in self.positions:
            if hasattr(self, "latest_prices") and sym in self.latest_prices:
                equity_prices[sym] = self.latest_prices[sym]
            else:
                equity_prices[sym] = self.positions[sym].avg_price

        return await self.portfolio.equity(equity_prices)

    async def _calculate_position_size(
        self,
        symbol: str,
        price: float,
        df: Any,
        equity: Decimal,
        signal_strength: float,
        side: str,
    ) -> int:
        """Calculate position size with various risk adjustments."""
        qty = 0

        if self.use_advanced_risk and self.advanced_risk:
            # Get ATR for stop loss calculation
            atr = df["atr"].iloc[-1] if df is not None and "atr" in df.columns else None

            sizing_result = await self.advanced_risk.calculate_position_size(
                symbol=symbol,
                signal_strength=abs(signal_strength),
                current_price=price,
                atr=atr,
            )

            if sizing_result["blocked"]:
                logger.warning(
                    f"Trade blocked by kill switch for {symbol}: {sizing_result['block_reason']}"
                )
                return 0

            qty = sizing_result["position_size"]

            # Log Kelly metrics
            if "kelly_metrics" in sizing_result:
                km = sizing_result["kelly_metrics"]
                logger.info(
                    f"Kelly metrics for {symbol}: Win rate={km['win_rate']:.2%}, "
                    f"Edge={km['edge']:.3f}, Kelly fraction={sizing_result['kelly_fraction']:.3f}"
                )

            for warning in sizing_result.get("warnings", []):
                logger.warning(f"Risk warning for {symbol}: {warning}")
        else:
            # Fallback to standard position sizing
            from ..utils.precise_pricing import PrecisePricing

            qty = self.risk.position_size(equity, PrecisePricing.to_decimal(price))

        # Scale by portfolio manager strategy weight
        if self.portfolio_manager and self.active_strategy_name:
            alloc = self.portfolio_manager.allocations.get(self.active_strategy_name)
            if alloc:
                qty = max(int(qty * alloc.current_weight), 0)

        # Apply correlation-based sizing
        if self.use_correlation_sizing and self.correlation_manager:
            adjusted_qty, sizing_reason = await self.correlation_manager.get_adjusted_position_size(
                symbol=symbol,
                base_size=qty,
                current_positions=self.positions,
                portfolio_value=float(equity),
            )
            if adjusted_qty != qty:
                logger.info(
                    f"Position size adjusted for {symbol}: {qty} -> {adjusted_qty} ({sizing_reason})"
                )
                qty = adjusted_qty

        return int(qty)

    async def _place_order_safe(self, order: Order) -> OrderResult:
        """Place order with circuit breaker protection."""
        if self._place_order:
            return await self._place_order(order)

        # Fallback: create a simple success result for paper trading
        from dataclasses import dataclass

        @dataclass
        class SimpleResult:
            ok: bool = True
            msg: str = ""
            fill_price: Optional[float] = None

        return SimpleResult(ok=True, fill_price=float(order.price))

    async def _record_trade(
        self, symbol: str, side: str, qty: int, fill_price: float, order_price: float
    ):
        """Record trade in database and monitors."""
        slippage = (fill_price - order_price) * qty if fill_price != order_price else 0

        await self.db.record_trade(symbol, side, qty, fill_price, slippage=slippage)

        if self.monitor:
            self.monitor.record_order_placed(symbol, qty)
            self.monitor.record_trade_executed(symbol, side, qty)

        if self.production_monitor:
            latency_ms = 10  # Simulated for paper trading
            self.production_monitor.record_order(symbol, True, latency_ms)

    async def _add_stop_loss(self, symbol: str, qty: int, fill_price: float):
        """Add stop-loss order for a new position."""
        if not self.stop_loss_monitor or not self.enable_stop_loss:
            return

        try:
            from ..stop_loss import StopType

            position = Position(
                symbol=symbol,
                quantity=qty,
                avg_price=fill_price,
                entry_time=datetime.now(),
            )
            await self.stop_loss_monitor.add_stop_loss(
                symbol=symbol,
                position=position,
                stop_percent=self.stop_loss_percent,
                stop_type=StopType.FIXED,
            )
            logger.info(f"Stop-loss order added for {symbol} at {self.stop_loss_percent:.1%}")
        except Exception as e:
            logger.error(f"Failed to add stop-loss for {symbol}: {e}")

    def _send_trade_update(self, symbol: str, side: str, qty: int, price: float):
        """Send trade update via WebSocket."""
        if self.ws_client:
            try:
                self.ws_client.send_trade_update(symbol, side, qty, price)
            except Exception as e:
                logger.debug(f"Could not send trade WebSocket update: {e}")
