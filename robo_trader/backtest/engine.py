"""
Event-driven backtesting engine for strategy validation.

This module provides realistic backtesting with:
- Transaction cost modeling
- Slippage simulation
- Historical data replay
- Performance tracking
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

from ..logger import get_logger
from ..strategies.framework import Strategy, Signal, SignalType
from ..features.engine import FeatureEngine
from ..data.pipeline import DataPipeline
from .metrics import calculate_metrics, PerformanceMetrics

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    min_commission: float = 1.0  # Minimum $1 commission
    use_spread: bool = True  # Use bid-ask spread
    allow_shorts: bool = False
    margin_requirement: float = 0.5  # 50% margin for shorts
    interest_rate: float = 0.02  # 2% annual interest on shorts
    data_frequency: str = "1min"  # Data granularity
    execution_delay: int = 0  # Bars delay for execution


@dataclass
class Position:
    """Track individual position."""

    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    commission_paid: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_commission: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    bars_held: int = 0
    max_profit: float = 0.0
    max_loss: float = 0.0

    def update_pnl(self, current_price: float) -> float:
        """Update P&L with current price."""
        if self.quantity > 0:  # Long
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:  # Short
            self.pnl = (self.entry_price - current_price) * abs(self.quantity)

        self.pnl -= self.commission_paid
        if self.exit_price:
            self.pnl -= self.exit_commission

        self.max_profit = max(self.max_profit, self.pnl)
        self.max_loss = min(self.max_loss, self.pnl)

        return self.pnl

    def close(self, exit_price: float, exit_time: datetime, commission: float) -> float:
        """Close position and calculate final P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_commission = commission

        # Calculate final P&L
        if self.quantity > 0:  # Long
            gross_pnl = (exit_price - self.entry_price) * self.quantity
        else:  # Short
            gross_pnl = (self.entry_price - exit_price) * abs(self.quantity)

        self.pnl = gross_pnl - self.commission_paid - commission
        self.return_pct = self.pnl / (self.entry_price * abs(self.quantity))
        self.bars_held = int((exit_time - self.entry_time).total_seconds() / 60)

        return self.pnl


@dataclass
class BacktestResult:
    """Results from backtesting."""

    strategy_name: str
    config: BacktestConfig
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    positions: List[Position]
    trades: pd.DataFrame
    signals: List[Signal]
    daily_returns: pd.Series
    monthly_returns: pd.Series
    drawdown_series: pd.Series
    statistics: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates realistic trading with:
    - Transaction costs
    - Slippage
    - Position tracking
    - Risk management
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.cash = config.initial_capital
        self.equity = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.pending_orders: Dict[str, Signal] = {}
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.all_signals: List[Signal] = []

        logger.info(
            "backtest.engine.initialized",
            initial_capital=config.initial_capital,
            start_date=config.start_date,
            end_date=config.end_date,
        )

    async def run(
        self,
        strategy: Strategy,
        data_pipeline: DataPipeline,
        feature_engine: FeatureEngine,
    ) -> BacktestResult:
        """
        Run backtest for strategy.

        Args:
            strategy: Strategy to test
            data_pipeline: Data source
            feature_engine: Feature calculator

        Returns:
            Backtest results
        """
        logger.info(
            "backtest.starting", strategy=strategy.name, symbols=strategy.symbols
        )

        # Get historical data
        historical_data = {}
        for symbol in strategy.symbols:
            data = await data_pipeline.get_historical_data(
                symbol,
                self.config.start_date,
                self.config.end_date,
                self.config.data_frequency,
            )
            if data is not None and not data.empty:
                historical_data[symbol] = data

        if not historical_data:
            logger.error("backtest.no_data")
            return self._create_empty_result(strategy.name)

        # Initialize strategy
        await strategy.initialize(historical_data)

        # Create time index from all data
        all_timestamps = set()
        for df in historical_data.values():
            all_timestamps.update(df.index.tolist())
        timestamps = sorted(all_timestamps)

        # Replay market data
        for i, timestamp in enumerate(timestamps):
            # Skip if outside test period
            if timestamp < self.config.start_date or timestamp > self.config.end_date:
                continue

            # Get current market snapshot
            market_data = {}
            for symbol, df in historical_data.items():
                if timestamp in df.index:
                    # Get data up to current time
                    market_data[symbol] = df.loc[:timestamp]

            if not market_data:
                continue

            # Update positions with current prices
            self._update_positions(market_data, timestamp)

            # Check stops and targets
            self._check_exit_conditions(market_data, timestamp)

            # Calculate features
            features = {}
            for symbol in market_data:
                feature_set = await feature_engine.calculate_features(
                    symbol, market_data[symbol]
                )
                if feature_set:
                    features[symbol] = feature_set

            # Generate signals
            signals = await strategy.generate_signals(market_data, features)
            self.all_signals.extend(signals)

            # Execute signals
            for signal in signals:
                await self._execute_signal(signal, market_data, timestamp)

            # Record equity
            self._update_equity(market_data, timestamp)
            self.equity_curve.append((timestamp, self.equity))

            # Log progress periodically
            if i % 1000 == 0:
                logger.debug(
                    "backtest.progress",
                    timestamp=timestamp,
                    equity=self.equity,
                    positions=len(self.positions),
                )

        # Close remaining positions at end
        self._close_all_positions(historical_data, timestamps[-1])

        # Create results
        result = self._create_result(strategy.name, historical_data)

        logger.info(
            "backtest.completed",
            strategy=strategy.name,
            total_return=result.metrics.total_return,
            sharpe_ratio=result.metrics.sharpe_ratio,
            max_drawdown=result.metrics.max_drawdown,
        )

        return result

    async def _execute_signal(
        self, signal: Signal, market_data: Dict[str, pd.DataFrame], timestamp: datetime
    ) -> None:
        """Execute trading signal."""
        symbol = signal.symbol

        if symbol not in market_data or market_data[symbol].empty:
            return

        current_bar = market_data[symbol].iloc[-1]

        # Apply slippage
        if signal.signal_type in [SignalType.BUY, SignalType.SCALE_IN]:
            execution_price = current_bar["close"] * (1 + self.config.slippage)
        elif signal.signal_type in [SignalType.SELL, SignalType.SCALE_OUT]:
            execution_price = current_bar["close"] * (1 - self.config.slippage)
        else:
            execution_price = current_bar["close"]

        # Calculate commission
        commission = max(
            self.config.min_commission,
            abs(signal.quantity or 100) * execution_price * self.config.commission,
        )

        # Execute based on signal type
        if signal.signal_type == SignalType.BUY:
            if symbol not in self.positions:
                # Calculate position size if not specified
                if signal.quantity is None:
                    available_cash = self.cash * 0.95  # Keep 5% reserve
                    signal.quantity = int(available_cash / execution_price)

                cost = signal.quantity * execution_price + commission

                if cost <= self.cash:
                    # Open long position
                    position = Position(
                        symbol=symbol,
                        quantity=signal.quantity,
                        entry_price=execution_price,
                        entry_time=timestamp,
                        commission_paid=commission,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                    )
                    self.positions[symbol] = position
                    self.cash -= cost

                    logger.debug(
                        "backtest.position_opened",
                        symbol=symbol,
                        quantity=signal.quantity,
                        price=execution_price,
                        cost=cost,
                    )

        elif signal.signal_type == SignalType.SELL:
            if symbol in self.positions:
                # Close long position
                position = self.positions[symbol]
                proceeds = position.quantity * execution_price - commission
                pnl = position.close(execution_price, timestamp, commission)

                self.cash += proceeds
                self.closed_positions.append(position)
                del self.positions[symbol]

                logger.debug(
                    "backtest.position_closed",
                    symbol=symbol,
                    pnl=pnl,
                    return_pct=position.return_pct,
                )

            elif self.config.allow_shorts and symbol not in self.positions:
                # Open short position
                if signal.quantity is None:
                    available_margin = self.cash * 0.45  # Use 45% of cash for margin
                    signal.quantity = -int(available_margin / execution_price)

                margin_required = (
                    abs(signal.quantity)
                    * execution_price
                    * self.config.margin_requirement
                )

                if margin_required <= self.cash:
                    position = Position(
                        symbol=symbol,
                        quantity=signal.quantity,  # Negative for short
                        entry_price=execution_price,
                        entry_time=timestamp,
                        commission_paid=commission,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                    )
                    self.positions[symbol] = position
                    self.cash -= margin_required + commission

                    logger.debug(
                        "backtest.short_opened",
                        symbol=symbol,
                        quantity=signal.quantity,
                        price=execution_price,
                        margin=margin_required,
                    )

        elif signal.signal_type == SignalType.CLOSE:
            if symbol in self.positions:
                position = self.positions[symbol]

                if position.quantity > 0:  # Close long
                    proceeds = position.quantity * execution_price - commission
                else:  # Close short
                    cost = abs(position.quantity) * execution_price + commission
                    proceeds = -cost + (
                        abs(position.quantity)
                        * position.entry_price
                        * self.config.margin_requirement
                    )

                pnl = position.close(execution_price, timestamp, commission)
                self.cash += proceeds
                self.closed_positions.append(position)
                del self.positions[symbol]

    def _update_positions(
        self, market_data: Dict[str, pd.DataFrame], timestamp: datetime
    ) -> None:
        """Update position P&L with current prices."""
        for symbol, position in self.positions.items():
            if symbol in market_data and not market_data[symbol].empty:
                current_price = market_data[symbol].iloc[-1]["close"]
                position.update_pnl(current_price)
                position.bars_held += 1

    def _check_exit_conditions(
        self, market_data: Dict[str, pd.DataFrame], timestamp: datetime
    ) -> None:
        """Check stop loss and take profit conditions."""
        positions_to_close = []

        for symbol, position in self.positions.items():
            if symbol not in market_data or market_data[symbol].empty:
                continue

            current_bar = market_data[symbol].iloc[-1]
            current_price = current_bar["close"]

            # Check stop loss
            if position.stop_loss:
                if position.quantity > 0 and current_price <= position.stop_loss:
                    positions_to_close.append((symbol, "stop_loss"))
                elif position.quantity < 0 and current_price >= position.stop_loss:
                    positions_to_close.append((symbol, "stop_loss"))

            # Check take profit
            if position.take_profit:
                if position.quantity > 0 and current_price >= position.take_profit:
                    positions_to_close.append((symbol, "take_profit"))
                elif position.quantity < 0 and current_price <= position.take_profit:
                    positions_to_close.append((symbol, "take_profit"))

        # Close positions that hit exits
        for symbol, reason in positions_to_close:
            position = self.positions[symbol]
            current_price = market_data[symbol].iloc[-1]["close"]

            # Apply slippage for stop orders
            if reason == "stop_loss":
                if position.quantity > 0:
                    execution_price = current_price * (1 - self.config.slippage * 2)
                else:
                    execution_price = current_price * (1 + self.config.slippage * 2)
            else:
                execution_price = current_price

            commission = max(
                self.config.min_commission,
                abs(position.quantity) * execution_price * self.config.commission,
            )

            if position.quantity > 0:
                proceeds = position.quantity * execution_price - commission
            else:
                proceeds = (
                    -abs(position.quantity) * execution_price
                    - commission
                    + (
                        abs(position.quantity)
                        * position.entry_price
                        * self.config.margin_requirement
                    )
                )

            pnl = position.close(execution_price, timestamp, commission)
            self.cash += proceeds
            self.closed_positions.append(position)
            del self.positions[symbol]

            logger.debug(
                f"backtest.{reason}_triggered",
                symbol=symbol,
                pnl=pnl,
                execution_price=execution_price,
            )

    def _update_equity(
        self, market_data: Dict[str, pd.DataFrame], timestamp: datetime
    ) -> None:
        """Update total equity value."""
        self.equity = self.cash

        for symbol, position in self.positions.items():
            if symbol in market_data and not market_data[symbol].empty:
                current_price = market_data[symbol].iloc[-1]["close"]
                position_value = position.quantity * current_price
                self.equity += position_value

    def _close_all_positions(
        self, historical_data: Dict[str, pd.DataFrame], timestamp: datetime
    ) -> None:
        """Close all remaining positions at end of backtest."""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]

            if symbol in historical_data and not historical_data[symbol].empty:
                final_price = historical_data[symbol].iloc[-1]["close"]
            else:
                final_price = position.entry_price

            commission = max(
                self.config.min_commission,
                abs(position.quantity) * final_price * self.config.commission,
            )

            pnl = position.close(final_price, timestamp, commission)

            if position.quantity > 0:
                proceeds = position.quantity * final_price - commission
            else:
                proceeds = (
                    -abs(position.quantity) * final_price
                    - commission
                    + (
                        abs(position.quantity)
                        * position.entry_price
                        * self.config.margin_requirement
                    )
                )

            self.cash += proceeds
            self.closed_positions.append(position)

            logger.debug("backtest.final_position_closed", symbol=symbol, pnl=pnl)

        self.positions.clear()

    def _create_result(
        self, strategy_name: str, historical_data: Dict[str, pd.DataFrame]
    ) -> BacktestResult:
        """Create backtest result object."""
        # Create equity curve
        equity_df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
        equity_df.set_index("timestamp", inplace=True)
        equity_series = equity_df["equity"]

        # Calculate returns
        returns = equity_series.pct_change().dropna()
        daily_returns = returns.resample("D").sum()
        monthly_returns = returns.resample("M").sum()

        # Create trades DataFrame
        trades_data = []
        for pos in self.closed_positions:
            trades_data.append(
                {
                    "symbol": pos.symbol,
                    "entry_time": pos.entry_time,
                    "exit_time": pos.exit_time,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "exit_price": pos.exit_price,
                    "pnl": pos.pnl,
                    "return_pct": pos.return_pct,
                    "bars_held": pos.bars_held,
                    "commission": pos.commission_paid + pos.exit_commission,
                }
            )

        trades_df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()

        # Calculate performance metrics
        metrics = calculate_metrics(
            returns=daily_returns,
            equity_curve=equity_series,
            trades=trades_df,
            initial_capital=self.config.initial_capital,
        )

        # Calculate drawdown series
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown_series = (cumulative - running_max) / running_max

        # Additional statistics
        statistics = {
            "total_trades": len(self.closed_positions),
            "winning_trades": len([p for p in self.closed_positions if p.pnl > 0]),
            "losing_trades": len([p for p in self.closed_positions if p.pnl < 0]),
            "avg_bars_held": (
                np.mean([p.bars_held for p in self.closed_positions])
                if self.closed_positions
                else 0
            ),
            "total_commission": sum(
                [p.commission_paid + p.exit_commission for p in self.closed_positions]
            ),
            "final_equity": self.equity,
            "total_return_pct": (self.equity - self.config.initial_capital)
            / self.config.initial_capital
            * 100,
        }

        return BacktestResult(
            strategy_name=strategy_name,
            config=self.config,
            metrics=metrics,
            equity_curve=equity_series,
            positions=self.closed_positions,
            trades=trades_df,
            signals=self.all_signals,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            drawdown_series=drawdown_series,
            statistics=statistics,
        )

    def _create_empty_result(self, strategy_name: str) -> BacktestResult:
        """Create empty result when no data available."""
        return BacktestResult(
            strategy_name=strategy_name,
            config=self.config,
            metrics=PerformanceMetrics(),
            equity_curve=pd.Series([self.config.initial_capital]),
            positions=[],
            trades=pd.DataFrame(),
            signals=[],
            daily_returns=pd.Series(),
            monthly_returns=pd.Series(),
            drawdown_series=pd.Series(),
            statistics={},
        )
