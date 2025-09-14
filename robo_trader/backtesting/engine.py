"""
Core backtesting engine with real data support and comprehensive metrics.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in the portfolio."""

    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    is_open: bool = True


@dataclass
class Trade:
    """Represents a completed trade."""

    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    commission: float
    duration_days: int
    trade_type: str  # 'long' or 'short'


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    equity_curve: pd.Series
    trades: List[Trade]
    positions: List[Position]
    metrics: Dict[str, float]
    daily_returns: pd.Series
    drawdown_series: pd.Series
    signals: pd.DataFrame


class BacktestEngine:
    """
    Comprehensive backtesting engine with real data support.

    Features:
    - Realistic execution with slippage and market impact
    - Position tracking and P&L calculation
    - Risk management and position sizing
    - Comprehensive performance metrics
    - Support for multiple assets
    """

    def __init__(
        self,
        strategy: Any,
        execution_simulator: Any,
        initial_capital: float = 100000,
        commission: float = 0.001,
        min_commission: float = 1.0,
        position_sizer: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        use_fractional_shares: bool = False,
        max_positions: int = 10,
        rebalance_frequency: str = "daily",
    ):
        """
        Initialize backtest engine.

        Args:
            strategy: Trading strategy instance
            execution_simulator: Execution simulator for realistic fills
            initial_capital: Starting capital
            commission: Commission rate (as decimal)
            min_commission: Minimum commission per trade
            position_sizer: Position sizing algorithm
            risk_manager: Risk management system
            use_fractional_shares: Allow fractional share trading
            max_positions: Maximum concurrent positions
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
        """
        self.strategy = strategy
        self.execution_simulator = execution_simulator
        self.initial_capital = initial_capital
        self.commission = commission
        self.min_commission = min_commission
        self.position_sizer = position_sizer
        self.risk_manager = risk_manager
        self.use_fractional_shares = use_fractional_shares
        self.max_positions = max_positions
        self.rebalance_frequency = rebalance_frequency

        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []

        # Performance tracking
        self.daily_returns: List[float] = []
        self.high_water_mark = initial_capital
        self.max_drawdown = 0.0

    def run(self, data: pd.DataFrame, symbols: Optional[List[str]] = None) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: Historical price data (MultiIndex with symbols if multiple assets)
            symbols: List of symbols to trade (if None, trades all in data)

        Returns:
            BacktestResult with performance metrics
        """
        # Prepare data
        if symbols is None:
            if isinstance(data.index, pd.MultiIndex):
                symbols = data.index.get_level_values(0).unique().tolist()
            else:
                symbols = ["SINGLE"]  # Single asset

        # Initialize strategy
        self.strategy.initialize(symbols=symbols)

        # Store signals for analysis
        all_signals = []

        # Iterate through time periods
        unique_dates = (
            data.index.get_level_values(-1).unique()
            if isinstance(data.index, pd.MultiIndex)
            else data.index.unique()
        )

        for i, timestamp in enumerate(unique_dates):
            try:
                # Get current market data
                current_data = self._get_current_data(data, timestamp, symbols)

                # Update portfolio value
                portfolio_value = self._calculate_portfolio_value(current_data)
                self.equity_curve.append(portfolio_value)
                self.timestamps.append(timestamp)

                # Check for rebalancing
                if self._should_rebalance(timestamp):
                    self._rebalance_portfolio(current_data)

                # Generate signals
                signals = self.strategy.generate_signals(current_data, self.positions)

                if signals:
                    all_signals.append({"timestamp": timestamp, "signals": signals})

                    # Process signals
                    self._process_signals(signals, current_data, timestamp)

                # Update positions
                self._update_positions(current_data, timestamp)

                # Risk management checks
                if self.risk_manager:
                    self._apply_risk_management(current_data)

                # Calculate daily returns
                if len(self.equity_curve) > 1:
                    daily_return = (portfolio_value - self.equity_curve[-1]) / self.equity_curve[-1]
                    self.daily_returns.append(daily_return)

            except Exception as e:
                logger.error(f"Error processing timestamp {timestamp}: {e}")
                continue

        # Close all remaining positions
        self._close_all_positions(current_data, timestamp)

        # Create results
        return self._create_results(all_signals)

    def _get_current_data(
        self, data: pd.DataFrame, timestamp: datetime, symbols: List[str]
    ) -> pd.DataFrame:
        """Get current market data for all symbols."""
        if isinstance(data.index, pd.MultiIndex):
            # Multi-asset data
            current = {}
            for symbol in symbols:
                if (symbol, timestamp) in data.index:
                    current[symbol] = data.loc[(symbol, timestamp)]
            return pd.DataFrame(current).T
        else:
            # Single asset data
            return data.loc[timestamp:timestamp]

    def _calculate_portfolio_value(self, current_data: pd.DataFrame) -> float:
        """Calculate total portfolio value."""
        positions_value = 0

        for symbol, position in self.positions.items():
            if position.is_open:
                if symbol in current_data.index:
                    current_price = current_data.loc[symbol, "close"]
                    positions_value += position.quantity * current_price

        return self.cash + positions_value

    def _should_rebalance(self, timestamp: datetime) -> bool:
        """Check if portfolio should be rebalanced."""
        if not self.timestamps:
            return False

        if self.rebalance_frequency == "daily":
            return True
        elif self.rebalance_frequency == "weekly":
            return timestamp.weekday() == 0  # Monday
        elif self.rebalance_frequency == "monthly":
            return timestamp.day == 1

        return False

    def _rebalance_portfolio(self, current_data: pd.DataFrame) -> None:
        """Rebalance portfolio based on strategy weights."""
        if hasattr(self.strategy, "get_target_weights"):
            target_weights = self.strategy.get_target_weights(current_data, self.positions)

            if target_weights:
                self._execute_rebalance(target_weights, current_data)

    def _execute_rebalance(
        self, target_weights: Dict[str, float], current_data: pd.DataFrame
    ) -> None:
        """Execute portfolio rebalancing."""
        portfolio_value = self._calculate_portfolio_value(current_data)

        for symbol, target_weight in target_weights.items():
            if symbol not in current_data.index:
                continue

            current_price = current_data.loc[symbol, "close"]
            target_value = portfolio_value * target_weight
            target_shares = (
                int(target_value / current_price)
                if not self.use_fractional_shares
                else target_value / current_price
            )

            current_shares = self.positions[symbol].quantity if symbol in self.positions else 0
            shares_to_trade = target_shares - current_shares

            if abs(shares_to_trade) > 0:
                if shares_to_trade > 0:
                    self._execute_buy(
                        symbol, shares_to_trade, current_data.loc[symbol], current_data.name
                    )
                else:
                    self._execute_sell(
                        symbol, abs(shares_to_trade), current_data.loc[symbol], current_data.name
                    )

    def _process_signals(
        self, signals: Dict[str, Any], current_data: pd.DataFrame, timestamp: datetime
    ) -> None:
        """Process trading signals."""
        for symbol, signal in signals.items():
            if symbol not in current_data.index:
                continue

            market_data = current_data.loc[symbol]

            if signal["action"] == "buy":
                quantity = self._calculate_position_size(symbol, market_data, signal)
                if quantity > 0:
                    self._execute_buy(symbol, quantity, market_data, timestamp)

            elif signal["action"] == "sell":
                if symbol in self.positions and self.positions[symbol].is_open:
                    self._execute_sell(
                        symbol, self.positions[symbol].quantity, market_data, timestamp
                    )

            elif signal["action"] == "close":
                if symbol in self.positions and self.positions[symbol].is_open:
                    self._close_position(symbol, market_data, timestamp)

    def _calculate_position_size(
        self, symbol: str, market_data: pd.Series, signal: Dict[str, Any]
    ) -> int:
        """Calculate position size for a trade."""
        if self.position_sizer:
            return self.position_sizer.calculate_size(
                symbol, market_data, signal, self.cash, self.positions
            )
        else:
            # Default sizing - equal weight
            max_position_value = self.cash / max(1, self.max_positions - len(self.positions))
            shares = int(max_position_value / market_data["close"])

            if not self.use_fractional_shares:
                return shares
            else:
                return max_position_value / market_data["close"]

    def _execute_buy(
        self, symbol: str, quantity: float, market_data: pd.Series, timestamp: datetime
    ) -> None:
        """Execute buy order."""
        if quantity <= 0:
            return

        # Simulate execution
        order = self.execution_simulator.simulate_execution(
            symbol=symbol,
            quantity=int(quantity),
            side="buy",
            order_type="market",
            price_data=pd.DataFrame([market_data]),
            timestamp=timestamp,
        )

        if not order.filled:
            return

        # Calculate cost
        total_cost = order.fill_price * quantity + order.execution_cost.commission

        if total_cost > self.cash:
            # Insufficient funds - reduce quantity
            affordable_quantity = int(
                (self.cash - order.execution_cost.commission) / order.fill_price
            )
            if affordable_quantity <= 0:
                return
            quantity = affordable_quantity
            total_cost = order.fill_price * quantity + order.execution_cost.commission

        # Update cash and positions
        self.cash -= total_cost

        if symbol in self.positions and self.positions[symbol].is_open:
            # Add to existing position
            position = self.positions[symbol]
            new_quantity = position.quantity + quantity
            new_entry_price = (
                (position.entry_price * position.quantity) + (order.fill_price * quantity)
            ) / new_quantity
            position.quantity = new_quantity
            position.entry_price = new_entry_price
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol, quantity=quantity, entry_price=order.fill_price, entry_time=timestamp
            )

    def _execute_sell(
        self, symbol: str, quantity: float, market_data: pd.Series, timestamp: datetime
    ) -> None:
        """Execute sell order."""
        if symbol not in self.positions or not self.positions[symbol].is_open:
            return

        position = self.positions[symbol]
        quantity = min(quantity, position.quantity)

        if quantity <= 0:
            return

        # Simulate execution
        order = self.execution_simulator.simulate_execution(
            symbol=symbol,
            quantity=int(quantity),
            side="sell",
            order_type="market",
            price_data=pd.DataFrame([market_data]),
            timestamp=timestamp,
        )

        if not order.filled:
            return

        # Calculate proceeds and P&L
        proceeds = order.fill_price * quantity - order.execution_cost.commission
        cost_basis = position.entry_price * quantity
        pnl = proceeds - cost_basis
        pnl_percent = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

        # Update cash
        self.cash += proceeds

        # Update or close position
        if quantity >= position.quantity:
            # Close entire position
            position.exit_price = order.fill_price
            position.exit_time = timestamp
            position.pnl = pnl
            position.is_open = False

            # Record trade
            self.trades.append(
                Trade(
                    symbol=symbol,
                    entry_time=position.entry_time,
                    exit_time=timestamp,
                    entry_price=position.entry_price,
                    exit_price=order.fill_price,
                    quantity=int(quantity),
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    commission=order.execution_cost.commission,
                    duration_days=(timestamp - position.entry_time).days,
                    trade_type="long",
                )
            )
        else:
            # Partial close
            position.quantity -= quantity

    def _close_position(self, symbol: str, market_data: pd.Series, timestamp: datetime) -> None:
        """Close an entire position."""
        if symbol in self.positions and self.positions[symbol].is_open:
            self._execute_sell(symbol, self.positions[symbol].quantity, market_data, timestamp)

    def _update_positions(self, current_data: pd.DataFrame, timestamp: datetime) -> None:
        """Update position values and check stops."""
        for symbol, position in self.positions.items():
            if not position.is_open or symbol not in current_data.index:
                continue

            current_price = current_data.loc[symbol, "close"]

            # Check stop loss if strategy has it
            if hasattr(self.strategy, "check_stop_loss"):
                if self.strategy.check_stop_loss(position, current_price):
                    self._close_position(symbol, current_data.loc[symbol], timestamp)

            # Check take profit if strategy has it
            if hasattr(self.strategy, "check_take_profit"):
                if self.strategy.check_take_profit(position, current_price):
                    self._close_position(symbol, current_data.loc[symbol], timestamp)

    def _apply_risk_management(self, current_data: pd.DataFrame) -> None:
        """Apply risk management rules."""
        if not self.risk_manager:
            return

        portfolio_value = self._calculate_portfolio_value(current_data)

        # Check drawdown
        if portfolio_value > self.high_water_mark:
            self.high_water_mark = portfolio_value

        drawdown = (self.high_water_mark - portfolio_value) / self.high_water_mark

        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Apply risk rules
        actions = self.risk_manager.check_risk(
            portfolio_value=portfolio_value,
            positions=self.positions,
            drawdown=drawdown,
            current_data=current_data,
        )

        if actions:
            self._execute_risk_actions(actions, current_data)

    def _execute_risk_actions(
        self, actions: List[Dict[str, Any]], current_data: pd.DataFrame
    ) -> None:
        """Execute risk management actions."""
        for action in actions:
            if action["type"] == "close_all":
                self._close_all_positions(current_data, current_data.name)
            elif action["type"] == "reduce_position":
                symbol = action["symbol"]
                reduction = action["reduction"]
                if symbol in self.positions:
                    quantity = int(self.positions[symbol].quantity * reduction)
                    self._execute_sell(
                        symbol, quantity, current_data.loc[symbol], current_data.name
                    )

    def _close_all_positions(self, current_data: pd.DataFrame, timestamp: datetime) -> None:
        """Close all open positions."""
        for symbol in list(self.positions.keys()):
            if self.positions[symbol].is_open and symbol in current_data.index:
                self._close_position(symbol, current_data.loc[symbol], timestamp)

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {
                "total_return": 0,
                "num_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "calmar_ratio": 0,
            }

        # Basic metrics
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital

        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Risk-adjusted metrics
        if self.daily_returns:
            returns_array = np.array(self.daily_returns)

            # Sharpe ratio (annualized)
            if len(returns_array) > 1:
                sharpe_ratio = np.sqrt(252) * np.mean(returns_array) / np.std(returns_array)
            else:
                sharpe_ratio = 0

            # Sortino ratio
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                sortino_ratio = np.sqrt(252) * np.mean(returns_array) / np.std(downside_returns)
            else:
                sortino_ratio = sharpe_ratio
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # Calmar ratio
        calmar_ratio = total_return / self.max_drawdown if self.max_drawdown > 0 else 0

        # Trade statistics
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        avg_duration = np.mean([t.duration_days for t in self.trades]) if self.trades else 0

        return {
            "total_return": total_return,
            "total_pnl": total_pnl,
            "num_trades": len(self.trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_duration_days": avg_duration,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "returns": total_return * 100,  # Percentage
            "final_equity": self.equity_curve[-1],
        }

    def _create_results(self, signals: List[Dict]) -> BacktestResult:
        """Create backtest results object."""
        # Create equity curve series
        equity_series = pd.Series(
            self.equity_curve, index=self.timestamps if self.timestamps else pd.DatetimeIndex([])
        )

        # Create daily returns series
        returns_series = pd.Series(
            self.daily_returns,
            index=self.timestamps[1:] if len(self.timestamps) > 1 else pd.DatetimeIndex([]),
        )

        # Calculate drawdown series
        drawdown_series = self._calculate_drawdown_series(equity_series)

        # Create signals DataFrame
        signals_df = pd.DataFrame(signals) if signals else pd.DataFrame()

        # Calculate metrics
        metrics = self.calculate_metrics()

        return BacktestResult(
            equity_curve=equity_series,
            trades=self.trades,
            positions=list(self.positions.values()),
            metrics=metrics,
            daily_returns=returns_series,
            drawdown_series=drawdown_series,
            signals=signals_df,
        )

    def _calculate_drawdown_series(self, equity_series: pd.Series) -> pd.Series:
        """Calculate drawdown series from equity curve."""
        if equity_series.empty:
            return pd.Series()

        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max

        return drawdown
