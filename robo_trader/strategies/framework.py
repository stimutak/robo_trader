"""
Base strategy framework for trading strategies.

This module provides the foundation for all trading strategies with:
- Standardized interface for signal generation
- Integration with risk management
- Performance tracking
- State management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..features.engine import FeatureSet
from ..logger import get_logger

logger = get_logger(__name__)


class SignalType(Enum):
    """Types of trading signals."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


@dataclass
class Signal:
    """Trading signal with metadata."""

    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float  # 0-1, confidence/strength of signal
    quantity: Optional[int] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    expected_value: Optional[float] = None
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        """Check if signal requires action."""
        return self.signal_type not in [SignalType.HOLD]

    def calculate_risk_reward(self) -> Optional[float]:
        """Calculate risk:reward ratio if prices are set."""
        if all([self.entry_price, self.stop_loss, self.take_profit]):
            risk = abs(self.entry_price - self.stop_loss)
            reward = abs(self.take_profit - self.entry_price)
            if risk > 0:
                self.risk_reward_ratio = reward / risk
                return self.risk_reward_ratio
        return None


@dataclass
class StrategyState:
    """State tracking for strategy."""

    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    signals_generated: int = 0
    signals_executed: int = 0
    last_signal_time: Optional[datetime] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    internal_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyMetrics:
    """Performance metrics for strategy evaluation."""

    total_signals: int = 0
    win_rate: float = 0.0
    average_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    recovery_factor: float = 0.0
    calmar_ratio: float = 0.0


class Strategy(ABC):
    """
    Base class for all trading strategies.

    Provides common functionality and enforces interface for:
    - Signal generation
    - Risk management integration
    - Performance tracking
    - State management
    """

    def __init__(
        self,
        name: str,
        symbols: List[str],
        lookback_period: int = 100,
        min_data_points: int = 50,
        position_sizing: str = "equal",
        max_positions: int = 10,
        enable_shorts: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize strategy.

        Args:
            name: Strategy identifier
            symbols: List of tradeable symbols
            lookback_period: Bars of history needed
            min_data_points: Minimum data points before generating signals
            position_sizing: Sizing method ('equal', 'volatility', 'risk_parity')
            max_positions: Maximum concurrent positions
            enable_shorts: Allow short positions
            config: Additional configuration
        """
        self.name = name
        self.symbols = symbols
        self.lookback_period = lookback_period
        self.min_data_points = min_data_points
        self.position_sizing = position_sizing
        self.max_positions = max_positions
        self.enable_shorts = enable_shorts
        self.config = config or {}

        self.state = StrategyState()
        self.metrics = StrategyMetrics()
        self._is_initialized = False

        logger.info(
            "strategy.initialized",
            strategy=name,
            symbols_count=len(symbols),
            lookback=lookback_period,
        )

    async def initialize(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """
        Initialize strategy with historical data.

        Args:
            historical_data: Dict of symbol -> DataFrame with OHLCV data
        """
        logger.info("strategy.initializing", strategy=self.name)

        # Validate data
        for symbol in self.symbols:
            if symbol not in historical_data:
                logger.warning("strategy.missing_data", strategy=self.name, symbol=symbol)
                continue

            df = historical_data[symbol]
            if len(df) < self.min_data_points:
                logger.warning(
                    "strategy.insufficient_data",
                    strategy=self.name,
                    symbol=symbol,
                    available=len(df),
                    required=self.min_data_points,
                )

        # Strategy-specific initialization
        await self._initialize(historical_data)
        self._is_initialized = True

        logger.info("strategy.initialized", strategy=self.name)

    @abstractmethod
    async def _initialize(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """
        Strategy-specific initialization.

        Args:
            historical_data: Historical OHLCV data by symbol
        """
        pass

    async def generate_signals(
        self, market_data: Dict[str, pd.DataFrame], features: Dict[str, FeatureSet]
    ) -> List[Signal]:
        """
        Generate trading signals from market data and features.

        Args:
            market_data: Current market data by symbol
            features: Calculated features by symbol

        Returns:
            List of trading signals
        """
        if not self._is_initialized:
            logger.error("strategy.not_initialized", strategy=self.name)
            return []

        try:
            # Generate signals
            signals = await self._generate_signals(market_data, features)

            # Validate and enhance signals
            validated_signals = []
            for signal in signals:
                # Calculate risk:reward if not set
                if signal.risk_reward_ratio is None:
                    signal.calculate_risk_reward()

                # Apply position limits
                if self._can_take_position(signal):
                    validated_signals.append(signal)
                    self.state.signals_generated += 1
                    self.state.last_signal_time = signal.timestamp

                    logger.info(
                        "strategy.signal_generated",
                        strategy=self.name,
                        symbol=signal.symbol,
                        signal_type=signal.signal_type.value,
                        strength=signal.strength,
                        risk_reward=signal.risk_reward_ratio,
                    )

            return validated_signals

        except Exception as e:
            logger.error("strategy.signal_generation_failed", strategy=self.name, error=str(e))
            return []

    @abstractmethod
    async def _generate_signals(
        self, market_data: Dict[str, pd.DataFrame], features: Dict[str, FeatureSet]
    ) -> List[Signal]:
        """
        Strategy-specific signal generation.

        Args:
            market_data: Current market data
            features: Calculated features

        Returns:
            List of raw signals
        """
        pass

    def _can_take_position(self, signal: Signal) -> bool:
        """
        Check if strategy can take the position.

        Args:
            signal: Proposed signal

        Returns:
            True if position can be taken
        """
        # Check position limits
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            current_positions = len(self.state.positions)
            if current_positions >= self.max_positions:
                logger.debug(
                    "strategy.position_limit_reached",
                    strategy=self.name,
                    current=current_positions,
                    max=self.max_positions,
                )
                return False

        # Check if shorts are allowed
        if signal.signal_type == SignalType.SELL and not self.enable_shorts:
            if signal.symbol not in self.state.positions:
                logger.debug(
                    "strategy.short_not_allowed",
                    strategy=self.name,
                    symbol=signal.symbol,
                )
                return False

        return True

    def update_position(self, symbol: str, position: Optional[Dict[str, Any]]) -> None:
        """
        Update strategy's position tracking.

        Args:
            symbol: Symbol to update
            position: Position data or None if closed
        """
        if position:
            self.state.positions[symbol] = position
        else:
            self.state.positions.pop(symbol, None)

    def calculate_position_size(
        self, signal: Signal, account_value: float, current_price: float
    ) -> int:
        """
        Calculate position size for signal.

        Args:
            signal: Trading signal
            account_value: Total account value
            current_price: Current asset price

        Returns:
            Number of shares to trade
        """
        if self.position_sizing == "equal":
            # Equal weight across max positions
            allocation = account_value / self.max_positions
            shares = int(allocation / current_price)

        elif self.position_sizing == "volatility":
            # Size inversely to volatility
            # This would need volatility data from features
            allocation = account_value / self.max_positions
            shares = int(allocation / current_price)

        elif self.position_sizing == "risk_parity":
            # Risk parity allocation
            # This would need risk metrics
            allocation = account_value / self.max_positions
            shares = int(allocation / current_price)

        else:
            # Default equal sizing
            allocation = account_value / self.max_positions
            shares = int(allocation / current_price)

        return max(1, shares)  # At least 1 share

    def get_metrics(self) -> StrategyMetrics:
        """Get current performance metrics."""
        return self.metrics

    def reset(self) -> None:
        """Reset strategy state."""
        self.state = StrategyState()
        self.metrics = StrategyMetrics()
        self._is_initialized = False
        logger.info("strategy.reset", strategy=self.name)
