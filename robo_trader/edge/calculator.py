"""
Edge Calculator for trade quality assessment.

This module implements Expected Value and risk:reward calculations
to quantify trade quality before position entry.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..logger import get_logger

logger = get_logger(__name__)


class ProbabilityMethod(Enum):
    """Methods for estimating win probability."""

    HISTORICAL = "historical"
    VOLATILITY = "volatility"
    TECHNICAL = "technical"
    ENSEMBLE = "ensemble"
    ML_MODEL = "ml_model"


@dataclass
class TradeEdge:
    """Complete edge metrics for a trade."""

    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float

    # Core metrics
    risk_reward_ratio: float
    expected_value: float
    win_probability: float
    loss_probability: float

    # Detailed metrics
    risk_amount: float
    reward_amount: float
    expected_profit: float
    expected_loss: float

    # Quality scores
    edge_score: float  # 0-100 composite score
    confidence: float  # 0-1 confidence in estimates

    # Additional context
    volatility: Optional[float] = None
    avg_true_range: Optional[float] = None
    historical_win_rate: Optional[float] = None
    breakeven_probability: Optional[float] = None
    kelly_fraction: Optional[float] = None

    @property
    def is_positive_ev(self) -> bool:
        """Check if trade has positive expected value."""
        return self.expected_value > 0

    @property
    def meets_minimum_rr(self, min_rr: float = 1.8) -> bool:
        """Check if trade meets minimum risk:reward ratio."""
        return self.risk_reward_ratio >= min_rr

    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe-like ratio for the trade."""
        if self.risk_amount > 0:
            return self.expected_value / self.risk_amount
        return 0.0


@dataclass
class EdgeMetrics:
    """Aggregate edge metrics for strategy evaluation."""

    total_trades: int
    positive_ev_trades: int
    negative_ev_trades: int
    avg_risk_reward: float
    avg_expected_value: float
    avg_win_probability: float
    best_edge_score: float
    worst_edge_score: float
    total_expected_value: float
    avg_kelly_fraction: float


class EdgeCalculator:
    """
    Calculate trade edge metrics including EV and risk:reward.

    Features:
    - Multiple probability estimation methods
    - Historical win rate tracking
    - Volatility-based probability models
    - Kelly criterion calculation
    - Composite edge scoring
    """

    def __init__(
        self,
        default_method: ProbabilityMethod = ProbabilityMethod.ENSEMBLE,
        lookback_days: int = 60,
        min_samples: int = 30,
        confidence_threshold: float = 0.6,
        volatility_window: int = 20,
    ):
        """
        Initialize EdgeCalculator.

        Args:
            default_method: Default probability estimation method
            lookback_days: Days of history for win rate calculation
            min_samples: Minimum trades for statistical significance
            confidence_threshold: Minimum confidence for valid estimates
            volatility_window: Period for volatility calculation
        """
        self.default_method = default_method
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.volatility_window = volatility_window

        # Track historical performance by symbol
        self.trade_history: Dict[str, List[Dict[str, Any]]] = {}
        self.win_rates: Dict[str, float] = {}
        self.avg_returns: Dict[str, Tuple[float, float]] = {}  # (avg_win, avg_loss)

        logger.info(
            "edge_calculator.initialized",
            method=default_method.value,
            lookback_days=lookback_days,
        )

    def calculate_edge(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        current_volatility: Optional[float] = None,
        atr: Optional[float] = None,
        technical_signals: Optional[Dict[str, float]] = None,
        method: Optional[ProbabilityMethod] = None,
    ) -> TradeEdge:
        """
        Calculate comprehensive edge metrics for a trade.

        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price
            take_profit: Take profit target
            current_volatility: Current volatility (annualized)
            atr: Average True Range
            technical_signals: Technical indicator values
            method: Probability estimation method

        Returns:
            TradeEdge with all calculated metrics
        """
        # Calculate basic risk:reward
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = abs(take_profit - entry_price)

        if risk_amount <= 0:
            logger.warning("edge_calculator.zero_risk", symbol=symbol)
            return self._create_invalid_edge(
                symbol, entry_price, stop_loss, take_profit
            )

        risk_reward_ratio = reward_amount / risk_amount

        # Estimate win probability
        method = method or self.default_method
        win_probability, confidence = self._estimate_probability(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volatility=current_volatility,
            atr=atr,
            technical_signals=technical_signals,
            method=method,
        )

        loss_probability = 1 - win_probability

        # Calculate expected value
        expected_profit = win_probability * reward_amount
        expected_loss = loss_probability * risk_amount
        expected_value = expected_profit - expected_loss

        # Calculate edge score (0-100)
        edge_score = self._calculate_edge_score(
            risk_reward_ratio=risk_reward_ratio,
            win_probability=win_probability,
            expected_value=expected_value,
            confidence=confidence,
        )

        # Calculate Kelly fraction if positive EV
        kelly_fraction = None
        if expected_value > 0 and risk_reward_ratio > 0:
            kelly_fraction = self._calculate_kelly_fraction(
                win_probability=win_probability, risk_reward_ratio=risk_reward_ratio
            )

        # Get historical win rate for this symbol
        historical_win_rate = self.win_rates.get(symbol)

        edge = TradeEdge(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            expected_value=expected_value,
            win_probability=win_probability,
            loss_probability=loss_probability,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            expected_profit=expected_profit,
            expected_loss=expected_loss,
            edge_score=edge_score,
            confidence=confidence,
            volatility=current_volatility,
            avg_true_range=atr,
            historical_win_rate=historical_win_rate,
            kelly_fraction=kelly_fraction,
        )

        logger.debug(
            "edge_calculator.calculated",
            symbol=symbol,
            risk_reward=risk_reward_ratio,
            expected_value=expected_value,
            win_probability=win_probability,
            edge_score=edge_score,
        )

        return edge

    def _estimate_probability(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        volatility: Optional[float],
        atr: Optional[float],
        technical_signals: Optional[Dict[str, float]],
        method: ProbabilityMethod,
    ) -> Tuple[float, float]:
        """
        Estimate win probability using specified method.

        Returns:
            Tuple of (probability, confidence)
        """
        if method == ProbabilityMethod.HISTORICAL:
            return self._historical_probability(symbol)

        elif method == ProbabilityMethod.VOLATILITY:
            return self._volatility_probability(
                entry_price, stop_loss, take_profit, volatility, atr
            )

        elif method == ProbabilityMethod.TECHNICAL:
            return self._technical_probability(technical_signals)

        elif method == ProbabilityMethod.ENSEMBLE:
            return self._ensemble_probability(
                symbol,
                entry_price,
                stop_loss,
                take_profit,
                volatility,
                atr,
                technical_signals,
            )

        else:
            # Default fallback
            return 0.5, 0.5

    def _historical_probability(self, symbol: str) -> Tuple[float, float]:
        """Calculate probability based on historical win rate."""
        if symbol not in self.win_rates:
            return 0.5, 0.3  # Low confidence default

        win_rate = self.win_rates[symbol]
        trades_count = len(self.trade_history.get(symbol, []))

        # Calculate confidence based on sample size
        confidence = min(trades_count / self.min_samples, 1.0) * 0.8

        return win_rate, confidence

    def _volatility_probability(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        volatility: Optional[float],
        atr: Optional[float],
    ) -> Tuple[float, float]:
        """
        Calculate probability using volatility/ATR model.

        Uses a simplified model based on price movement distributions.
        """
        if not volatility and not atr:
            return 0.5, 0.3

        # Use ATR or volatility for price movement estimation
        if atr:
            daily_move = atr
        else:
            # Convert annual volatility to daily
            daily_move = entry_price * (volatility / np.sqrt(252))

        # Calculate required moves
        stop_distance = abs(entry_price - stop_loss)
        target_distance = abs(take_profit - entry_price)

        # Simple probability model based on distance/volatility
        # Assumes normally distributed returns
        stop_z_score = stop_distance / daily_move
        target_z_score = target_distance / daily_move

        # Probability of hitting target before stop
        # Simplified - in reality would use more complex model
        if target_z_score < stop_z_score:
            # Target is closer - higher probability
            base_prob = 0.5 + (0.2 * (stop_z_score - target_z_score))
        else:
            # Stop is closer - lower probability
            base_prob = 0.5 - (0.2 * (target_z_score - stop_z_score))

        probability = np.clip(base_prob, 0.2, 0.8)
        confidence = 0.6  # Moderate confidence in model

        return probability, confidence

    def _technical_probability(
        self, technical_signals: Optional[Dict[str, float]]
    ) -> Tuple[float, float]:
        """
        Estimate probability from technical indicators.
        """
        if not technical_signals:
            return 0.5, 0.3

        score = 0.5  # Base probability
        signals_used = 0

        # RSI signal
        if "rsi" in technical_signals:
            rsi = technical_signals["rsi"]
            if rsi < 30:
                score += 0.15  # Oversold - bullish
            elif rsi > 70:
                score -= 0.15  # Overbought - bearish
            signals_used += 1

        # MACD signal
        if "macd_histogram" in technical_signals:
            macd_hist = technical_signals["macd_histogram"]
            if macd_hist > 0:
                score += 0.1
            else:
                score -= 0.1
            signals_used += 1

        # Trend strength
        if "trend_strength" in technical_signals:
            trend = technical_signals["trend_strength"]
            score += trend * 0.2  # -1 to 1 scaled impact
            signals_used += 1

        probability = np.clip(score, 0.2, 0.8)
        confidence = min(signals_used / 5, 1.0) * 0.7

        return probability, confidence

    def _ensemble_probability(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        volatility: Optional[float],
        atr: Optional[float],
        technical_signals: Optional[Dict[str, float]],
    ) -> Tuple[float, float]:
        """
        Combine multiple probability estimation methods.
        """
        probabilities = []
        weights = []

        # Historical probability
        hist_prob, hist_conf = self._historical_probability(symbol)
        if hist_conf > 0.3:
            probabilities.append(hist_prob)
            weights.append(hist_conf * 2)  # Weight historical more

        # Volatility probability
        vol_prob, vol_conf = self._volatility_probability(
            entry_price, stop_loss, take_profit, volatility, atr
        )
        probabilities.append(vol_prob)
        weights.append(vol_conf)

        # Technical probability
        tech_prob, tech_conf = self._technical_probability(technical_signals)
        if tech_conf > 0.3:
            probabilities.append(tech_prob)
            weights.append(tech_conf)

        # Weighted average
        if probabilities:
            weights = np.array(weights)
            probabilities = np.array(probabilities)
            weighted_prob = np.average(probabilities, weights=weights)

            # Confidence is average of component confidences
            avg_confidence = np.mean([hist_conf, vol_conf, tech_conf])

            return weighted_prob, avg_confidence

        return 0.5, 0.3

    def _calculate_edge_score(
        self,
        risk_reward_ratio: float,
        win_probability: float,
        expected_value: float,
        confidence: float,
    ) -> float:
        """
        Calculate composite edge score (0-100).

        Combines multiple factors into single quality score.
        """
        score = 0.0

        # Risk:reward component (0-40 points)
        if risk_reward_ratio >= 3:
            rr_score = 40
        elif risk_reward_ratio >= 2:
            rr_score = 30
        elif risk_reward_ratio >= 1.5:
            rr_score = 20
        elif risk_reward_ratio >= 1:
            rr_score = 10
        else:
            rr_score = 0

        score += rr_score

        # Win probability component (0-30 points)
        prob_score = win_probability * 30
        score += prob_score

        # Expected value component (0-20 points)
        if expected_value > 0:
            ev_score = min(expected_value * 10, 20)
        else:
            ev_score = max(expected_value * 5, -10)
        score += ev_score

        # Confidence component (0-10 points)
        conf_score = confidence * 10
        score += conf_score

        return np.clip(score, 0, 100)

    def _calculate_kelly_fraction(
        self, win_probability: float, risk_reward_ratio: float
    ) -> float:
        """
        Calculate Kelly Criterion fraction for position sizing.

        Kelly % = (p * b - q) / b
        where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = risk:reward ratio
        """
        q = 1 - win_probability
        kelly = (win_probability * risk_reward_ratio - q) / risk_reward_ratio

        # Apply Kelly fraction cap (typically 25% max)
        return np.clip(kelly, 0, 0.25)

    def update_history(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        is_win: bool,
        return_pct: float,
    ) -> None:
        """
        Update historical trade data for probability estimation.

        Args:
            symbol: Trading symbol
            entry_price: Entry price of trade
            exit_price: Exit price of trade
            is_win: Whether trade was profitable
            return_pct: Percentage return of trade
        """
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []

        trade = {
            "timestamp": datetime.now(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "is_win": is_win,
            "return_pct": return_pct,
        }

        self.trade_history[symbol].append(trade)

        # Limit history size
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        self.trade_history[symbol] = [
            t for t in self.trade_history[symbol] if t["timestamp"] > cutoff_date
        ]

        # Update win rate
        if len(self.trade_history[symbol]) >= 1:
            wins = sum(1 for t in self.trade_history[symbol] if t["is_win"])
            self.win_rates[symbol] = wins / len(self.trade_history[symbol])

            # Update average returns
            winning_trades = [
                t["return_pct"] for t in self.trade_history[symbol] if t["is_win"]
            ]
            losing_trades = [
                abs(t["return_pct"])
                for t in self.trade_history[symbol]
                if not t["is_win"]
            ]

            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0

            self.avg_returns[symbol] = (avg_win, avg_loss)

            logger.debug(
                "edge_calculator.history_updated",
                symbol=symbol,
                win_rate=self.win_rates[symbol],
                trades_count=len(self.trade_history[symbol]),
            )

    def get_aggregate_metrics(self, edges: List[TradeEdge]) -> EdgeMetrics:
        """
        Calculate aggregate metrics from multiple edges.

        Args:
            edges: List of TradeEdge objects

        Returns:
            Aggregate EdgeMetrics
        """
        if not edges:
            return EdgeMetrics(
                total_trades=0,
                positive_ev_trades=0,
                negative_ev_trades=0,
                avg_risk_reward=0,
                avg_expected_value=0,
                avg_win_probability=0,
                best_edge_score=0,
                worst_edge_score=0,
                total_expected_value=0,
                avg_kelly_fraction=0,
            )

        positive_ev = sum(1 for e in edges if e.is_positive_ev)
        negative_ev = len(edges) - positive_ev

        kelly_fractions = [e.kelly_fraction for e in edges if e.kelly_fraction]

        return EdgeMetrics(
            total_trades=len(edges),
            positive_ev_trades=positive_ev,
            negative_ev_trades=negative_ev,
            avg_risk_reward=np.mean([e.risk_reward_ratio for e in edges]),
            avg_expected_value=np.mean([e.expected_value for e in edges]),
            avg_win_probability=np.mean([e.win_probability for e in edges]),
            best_edge_score=max(e.edge_score for e in edges),
            worst_edge_score=min(e.edge_score for e in edges),
            total_expected_value=sum(e.expected_value for e in edges),
            avg_kelly_fraction=np.mean(kelly_fractions) if kelly_fractions else 0,
        )

    def _create_invalid_edge(
        self, symbol: str, entry_price: float, stop_loss: float, take_profit: float
    ) -> TradeEdge:
        """Create edge with invalid/zero values."""
        return TradeEdge(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=0,
            expected_value=0,
            win_probability=0,
            loss_probability=1,
            risk_amount=0,
            reward_amount=0,
            expected_profit=0,
            expected_loss=0,
            edge_score=0,
            confidence=0,
        )
