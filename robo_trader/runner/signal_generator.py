"""
Signal Generator Module - Trading signal generation from multiple sources.

This module handles:
- ML Enhanced strategy signals
- AI Analyst signals from news
- SMA crossover fallback signals
- Mean reversion signals
- Correlation tracking for position sizing

Extracted from runner_async.py to improve modularity.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

import pandas as pd

from ..logger import get_logger
from ..monitoring.performance import Timer

if TYPE_CHECKING:
    from ..analytics.correlation import CorrelationTracker
    from ..monitoring.performance import PerformanceMonitor
    from ..strategies.mean_reversion import MeanReversionStrategy
    from ..strategies.ml_enhanced_strategy import MLEnhancedStrategy
    from ..strategies.ml_strategy import MLStrategy

logger = get_logger(__name__)


class SignalResult:
    """Result of signal generation."""

    def __init__(
        self,
        signal: int,
        confidence: float = 0.5,
        source: str = "UNKNOWN",
        position_size: float = 0.02,
        stop_loss: float = 0.02,
        take_profit: float = 0.05,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.signal = signal  # 1=BUY, -1=SELL, 0=HOLD
        self.confidence = confidence
        self.source = source
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.metadata = metadata or {}

    @property
    def action(self) -> str:
        if self.signal == 1:
            return "BUY"
        elif self.signal == -1:
            return "SELL"
        return "HOLD"


def sma_crossover_signals(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.DataFrame:
    """Generate SMA crossover signals."""
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(window=fast).mean()
    df["sma_slow"] = df["close"].rolling(window=slow).mean()

    # Generate signals
    df["signal"] = 0
    df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
    df.loc[df["sma_fast"] < df["sma_slow"], "signal"] = -1

    return df


class SignalGenerator:
    """
    Generates trading signals from multiple sources.

    Signal sources (in order of priority):
    1. ML Enhanced Strategy - Full ML pipeline with regime detection
    2. AI Analyst - News-based sentiment analysis
    3. Mean Reversion - Statistical arbitrage signals
    4. SMA Crossover - Simple technical fallback

    Usage:
        generator = SignalGenerator(
            ml_enhanced_strategy=ml_strategy,
            ai_analyst=ai_analyst,
            sma_fast=10,
            sma_slow=30,
        )
        result = await generator.generate_signal(symbol, df)
    """

    def __init__(
        self,
        ml_enhanced_strategy: Optional[MLEnhancedStrategy] = None,
        ml_strategy: Optional[MLStrategy] = None,
        ai_analyst: Optional[Any] = None,
        mean_reversion_strategy: Optional[MeanReversionStrategy] = None,
        correlation_tracker: Optional[CorrelationTracker] = None,
        monitor: Optional[PerformanceMonitor] = None,
        sma_fast: int = 10,
        sma_slow: int = 30,
        use_ml_enhanced: bool = True,
        use_ml_strategy: bool = False,
        use_correlation_sizing: bool = False,
    ):
        self.ml_enhanced_strategy = ml_enhanced_strategy
        self.ml_strategy = ml_strategy
        self.ai_analyst = ai_analyst
        self.mean_reversion_strategy = mean_reversion_strategy
        self.correlation_tracker = correlation_tracker
        self.monitor = monitor
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.use_ml_enhanced = use_ml_enhanced
        self.use_ml_strategy = use_ml_strategy
        self.use_correlation_sizing = use_correlation_sizing

        # News cache for AI analyst
        self.news_cache: Dict[str, Any] = {}
        self.last_news_fetch: Optional[datetime] = None

        # AI opportunities from news scan
        self._ai_opportunities: Dict[str, Dict] = {}

        # ML predictions tracking for dashboard
        self._ml_predictions: Dict[str, Dict] = {}

    def set_ai_opportunities(self, opportunities: Dict[str, Dict]) -> None:
        """Set AI-identified opportunities from news scanning."""
        self._ai_opportunities = opportunities

    def get_ml_predictions(self) -> Dict[str, Dict]:
        """Get ML predictions for dashboard display."""
        return self._ml_predictions

    async def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        portfolio_value: float = 100000,
    ) -> SignalResult:
        """
        Generate trading signal for a symbol.

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            portfolio_value: Current portfolio value for sizing

        Returns:
            SignalResult with signal, confidence, and metadata
        """
        if df is None or df.empty:
            return SignalResult(signal=0, source="NO_DATA")

        timer_context = Timer("signal_generation", self.monitor) if self.monitor else None

        try:
            if timer_context:
                timer_context.__enter__()

            # Try ML Enhanced Strategy first
            if self.use_ml_enhanced and self.ml_enhanced_strategy:
                result = await self._generate_ml_enhanced_signal(symbol, df)
                if result.signal != 0:
                    return result

                # Check AI opportunities if ML returned HOLD
                if symbol in self._ai_opportunities:
                    opp = self._ai_opportunities[symbol]
                    result = SignalResult(
                        signal=1,
                        confidence=opp.get("confidence", 0.7),
                        source="AI_OPPORTUNITY",
                        metadata={"reason": opp.get("reason", "AI identified")},
                    )
                    self._ml_predictions[symbol] = {
                        "signal": 1,
                        "confidence": result.confidence,
                        "action": "BUY",
                        "source": "AI_OPPORTUNITY",
                        "timestamp": datetime.now().isoformat(),
                    }
                    logger.info(
                        f"AI OPPORTUNITY BUY for {symbol}: "
                        f"conf={result.confidence:.0%} - {opp.get('reason', '')}"
                    )
                    return result

                # Try AI Analyst for news-based signals
                if self.ai_analyst:
                    ai_result = await self._generate_ai_signal(symbol, df)
                    if ai_result.signal != 0:
                        return ai_result

            # Try ML Strategy (older version)
            elif self.use_ml_strategy and self.ml_strategy:
                result = await self._generate_ml_signal(symbol, df, portfolio_value)
                if result.signal != 0:
                    return result

            # Fallback to SMA crossover
            return self._generate_sma_signal(symbol, df)

        finally:
            if timer_context:
                timer_context.__exit__(None, None, None)

    async def _generate_ml_enhanced_signal(self, symbol: str, df: pd.DataFrame) -> SignalResult:
        """Generate signal using ML Enhanced Strategy."""
        try:
            signal_obj = await self.ml_enhanced_strategy.analyze(symbol, df)

            if signal_obj:
                signal_value = (
                    1 if signal_obj.action == "BUY" else (-1 if signal_obj.action == "SELL" else 0)
                )
                result = SignalResult(
                    signal=signal_value,
                    confidence=signal_obj.confidence,
                    source="ML_ENHANCED",
                    position_size=signal_obj.features.get("position_size", 0.02),
                    stop_loss=signal_obj.features.get("stop_loss", 0.02),
                    take_profit=signal_obj.features.get("take_profit", 0.05),
                )

                self._ml_predictions[symbol] = {
                    "signal": signal_value,
                    "confidence": signal_obj.confidence,
                    "action": signal_obj.action,
                    "source": "ML_ENHANCED",
                    "timestamp": datetime.now().isoformat(),
                }

                return result

            # ML returned no signal
            self._ml_predictions[symbol] = {
                "signal": 0,
                "confidence": 0.5,
                "action": "HOLD",
                "source": "ML_NO_SIGNAL",
                "timestamp": datetime.now().isoformat(),
            }
            return SignalResult(signal=0, confidence=0.5, source="ML_NO_SIGNAL")

        except Exception as e:
            logger.error(f"ML Enhanced signal error for {symbol}: {e}")
            return SignalResult(signal=0, source="ML_ERROR")

    async def _generate_ai_signal(self, symbol: str, df: pd.DataFrame) -> SignalResult:
        """Generate signal using AI Analyst from news."""
        try:
            from ..utils.news_fetcher import fetch_rss_news
        except ImportError:
            return SignalResult(signal=0, source="AI_UNAVAILABLE")

        try:
            # Refresh news cache if stale (5 min)
            now = datetime.now()
            if self.last_news_fetch is None or (now - self.last_news_fetch) > timedelta(minutes=5):
                self.news_cache = {item["title"]: item for item in fetch_rss_news(max_items=15)}
                self.last_news_fetch = now
                logger.info(f"Fetched {len(self.news_cache)} news items")

            # Find relevant news for this symbol
            symbol_news = [
                n for n in self.news_cache.values() if symbol.upper() in n["title"].upper()
            ]

            # Also check general market news
            market_keywords = ["market", "fed", "economy", "inflation", "stocks", "rally", "crash"]
            market_news = [
                n
                for n in self.news_cache.values()
                if any(kw in n["title"].lower() for kw in market_keywords)
            ]

            relevant_news = symbol_news or market_news[:3]

            if relevant_news:
                news_text = "\n".join(
                    [f"- {n['title']} ({n['source']})" for n in relevant_news[:3]]
                )

                analysis = self.ai_analyst.analyze_market_event(
                    symbol=symbol,
                    event_text=news_text,
                    market_data={
                        "price": float(df["close"].iloc[-1]) if len(df) > 0 else 0,
                        "change_5d": (
                            float((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100)
                            if len(df) > 5
                            else 0
                        ),
                    },
                )

                if analysis.suggested_action in ["buy", "sell"] and analysis.confidence > 0.5:
                    signal_value = 1 if analysis.suggested_action == "buy" else -1
                    result = SignalResult(
                        signal=signal_value,
                        confidence=analysis.confidence,
                        source="AI_ANALYST",
                        metadata={"reasoning": analysis.reasoning[:100]},
                    )

                    self._ml_predictions[symbol] = {
                        "signal": signal_value,
                        "confidence": analysis.confidence,
                        "action": analysis.suggested_action.upper(),
                        "source": "AI_ANALYST",
                        "timestamp": datetime.now().isoformat(),
                    }

                    logger.info(
                        f"AI {analysis.suggested_action.upper()} signal for {symbol}: "
                        f"{analysis.reasoning[:80]}"
                    )
                    return result

        except Exception as e:
            logger.warning(f"AI analysis failed for {symbol}: {e}")

        return SignalResult(signal=0, source="AI_NO_SIGNAL")

    async def _generate_ml_signal(
        self, symbol: str, df: pd.DataFrame, portfolio_value: float
    ) -> SignalResult:
        """Generate signal using ML Strategy (older version)."""
        try:
            market_data_dict = {
                "symbol": symbol,
                "data": df,
                "price": float(df["close"].iloc[-1]) if len(df) > 0 else 0,
                "portfolio_value": portfolio_value,
                "atr": df["close"].rolling(14).std().iloc[-1] if len(df) > 14 else 2,
            }

            signal_obj = await self.ml_strategy.generate_signal(
                symbol=symbol, market_data=market_data_dict
            )

            if signal_obj:
                signal_value = (
                    1 if signal_obj.action == "BUY" else (-1 if signal_obj.action == "SELL" else 0)
                )
                return SignalResult(
                    signal=signal_value,
                    confidence=signal_obj.confidence,
                    source="ML_ENSEMBLE",
                )

        except Exception as e:
            logger.error(f"ML Strategy signal error for {symbol}: {e}")

        return SignalResult(signal=0, source="ML_NO_SIGNAL")

    def _generate_sma_signal(self, symbol: str, df: pd.DataFrame) -> SignalResult:
        """Generate signal using SMA crossover (fallback)."""
        try:
            signals = sma_crossover_signals(
                pd.DataFrame({"close": df["close"]}),
                fast=self.sma_fast,
                slow=self.sma_slow,
            )

            if len(signals) > 0:
                signal_value = int(signals["signal"].iloc[-1])
                if signal_value != 0:
                    logger.info(f"SMA crossover signal for {symbol}: {signal_value}")
                    return SignalResult(
                        signal=signal_value,
                        confidence=0.6,
                        source="SMA_CROSSOVER",
                    )

        except Exception as e:
            logger.error(f"SMA signal error for {symbol}: {e}")

        return SignalResult(signal=0, source="NO_SIGNAL")

    async def generate_mean_reversion_signal(
        self, symbol: str, df: pd.DataFrame
    ) -> Optional[SignalResult]:
        """Generate mean reversion signal if strategy is enabled."""
        if not self.mean_reversion_strategy:
            return None

        try:
            from ..features.engine import FeatureSet

            feature_set = FeatureSet(timestamp=pd.Timestamp.now(tz="UTC"), symbol=symbol)

            if len(df) >= 20:
                # Calculate basic features
                feature_set.bb_upper = (
                    df["close"].rolling(20).mean().iloc[-1]
                    + 2 * df["close"].rolling(20).std().iloc[-1]
                )
                feature_set.bb_middle = df["close"].rolling(20).mean().iloc[-1]
                feature_set.bb_lower = (
                    df["close"].rolling(20).mean().iloc[-1]
                    - 2 * df["close"].rolling(20).std().iloc[-1]
                )
                feature_set.atr = (
                    df["close"].rolling(14).std().iloc[-1]
                    if len(df) >= 14
                    else df["close"].iloc[-1] * 0.02
                )

                # RSI calculation
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                feature_set.rsi = (
                    100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50
                )

                # Generate mean reversion signals
                mr_signals = await self.mean_reversion_strategy._generate_signals(
                    {symbol: df}, {symbol: feature_set}
                )

                if mr_signals:
                    for mr_signal in mr_signals:
                        logger.info(
                            "mean_reversion_signal",
                            symbol=symbol,
                            signal_type=mr_signal.signal_type.value,
                            strength=mr_signal.strength,
                            reversion_score=mr_signal.metadata.get("reversion_score", 0),
                        )

                        if mr_signal.strength > 0.7:
                            signal_value = 1 if mr_signal.signal_type.value == "BUY" else -1
                            return SignalResult(
                                signal=signal_value,
                                confidence=mr_signal.strength,
                                source="MEAN_REVERSION",
                                metadata={
                                    "reversion_score": mr_signal.metadata.get("reversion_score", 0)
                                },
                            )

        except Exception as e:
            logger.error(f"Mean reversion signal error for {symbol}: {e}")

        return None

    def update_correlation_tracker(self, symbol: str, df: pd.DataFrame, sector: str = "Unknown"):
        """Update correlation tracker with price data."""
        if self.use_correlation_sizing and self.correlation_tracker:
            self.correlation_tracker.add_price_series(
                symbol=symbol, prices=df["close"], sector=sector
            )
