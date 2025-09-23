#!/usr/bin/env python3
"""
Phase 2 Integration Tests - Smart Data Pipeline

Tests all Phase 2 components:
- Real-time data pipeline
- Feature engineering
- Technical indicators
- Data validation
- Database integration
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robo_trader.config import load_config
from robo_trader.data.pipeline import BarData, DataPipeline, DataSubscriber, TickData
from robo_trader.data.validation import DataValidator, MarketHoursValidator
from robo_trader.features.engine import FeatureEngine
from robo_trader.features.indicators import IndicatorConfig, TechnicalIndicators


class MockSubscriber(DataSubscriber):
    """Test subscriber for pipeline testing."""

    def __init__(self):
        super().__init__("test_subscriber")
        self.ticks_received = 0
        self.bars_received = 0
        self.last_tick = None
        self.last_bar = None

    async def on_tick(self, tick: TickData) -> None:
        """Handle tick data."""
        self.ticks_received += 1
        self.last_tick = tick

    async def on_bar(self, bar: BarData) -> None:
        """Handle bar data."""
        self.bars_received += 1
        self.last_bar = bar


def print_header(title: str):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"        {details}")


@pytest.mark.asyncio
async def test_data_pipeline():
    """Test data pipeline functionality."""
    print_header("Testing Data Pipeline")

    try:
        config = load_config()
        pipeline = DataPipeline(config)

        # Test 1: Pipeline initialization
        assert pipeline is not None, "Pipeline creation failed"
        assert len(pipeline.tick_buffer) == len(config.symbols), "Buffer initialization failed"
        print_result(
            "Pipeline initialization", True, f"Created buffers for {len(config.symbols)} symbols"
        )

        # Test 2: Start pipeline
        await pipeline.start()
        assert pipeline.running, "Pipeline failed to start"
        print_result("Pipeline start", True)

        # Test 3: Subscribe to events
        subscriber = MockSubscriber()
        pipeline.subscribe(subscriber)

        # Wait for some mock data
        await asyncio.sleep(1)

        assert subscriber.ticks_received > 0, "No ticks received"
        print_result("Data streaming", True, f"Received {subscriber.ticks_received} ticks")

        # Test 4: Historical data fetch
        hist_data = await pipeline.get_historical_data("AAPL", "5 D", "5 mins")
        assert hist_data is not None, "Historical data fetch failed"
        assert len(hist_data) > 0, "No historical data returned"
        print_result("Historical data", True, f"Fetched {len(hist_data)} bars")

        # Test 5: Tick aggregation
        bar = await pipeline.aggregate_ticks_to_bars("AAPL", timedelta(minutes=1))
        print_result(
            "Tick aggregation",
            bar is not None,
            "Successfully aggregated ticks to bars" if bar else "",
        )

        # Test 6: Pipeline metrics
        metrics = pipeline.get_metrics()
        assert metrics["ticks_received"] > 0, "No metrics recorded"
        print_result(
            "Pipeline metrics",
            True,
            f"Ticks: {metrics['ticks_received']}, Subscribers: {metrics['subscribers']}",
        )

        # Stop pipeline
        await pipeline.stop()

    except Exception as e:
        print_result("Data pipeline", False, str(e))
        raise


@pytest.mark.asyncio
async def test_feature_engine():
    """Test feature calculation engine."""
    print_header("Testing Feature Engine")

    try:
        config = load_config()
        engine = FeatureEngine(config)

        # Test 1: Engine initialization
        assert engine is not None, "Engine creation failed"
        print_result("Feature engine initialization", True)

        # Test 2: Start engine
        await engine.start()
        assert engine.running, "Engine failed to start"
        print_result("Engine start", True)

        # Test 3: Add sample data
        # Create sample price data
        dates = pd.date_range(end=datetime.now(), periods=100, freq="5min")
        sample_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.randn(100) + 100,
                "high": np.random.randn(100) + 101,
                "low": np.random.randn(100) + 99,
                "close": np.random.randn(100) + 100,
                "volume": np.random.uniform(100000, 1000000, 100),
            }
        )

        engine.price_data["TEST"] = sample_data

        # Test 4: Calculate features
        features = await engine.calculate_features("TEST")
        assert features is not None, "Feature calculation failed"
        assert features.symbol == "TEST", "Incorrect symbol"

        non_null = features.get_non_null_features()
        print_result("Feature calculation", True, f"Calculated {len(non_null)} features")

        # Test 5: Check specific indicators
        has_rsi = features.rsi is not None
        has_macd = features.macd_line is not None
        has_bb = features.bb_upper is not None

        print_result(
            "Technical indicators",
            has_rsi or has_macd or has_bb,
            f"RSI: {has_rsi}, MACD: {has_macd}, BB: {has_bb}",
        )

        # Test 6: Feature caching
        cached = await engine.calculate_features("TEST")
        assert engine.metrics["cache_hits"] > 0, "Cache not working"
        print_result(
            "Feature caching",
            True,
            f"Cache hit rate: {engine.metrics['cache_hits']/(engine.metrics['cache_hits']+engine.metrics['cache_misses']):.1%}",
        )

        # Stop engine
        await engine.stop()

    except Exception as e:
        print_result("Feature engine", False, str(e))
        raise


def test_technical_indicators():
    """Test technical indicator calculations."""
    print_header("Testing Technical Indicators")

    try:
        indicators = TechnicalIndicators()

        # Create sample data
        prices = pd.Series(np.random.randn(100) + 100)
        df = pd.DataFrame(
            {
                "high": prices + np.random.rand(100),
                "low": prices - np.random.rand(100),
                "close": prices,
                "volume": np.random.uniform(100000, 1000000, 100),
            }
        )
        df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])

        # Test 1: RSI
        rsi = indicators.rsi(prices)
        assert rsi is not None, "RSI calculation failed"
        assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"
        print_result("RSI calculation", True, f"RSI = {rsi:.2f}")

        # Test 2: MACD
        macd = indicators.macd(prices)
        assert macd is not None, "MACD calculation failed"
        assert all(k in macd for k in ["macd", "signal", "histogram"]), "MACD missing components"
        print_result(
            "MACD calculation", True, f"MACD = {macd['macd']:.4f}, Signal = {macd['signal']:.4f}"
        )

        # Test 3: Bollinger Bands
        bb = indicators.bollinger_bands(prices)
        assert bb is not None, "Bollinger Bands calculation failed"
        assert bb["lower"] < bb["middle"] < bb["upper"], "Invalid BB relationship"
        print_result(
            "Bollinger Bands",
            True,
            f"Upper = {bb['upper']:.2f}, Middle = {bb['middle']:.2f}, Lower = {bb['lower']:.2f}",
        )

        # Test 4: ATR
        atr = indicators.atr(df)
        assert atr is not None, "ATR calculation failed"
        assert atr > 0, f"Invalid ATR: {atr}"
        print_result("ATR calculation", True, f"ATR = {atr:.4f}")

        # Test 5: Volume indicators
        obv = indicators.obv(df)
        assert obv is not None, "OBV calculation failed"
        print_result("OBV calculation", True, f"OBV = {obv:,.0f}")

        vwap = indicators.vwap(df)
        assert vwap is not None, "VWAP calculation failed"
        print_result("VWAP calculation", True, f"VWAP = {vwap:.2f}")

        # Test 6: Additional indicators
        stoch = indicators.stochastic(df)
        assert stoch is not None, "Stochastic calculation failed"
        print_result("Stochastic", True, f"K = {stoch['k']:.2f}, D = {stoch['d']:.2f}")

    except Exception as e:
        print_result("Technical indicators", False, str(e))
        raise


def test_data_validation():
    """Test data validation and quality checks."""
    print_header("Testing Data Validation")

    try:
        config = load_config()
        validator = DataValidator(config)

        # Test 1: Valid tick
        valid_tick = TickData(
            timestamp=datetime.now(),
            symbol="AAPL",
            bid=150.00,
            ask=150.05,
            last=150.02,
            bid_size=100,
            ask_size=100,
            last_size=100,
            volume=1000000,
        )

        results = validator.validate_tick(valid_tick)
        all_valid = all(r.is_valid for r in results if r.check_name != "market_hours")
        print_result(
            "Valid tick validation",
            all_valid,
            f"{len([r for r in results if r.is_valid])} checks passed",
        )

        # Test 2: Invalid tick (inverted market)
        invalid_tick = TickData(
            timestamp=datetime.now(),
            symbol="AAPL",
            bid=150.05,
            ask=150.00,  # Bid > Ask
            last=150.02,
            bid_size=100,
            ask_size=100,
            last_size=100,
            volume=1000000,
        )

        results = validator.validate_tick(invalid_tick)
        has_error = any(not r.is_valid and r.severity == "error" for r in results)
        print_result("Invalid tick detection", has_error, "Correctly identified inverted market")

        # Test 3: Outlier detection
        outlier_tick = TickData(
            timestamp=datetime.now(),
            symbol="AAPL",
            bid=500.00,  # Way outside normal range
            ask=500.05,
            last=500.02,
            bid_size=100,
            ask_size=100,
            last_size=100,
            volume=1000000,
        )

        # Add some history first
        for i in range(50):
            normal_tick = TickData(
                timestamp=datetime.now(),
                symbol="AAPL",
                bid=150.00 + np.random.randn(),
                ask=150.05 + np.random.randn(),
                last=150.02 + np.random.randn(),
                bid_size=100,
                ask_size=100,
                last_size=100,
                volume=1000000,
            )
            validator.validate_tick(normal_tick)

        results = validator.validate_tick(outlier_tick)
        outlier_detected = any(
            "outlier" in r.check_name.lower() and not r.is_valid for r in results
        )
        print_result("Outlier detection", outlier_detected, "Correctly identified price outlier")

        # Test 4: Market hours validation
        market_validator = MarketHoursValidator()

        # Regular hours
        regular_time = datetime(2025, 8, 26, 10, 30)  # Tuesday 10:30 AM
        is_open = market_validator.is_market_open(regular_time)
        print_result(
            "Market hours (regular)",
            is_open,
            f"Session: {market_validator.get_session_type(regular_time)}",
        )

        # After hours
        after_time = datetime(2025, 8, 26, 18, 0)  # Tuesday 6:00 PM
        is_extended = market_validator.is_market_open(after_time, extended=True)
        print_result(
            "Market hours (extended)",
            is_extended,
            f"Session: {market_validator.get_session_type(after_time)}",
        )

        # Weekend
        weekend_time = datetime(2025, 8, 30, 10, 0)  # Saturday
        is_closed = not market_validator.is_market_open(weekend_time, extended=True)
        print_result("Market hours (weekend)", is_closed, "Correctly identified closed market")

        # Test 5: Data quality metrics
        metrics = validator.get_metrics()
        assert metrics.total_ticks > 0, "No ticks processed"
        print_result(
            "Data quality metrics",
            True,
            f"Validity rate: {metrics.validity_rate:.1f}%, "
            f"Outliers: {metrics.outliers_detected}",
        )

    except Exception as e:
        print_result("Data validation", False, str(e))
        raise


def test_bar_validation():
    """Test OHLCV bar validation."""
    print_header("Testing Bar Validation")

    try:
        config = load_config()
        validator = DataValidator(config)

        # Test 1: Valid bar
        valid_bar = BarData(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=150.00,
            high=151.00,
            low=149.00,
            close=150.50,
            volume=1000000,
        )

        results = validator.validate_bar(valid_bar)
        all_valid = all(r.is_valid for r in results)
        print_result("Valid bar", all_valid, "All OHLCV relationships correct")

        # Test 2: Invalid bar (high < low)
        invalid_bar = BarData(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=150.00,
            high=149.00,  # High < Low
            low=151.00,
            close=150.50,
            volume=1000000,
        )

        results = validator.validate_bar(invalid_bar)
        has_error = any(not r.is_valid for r in results)
        print_result("Invalid bar detection", has_error, "Correctly identified invalid OHLC")

        # Test 3: Zero volume bar
        zero_vol_bar = BarData(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=150.00,
            high=151.00,
            low=149.00,
            close=150.50,
            volume=0,
        )

        results = validator.validate_bar(zero_vol_bar)
        has_warning = any(r.severity == "warning" for r in results)
        print_result("Zero volume detection", has_warning, "Correctly flagged zero volume")

    except Exception as e:
        print_result("Bar validation", False, str(e))
        raise


@pytest.mark.asyncio
async def test_integration():
    """Test integration of all Phase 2 components."""
    print_header("Testing Phase 2 Integration")

    try:
        config = load_config()

        # Create components
        pipeline = DataPipeline(config)
        feature_engine = FeatureEngine(config)
        validator = DataValidator(config)

        # Connect feature engine to pipeline
        pipeline.subscribe(feature_engine)

        # Start components
        await pipeline.start()
        await feature_engine.start()

        print_result("Component integration", True, "Pipeline connected to feature engine")

        # Wait for data flow
        await asyncio.sleep(2)

        # Check feature calculation
        features_calculated = feature_engine.metrics["features_calculated"] > 0
        print_result(
            "End-to-end data flow",
            features_calculated,
            f"Features calculated: {feature_engine.metrics['features_calculated']}",
        )

        # Check data validation
        test_tick = TickData(
            timestamp=datetime.now(),
            symbol="AAPL",
            bid=150.00,
            ask=150.05,
            last=150.02,
            bid_size=100,
            ask_size=100,
            last_size=100,
            volume=1000000,
        )

        validation_results = validator.validate_tick(test_tick)
        validation_working = len(validation_results) > 0
        print_result(
            "Validation integration",
            validation_working,
            f"Performed {len(validation_results)} validation checks",
        )

        # Stop components
        await pipeline.stop()
        await feature_engine.stop()

    except Exception as e:
        print_result("Integration test", False, str(e))
        raise


async def main():
    """Run all Phase 2 tests."""
    print("\n" + "=" * 60)
    print("  PHASE 2 TEST SUITE - Smart Data Pipeline")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Data Pipeline", await test_data_pipeline()))
    results.append(("Feature Engine", await test_feature_engine()))
    results.append(("Technical Indicators", test_technical_indicators()))
    results.append(("Data Validation", test_data_validation()))
    results.append(("Bar Validation", test_bar_validation()))
    results.append(("Integration", await test_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All Phase 2 tests passed! Data pipeline ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")

    return passed == total


if __name__ == "__main__":
    # Run async main
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
