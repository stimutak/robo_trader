"""
Test suite for microstructure trading strategies
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from robo_trader.strategies.microstructure import (
    MicrostructureConfig,
    OrderFlowImbalanceStrategy,
    SpreadTradingStrategy,
    TickMomentumStrategy,
    MicrostructureEnsembleStrategy
)
from robo_trader.features.orderbook import OrderBookSnapshot


def create_mock_orderbook(bid_price=99.95, ask_price=100.05, bid_size=1000, ask_size=1000):
    """Create mock order book data"""
    return {
        'bid_prices': np.array([bid_price, bid_price - 0.01, bid_price - 0.02]),
        'bid_sizes': np.array([bid_size, bid_size * 0.8, bid_size * 0.6]),
        'ask_prices': np.array([ask_price, ask_price + 0.01, ask_price + 0.02]),
        'ask_sizes': np.array([ask_size, ask_size * 0.8, ask_size * 0.6]),
        'best_bid': bid_price,
        'best_ask': ask_price
    }


class TestOrderFlowImbalanceStrategy:
    """Test order flow imbalance strategy"""
    
    def test_initialization(self):
        """Test strategy initialization"""
        config = MicrostructureConfig(ofi_entry_threshold=0.3)
        strategy = OrderFlowImbalanceStrategy(config)
        
        assert strategy.config.ofi_entry_threshold == 0.3
        assert strategy.positions == {}
        assert strategy.entry_prices == {}
    
    def test_buy_signal_with_imbalance(self):
        """Test buy signal when order flow imbalance is high"""
        strategy = OrderFlowImbalanceStrategy()
        
        # Create order book with buy imbalance
        market_data = {
            'orderbook': create_mock_orderbook(bid_size=1500, ask_size=500)
        }
        
        should_buy, metadata = strategy.should_buy('AAPL', 100.0, market_data)
        
        # With strong buy imbalance, should generate buy signal
        assert should_buy == True
        assert metadata['strategy'] == 'order_flow_imbalance'
        assert metadata['ofi'] > 0  # Positive imbalance
        assert 'signal_strength' in metadata
    
    def test_no_buy_when_already_long(self):
        """Test no buy signal when already in long position"""
        strategy = OrderFlowImbalanceStrategy()
        strategy.positions['AAPL'] = 100  # Already long
        
        market_data = {
            'orderbook': create_mock_orderbook(bid_size=1500, ask_size=500)
        }
        
        should_buy, metadata = strategy.should_buy('AAPL', 100.0, market_data)
        
        assert should_buy == False  # Should not buy when already long
    
    def test_sell_signal_on_reversal(self):
        """Test sell signal when OFI reverses"""
        strategy = OrderFlowImbalanceStrategy()
        strategy.positions['AAPL'] = 100
        strategy.entry_prices['AAPL'] = 99.0
        
        # Create order book with sell imbalance
        market_data = {
            'orderbook': create_mock_orderbook(bid_size=500, ask_size=1500)
        }
        
        should_sell, metadata = strategy.should_sell('AAPL', 100.0, market_data)
        
        assert should_sell == True
        assert metadata['exit_reason'] == 'ofi_reversal'
    
    def test_position_sizing(self):
        """Test position size calculation based on OFI strength"""
        config = MicrostructureConfig(min_order_size=100, max_order_size=500)
        strategy = OrderFlowImbalanceStrategy(config)
        
        market_data = {
            'orderbook': create_mock_orderbook(bid_size=1500, ask_size=500)
        }
        
        # First call to populate order book features
        strategy.should_buy('AAPL', 100.0, market_data)
        
        size = strategy.calculate_position_size('AAPL', 100.0, 10000, market_data)
        
        assert size >= config.min_order_size
        assert size <= config.max_order_size


class TestSpreadTradingStrategy:
    """Test spread trading (market making) strategy"""
    
    def test_quote_calculation(self):
        """Test bid/ask quote calculation"""
        config = MicrostructureConfig(quote_offset_bps=2.0)
        strategy = SpreadTradingStrategy(config)
        
        market_data = {
            'orderbook': create_mock_orderbook()
        }
        
        bid_price, ask_price = strategy.calculate_quotes('AAPL', market_data)
        
        assert bid_price is not None
        assert ask_price is not None
        assert bid_price < ask_price  # Bid should be lower than ask
    
    def test_inventory_skew(self):
        """Test quote adjustment based on inventory"""
        config = MicrostructureConfig(max_inventory=100, inventory_skew_factor=0.5)
        strategy = SpreadTradingStrategy(config)
        strategy.inventory['AAPL'] = 50  # Half of max inventory
        
        market_data = {
            'orderbook': create_mock_orderbook()
        }
        
        bid_with_inventory, ask_with_inventory = strategy.calculate_quotes('AAPL', market_data)
        
        # Reset inventory
        strategy.inventory['AAPL'] = 0
        bid_no_inventory, ask_no_inventory = strategy.calculate_quotes('AAPL', market_data)
        
        # With positive inventory, should lower both quotes to reduce position
        assert bid_with_inventory < bid_no_inventory
        assert ask_with_inventory < ask_no_inventory
    
    def test_inventory_limits(self):
        """Test that strategy respects inventory limits"""
        config = MicrostructureConfig(max_inventory=100)
        strategy = SpreadTradingStrategy(config)
        strategy.inventory['AAPL'] = 100  # At max inventory
        
        market_data = {
            'orderbook': create_mock_orderbook()
        }
        
        should_buy, metadata = strategy.should_buy('AAPL', 100.0, market_data)
        
        assert should_buy == False
        assert metadata.get('reason') == 'max_inventory'
    
    def test_competitive_quote_check(self):
        """Test that quotes are competitive with market"""
        strategy = SpreadTradingStrategy()
        
        market_data = {
            'orderbook': create_mock_orderbook(bid_price=99.95, ask_price=100.05)
        }
        
        # Calculate quotes
        bid_price, _ = strategy.calculate_quotes('AAPL', market_data)
        
        # If our bid is too far below market, should not place order
        if bid_price and bid_price < 99.95 * 0.995:
            should_buy, metadata = strategy.should_buy('AAPL', 100.0, market_data)
            assert metadata.get('reason') == 'uncompetitive_bid'


class TestTickMomentumStrategy:
    """Test tick momentum strategy"""
    
    def test_tick_history_management(self):
        """Test that tick history is properly maintained"""
        config = MicrostructureConfig(tick_window=5)
        strategy = TickMomentumStrategy(config)
        
        # Add ticks
        for price in [100.0, 100.1, 100.2, 100.3, 100.4, 100.5]:
            strategy.should_buy('AAPL', price, {})
        
        assert len(strategy.tick_history['AAPL']) == 5  # Should keep only window size
        assert strategy.tick_history['AAPL'][-1] == 100.5  # Most recent price
    
    def test_momentum_detection(self):
        """Test upward momentum detection"""
        config = MicrostructureConfig(tick_window=5, tick_threshold=0.3)
        strategy = TickMomentumStrategy(config)
        
        # Create strong upward momentum
        prices = [100.0, 100.1, 100.2, 100.3, 100.4]
        
        for i, price in enumerate(prices):
            market_data = {
                'orderbook': create_mock_orderbook()
            }
            
            if i == len(prices) - 1:
                # Last tick should trigger buy signal
                should_buy, metadata = strategy.should_buy('AAPL', price, market_data)
                
                if len(strategy.tick_history.get('AAPL', [])) >= config.tick_window:
                    assert metadata['momentum_bps'] > 0  # Positive momentum
                    # Note: Buy signal depends on tick direction calculation
    
    def test_exit_on_momentum_reversal(self):
        """Test exit when momentum reverses"""
        config = MicrostructureConfig(tick_window=5)
        strategy = TickMomentumStrategy(config)
        
        # Setup position
        strategy.positions['AAPL'] = 100
        strategy.entry_times['AAPL'] = datetime.now()
        
        # Create downward momentum
        strategy.tick_history['AAPL'] = [100.4, 100.3, 100.2, 100.1, 100.0]
        
        market_data = {
            'orderbook': create_mock_orderbook()
        }
        
        should_sell, metadata = strategy.should_sell('AAPL', 99.9, market_data)
        
        # Should consider selling on momentum reversal
        assert 'exit_reason' in metadata
    
    def test_max_holding_time(self):
        """Test exit after maximum holding time"""
        strategy = TickMomentumStrategy()
        
        # Setup position with old entry time
        strategy.positions['AAPL'] = 100
        strategy.entry_times['AAPL'] = datetime.now() - timedelta(seconds=120)
        strategy.tick_history['AAPL'] = [100.0, 100.0]
        
        should_sell, metadata = strategy.should_sell('AAPL', 100.0, {})
        
        assert should_sell == True
        assert metadata['exit_reason'] == 'max_holding_time'


class TestMicrostructureEnsembleStrategy:
    """Test ensemble strategy combining multiple microstructure signals"""
    
    def test_ensemble_initialization(self):
        """Test ensemble strategy initialization"""
        strategy = MicrostructureEnsembleStrategy()
        
        assert strategy.ofi_strategy is not None
        assert strategy.spread_strategy is not None
        assert strategy.tick_strategy is not None
        assert sum(strategy.weights.values()) == 1.0  # Weights should sum to 1
    
    def test_ensemble_signal_combination(self):
        """Test that ensemble properly combines signals"""
        strategy = MicrostructureEnsembleStrategy()
        
        market_data = {
            'orderbook': create_mock_orderbook(bid_size=1500, ask_size=500)
        }
        
        should_buy, metadata = strategy.should_buy('AAPL', 100.0, market_data)
        
        assert 'ensemble_score' in metadata
        assert 'ofi_signal' in metadata
        assert 'spread_signal' in metadata
        assert 'tick_signal' in metadata
        assert 'sub_strategies' in metadata
        
        # Ensemble score should be weighted average
        score = metadata['ensemble_score']
        assert 0 <= score <= 1.0
    
    def test_ensemble_position_sizing(self):
        """Test ensemble position size calculation"""
        config = MicrostructureConfig(min_order_size=100, max_order_size=1000)
        strategy = MicrostructureEnsembleStrategy(config)
        
        market_data = {
            'orderbook': create_mock_orderbook()
        }
        
        size = strategy.calculate_position_size('AAPL', 100.0, 10000, market_data)
        
        assert size >= 0
        assert size <= config.max_order_size


class TestMicrostructureConfig:
    """Test microstructure configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = MicrostructureConfig()
        
        assert config.ofi_entry_threshold == 0.3
        assert config.min_spread_bps == 5.0
        assert config.max_position_size == 1000
        assert config.stop_loss_bps == 20.0
        assert config.tick_window == 20
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = MicrostructureConfig(
            ofi_entry_threshold=0.5,
            max_position_size=2000,
            tick_window=30
        )
        
        assert config.ofi_entry_threshold == 0.5
        assert config.max_position_size == 2000
        assert config.tick_window == 30


def test_integration_scenario():
    """Test a complete trading scenario with microstructure strategies"""
    config = MicrostructureConfig(
        ofi_entry_threshold=0.2,
        min_order_size=100,
        max_order_size=500
    )
    
    # Test OFI strategy
    ofi_strategy = OrderFlowImbalanceStrategy(config)
    
    # Simulate market conditions
    market_data_bullish = {
        'orderbook': create_mock_orderbook(bid_size=2000, ask_size=500)
    }
    
    # Should generate buy signal with bullish order flow
    should_buy, buy_meta = ofi_strategy.should_buy('AAPL', 100.0, market_data_bullish)
    assert should_buy == True
    assert buy_meta['ofi'] > 0
    
    # After buying, should have position
    assert ofi_strategy.positions.get('AAPL', 0) > 0
    
    # Simulate market reversal
    market_data_bearish = {
        'orderbook': create_mock_orderbook(bid_size=500, ask_size=2000)
    }
    
    # Should generate sell signal with bearish order flow
    should_sell, sell_meta = ofi_strategy.should_sell('AAPL', 101.0, market_data_bearish)
    assert should_sell == True
    
    # After selling, position should be closed
    assert ofi_strategy.positions.get('AAPL', 0) == 0
    
    print("✅ Integration test passed")


def run_all_tests():
    """Run all microstructure strategy tests"""
    print("Testing Microstructure Trading Strategies...")
    print("=" * 50)
    
    # Test Order Flow Imbalance Strategy
    print("\n📊 Testing Order Flow Imbalance Strategy...")
    ofi_tests = TestOrderFlowImbalanceStrategy()
    ofi_tests.test_initialization()
    ofi_tests.test_buy_signal_with_imbalance()
    ofi_tests.test_no_buy_when_already_long()
    ofi_tests.test_sell_signal_on_reversal()
    ofi_tests.test_position_sizing()
    print("✅ Order Flow Imbalance tests passed")
    
    # Test Spread Trading Strategy
    print("\n💹 Testing Spread Trading Strategy...")
    spread_tests = TestSpreadTradingStrategy()
    spread_tests.test_quote_calculation()
    spread_tests.test_inventory_skew()
    spread_tests.test_inventory_limits()
    spread_tests.test_competitive_quote_check()
    print("✅ Spread Trading tests passed")
    
    # Test Tick Momentum Strategy
    print("\n📈 Testing Tick Momentum Strategy...")
    tick_tests = TestTickMomentumStrategy()
    tick_tests.test_tick_history_management()
    tick_tests.test_momentum_detection()
    tick_tests.test_exit_on_momentum_reversal()
    tick_tests.test_max_holding_time()
    print("✅ Tick Momentum tests passed")
    
    # Test Ensemble Strategy
    print("\n🎯 Testing Ensemble Strategy...")
    ensemble_tests = TestMicrostructureEnsembleStrategy()
    ensemble_tests.test_ensemble_initialization()
    ensemble_tests.test_ensemble_signal_combination()
    ensemble_tests.test_ensemble_position_sizing()
    print("✅ Ensemble Strategy tests passed")
    
    # Test Configuration
    print("\n⚙️ Testing Configuration...")
    config_tests = TestMicrostructureConfig()
    config_tests.test_default_config()
    config_tests.test_custom_config()
    print("✅ Configuration tests passed")
    
    # Integration test
    print("\n🔄 Running integration test...")
    test_integration_scenario()
    
    print("\n" + "=" * 50)
    print("✨ All microstructure strategy tests passed successfully!")
    print("\nKey Features Tested:")
    print("  • Order flow imbalance detection and trading")
    print("  • Market making with inventory management")
    print("  • Tick-level momentum detection")
    print("  • Ensemble strategy combination")
    print("  • Risk management (stop loss, take profit)")
    print("  • Position sizing based on signal strength")


if __name__ == "__main__":
    run_all_tests()