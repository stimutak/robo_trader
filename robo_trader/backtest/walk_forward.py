"""
Enhanced walk-forward backtesting with realistic execution simulation.

This module implements:
- Walk-forward analysis with rolling windows
- Realistic execution simulation (slippage, market impact)
- Comprehensive performance metrics
- Out-of-sample validation
- Monte Carlo simulation for robustness testing
- Feature engineering integration (M1)
- Correlation-based position sizing (M5)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..logger import get_logger
from .engine import BacktestEngine
from .metrics import PerformanceMetrics
from ..features import (
    MomentumIndicators,
    TrendIndicators,
    VolatilityIndicators,
    VolumeIndicators
)
from ..correlation import CorrelationTracker
from ..analysis.correlation_integration import CorrelationBasedPositionSizer


@dataclass
class ExecutionSimulator:
    """Realistic execution simulation for backtesting."""
    
    # Market impact parameters
    temporary_impact_factor: float = 0.1  # basis points per unit of participation rate
    permanent_impact_factor: float = 0.05  # basis points permanent impact
    
    # Slippage parameters
    base_slippage_bps: float = 2.0
    volatility_multiplier: float = 0.5  # Additional slippage based on volatility
    
    # Liquidity parameters
    participation_rate: float = 0.1  # Max % of volume we can trade
    min_liquidity_ratio: float = 0.01  # Min volume/ADV ratio required
    
    # Transaction costs
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    
    def simulate_execution(
        self,
        side: str,
        quantity: int,
        price: float,
        volatility: float,
        avg_volume: float,
        spread: float,
        order_book_imbalance: float = 0.0
    ) -> Dict[str, float]:
        """
        Simulate realistic order execution.
        
        Args:
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Current market price
            volatility: Current volatility
            avg_volume: Average daily volume
            spread: Bid-ask spread
            order_book_imbalance: Imbalance in order book (-1 to 1)
            
        Returns:
            Execution details including fill price and costs
        """
        # Calculate participation rate
        if avg_volume > 0:
            participation = quantity / avg_volume
        else:
            participation = 1.0
        
        # Calculate market impact (square root model)
        temporary_impact = self.temporary_impact_factor * np.sqrt(participation) / 10000
        permanent_impact = self.permanent_impact_factor * participation / 10000
        
        # Calculate slippage
        base_slippage = self.base_slippage_bps / 10000
        volatility_slippage = self.volatility_multiplier * volatility / 10000
        
        # Adjust for order book imbalance
        imbalance_adjustment = order_book_imbalance * 0.0005  # 5 bps max adjustment
        
        # Total price impact
        if side == 'BUY':
            impact_multiplier = 1 + temporary_impact + permanent_impact + base_slippage + volatility_slippage + max(0, imbalance_adjustment)
        else:
            impact_multiplier = 1 - temporary_impact - permanent_impact - base_slippage - volatility_slippage + min(0, imbalance_adjustment)
        
        # Calculate fill price
        fill_price = price * impact_multiplier
        
        # Add half spread
        if side == 'BUY':
            fill_price += spread / 2
        else:
            fill_price -= spread / 2
        
        # Calculate commission
        commission = max(self.min_commission, quantity * self.commission_per_share)
        
        # Calculate total slippage
        if side == 'BUY':
            slippage = (fill_price - price) * quantity
        else:
            slippage = (price - fill_price) * quantity
        
        return {
            'fill_price': fill_price,
            'slippage': slippage,
            'market_impact': (temporary_impact + permanent_impact) * price * quantity,
            'commission': commission,
            'total_cost': slippage + commission,
            'participation_rate': participation
        }
    
    def can_execute(
        self,
        quantity: int,
        avg_volume: float,
        current_volume: float
    ) -> Tuple[bool, str]:
        """
        Check if order can be executed given liquidity constraints.
        
        Returns:
            Tuple of (can_execute, reason)
        """
        if avg_volume <= 0:
            return False, "No liquidity data available"
        
        # Check participation rate
        if quantity / avg_volume > self.participation_rate:
            return False, f"Order exceeds max participation rate ({self.participation_rate * 100}%)"
        
        # Check minimum liquidity
        if current_volume / avg_volume < self.min_liquidity_ratio:
            return False, "Insufficient liquidity"
        
        return True, "OK"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    
    # Window configuration
    train_window_days: int = 252  # 1 year training
    test_window_days: int = 63  # 3 months testing
    step_days: int = 21  # 1 month step
    min_train_samples: int = 100
    
    # Optimization settings
    optimization_metric: str = 'sharpe_ratio'
    min_trades: int = 30
    
    # Validation settings
    confidence_level: float = 0.95
    monte_carlo_simulations: int = 1000
    
    # Execution settings
    execution_simulator: ExecutionSimulator = field(default_factory=ExecutionSimulator)
    
    # Feature engineering settings (M1 integration)
    use_technical_features: bool = True
    momentum_window: int = 14
    trend_window: int = 20
    volatility_window: int = 20
    volume_window: int = 20
    
    # Correlation settings (M5 integration)
    use_correlation_sizing: bool = True
    max_correlation: float = 0.7
    correlation_penalty_factor: float = 0.5


class WalkForwardBacktest:
    """Enhanced walk-forward backtesting framework."""
    
    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.logger = get_logger("backtest.walk_forward")
        self.execution_sim = self.config.execution_simulator
        
        # Results storage
        self.window_results: List[Dict] = []
        self.oos_results: List[Dict] = []  # Out-of-sample results
        self.parameter_stability: Dict[str, List] = {}
        
        # Initialize feature engineering components (M1)
        if self.config.use_technical_features:
            self.momentum_indicators = MomentumIndicators(window_size=self.config.momentum_window)
            self.trend_indicators = TrendIndicators()
            self.volatility_indicators = VolatilityIndicators(window_size=self.config.volatility_window)
            self.volume_indicators = VolumeIndicators(window_size=self.config.volume_window)
        
        # Initialize correlation components (M5)
        if self.config.use_correlation_sizing:
            self.correlation_tracker = CorrelationTracker(
                correlation_threshold=self.config.max_correlation
            )
            self.position_sizer = CorrelationBasedPositionSizer(
                correlation_tracker=self.correlation_tracker,
                max_correlation=self.config.max_correlation,
                correlation_penalty_factor=self.config.correlation_penalty_factor
            )
        else:
            self.correlation_tracker = None
            self.position_sizer = None
        
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical features for backtesting (M1 integration).
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with features added
        """
        if not self.config.use_technical_features:
            return data
        
        features = data.copy()
        
        # Generate momentum features
        if self.momentum_indicators.validate_data(data):
            momentum_features = self.momentum_indicators.calculate(data)
            for name, feature in momentum_features.items():
                features[f'momentum_{name}'] = feature.value
        
        # Generate trend features
        if self.trend_indicators.validate_data(data):
            trend_features = self.trend_indicators.calculate(data)
            for name, feature in trend_features.items():
                if name in ['sma_20', 'ema_20', 'macd', 'adx']:  # Key features
                    features[f'trend_{name}'] = feature.value
        
        # Generate volatility features
        if self.volatility_indicators.validate_data(data):
            volatility_features = self.volatility_indicators.calculate(data)
            for name, feature in volatility_features.items():
                features[f'volatility_{name}'] = feature.value
        
        # Generate volume features
        if self.volume_indicators.validate_data(data):
            volume_features = self.volume_indicators.calculate(data)
            for name, feature in volume_features.items():
                features[f'volume_{name}'] = feature.value
        
        # Add derived features
        if 'trend_sma_20' in features.columns:
            features['price_to_sma'] = features['close'] / features['trend_sma_20']
        
        features['volume_ratio'] = features['volume'] / features['volume'].rolling(20).mean()
        features['spread'] = (features['high'] - features['low']) / features['close']
        features['overnight_gap'] = (features['open'] - features['close'].shift(1)) / features['close'].shift(1)
        
        return features
    
    def create_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Create walk-forward windows.
        
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        current_date = start_date + timedelta(days=self.config.train_window_days)
        
        while current_date + timedelta(days=self.config.test_window_days) <= end_date:
            train_start = current_date - timedelta(days=self.config.train_window_days)
            train_end = current_date
            test_start = current_date
            test_end = min(
                current_date + timedelta(days=self.config.test_window_days),
                end_date
            )
            
            windows.append((train_start, train_end, test_start, test_end))
            current_date += timedelta(days=self.config.step_days)
        
        return windows
    
    async def run_single_window(
        self,
        engine: BacktestEngine,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        parameters: Dict[str, Any],
        window_id: int
    ) -> Dict[str, Any]:
        """Run backtest for a single window."""
        try:
            # Train on in-sample data
            train_metrics = await self._run_backtest_period(
                engine, train_data, parameters, is_training=True
            )
            
            # Test on out-of-sample data
            test_metrics = await self._run_backtest_period(
                engine, test_data, parameters, is_training=False
            )
            
            # Calculate stability metrics
            stability = self._calculate_stability_metrics(train_metrics, test_metrics)
            
            return {
                'window_id': window_id,
                'parameters': parameters,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'stability': stability,
                'degradation': (train_metrics['sharpe_ratio'] - test_metrics['sharpe_ratio']) 
                             / max(abs(train_metrics['sharpe_ratio']), 1e-6)
            }
            
        except Exception as e:
            self.logger.error(f"Window {window_id} failed: {e}")
            return None
    
    async def _run_backtest_period(
        self,
        engine: BacktestEngine,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        is_training: bool
    ) -> Dict[str, float]:
        """Run backtest for a period with execution simulation."""
        trades = []
        portfolio_values = []
        positions = {}
        cash = 100000  # Starting cash
        
        # Generate features for the data (M1 integration)
        featured_data = self.generate_features(data)
        
        # Update correlation tracker if enabled (M5 integration)
        if self.correlation_tracker and 'symbol' in data.columns:
            symbol = data['symbol'].iloc[0]
            self.correlation_tracker.add_price_series(
                symbol=symbol,
                prices=data['close']
            )
        
        for i, row in featured_data.iterrows():
            # Generate signal using featured data
            signal = engine.strategy.generate_signal(featured_data.loc[:i], parameters)
            
            if signal != 0:
                # Get market conditions
                volatility = data.loc[:i, 'close'].pct_change().std() * np.sqrt(252)
                avg_volume = data.loc[:i, 'volume'].mean()
                current_volume = row['volume']
                spread = row.get('spread', row['close'] * 0.001)  # Default 10 bps
                
                # Calculate base position size
                base_position_size = int(cash * 0.02 / row['close'])  # 2% position
                
                # Apply correlation-based sizing if enabled (M5 integration)
                if self.position_sizer and positions:
                    # Convert positions for correlation sizer
                    position_objects = {
                        s: type('Position', (), {
                            'symbol': s,
                            'quantity': p['quantity'],
                            'avg_price': p['entry_price'],
                            'notional_value': p['quantity'] * p['entry_price']
                        })() for s, p in positions.items()
                    }
                    
                    position_size, sizing_reason = await self.position_sizer.calculate_position_size(
                        symbol=data.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in data else 'UNKNOWN',
                        base_size=base_position_size,
                        current_positions=position_objects,
                        portfolio_value=cash + sum(p['quantity'] * row['close'] for p in positions.values())
                    )
                    self.logger.debug(f"Position sizing: {base_position_size} -> {position_size} ({sizing_reason})")
                else:
                    position_size = base_position_size
                
                # Check if we can execute
                can_exec, reason = self.execution_sim.can_execute(
                    position_size, avg_volume, current_volume
                )
                
                if can_exec:
                    # Simulate execution
                    exec_result = self.execution_sim.simulate_execution(
                        side='BUY' if signal > 0 else 'SELL',
                        quantity=position_size,
                        price=row['close'],
                        volatility=volatility,
                        avg_volume=avg_volume,
                        spread=spread
                    )
                    
                    # Record trade
                    trades.append({
                        'timestamp': i,
                        'side': 'BUY' if signal > 0 else 'SELL',
                        'quantity': position_size,
                        'price': row['close'],
                        'fill_price': exec_result['fill_price'],
                        'slippage': exec_result['slippage'],
                        'commission': exec_result['commission'],
                        'signal': signal
                    })
                    
                    # Update positions
                    symbol = data.get('symbol', 'UNKNOWN').iloc[0] if 'symbol' in data else 'UNKNOWN'
                    if signal > 0:
                        positions[symbol] = {
                            'quantity': position_size,
                            'entry_price': exec_result['fill_price']
                        }
                        cash -= exec_result['fill_price'] * position_size + exec_result['commission']
                    else:
                        if symbol in positions:
                            cash += exec_result['fill_price'] * positions[symbol]['quantity'] - exec_result['commission']
                            del positions[symbol]
            
            # Calculate portfolio value
            portfolio_value = cash
            for sym, pos in positions.items():
                portfolio_value += pos['quantity'] * row['close']
            portfolio_values.append(portfolio_value)
        
        # Calculate metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        return {
            'total_return': (portfolio_values[-1] / 100000 - 1) * 100 if portfolio_values else 0,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'win_rate': self._calculate_win_rate(trades),
            'avg_trade': self._calculate_avg_trade(trades),
            'total_trades': len(trades),
            'total_slippage': sum(t['slippage'] for t in trades),
            'total_commission': sum(t['commission'] for t in trades)
        }
    
    def _calculate_stability_metrics(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate stability between training and test periods."""
        stability_scores = {}
        
        for metric in ['sharpe_ratio', 'win_rate', 'avg_trade']:
            if metric in train_metrics and metric in test_metrics:
                train_val = train_metrics[metric]
                test_val = test_metrics[metric]
                
                if abs(train_val) > 1e-6:
                    stability_scores[f'{metric}_stability'] = 1 - abs(train_val - test_val) / abs(train_val)
                else:
                    stability_scores[f'{metric}_stability'] = 0
        
        return stability_scores
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not values:
            return 0
        
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0
        
        wins = 0
        for i in range(0, len(trades) - 1, 2):  # Pairs of entry/exit
            if i + 1 < len(trades):
                entry = trades[i]
                exit = trades[i + 1]
                if entry['side'] == 'BUY':
                    profit = (exit['fill_price'] - entry['fill_price']) * entry['quantity']
                else:
                    profit = (entry['fill_price'] - exit['fill_price']) * entry['quantity']
                if profit > 0:
                    wins += 1
        
        total_pairs = len(trades) // 2
        return wins / total_pairs * 100 if total_pairs > 0 else 0
    
    def _calculate_avg_trade(self, trades: List[Dict]) -> float:
        """Calculate average trade profit."""
        if not trades or len(trades) < 2:
            return 0
        
        profits = []
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                entry = trades[i]
                exit = trades[i + 1]
                if entry['side'] == 'BUY':
                    profit = (exit['fill_price'] - entry['fill_price']) * entry['quantity']
                else:
                    profit = (entry['fill_price'] - exit['fill_price']) * entry['quantity']
                profits.append(profit)
        
        return np.mean(profits) if profits else 0
    
    async def run_monte_carlo_validation(
        self,
        results: pd.DataFrame,
        n_simulations: int = None
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for robustness testing.
        
        Args:
            results: Historical results to bootstrap from
            n_simulations: Number of simulations
            
        Returns:
            Monte Carlo statistics
        """
        n_simulations = n_simulations or self.config.monte_carlo_simulations
        
        returns = results['returns'].values
        simulated_paths = []
        
        for _ in range(n_simulations):
            # Bootstrap returns
            bootstrapped = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate cumulative path
            path = (1 + bootstrapped).cumprod()
            simulated_paths.append(path)
        
        simulated_paths = np.array(simulated_paths)
        
        # Calculate statistics
        final_values = simulated_paths[:, -1]
        max_drawdowns = [self._calculate_max_drawdown(path.tolist()) for path in simulated_paths]
        
        confidence_level = self.config.confidence_level
        
        return {
            'expected_return': np.mean(final_values - 1) * 100,
            'return_std': np.std(final_values - 1) * 100,
            'var': np.percentile(final_values - 1, (1 - confidence_level) * 100) * 100,
            'cvar': np.mean([r for r in (final_values - 1) * 100 
                           if r <= np.percentile(final_values - 1, (1 - confidence_level) * 100) * 100]),
            'expected_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown': np.max(max_drawdowns),
            'probability_profit': np.mean(final_values > 1) * 100,
            'median_return': np.median(final_values - 1) * 100
        }
    
    def calculate_parameter_sensitivity(
        self,
        results: List[Dict],
        parameter_name: str
    ) -> Dict[str, float]:
        """
        Calculate sensitivity of results to parameter changes.
        
        Returns:
            Sensitivity metrics for the parameter
        """
        if not results:
            return {}
        
        # Group results by parameter value
        param_groups = {}
        for result in results:
            if result and 'parameters' in result:
                param_value = result['parameters'].get(parameter_name)
                if param_value is not None:
                    if param_value not in param_groups:
                        param_groups[param_value] = []
                    param_groups[param_value].append(result['test_metrics']['sharpe_ratio'])
        
        if len(param_groups) < 2:
            return {}
        
        # Calculate correlation between parameter value and performance
        param_values = []
        performances = []
        
        for value, sharpes in param_groups.items():
            param_values.extend([value] * len(sharpes))
            performances.extend(sharpes)
        
        correlation = np.corrcoef(param_values, performances)[0, 1] if len(param_values) > 1 else 0
        
        # Calculate variance across parameter values
        variances = [np.var(sharpes) for sharpes in param_groups.values()]
        
        return {
            'correlation': correlation,
            'avg_variance': np.mean(variances),
            'parameter_impact': abs(correlation) * np.mean(variances)
        }
    
    async def run_multi_symbol_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        engine: BacktestEngine,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run backtest on multiple symbols with correlation awareness.
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            engine: Backtest engine
            parameters: Strategy parameters
            
        Returns:
            Backtest results
        """
        self.logger.info(f"Running multi-symbol backtest for {len(data)} symbols")
        
        # Update correlation tracker with all symbols
        if self.correlation_tracker:
            for symbol, df in data.items():
                self.correlation_tracker.add_price_series(
                    symbol=symbol,
                    prices=df['close']
                )
        
        # Run walk-forward for primary symbol
        primary_symbol = list(data.keys())[0]
        primary_data = data[primary_symbol]
        
        # Create windows
        windows = self.create_windows(
            start_date=primary_data.index[0],
            end_date=primary_data.index[-1]
        )
        
        # Run windows
        for i, window in enumerate(windows):
            result = await self.run_single_window(
                engine=engine,
                train_data=primary_data.loc[window[0]:window[1]],
                test_data=primary_data.loc[window[2]:window[3]],
                parameters=parameters,
                window_id=i
            )
            if result:
                self.window_results.append(result)
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive walk-forward analysis report."""
        if not self.window_results:
            return {'error': 'No results available'}
        
        # Calculate aggregate metrics
        oos_sharpes = [r['test_metrics']['sharpe_ratio'] 
                      for r in self.window_results if r]
        oos_returns = [r['test_metrics']['total_return'] 
                      for r in self.window_results if r]
        
        return {
            'summary': {
                'total_windows': len(self.window_results),
                'avg_oos_sharpe': np.mean(oos_sharpes) if oos_sharpes else 0,
                'avg_oos_return': np.mean(oos_returns) if oos_returns else 0,
                'consistency': np.std(oos_sharpes) if len(oos_sharpes) > 1 else 0,
                'positive_windows': sum(1 for s in oos_sharpes if s > 0),
                'win_rate': sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes) * 100 
                          if oos_sharpes else 0
            },
            'stability': {
                'avg_degradation': np.mean([r['degradation'] for r in self.window_results if r]),
                'parameter_variance': {
                    param: np.std(values) 
                    for param, values in self.parameter_stability.items()
                }
            },
            'execution_costs': {
                'avg_slippage': np.mean([r['test_metrics']['total_slippage'] 
                                        for r in self.window_results if r]),
                'avg_commission': np.mean([r['test_metrics']['total_commission'] 
                                         for r in self.window_results if r])
            },
            'windows': self.window_results
        }