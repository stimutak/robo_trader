"""
Kelly Criterion position sizing for optimal capital allocation.

The Kelly formula determines optimal bet size based on:
- Probability of winning (from AI conviction)
- Expected payoff ratio
- Available capital
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from robo_trader.logger import get_logger

logger = get_logger(__name__)


@dataclass
class KellyParameters:
    """Parameters for Kelly Criterion calculation."""
    win_probability: float  # p: probability of winning (0 to 1)
    win_return: float  # b: net odds received on win (e.g., 0.05 for 5% gain)
    loss_return: float  # typically -1.0 for total loss, or -0.02 for 2% stop loss
    
    def validate(self) -> bool:
        """Validate parameters are in valid ranges."""
        if not 0 <= self.win_probability <= 1:
            return False
        if self.win_return <= 0:
            return False
        if self.loss_return >= 0:
            return False
        return True


class KellyCalculator:
    """
    Calculate optimal position sizes using Kelly Criterion.
    
    Kelly formula: f* = (p*b - q)/b
    where:
    - f* = fraction of capital to bet
    - p = probability of winning
    - q = probability of losing (1-p)
    - b = ratio of win amount to loss amount
    """
    
    def __init__(
        self,
        max_kelly_fraction: float = 0.25,  # Cap at 25% of capital (quarter Kelly)
        min_conviction: float = 0.55,  # Minimum conviction to trade
        capital: float = 100000.0  # Available capital
    ):
        """
        Initialize Kelly calculator.
        
        Args:
            max_kelly_fraction: Maximum fraction of capital to risk (Kelly cap)
            min_conviction: Minimum AI conviction required to trade
            capital: Total available capital
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.min_conviction = min_conviction
        self.capital = capital
        
        # Track historical performance for adaptive Kelly
        self.trade_history: List[Dict] = []
        self.win_rate_estimate = 0.5  # Start with neutral assumption
        
    def calculate_kelly_fraction(self, params: KellyParameters) -> float:
        """
        Calculate raw Kelly fraction.
        
        Args:
            params: Kelly parameters
            
        Returns:
            Optimal fraction of capital to bet
        """
        if not params.validate():
            logger.error(f"Invalid Kelly parameters: {params}")
            return 0.0
            
        p = params.win_probability
        q = 1 - p
        
        # Calculate ratio of win to loss
        # b = win_amount / loss_amount
        b = abs(params.win_return / params.loss_return)
        
        # Kelly formula
        kelly_fraction = (p * b - q) / b
        
        # Only bet if positive expectation
        if kelly_fraction <= 0:
            return 0.0
            
        return kelly_fraction
        
    def calculate_position_size(
        self,
        conviction: float,
        expected_return: float,
        stop_loss_pct: float = 0.02,
        current_price: float = 100.0
    ) -> Tuple[int, float]:
        """
        Calculate position size based on AI conviction.
        
        Args:
            conviction: AI conviction (0 to 1)
            expected_return: Expected return if correct (e.g., 0.05 for 5%)
            stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%)
            current_price: Current asset price
            
        Returns:
            Tuple of (shares, dollar_amount)
        """
        # Skip if conviction too low
        if conviction < self.min_conviction:
            logger.debug(f"Conviction {conviction:.2%} below minimum {self.min_conviction:.2%}")
            return 0, 0.0
            
        # Create Kelly parameters
        params = KellyParameters(
            win_probability=conviction,
            win_return=expected_return,
            loss_return=-stop_loss_pct
        )
        
        # Calculate raw Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(params)
        
        # Apply fractional Kelly (more conservative)
        # Using 1/4 Kelly is common for real trading
        adjusted_fraction = min(kelly_fraction * 0.25, self.max_kelly_fraction)
        
        # Apply conviction scaling (higher conviction = closer to full Kelly)
        conviction_scale = (conviction - self.min_conviction) / (1.0 - self.min_conviction)
        final_fraction = adjusted_fraction * (0.5 + 0.5 * conviction_scale)
        
        # Calculate dollar amount
        position_value = self.capital * final_fraction
        
        # Calculate shares
        shares = int(position_value / current_price)
        
        logger.info(
            f"Kelly calculation: conviction={conviction:.2%}, "
            f"kelly={kelly_fraction:.2%}, adjusted={final_fraction:.2%}, "
            f"shares={shares} @ ${current_price:.2f}"
        )
        
        return shares, position_value
        
    def calculate_multi_asset_allocation(
        self,
        opportunities: List[Dict]
    ) -> Dict[str, Tuple[int, float]]:
        """
        Allocate capital across multiple opportunities.
        
        Args:
            opportunities: List of dicts with keys:
                - symbol: str
                - conviction: float
                - expected_return: float
                - stop_loss: float
                - price: float
                
        Returns:
            Dict of symbol -> (shares, dollar_amount)
        """
        # Sort by conviction * expected_return (expected value)
        opportunities = sorted(
            opportunities,
            key=lambda x: x['conviction'] * x['expected_return'],
            reverse=True
        )
        
        allocations = {}
        remaining_capital = self.capital
        
        for opp in opportunities:
            if remaining_capital <= 0:
                break
                
            # Use remaining capital for position sizing
            temp_calc = KellyCalculator(
                max_kelly_fraction=self.max_kelly_fraction,
                min_conviction=self.min_conviction,
                capital=remaining_capital
            )
            
            shares, position_value = temp_calc.calculate_position_size(
                conviction=opp['conviction'],
                expected_return=opp['expected_return'],
                stop_loss_pct=opp.get('stop_loss', 0.02),
                current_price=opp['price']
            )
            
            if shares > 0:
                allocations[opp['symbol']] = (shares, position_value)
                remaining_capital -= position_value
                
        return allocations
        
    def update_performance(self, won: bool, return_pct: float):
        """
        Update historical performance for adaptive Kelly.
        
        Args:
            won: Whether the trade was profitable
            return_pct: Actual return percentage
        """
        self.trade_history.append({
            'won': won,
            'return': return_pct
        })
        
        # Update win rate estimate (simple moving average)
        if len(self.trade_history) >= 10:
            recent_trades = self.trade_history[-20:]  # Last 20 trades
            wins = sum(1 for t in recent_trades if t['won'])
            self.win_rate_estimate = wins / len(recent_trades)
            
    def suggest_conviction_adjustment(self, ai_conviction: float) -> float:
        """
        Adjust AI conviction based on historical performance.
        
        Args:
            ai_conviction: Raw AI conviction
            
        Returns:
            Adjusted conviction incorporating historical win rate
        """
        if len(self.trade_history) < 10:
            return ai_conviction  # Not enough data
            
        # Blend AI conviction with historical win rate
        # More weight on AI conviction (70%) vs history (30%)
        adjusted = 0.7 * ai_conviction + 0.3 * self.win_rate_estimate
        
        # Apply Bayesian-like shrinkage toward 0.5 (neutral)
        shrinkage_factor = min(len(self.trade_history) / 50, 1.0)
        final_conviction = 0.5 + (adjusted - 0.5) * shrinkage_factor
        
        return final_conviction


def calculate_optimal_position(
    conviction: float,
    expected_move: float,
    capital: float = 100000,
    max_risk_per_trade: float = 0.02
) -> Dict[str, float]:
    """
    Simple interface for position sizing.
    
    Args:
        conviction: AI conviction (0 to 1)
        expected_move: Expected price move (e.g., 0.05 for 5%)
        capital: Available capital
        max_risk_per_trade: Maximum risk per trade (e.g., 0.02 for 2%)
        
    Returns:
        Dict with position sizing details
    """
    calculator = KellyCalculator(capital=capital)
    
    # Assume stop loss is 1/3 of expected move
    stop_loss = expected_move / 3
    
    shares, position_value = calculator.calculate_position_size(
        conviction=conviction,
        expected_return=expected_move,
        stop_loss_pct=stop_loss,
        current_price=100.0  # Normalized price
    )
    
    return {
        'shares': shares,
        'position_value': position_value,
        'position_pct': position_value / capital,
        'kelly_fraction': position_value / capital,
        'risk_amount': position_value * stop_loss,
        'expected_profit': position_value * expected_move
    }


def main():
    """Test Kelly calculations."""
    print("Kelly Criterion Position Sizing Tests\n")
    print("=" * 50)
    
    # Test 1: High conviction trade
    print("\nTest 1: High Conviction Trade")
    print("-" * 30)
    result = calculate_optimal_position(
        conviction=0.75,
        expected_move=0.05,
        capital=100000
    )
    for key, value in result.items():
        if 'shares' in key:
            print(f"{key}: {value}")
        else:
            print(f"{key}: ${value:,.2f}" if 'value' in key or 'amount' in key 
                  else f"{key}: {value:.2%}")
    
    # Test 2: Medium conviction
    print("\nTest 2: Medium Conviction Trade")
    print("-" * 30)
    result = calculate_optimal_position(
        conviction=0.60,
        expected_move=0.03,
        capital=100000
    )
    for key, value in result.items():
        if 'shares' in key:
            print(f"{key}: {value}")
        else:
            print(f"{key}: ${value:,.2f}" if 'value' in key or 'amount' in key 
                  else f"{key}: {value:.2%}")
    
    # Test 3: Multi-asset allocation
    print("\nTest 3: Multi-Asset Allocation")
    print("-" * 30)
    
    calculator = KellyCalculator(capital=100000)
    
    opportunities = [
        {'symbol': 'AAPL', 'conviction': 0.70, 'expected_return': 0.04, 'price': 150},
        {'symbol': 'TSLA', 'conviction': 0.65, 'expected_return': 0.06, 'price': 200},
        {'symbol': 'NVDA', 'conviction': 0.80, 'expected_return': 0.03, 'price': 400},
    ]
    
    allocations = calculator.calculate_multi_asset_allocation(opportunities)
    
    total_allocated = 0
    for symbol, (shares, value) in allocations.items():
        print(f"{symbol}: {shares} shares = ${value:,.2f}")
        total_allocated += value
        
    print(f"\nTotal Allocated: ${total_allocated:,.2f} ({total_allocated/100000:.1%} of capital)")
    
    # Test 4: Adaptive Kelly with history
    print("\nTest 4: Adaptive Kelly with Performance History")
    print("-" * 30)
    
    calc = KellyCalculator(capital=100000)
    
    # Simulate some trade history
    trade_results = [
        (True, 0.03), (True, 0.02), (False, -0.01), (True, 0.04),
        (False, -0.02), (True, 0.03), (True, 0.05), (False, -0.01),
        (True, 0.02), (True, 0.03), (True, 0.04), (False, -0.02)
    ]
    
    for won, return_pct in trade_results:
        calc.update_performance(won, return_pct)
        
    print(f"Historical win rate: {calc.win_rate_estimate:.1%}")
    
    ai_conviction = 0.70
    adjusted = calc.suggest_conviction_adjustment(ai_conviction)
    print(f"AI conviction: {ai_conviction:.1%}")
    print(f"Adjusted conviction: {adjusted:.1%}")


if __name__ == "__main__":
    main()