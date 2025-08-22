"""
Edge calculation and Expected Value (EV) modeling.
Ensures trades have positive expectancy after costs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeSetup:
    """Definition of a trade setup type."""
    name: str
    description: str
    typical_win_rate: float  # Historical win rate
    typical_rr_ratio: float  # Risk:reward ratio
    typical_holding_period_hours: int
    
    def get_base_ev(self) -> float:
        """Get base expected value as percentage."""
        # EV = (win_rate * reward) - ((1 - win_rate) * risk)
        # Where risk = 1 and reward = rr_ratio
        ev = (self.typical_win_rate * self.typical_rr_ratio) - ((1 - self.typical_win_rate) * 1)
        return ev * 100  # Return as percentage


@dataclass
class EVCalculation:
    """Expected value calculation for a trade."""
    symbol: str
    p_win: float  # Probability of winning (0-1)
    avg_win_pct: float  # Average win as percentage
    avg_loss_pct: float  # Average loss as percentage (positive number)
    commission_bps: float  # Commission in basis points
    slippage_bps: float  # Expected slippage in bps
    time_decay_factor: float  # Decay factor for time value
    
    # Calculated fields
    gross_ev_pct: float = 0
    net_ev_pct: float = 0
    risk_reward_ratio: float = 0
    kelly_fraction: float = 0
    sharpe_estimate: float = 0
    
    def calculate(self) -> 'EVCalculation':
        """Calculate all EV metrics."""
        # Gross EV before costs
        self.gross_ev_pct = (self.p_win * self.avg_win_pct) - ((1 - self.p_win) * self.avg_loss_pct)
        
        # Net EV after costs (costs apply on entry and exit)
        total_cost_pct = (self.commission_bps + self.slippage_bps) * 2 / 100
        self.net_ev_pct = self.gross_ev_pct - total_cost_pct
        
        # Apply time decay
        self.net_ev_pct *= self.time_decay_factor
        
        # Risk:reward ratio
        if self.avg_loss_pct > 0:
            self.risk_reward_ratio = self.avg_win_pct / self.avg_loss_pct
        
        # Kelly fraction (simplified)
        if self.avg_loss_pct > 0:
            self.kelly_fraction = (self.p_win * self.risk_reward_ratio - (1 - self.p_win)) / self.risk_reward_ratio
            self.kelly_fraction = max(0, min(self.kelly_fraction, 0.25))  # Cap at 25% Kelly
        
        # Sharpe estimate (simplified)
        if self.avg_loss_pct > 0:
            expected_return = self.net_ev_pct / 100
            variance = (
                self.p_win * (self.avg_win_pct/100 - expected_return)**2 +
                (1 - self.p_win) * (-self.avg_loss_pct/100 - expected_return)**2
            )
            if variance > 0:
                self.sharpe_estimate = expected_return / np.sqrt(variance)
        
        return self
    
    def meets_threshold(self, min_ev_pct: float = 0, min_rr: float = 1.8) -> bool:
        """Check if EV meets minimum thresholds."""
        return self.net_ev_pct > min_ev_pct and self.risk_reward_ratio >= min_rr


class EdgeCalculator:
    """
    Calculate trading edge and expected value.
    Tracks historical performance by setup type.
    """
    
    # Predefined setup types
    SETUPS = {
        "breakout": TradeSetup(
            "breakout", "Price breaking key resistance", 
            0.45, 2.5, 48
        ),
        "mean_reversion": TradeSetup(
            "mean_reversion", "Reversion to mean from extreme",
            0.65, 1.5, 24
        ),
        "momentum": TradeSetup(
            "momentum", "Continuation of strong trend",
            0.50, 2.2, 72
        ),
        "earnings_beat": TradeSetup(
            "earnings_beat", "Post-earnings momentum",
            0.55, 2.0, 36
        ),
        "earnings_miss": TradeSetup(
            "earnings_miss", "Earnings disappointment short",
            0.52, 1.8, 24
        ),
        "fed_dovish": TradeSetup(
            "fed_dovish", "Fed dovish surprise",
            0.60, 3.0, 96
        ),
        "fed_hawkish": TradeSetup(
            "fed_hawkish", "Fed hawkish surprise",
            0.58, 2.5, 72
        ),
        "options_flow": TradeSetup(
            "options_flow", "Following smart money options",
            0.48, 2.0, 48
        ),
        "insider_buying": TradeSetup(
            "insider_buying", "Following insider purchases",
            0.53, 2.8, 120
        ),
        "short_squeeze": TradeSetup(
            "short_squeeze", "High short interest squeeze",
            0.42, 3.5, 36
        ),
        "sector_rotation": TradeSetup(
            "sector_rotation", "Sector leadership change",
            0.50, 2.0, 96
        )
    }
    
    def __init__(self):
        """Initialize edge calculator."""
        # Track performance by setup
        self.setup_performance: Dict[str, List[float]] = {
            setup: [] for setup in self.SETUPS
        }
        
        # Track overall statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl_pct = 0
        
        logger.info(f"EdgeCalculator initialized with {len(self.SETUPS)} setup types")
    
    def calculate_ev(
        self,
        symbol: str,
        setup_type: str,
        conviction: float,  # 0-1 conviction score
        current_volatility: float,  # Current IV or HV
        spread_bps: float,
        time_to_catalyst_hours: Optional[int] = None,
        custom_p_win: Optional[float] = None,
        custom_rr: Optional[float] = None
    ) -> EVCalculation:
        """
        Calculate expected value for a trade.
        
        Args:
            symbol: Trading symbol
            setup_type: Type of setup
            conviction: Conviction score (0-1)
            current_volatility: Current volatility (for sizing wins/losses)
            spread_bps: Current bid-ask spread in bps
            time_to_catalyst_hours: Hours until catalyst
            custom_p_win: Override probability of win
            custom_rr: Override risk:reward ratio
            
        Returns:
            EVCalculation with all metrics
        """
        # Get base setup or use default
        setup = self.SETUPS.get(setup_type, self.SETUPS["momentum"])
        
        # Adjust p_win based on conviction
        base_p_win = custom_p_win or setup.typical_win_rate
        p_win = base_p_win * (0.8 + 0.4 * conviction)  # Scale by conviction
        p_win = min(0.85, max(0.15, p_win))  # Cap between 15% and 85%
        
        # Adjust risk:reward based on setup
        rr_ratio = custom_rr or setup.typical_rr_ratio
        
        # Calculate win/loss percentages based on volatility
        # Higher volatility = larger wins and losses
        volatility_factor = max(0.5, min(2.0, current_volatility / 20))  # Normalize around 20% vol
        
        avg_win_pct = rr_ratio * 2.0 * volatility_factor  # Base 2% win scaled by RR and vol
        avg_loss_pct = 2.0 * volatility_factor  # Base 2% loss scaled by vol
        
        # Calculate slippage based on spread and volatility
        base_slippage = spread_bps / 2  # Half spread as base
        volatility_slippage = volatility_factor * 2  # Additional slippage from vol
        slippage_bps = base_slippage + volatility_slippage
        
        # Time decay factor
        if time_to_catalyst_hours:
            # Decay factor: 1.0 at catalyst, decays over time
            hours_since_ideal = max(0, time_to_catalyst_hours - setup.typical_holding_period_hours)
            time_decay_factor = np.exp(-hours_since_ideal / (setup.typical_holding_period_hours * 4))
        else:
            time_decay_factor = 1.0
        
        # Create and calculate EV
        ev_calc = EVCalculation(
            symbol=symbol,
            p_win=p_win,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            commission_bps=0.5,  # Assume 0.5bp commission
            slippage_bps=slippage_bps,
            time_decay_factor=time_decay_factor
        ).calculate()
        
        logger.debug(
            f"EV for {symbol} {setup_type}: "
            f"P(win)={p_win:.1%}, RR={rr_ratio:.1f}, "
            f"NetEV={ev_calc.net_ev_pct:.2f}%"
        )
        
        return ev_calc
    
    def update_performance(
        self,
        setup_type: str,
        actual_pnl_pct: float,
        was_winner: bool
    ):
        """
        Update performance tracking for a setup.
        
        Args:
            setup_type: Type of setup
            actual_pnl_pct: Actual P&L as percentage
            was_winner: Whether trade was profitable
        """
        if setup_type in self.setup_performance:
            self.setup_performance[setup_type].append(actual_pnl_pct)
        
        self.total_trades += 1
        if was_winner:
            self.winning_trades += 1
        self.total_pnl_pct += actual_pnl_pct
        
        # Keep only last 100 trades per setup
        if len(self.setup_performance[setup_type]) > 100:
            self.setup_performance[setup_type].pop(0)
    
    def get_setup_stats(self, setup_type: str) -> Dict:
        """Get performance statistics for a setup type."""
        if setup_type not in self.setup_performance:
            return {}
        
        results = self.setup_performance[setup_type]
        if not results:
            return {}
        
        wins = [r for r in results if r > 0]
        losses = [r for r in results if r <= 0]
        
        stats = {
            "total_trades": len(results),
            "win_rate": len(wins) / len(results) if results else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "total_pnl": sum(results),
            "sharpe": np.mean(results) / np.std(results) if len(results) > 1 and np.std(results) > 0 else 0
        }
        
        return stats
    
    def get_best_setups(self, min_trades: int = 10) -> List[Tuple[str, float]]:
        """
        Get best performing setup types.
        
        Args:
            min_trades: Minimum trades to qualify
            
        Returns:
            List of (setup_type, avg_pnl) sorted by performance
        """
        performances = []
        
        for setup_type, results in self.setup_performance.items():
            if len(results) >= min_trades:
                avg_pnl = np.mean(results)
                performances.append((setup_type, avg_pnl))
        
        # Sort by average P&L
        performances.sort(key=lambda x: x[1], reverse=True)
        
        return performances
    
    def calculate_optimal_size(
        self,
        ev_calc: EVCalculation,
        max_risk_bps: int = 50,
        kelly_fraction: float = 0.25
    ) -> int:
        """
        Calculate optimal position size in risk basis points.
        
        Args:
            ev_calc: EV calculation
            max_risk_bps: Maximum risk in bps
            kelly_fraction: Fraction of Kelly to use
            
        Returns:
            Optimal size in risk basis points
        """
        if ev_calc.net_ev_pct <= 0:
            return 0
        
        # Use Kelly criterion
        kelly_size = ev_calc.kelly_fraction * kelly_fraction * 10000  # Convert to bps
        
        # Apply conviction scaling
        # Higher conviction = closer to full Kelly
        conviction_factor = 0.5 + 0.5 * min(1.0, ev_calc.net_ev_pct / 2.0)
        
        optimal_bps = int(kelly_size * conviction_factor)
        
        # Cap at max risk
        return min(optimal_bps, max_risk_bps)
    
    def get_regime_adjustment(self, market_regime: str) -> float:
        """
        Get EV adjustment factor based on market regime.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Adjustment multiplier for EV
        """
        regime_factors = {
            "bull_trend": 1.1,      # Favorable for longs
            "bear_trend": 0.9,      # Reduce long EV
            "high_volatility": 0.8, # Reduce EV in chaos
            "low_volatility": 1.0,  # Normal
            "risk_on": 1.05,        # Slight boost
            "risk_off": 0.85,       # Reduce risk
        }
        
        return regime_factors.get(market_regime, 1.0)