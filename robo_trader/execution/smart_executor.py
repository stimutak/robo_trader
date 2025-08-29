"""Smart Execution Algorithms for RoboTrader.

This module implements sophisticated order execution algorithms:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Adaptive lot sizing
- Market impact minimization
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import structlog

from ..config import Config
from ..risk import Position

logger = structlog.get_logger(__name__)


class ExecutionAlgorithm(Enum):
    """Available execution algorithms."""
    
    MARKET = "market"          # Immediate execution
    TWAP = "twap"              # Time-weighted average price
    VWAP = "vwap"              # Volume-weighted average price
    ICEBERG = "iceberg"        # Show only part of order
    ADAPTIVE = "adaptive"      # Dynamically adjust based on conditions
    SNIPER = "sniper"          # Wait for optimal conditions


@dataclass
class ExecutionParams:
    """Parameters for smart execution."""
    
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    duration_minutes: int = 30
    slice_count: int = 10
    max_participation: float = 0.1  # Max % of volume
    urgency: float = 0.5  # 0=patient, 1=aggressive
    min_slice_size: int = 100
    max_slice_size: int = 10000
    iceberg_display_ratio: float = 0.2
    adaptive_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Execution plan with timing and sizing."""
    
    symbol: str
    side: str  # BUY or SELL
    total_quantity: int
    slices: List[Dict[str, Any]]
    algorithm: ExecutionAlgorithm
    estimated_duration: timedelta
    estimated_cost: float
    market_impact_bps: float


@dataclass
class ExecutionResult:
    """Result of smart execution."""
    
    symbol: str
    side: str
    requested_quantity: int
    executed_quantity: int
    average_price: float
    slippage_bps: float
    execution_time_ms: float
    fills: List[Dict[str, Any]]
    success: bool
    message: str


class SmartExecutor:
    """Smart order execution with advanced algorithms."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logger.bind(component="smart_executor")
        
        # Market data cache
        self.volume_profiles: Dict[str, pd.DataFrame] = {}
        self.spread_history: Dict[str, List[float]] = {}
        self.recent_trades: Dict[str, List[Dict]] = {}
        
        # Execution tracking
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # Market impact model parameters
        self.impact_params = {
            'temporary_impact': 0.1,  # bps per % of ADV
            'permanent_impact': 0.05,  # bps per % of ADV
            'decay_rate': 0.5,         # Impact decay over time
        }
    
    async def create_execution_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        params: Optional[ExecutionParams] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """Create an execution plan based on algorithm and market conditions."""
        
        if params is None:
            params = ExecutionParams()
        
        self.logger.info(
            "Creating execution plan",
            symbol=symbol,
            side=side,
            quantity=quantity,
            algorithm=params.algorithm.value
        )
        
        # Choose execution strategy
        if params.algorithm == ExecutionAlgorithm.TWAP:
            plan = await self._create_twap_plan(symbol, side, quantity, params, market_data)
        elif params.algorithm == ExecutionAlgorithm.VWAP:
            plan = await self._create_vwap_plan(symbol, side, quantity, params, market_data)
        elif params.algorithm == ExecutionAlgorithm.ICEBERG:
            plan = await self._create_iceberg_plan(symbol, side, quantity, params, market_data)
        elif params.algorithm == ExecutionAlgorithm.ADAPTIVE:
            plan = await self._create_adaptive_plan(symbol, side, quantity, params, market_data)
        else:
            # Default to market order
            plan = await self._create_market_plan(symbol, side, quantity, market_data)
        
        # Estimate market impact
        plan.market_impact_bps = self._estimate_market_impact(
            symbol, quantity, plan.estimated_duration, market_data
        )
        
        self.active_plans[f"{symbol}_{datetime.now().isoformat()}"] = plan
        
        return plan
    
    async def _create_twap_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        params: ExecutionParams,
        market_data: Optional[Dict[str, Any]]
    ) -> ExecutionPlan:
        """Create TWAP (Time-Weighted Average Price) execution plan."""
        
        # Calculate slice size and timing
        slice_size = max(
            params.min_slice_size,
            min(params.max_slice_size, quantity // params.slice_count)
        )
        
        actual_slices = quantity // slice_size
        remainder = quantity % slice_size
        
        # Create time-weighted slices
        interval_seconds = (params.duration_minutes * 60) / actual_slices
        
        slices = []
        current_time = datetime.now()
        
        for i in range(actual_slices):
            slice_qty = slice_size + (remainder if i == actual_slices - 1 else 0)
            
            slices.append({
                'time': current_time + timedelta(seconds=i * interval_seconds),
                'quantity': slice_qty,
                'type': 'LIMIT',
                'price_offset': 0,  # Will be set at execution time
                'urgency': params.urgency
            })
        
        return ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            slices=slices,
            algorithm=ExecutionAlgorithm.TWAP,
            estimated_duration=timedelta(minutes=params.duration_minutes),
            estimated_cost=0,  # Will be calculated
            market_impact_bps=0  # Will be calculated
        )
    
    async def _create_vwap_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        params: ExecutionParams,
        market_data: Optional[Dict[str, Any]]
    ) -> ExecutionPlan:
        """Create VWAP (Volume-Weighted Average Price) execution plan."""
        
        # Get historical volume profile
        volume_profile = await self._get_volume_profile(symbol)
        
        if volume_profile is None or volume_profile.empty:
            # Fall back to TWAP if no volume data
            return await self._create_twap_plan(symbol, side, quantity, params, market_data)
        
        # Distribute quantity according to typical volume pattern
        total_volume = volume_profile['volume'].sum()
        
        slices = []
        remaining = quantity
        current_time = datetime.now()
        
        for idx, row in volume_profile.iterrows():
            # Calculate proportion based on historical volume
            volume_weight = row['volume'] / total_volume
            slice_qty = min(remaining, int(quantity * volume_weight))
            
            if slice_qty >= params.min_slice_size:
                slices.append({
                    'time': current_time + timedelta(minutes=idx * 5),  # 5-min intervals
                    'quantity': slice_qty,
                    'type': 'LIMIT',
                    'price_offset': 0,
                    'urgency': params.urgency * (1 + volume_weight),  # More urgent during high volume
                    'expected_volume': row['volume']
                })
                remaining -= slice_qty
        
        # Add remainder to last slice
        if remaining > 0 and slices:
            slices[-1]['quantity'] += remaining
        
        return ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            slices=slices,
            algorithm=ExecutionAlgorithm.VWAP,
            estimated_duration=timedelta(minutes=len(slices) * 5),
            estimated_cost=0,
            market_impact_bps=0
        )
    
    async def _create_iceberg_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        params: ExecutionParams,
        market_data: Optional[Dict[str, Any]]
    ) -> ExecutionPlan:
        """Create iceberg order execution plan."""
        
        # Display only a fraction of total order
        display_size = max(
            params.min_slice_size,
            int(quantity * params.iceberg_display_ratio)
        )
        
        # Create hidden slices
        slices = []
        remaining = quantity
        slice_num = 0
        
        while remaining > 0:
            slice_qty = min(display_size, remaining)
            
            slices.append({
                'time': datetime.now() + timedelta(seconds=slice_num * 30),
                'quantity': slice_qty,
                'type': 'LIMIT',
                'price_offset': 0,
                'urgency': params.urgency,
                'hidden': slice_num > 0,  # First slice visible, rest hidden
                'display_size': display_size if slice_qty > display_size else slice_qty
            })
            
            remaining -= slice_qty
            slice_num += 1
        
        return ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            slices=slices,
            algorithm=ExecutionAlgorithm.ICEBERG,
            estimated_duration=timedelta(seconds=len(slices) * 30),
            estimated_cost=0,
            market_impact_bps=0
        )
    
    async def _create_adaptive_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        params: ExecutionParams,
        market_data: Optional[Dict[str, Any]]
    ) -> ExecutionPlan:
        """Create adaptive execution plan based on real-time conditions."""
        
        # Analyze current market conditions
        conditions = await self._analyze_market_conditions(symbol, market_data)
        
        # Choose strategy based on conditions
        if conditions.get('high_volatility', False):
            # Use smaller slices in volatile markets
            params.slice_count = params.slice_count * 2
            params.urgency = max(0.3, params.urgency - 0.2)
            return await self._create_twap_plan(symbol, side, quantity, params, market_data)
            
        elif conditions.get('low_liquidity', False):
            # Use iceberg for low liquidity
            params.iceberg_display_ratio = 0.1
            return await self._create_iceberg_plan(symbol, side, quantity, params, market_data)
            
        elif conditions.get('trending', False):
            # Be more aggressive in trending markets
            params.urgency = min(1.0, params.urgency + 0.3)
            params.duration_minutes = max(10, params.duration_minutes // 2)
            return await self._create_vwap_plan(symbol, side, quantity, params, market_data)
            
        else:
            # Default to VWAP in normal conditions
            return await self._create_vwap_plan(symbol, side, quantity, params, market_data)
    
    async def _create_market_plan(
        self,
        symbol: str,
        side: str,
        quantity: int,
        market_data: Optional[Dict[str, Any]]
    ) -> ExecutionPlan:
        """Create simple market order execution plan."""
        
        return ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            slices=[{
                'time': datetime.now(),
                'quantity': quantity,
                'type': 'MARKET',
                'price_offset': 0,
                'urgency': 1.0
            }],
            algorithm=ExecutionAlgorithm.MARKET,
            estimated_duration=timedelta(seconds=1),
            estimated_cost=0,
            market_impact_bps=0
        )
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        executor: Any  # The actual order executor (paper or live)
    ) -> ExecutionResult:
        """Execute a smart execution plan."""
        
        start_time = datetime.now()
        fills = []
        total_executed = 0
        total_value = 0
        
        self.logger.info(
            "Executing plan",
            symbol=plan.symbol,
            algorithm=plan.algorithm.value,
            slices=len(plan.slices)
        )
        
        try:
            for slice_config in plan.slices:
                # Wait until scheduled time
                wait_time = (slice_config['time'] - datetime.now()).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                # Adjust price based on urgency and market conditions
                current_price = await self._get_current_price(plan.symbol)
                
                if slice_config['type'] == 'LIMIT':
                    # Set limit price based on urgency
                    if plan.side == 'BUY':
                        limit_price = current_price * (1 + 0.001 * slice_config['urgency'])
                    else:
                        limit_price = current_price * (1 - 0.001 * slice_config['urgency'])
                else:
                    limit_price = None
                
                # Execute slice
                fill = await executor.execute_order(
                    symbol=plan.symbol,
                    side=plan.side,
                    quantity=slice_config['quantity'],
                    order_type=slice_config['type'],
                    limit_price=limit_price
                )
                
                if fill and fill.get('executed_quantity', 0) > 0:
                    fills.append(fill)
                    total_executed += fill['executed_quantity']
                    total_value += fill['executed_quantity'] * fill['price']
                    
                    self.logger.info(
                        "Slice executed",
                        symbol=plan.symbol,
                        quantity=fill['executed_quantity'],
                        price=fill['price']
                    )
                
                # Check if we should abort
                if total_executed >= plan.total_quantity * 0.95:
                    break
            
            # Calculate results
            if total_executed > 0:
                avg_price = total_value / total_executed
                slippage_bps = self._calculate_slippage(
                    plan.symbol,
                    plan.side,
                    avg_price,
                    await self._get_current_price(plan.symbol)
                )
                success = True
                message = f"Executed {total_executed}/{plan.total_quantity} @ {avg_price:.2f}"
            else:
                avg_price = 0
                slippage_bps = 0
                success = False
                message = "No fills executed"
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = ExecutionResult(
                symbol=plan.symbol,
                side=plan.side,
                requested_quantity=plan.total_quantity,
                executed_quantity=total_executed,
                average_price=avg_price,
                slippage_bps=slippage_bps,
                execution_time_ms=execution_time,
                fills=fills,
                success=success,
                message=message
            )
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                symbol=plan.symbol,
                side=plan.side,
                requested_quantity=plan.total_quantity,
                executed_quantity=total_executed,
                average_price=total_value / total_executed if total_executed > 0 else 0,
                slippage_bps=0,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                fills=fills,
                success=False,
                message=str(e)
            )
    
    async def _get_volume_profile(self, symbol: str) -> pd.DataFrame:
        """Get historical intraday volume profile."""
        
        # Check cache
        if symbol in self.volume_profiles:
            return self.volume_profiles[symbol]
        
        # In production, fetch from market data provider
        # For now, return synthetic profile
        times = pd.date_range('09:30', '16:00', freq='5min')
        
        # U-shaped volume profile (high at open/close)
        volumes = []
        for t in times:
            hour = t.hour + t.minute / 60
            if hour < 10:  # Morning
                vol = 1000 * (10 - hour)
            elif hour > 15.5:  # Close
                vol = 1000 * (hour - 15)
            else:  # Midday
                vol = 500
            volumes.append(max(100, int(vol + np.random.normal(0, 100))))
        
        profile = pd.DataFrame({
            'time': times,
            'volume': volumes
        })
        
        self.volume_profiles[symbol] = profile
        return profile
    
    async def _analyze_market_conditions(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze current market conditions for adaptive execution."""
        
        conditions = {
            'high_volatility': False,
            'low_liquidity': False,
            'trending': False,
            'wide_spread': False
        }
        
        if market_data:
            # Check volatility
            if market_data.get('volatility', 0) > 0.02:
                conditions['high_volatility'] = True
            
            # Check liquidity (volume)
            if market_data.get('volume', 0) < 1000000:
                conditions['low_liquidity'] = True
            
            # Check trend
            if abs(market_data.get('price_change_pct', 0)) > 0.01:
                conditions['trending'] = True
            
            # Check spread
            spread_bps = market_data.get('spread_bps', 0)
            if spread_bps > 10:
                conditions['wide_spread'] = True
        
        return conditions
    
    def _estimate_market_impact(
        self,
        symbol: str,
        quantity: int,
        duration: timedelta,
        market_data: Optional[Dict[str, Any]]
    ) -> float:
        """Estimate market impact in basis points."""
        
        if not market_data:
            return 5.0  # Default 5 bps
        
        # Get average daily volume
        adv = market_data.get('avg_volume', 1000000)
        
        # Calculate participation rate
        participation = quantity / (adv * duration.total_seconds() / 86400)
        
        # Square-root market impact model
        temporary_impact = self.impact_params['temporary_impact'] * np.sqrt(participation)
        permanent_impact = self.impact_params['permanent_impact'] * participation
        
        total_impact = temporary_impact + permanent_impact
        
        return min(50, total_impact * 10000)  # Cap at 50 bps
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        # In production, fetch from market data
        # For now, return mock price
        return 100.0
    
    def _calculate_slippage(
        self,
        symbol: str,
        side: str,
        executed_price: float,
        expected_price: float
    ) -> float:
        """Calculate slippage in basis points."""
        
        if expected_price == 0:
            return 0
        
        if side == 'BUY':
            # Negative slippage if we paid more than expected
            slippage = (executed_price - expected_price) / expected_price
        else:
            # Negative slippage if we sold for less than expected
            slippage = (expected_price - executed_price) / expected_price
        
        return slippage * 10000  # Convert to basis points
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        
        if not self.execution_history:
            return {
                'total_executions': 0,
                'success_rate': 0,
                'avg_slippage_bps': 0,
                'avg_execution_time_ms': 0
            }
        
        successful = [r for r in self.execution_history if r.success]
        
        return {
            'total_executions': len(self.execution_history),
            'success_rate': len(successful) / len(self.execution_history),
            'avg_slippage_bps': np.mean([r.slippage_bps for r in successful]) if successful else 0,
            'avg_execution_time_ms': np.mean([r.execution_time_ms for r in self.execution_history]),
            'total_volume': sum(r.executed_quantity for r in self.execution_history),
            'algorithms_used': {
                'twap': sum(1 for p in self.active_plans.values() if p.algorithm == ExecutionAlgorithm.TWAP),
                'vwap': sum(1 for p in self.active_plans.values() if p.algorithm == ExecutionAlgorithm.VWAP),
                'iceberg': sum(1 for p in self.active_plans.values() if p.algorithm == ExecutionAlgorithm.ICEBERG),
                'adaptive': sum(1 for p in self.active_plans.values() if p.algorithm == ExecutionAlgorithm.ADAPTIVE),
            }
        }