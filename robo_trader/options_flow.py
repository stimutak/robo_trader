"""
Options flow analysis for detecting unusual activity and institutional positioning.
Fetches options data from IB API and identifies smart money moves.
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ib_insync import Option, OptionChain
from .logger import get_logger
from .ibkr_client import IBKRClient

logger = get_logger(__name__)


@dataclass
class OptionsFlowSignal:
    """Unusual options activity signal."""
    symbol: str
    timestamp: datetime
    strike: float
    expiry: str
    option_type: str  # 'CALL' or 'PUT'
    volume: int
    open_interest: int
    volume_oi_ratio: float
    premium: float
    delta: float
    implied_vol: float
    signal_type: str  # 'unusual_volume', 'sweep', 'block', 'high_premium'
    confidence: float  # 0-100
    interpretation: str


class OptionsFlowAnalyzer:
    """Analyzes options flow to detect institutional activity."""
    
    # Thresholds for unusual activity
    UNUSUAL_VOLUME_RATIO = 2.0  # Volume > 2x open interest
    HIGH_PREMIUM_THRESHOLD = 100000  # $100k+ single trades
    BLOCK_TRADE_SIZE = 100  # 100+ contracts
    SWEEP_TIME_WINDOW = 60  # Seconds for sweep detection
    
    def __init__(self, ib_client: IBKRClient):
        """
        Initialize options flow analyzer.
        
        Args:
            ib_client: Connected IB client
        """
        self.ib_client = ib_client
        self.flow_cache: Dict[str, List[Dict]] = {}
        self.sweep_detector: Dict[str, List[Tuple[datetime, int]]] = {}
        
    async def scan_options_flow(self, symbols: List[str]) -> List[OptionsFlowSignal]:
        """
        Scan for unusual options activity across symbols.
        
        Args:
            symbols: List of underlying symbols to scan
            
        Returns:
            List of unusual activity signals
        """
        signals = []
        
        for symbol in symbols:
            try:
                # Get options chain from IB
                chain_signals = await self._analyze_symbol_options(symbol)
                signals.extend(chain_signals)
                
            except Exception as e:
                logger.error(f"Error scanning options for {symbol}: {e}")
                
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        if signals:
            logger.info(f"Found {len(signals)} unusual options activities")
            for signal in signals[:5]:  # Log top 5
                logger.info(
                    f"  {signal.symbol} {signal.option_type} {signal.strike} "
                    f"{signal.expiry}: {signal.signal_type} "
                    f"(confidence: {signal.confidence:.0f}%)"
                )
                
        return signals
        
    async def _analyze_symbol_options(self, symbol: str) -> List[OptionsFlowSignal]:
        """Analyze options chain for a single symbol."""
        signals = []
        
        try:
            # Qualify the stock contract
            stock = await self.ib_client.qualify_stock(symbol)
            
            # Get options chain
            chains = await self.ib_client.ib.reqSecDefOptParamsAsync(
                stock.symbol,
                stock.exchange,
                stock.secType,
                stock.conId
            )
            
            if not chains:
                return signals
                
            chain = chains[0]
            
            # Get near-term expirations (next 30 days)
            today = datetime.now()
            cutoff = today + timedelta(days=30)
            
            near_expirations = [
                exp for exp in chain.expirations 
                if datetime.strptime(exp, '%Y%m%d') <= cutoff
            ][:3]  # Top 3 nearest expirations
            
            for expiry in near_expirations:
                # Analyze calls and puts separately
                for right in ['C', 'P']:
                    option_signals = await self._analyze_option_strikes(
                        symbol, expiry, right, chain.strikes
                    )
                    signals.extend(option_signals)
                    
        except Exception as e:
            logger.debug(f"Error analyzing {symbol} options: {e}")
            
        return signals
        
    async def _analyze_option_strikes(
        self, 
        symbol: str, 
        expiry: str, 
        right: str,
        strikes: List[float]
    ) -> List[OptionsFlowSignal]:
        """Analyze specific option strikes for unusual activity."""
        signals = []
        
        # Get current stock price for ATM calculation
        bars = await self.ib_client.fetch_recent_bars(symbol, "1 D", "5 mins")
        if bars.empty:
            return signals
            
        current_price = bars.iloc[-1]['close']
        
        # Focus on near-the-money strikes (within 10% of current price)
        atm_strikes = [
            s for s in strikes 
            if 0.9 * current_price <= s <= 1.1 * current_price
        ][:5]  # Top 5 strikes near ATM
        
        for strike in atm_strikes:
            try:
                # Create option contract
                option = Option(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=expiry,
                    strike=strike,
                    right=right,
                    exchange='SMART'
                )
                
                # Qualify and get market data
                qualified = self.ib_client.ib.qualifyContracts(option)
                if not qualified:
                    continue
                    
                option_contract = qualified[0]
                
                # Request market data
                ticker = self.ib_client.ib.reqMktData(option_contract, '', False, False)
                await asyncio.sleep(2)  # Wait for data
                
                # Analyze for unusual activity
                signal = self._detect_unusual_activity(
                    symbol, expiry, right, strike, ticker, current_price
                )
                
                if signal:
                    signals.append(signal)
                    
                # Cancel market data
                self.ib_client.ib.cancelMktData(option_contract)
                
            except Exception as e:
                logger.debug(f"Error analyzing {symbol} {strike} {right}: {e}")
                
        return signals
        
    def _detect_unusual_activity(
        self,
        symbol: str,
        expiry: str,
        right: str,
        strike: float,
        ticker,
        stock_price: float
    ) -> Optional[OptionsFlowSignal]:
        """Detect unusual activity patterns in option data."""
        
        # Check if we have sufficient data
        if not ticker.volume or not ticker.openInterest:
            return None
            
        volume = ticker.volume
        open_interest = ticker.openInterest
        
        # Skip if no meaningful activity
        if volume < 10:
            return None
            
        # Calculate metrics
        volume_oi_ratio = volume / max(open_interest, 1)
        
        # Estimate premium (rough calculation)
        if ticker.last:
            premium = ticker.last * volume * 100  # 100 shares per contract
        else:
            return None
            
        # Calculate moneyness
        if right == 'C':
            moneyness = (stock_price - strike) / stock_price
            option_type = 'CALL'
        else:
            moneyness = (strike - stock_price) / stock_price
            option_type = 'PUT'
            
        # Detect signal types
        signal_type = None
        confidence = 0.0
        interpretation = ""
        
        # Check for unusual volume
        if volume_oi_ratio > self.UNUSUAL_VOLUME_RATIO:
            signal_type = 'unusual_volume'
            confidence = min(volume_oi_ratio * 20, 90)  # Cap at 90%
            interpretation = (
                f"Volume {volume_oi_ratio:.1f}x open interest suggests "
                f"new institutional positioning"
            )
            
        # Check for high premium trades
        elif premium > self.HIGH_PREMIUM_THRESHOLD:
            signal_type = 'high_premium'
            confidence = min(premium / self.HIGH_PREMIUM_THRESHOLD * 30, 80)
            interpretation = (
                f"${premium:,.0f} premium indicates large institutional bet"
            )
            
        # Check for block trades
        elif volume >= self.BLOCK_TRADE_SIZE:
            signal_type = 'block'
            confidence = min(volume / self.BLOCK_TRADE_SIZE * 25, 70)
            interpretation = (
                f"{volume} contract block suggests institutional accumulation"
            )
            
        # Check for sweep patterns (multiple strikes hit quickly)
        if signal_type and self._is_sweep_pattern(symbol, expiry, right, volume):
            signal_type = 'sweep'
            confidence = min(confidence + 20, 95)
            interpretation += " - SWEEP DETECTED across multiple strikes"
            
        if not signal_type:
            return None
            
        # Add directional bias to interpretation
        if option_type == 'CALL' and moneyness > -0.05:  # Near or ITM calls
            interpretation += f" - BULLISH on {symbol}"
        elif option_type == 'PUT' and moneyness > -0.05:  # Near or ITM puts
            interpretation += f" - BEARISH on {symbol}"
            
        return OptionsFlowSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            volume=volume,
            open_interest=open_interest,
            volume_oi_ratio=volume_oi_ratio,
            premium=premium,
            delta=ticker.modelGreeks.delta if ticker.modelGreeks else 0,
            implied_vol=ticker.modelGreeks.impliedVol if ticker.modelGreeks else 0,
            signal_type=signal_type,
            confidence=confidence,
            interpretation=interpretation
        )
        
    def _is_sweep_pattern(
        self, 
        symbol: str, 
        expiry: str, 
        right: str, 
        volume: int
    ) -> bool:
        """
        Detect sweep patterns (rapid buying across multiple strikes).
        
        A sweep occurs when large orders hit multiple strikes within
        a short time window, indicating urgency.
        """
        key = f"{symbol}_{expiry}_{right}"
        now = datetime.now()
        
        if key not in self.sweep_detector:
            self.sweep_detector[key] = []
            
        # Add current activity
        self.sweep_detector[key].append((now, volume))
        
        # Clean old entries
        cutoff = now - timedelta(seconds=self.SWEEP_TIME_WINDOW)
        self.sweep_detector[key] = [
            (t, v) for t, v in self.sweep_detector[key] if t > cutoff
        ]
        
        # Check if we have multiple strikes hit recently
        if len(self.sweep_detector[key]) >= 3:  # 3+ strikes
            total_volume = sum(v for _, v in self.sweep_detector[key])
            if total_volume >= self.BLOCK_TRADE_SIZE * 2:  # Significant size
                return True
                
        return False
        
    def get_flow_summary(self, signals: List[OptionsFlowSignal]) -> Dict:
        """
        Generate summary of options flow.
        
        Args:
            signals: List of flow signals
            
        Returns:
            Summary statistics
        """
        if not signals:
            return {
                'total_signals': 0,
                'bullish_flow': 0,
                'bearish_flow': 0,
                'top_symbols': [],
                'total_premium': 0
            }
            
        bullish = [s for s in signals if s.option_type == 'CALL']
        bearish = [s for s in signals if s.option_type == 'PUT']
        
        # Group by symbol
        symbol_counts = {}
        for signal in signals:
            if signal.symbol not in symbol_counts:
                symbol_counts[signal.symbol] = {'count': 0, 'premium': 0}
            symbol_counts[signal.symbol]['count'] += 1
            symbol_counts[signal.symbol]['premium'] += signal.premium
            
        top_symbols = sorted(
            symbol_counts.items(),
            key=lambda x: x[1]['premium'],
            reverse=True
        )[:5]
        
        return {
            'total_signals': len(signals),
            'bullish_flow': len(bullish),
            'bearish_flow': len(bearish),
            'bullish_premium': sum(s.premium for s in bullish),
            'bearish_premium': sum(s.premium for s in bearish),
            'top_symbols': [
                {
                    'symbol': sym,
                    'signals': data['count'],
                    'premium': data['premium']
                }
                for sym, data in top_symbols
            ],
            'total_premium': sum(s.premium for s in signals),
            'avg_confidence': np.mean([s.confidence for s in signals]) if signals else 0
        }


async def main():
    """Test options flow analyzer."""
    from .config import load_config
    
    cfg = load_config()
    
    # Setup IB client with different ID to avoid conflicts
    ib_client = IBKRClient(cfg.ibkr_host, cfg.ibkr_port, 3)
    await ib_client.connect(readonly=True)
    
    # Create analyzer
    analyzer = OptionsFlowAnalyzer(ib_client)
    
    # Scan for unusual activity
    symbols = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA']
    logger.info(f"Scanning options flow for {symbols}...")
    
    signals = await analyzer.scan_options_flow(symbols)
    
    # Show summary
    summary = analyzer.get_flow_summary(signals)
    logger.info(f"Options Flow Summary:")
    logger.info(f"  Total signals: {summary['total_signals']}")
    if summary['total_signals'] > 0:
        logger.info(f"  Bullish: {summary['bullish_flow']} (${summary.get('bullish_premium', 0):,.0f})")
        logger.info(f"  Bearish: {summary['bearish_flow']} (${summary.get('bearish_premium', 0):,.0f})")
        logger.info(f"  Top symbols: {summary['top_symbols']}")
    else:
        logger.info("  No unusual options activity detected")
    
    # Show top signals
    for signal in signals[:5]:
        logger.info(f"\n{signal.interpretation}")
        logger.info(
            f"  {signal.symbol} {signal.option_type} {signal.strike} {signal.expiry}"
            f" | Volume: {signal.volume} | Premium: ${signal.premium:,.0f}"
        )
    
    ib_client.ib.disconnect()


if __name__ == "__main__":
    asyncio.run(main())