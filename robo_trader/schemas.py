"""
Trading decision schemas for structured LLM output.
Enforces consistent JSON format from AI models.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import hashlib
import json


class TradingMode(str, Enum):
    """Valid trading decision modes."""
    TRADE = "trade"
    ADJUST = "adjust"
    EXIT = "exit"
    NEUTRAL = "neutral"
    WATCHLIST = "watchlist"


class EntryType(str, Enum):
    """Order entry types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class Direction(str, Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class ComplianceCheck(BaseModel):
    """Compliance validation status."""
    liquidity_ok: bool = Field(description="Meets minimum ADV requirements")
    spread_ok: bool = Field(description="Spread within acceptable range")
    borrow_ok: bool = Field(description="Shares available to borrow if short")
    correlation_ok: bool = Field(description="Within correlation bucket limits")


class RiskState(BaseModel):
    """Current risk metrics."""
    day_dd_bps: int = Field(description="Daily drawdown in basis points")
    week_dd_bps: int = Field(description="Weekly drawdown in basis points")
    cash_pct: float = Field(description="Cash percentage of portfolio")
    open_positions: int = Field(default=0, description="Number of open positions")
    total_exposure_pct: float = Field(default=0.0, description="Total exposure as % of equity")


class TradeRecommendation(BaseModel):
    """Detailed trade recommendation."""
    symbol: str = Field(description="Trading symbol")
    direction: Direction = Field(description="Trade direction")
    entry_type: EntryType = Field(description="Order type for entry")
    entry_price: float = Field(description="Entry price level")
    position_size_bps: int = Field(description="Position size in risk basis points (max 50)")
    shares: Optional[int] = Field(default=None, description="Number of shares/contracts")
    stop_loss: float = Field(description="Stop loss price")
    time_stop_hours: int = Field(description="Time stop in hours")
    targets: List[float] = Field(description="Take profit targets")
    thesis: str = Field(description="Trade setup and catalyst")
    risk_reward: float = Field(description="Risk:reward ratio")
    p_win: float = Field(description="Probability of win (0-1)")
    expected_value_pct: float = Field(description="Expected value as % of risk")
    
    @validator('position_size_bps')
    def validate_position_size(cls, v):
        if v > 50:  # Max 0.50% risk per trade
            raise ValueError("Position size exceeds 50 bps (0.50%) limit")
        return v
    
    @validator('risk_reward')
    def validate_risk_reward(cls, v):
        if v < 1.8:
            raise ValueError("Risk:reward must be >= 1.8:1")
        return v


class WatchlistItem(BaseModel):
    """Watchlist entry with trigger conditions."""
    symbol: str
    trigger: str = Field(description="Condition that would trigger action")
    target_conviction: int = Field(description="Expected conviction if triggered")
    notes: str = Field(default="", description="Additional context")


class CostEstimate(BaseModel):
    """Transaction cost estimates."""
    commission_bps: float = Field(default=0.5, description="Commission in basis points")
    slippage_bps: float = Field(default=2.0, description="Expected slippage in bps")
    total_bps: float = Field(default=2.5, description="Total cost in bps")


class TradingDecision(BaseModel):
    """
    Complete trading decision output from LLM.
    This is the ONLY valid output format for trading decisions.
    """
    mode: TradingMode = Field(description="Decision mode")
    timestamp_utc: datetime = Field(default_factory=datetime.utcnow)
    universe_checked: List[str] = Field(description="Symbols analyzed")
    
    # Trade details (required if mode=trade)
    recommendation: Optional[TradeRecommendation] = None
    
    # Risk and compliance
    conviction: int = Field(description="Conviction score 0-100")
    compliance_checks: ComplianceCheck
    risk_state: RiskState
    costs: CostEstimate = Field(default_factory=CostEstimate)
    
    # Watchlist and notes
    watchlist: List[WatchlistItem] = Field(default_factory=list)
    notes: str = Field(default="", description="Additional context or warnings")
    
    # Metadata
    model_id: str = Field(default="claude-3-5-sonnet-latest")
    aggressiveness_level: int = Field(default=1, ge=0, le=3)
    prompt_version: str = Field(default="v2_decisive")
    
    @validator('conviction')
    def validate_conviction(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Conviction must be between 0 and 100")
        return v
    
    @validator('recommendation')
    def validate_recommendation(cls, v, values):
        mode = values.get('mode')
        if mode == TradingMode.TRADE and v is None:
            raise ValueError("Trade mode requires a recommendation")
        return v
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.json(indent=2, default=str)
    
    def get_prompt_hash(self, prompt: str) -> str:
        """Generate hash of the prompt used."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def should_execute(self) -> bool:
        """Determine if this decision should result in execution."""
        if self.mode != TradingMode.TRADE:
            return False
        
        if self.conviction < 55:  # Base threshold
            return False
        
        if not self.compliance_checks.liquidity_ok:
            return False
        
        if not self.compliance_checks.correlation_ok:
            return False
        
        if self.recommendation and self.recommendation.expected_value_pct <= 0:
            return False
        
        return True
    
    def get_action_summary(self) -> str:
        """Get human-readable action summary."""
        if self.mode == TradingMode.NEUTRAL:
            return f"NEUTRAL: {self.notes}"
        elif self.mode == TradingMode.TRADE and self.recommendation:
            rec = self.recommendation
            return (f"{rec.direction.value.upper()} {rec.symbol} @ {rec.entry_price:.2f} "
                   f"(Stop: {rec.stop_loss:.2f}, Target: {rec.targets[0]:.2f}, "
                   f"Conviction: {self.conviction}%)")
        elif self.mode == TradingMode.WATCHLIST:
            return f"WATCHING: {len(self.watchlist)} symbols"
        else:
            return f"{self.mode.value.upper()}: {self.notes}"


class MarketData(BaseModel):
    """Market data input for LLM."""
    symbol: str
    price: float
    volume: int
    avg_volume: int
    atr: float = Field(description="Average True Range")
    adv: float = Field(description="Average Daily Volume in dollars")
    spread_pct: float = Field(description="Bid-ask spread as % of mid")
    shortable: bool = Field(default=True)
    borrow_rate: float = Field(default=0.0, description="Borrow rate if short")
    iv_rank: Optional[float] = Field(default=None, description="Implied volatility rank")
    
    # Technical indicators
    rsi: Optional[float] = None
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    
    # Fundamentals
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    earnings_date: Optional[str] = None


class OptionsFlow(BaseModel):
    """Options flow data."""
    symbol: str
    total_volume: int
    put_call_ratio: float
    unusual_activity: List[Dict[str, Any]] = Field(default_factory=list)
    net_premium: float = Field(description="Net premium flow")
    smart_money_sentiment: str = Field(default="neutral", description="bullish/bearish/neutral")


class NewsEvent(BaseModel):
    """News or event data."""
    headline: str
    summary: str
    source: str
    timestamp: datetime
    relevance_score: float = Field(ge=0, le=1)
    sentiment_score: float = Field(ge=-1, le=1)
    symbols: List[str] = Field(default_factory=list)
    event_type: str = Field(default="news", description="news/earnings/filing/fed")


@dataclass
class DecisionMetadata:
    """Metadata to store with each decision."""
    decision_id: str
    prompt_hash: str
    model_id: str
    prompt_version: str
    timestamp: datetime
    latency_ms: int
    raw_response: str
    parsed_decision: TradingDecision
    market_snapshot: Dict[str, Any]
    news_context: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "decision_id": self.decision_id,
            "prompt_hash": self.prompt_hash,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "raw_response": self.raw_response,
            "parsed_decision": json.loads(self.parsed_decision.to_json()),
            "market_snapshot": self.market_snapshot,
            "news_context": self.news_context
        }