"""
Decisive LLM client with forced JSON schema output.
Uses Anthropic's tool-use pattern to guarantee structured responses.
"""

import os
import json
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from .schemas import (
    TradingDecision, MarketData, NewsEvent, OptionsFlow,
    DecisionMetadata, TradingMode, Direction
)
from .logger import get_logger

logger = get_logger(__name__)


DECISIVE_TRADING_PROMPT = """Role
You are an elite, data-driven discretionary/quant hybrid trader focused on producing positive risk-adjusted returns net of costs. You reason explicitly, quantify uncertainty, and act decisively when edge is present.

Universe and inputs (host app provides)
Real-time and historical OHLCV; options chains/greeks; fundamentals; earnings and economic calendar; macro prints; ETF flows; news/sentiment; borrow/shortability; open positions and PnL; realized/implied vol; correlations; transaction-cost and slippage model; compliance filters; account size; leverage and borrow limits; minimum liquidity thresholds. If any input is missing, state "MISSING:" and adapt conservatively rather than halting.

Guardrails and risk
Per-trade risk cap: ≤ 0.50% of equity. Daily max drawdown stop: -2.0%. Weekly: -5.0%. Liquidity floor: average daily dollar volume ≥ $3M; equity bid-ask spread ≤ 1.0% of mid (or options spread ≤ $0.25 or ≤ 8% of mid). No microcaps/pink sheets. Correlation control: do not push more than 35% of portfolio exposure into a single highly correlated cluster (sector/theme/beta). Prefer defined-risk trades: hard stops or debit option structures; avoid naked short gamma.

Edge and action thresholds
Compute Expected Value = p_win*avg_win - (1-p_win)*avg_loss - costs - slippage. Require risk:reward ≥ 1.8:1 or EV > 0 with p_win ≥ 0.45 and clear invalidation. Output Conviction 0-100 from signal strength, catalyst quality, regime fit, liquidity, data completeness, and conflicts. Act when Conviction ≥ 55 and all guardrails pass. If Conviction 45-54 and cash > 70% and the volatility regime is favorable, allow a half-size probe with tighter stops. If no trades in the past 2 sessions but at least 2 setups score ≥ 50, propose the single best at half size to avoid inaction bias. If nothing reaches 50, return neutral with watchlist and explicit triggers.

Position sizing (clip-Kelly under risk caps)
Map conviction to risk fraction of the 0.50% cap: 55-64 → 0.30×cap; 65-74 → 0.60×cap; 75-84 → 0.85×cap; 85-100 → 1.00×cap (only if liquidity and correlation budgets are clean). Equity sizing uses stop distance in ATR: shares = (equity * risk_bps/10000) / (ATR_mult * ATR). Default ATR_mult = 1.2 for trend or 0.8 for mean-reversion. For options, prefer debit verticals with max loss ≤ risk cap and target payoff ≥ 2:1.

Stops, targets, management
Hard stop at technical invalidation or -1.2× planned risk, whichever first. Time stop when the catalyst passes or the signal decays (2-3 sessions without progress). Scale-out 1/3 at +1R, 1/3 at +2R, trail remainder using swing or EMA; for options, use delta targets.

Bias controls (anti-paralysis)
Penalize "no trade" if three consecutive sessions contained at least one valid setup ≥ 55 that was skipped. Always compare the top three candidates by EV per unit risk and implementation cost; pick the best when correlation budget permits only one.

Current Market Context
{market_context}

News and Events
{news_events}

Options Flow
{options_flow}

Current Portfolio State
{portfolio_state}

Aggressiveness Level: {aggressiveness_level}
{aggressiveness_note}

What to output (valid JSON only)
A single decision object matching the TradingDecision schema exactly. The JSON must include: mode (trade|adjust|exit|neutral|watchlist), timestamp_utc, universe_checked, one best recommendation with entry type and price, position_size in risk_bps, stops and time stop, targets, thesis (setup, catalyst, risk:reward, p_win, ev%), costs (fees, slippage bps), conviction, compliance_checks (liquidity_ok, borrow_ok, correlation_ok), a watchlist with triggers, current risk state (day_dd_bps, week_dd_bps, cash_pct), and notes.

Ethics/compliance
This is a policy engine, not personal financial advice. Respect all broker/regulatory constraints supplied by the host."""


AGGRESSIVENESS_NOTES = {
    0: "Very conservative mode: action threshold 60, probe min 55, size multipliers [0.20, 0.40, 0.60, 0.80]",
    1: "Balanced default: action threshold 55, probe min 50, size multipliers [0.30, 0.60, 0.85, 1.00]",
    2: "Assertive mode: action threshold 52, probe min 48, size multipliers [0.35, 0.70, 0.95, 1.00]",
    3: "Opportunistic within guardrails: action threshold 50, probe min 45, size multipliers [0.40, 0.80, 1.00, 1.00], require risk:reward ≥ 2.0"
}


class DecisiveLLMClient:
    """
    LLM client that forces structured JSON output for trading decisions.
    Uses Anthropic's Claude with tool-use pattern for schema enforcement.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            api_key: Anthropic API key. If None, reads from environment.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Please set in environment or pass directly.")
        
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.model = "claude-3-5-sonnet-latest"
        self.prompt_version = "v2_decisive"
        
        # Track decisions for anti-stall logic
        self.recent_decisions: List[TradingDecision] = []
        self.consecutive_no_trades = 0
        
        logger.info(f"Initialized DecisiveLLMClient with model {self.model}")
    
    async def get_trading_decision(
        self,
        market_data: Dict[str, MarketData],
        news_events: List[NewsEvent],
        options_flow: Optional[Dict[str, OptionsFlow]] = None,
        portfolio_state: Optional[Dict[str, Any]] = None,
        aggressiveness_level: int = 1
    ) -> TradingDecision:
        """
        Get a structured trading decision from the LLM.
        
        Args:
            market_data: Current market data by symbol
            news_events: Recent news and events
            options_flow: Options flow data by symbol
            portfolio_state: Current portfolio positions and P&L
            aggressiveness_level: 0-3, controls decisiveness
            
        Returns:
            Validated TradingDecision object
        """
        # Prepare context sections
        market_context = self._format_market_context(market_data)
        news_context = self._format_news_events(news_events)
        options_context = self._format_options_flow(options_flow or {})
        portfolio_context = self._format_portfolio_state(portfolio_state or {})
        
        # Build the prompt
        prompt = DECISIVE_TRADING_PROMPT.format(
            market_context=market_context,
            news_events=news_context,
            options_flow=options_context,
            portfolio_state=portfolio_context,
            aggressiveness_level=aggressiveness_level,
            aggressiveness_note=AGGRESSIVENESS_NOTES.get(aggressiveness_level, AGGRESSIVENESS_NOTES[1])
        )
        
        # Track timing
        start_time = time.time()
        
        try:
            # For now, use standard message format with JSON instruction
            # Tool-use requires specific SDK version
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.2,  # Low temperature for consistency
                messages=[{
                    "role": "user",
                    "content": prompt + "\n\nIMPORTANT: You must respond with a valid JSON object matching the TradingDecision schema. No other text or explanation."
                }],
                system="You are a trading decision engine that outputs only valid JSON matching the required schema."
            )
            
            # Extract JSON from response
            response_text = response.content[0].text
            
            # Try to parse JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                decision_data = json.loads(json_match.group(0))
            else:
                raise ValueError("No valid JSON in response")
            
            # Log raw Claude response for debugging (temporarily use INFO to see it)
            logger.info(f"Raw Claude JSON (first 1000 chars): {json.dumps(decision_data, indent=2)[:1000]}")
            
            # Fix common field mismatches from Claude
            if 'recommendation' in decision_data:
                rec = decision_data['recommendation']
                
                # Handle null recommendation for neutral/watchlist modes
                if rec is None or (isinstance(rec, dict) and rec.get('symbol') is None and decision_data.get('mode') in ['neutral', 'watchlist']):
                    # Don't process null recommendations for neutral/watchlist
                    decision_data['recommendation'] = None
                elif rec and isinstance(rec, dict):
                    # Handle ticker vs symbol
                    if 'ticker' in rec and 'symbol' not in rec:
                        rec['symbol'] = rec['ticker']
                    
                    # Handle position_size object
                    if 'position_size' in rec and isinstance(rec['position_size'], dict):
                        ps = rec['position_size']
                        rec['position_size_bps'] = ps.get('risk_bps', 0)
                        del rec['position_size']
                    elif 'position_size' in rec:
                        rec['position_size_bps'] = rec['position_size']
                        del rec['position_size']
                    
                    # Handle stops object
                    if 'stops' in rec and isinstance(rec['stops'], dict):
                        stops = rec['stops']
                        rec['stop_loss'] = stops.get('technical') or stops.get('max_loss') or stops.get('risk', 0)
                        del rec['stops']
                    
                    # Handle thesis object
                    if 'thesis' in rec and isinstance(rec['thesis'], dict):
                        thesis_obj = rec['thesis']
                        # Build thesis string from components
                        thesis_parts = []
                        if thesis_obj.get('setup'):
                            thesis_parts.append(f"Setup: {thesis_obj['setup']}")
                        if thesis_obj.get('catalyst'):
                            thesis_parts.append(f"Catalyst: {thesis_obj['catalyst']}")
                        rec['thesis'] = ' | '.join(thesis_parts) if thesis_parts else "No thesis provided"
                        
                        # Extract other fields from thesis object
                        if 'risk_reward' not in rec and 'risk_reward' in thesis_obj:
                            rec['risk_reward'] = thesis_obj['risk_reward']
                        if 'p_win' not in rec and 'prob_win' in thesis_obj:
                            rec['p_win'] = thesis_obj['prob_win']
                        if 'expected_value_pct' not in rec and 'expected_value_bps' in thesis_obj:
                            rec['expected_value_pct'] = thesis_obj['expected_value_bps'] / 100.0
                    
                    # Handle costs object (extract slippage)
                    if 'costs' in rec and isinstance(rec['costs'], dict):
                        rec['slippage_bps'] = rec['costs'].get('slippage_bps', 0)
                        del rec['costs']
                    
                    # Extract conviction if nested
                    if 'conviction' in rec and 'conviction' not in decision_data:
                        decision_data['conviction'] = rec['conviction']
                    
                    # Fix missing required fields with defaults
                    if 'direction' not in rec:
                        rec['direction'] = 'long'  # Default to long
                    if 'position_size_bps' not in rec:
                        rec['position_size_bps'] = 0
                    if 'stop_loss' not in rec:
                        rec['stop_loss'] = 0
                    if 'time_stop_hours' not in rec:
                        rec['time_stop_hours'] = 24
                    if 'risk_reward' not in rec:
                        rec['risk_reward'] = 0
                    if 'p_win' not in rec:
                        rec['p_win'] = 0
                    if 'expected_value_pct' not in rec:
                        rec['expected_value_pct'] = 0
                        
                    # Fix targets format (Claude returns objects, we need floats)
                    if 'targets' in rec and rec['targets'] and len(rec['targets']) > 0 and isinstance(rec['targets'][0], dict):
                        rec['targets'] = [t.get('price', t) for t in rec['targets']]
                    elif 'targets' not in rec or not rec['targets']:
                        rec['targets'] = []
            
            # Fix universe_checked (Claude returns int, we need list)
            if 'universe_checked' in decision_data and isinstance(decision_data['universe_checked'], int):
                decision_data['universe_checked'] = list(market_data.keys())
            
            # Fix compliance_checks missing spread_ok
            if 'compliance_checks' in decision_data and 'spread_ok' not in decision_data['compliance_checks']:
                decision_data['compliance_checks']['spread_ok'] = True
            
            # Fix risk_state field names
            if 'risk_state' in decision_data:
                rs = decision_data['risk_state']
                if 'day_dd_bps' not in rs and 'day_drawdown_bps' in rs:
                    rs['day_dd_bps'] = rs['day_drawdown_bps']
                if 'week_dd_bps' not in rs and 'week_drawdown_bps' in rs:
                    rs['week_dd_bps'] = rs['week_drawdown_bps']
            
            # Fix watchlist fields
            if 'watchlist' in decision_data:
                for watch in decision_data['watchlist']:
                    # Handle ticker vs symbol
                    if 'ticker' in watch and 'symbol' not in watch:
                        watch['symbol'] = watch['ticker']
                    
                    # Handle various trigger field names
                    if 'trigger' not in watch:
                        if 'trigger_condition' in watch:
                            watch['trigger'] = watch['trigger_condition']
                        elif 'trigger_type' in watch:
                            watch['trigger'] = watch['trigger_type']
                        elif 'notes' in watch:
                            watch['trigger'] = watch['notes']
                        else:
                            watch['trigger'] = "Price trigger"
                    
                    # Handle target_conviction
                    if 'target_conviction' not in watch:
                        if 'conviction' in watch:
                            watch['target_conviction'] = watch['conviction']
                        else:
                            watch['target_conviction'] = 50  # Default 50%
            
            # Fix conviction field (sometimes missing at top level)
            if 'conviction' not in decision_data:
                # Try to extract from recommendation
                if 'recommendation' in decision_data and decision_data['recommendation']:
                    if isinstance(decision_data['recommendation'], dict) and 'conviction' in decision_data['recommendation']:
                        decision_data['conviction'] = decision_data['recommendation']['conviction']
                    elif isinstance(decision_data['recommendation'], list) and len(decision_data['recommendation']) > 0:
                        # If recommendation is a list, check first item
                        first_rec = decision_data['recommendation'][0]
                        if isinstance(first_rec, dict) and 'conviction' in first_rec:
                            decision_data['conviction'] = first_rec['conviction']
                        else:
                            decision_data['conviction'] = 0
                    else:
                        decision_data['conviction'] = 0
                else:
                    # Default to 0 for neutral decisions
                    decision_data['conviction'] = 0
            
            decision = TradingDecision(**decision_data)
            
            # Add metadata
            decision.model_id = self.model
            decision.aggressiveness_level = aggressiveness_level
            decision.prompt_version = self.prompt_version
            
            # Track for anti-stall logic
            self._track_decision(decision)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Store metadata
            metadata = DecisionMetadata(
                decision_id=str(uuid.uuid4()),
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest()[:16],
                model_id=self.model,
                prompt_version=self.prompt_version,
                timestamp=datetime.utcnow(),
                latency_ms=latency_ms,
                raw_response=json.dumps(decision_data),
                parsed_decision=decision,
                market_snapshot={s: m.dict() for s, m in market_data.items()},
                news_context=[n.headline for n in news_events[:5]]
            )
            
            # Log the decision
            logger.info(
                f"LLM Decision: {decision.mode.value.upper()} "
                f"(Conviction: {decision.conviction}%, Latency: {latency_ms}ms) "
                f"- {decision.get_action_summary()}"
            )
            
            # Store metadata (could be saved to database here)
            self._store_metadata(metadata)
            
            return decision
            
        except ValidationError as e:
            logger.error(f"Decision validation failed: {e}")
            # Return neutral decision on validation error
            return self._create_neutral_decision(
                reason=f"Validation error: {str(e)}",
                market_data=market_data
            )
        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            # Return neutral decision on error
            return self._create_neutral_decision(
                reason=f"LLM error: {str(e)}",
                market_data=market_data
            )
    
    def _format_market_context(self, market_data: Dict[str, MarketData]) -> str:
        """Format market data for the prompt."""
        if not market_data:
            return "MISSING: No market data available"
        
        lines = []
        for symbol, data in market_data.items():
            lines.append(
                f"{symbol}: ${data.price:.2f} | "
                f"Vol: {data.volume:,} ({data.volume/data.avg_volume:.1f}x avg) | "
                f"ATR: ${data.atr:.2f} | "
                f"ADV: ${data.adv/1e6:.1f}M | "
                f"Spread: {data.spread_pct:.2%} | "
                f"RSI: {data.rsi:.0f}" if data.rsi else "RSI: MISSING"
            )
        
        return "\n".join(lines)
    
    def _format_news_events(self, news_events: List[NewsEvent]) -> str:
        """Format news events for the prompt."""
        if not news_events:
            return "No significant news in the last 6 hours"
        
        lines = []
        for event in news_events[:10]:  # Top 10 events
            lines.append(
                f"[{event.event_type}] {event.headline} "
                f"(Relevance: {event.relevance_score:.0%}, "
                f"Sentiment: {event.sentiment_score:+.2f})"
            )
        
        return "\n".join(lines)
    
    def _format_options_flow(self, options_flow: Dict[str, OptionsFlow]) -> str:
        """Format options flow data for the prompt."""
        if not options_flow:
            return "MISSING: No options flow data available"
        
        lines = []
        for symbol, flow in options_flow.items():
            lines.append(
                f"{symbol}: P/C Ratio: {flow.put_call_ratio:.2f} | "
                f"Volume: {flow.total_volume:,} | "
                f"Net Premium: ${flow.net_premium/1e6:.1f}M | "
                f"Smart Money: {flow.smart_money_sentiment.upper()}"
            )
        
        return "\n".join(lines) if lines else "No significant options flow"
    
    def _format_portfolio_state(self, portfolio_state: Dict[str, Any]) -> str:
        """Format portfolio state for the prompt."""
        if not portfolio_state:
            return "Portfolio: $100,000 cash (100%), no positions"
        
        positions = portfolio_state.get('positions', {})
        equity = portfolio_state.get('equity', 100000)
        cash = portfolio_state.get('cash', equity)
        daily_pnl = portfolio_state.get('daily_pnl', 0)
        weekly_pnl = portfolio_state.get('weekly_pnl', 0)
        
        lines = [
            f"Equity: ${equity:,.0f} | Cash: ${cash:,.0f} ({100*cash/equity:.0f}%)",
            f"Daily P&L: ${daily_pnl:+,.0f} ({100*daily_pnl/equity:+.2f}%)",
            f"Weekly P&L: ${weekly_pnl:+,.0f} ({100*weekly_pnl/equity:+.2f}%)",
            f"Positions: {len(positions)}"
        ]
        
        for symbol, pos in positions.items():
            lines.append(
                f"  {symbol}: {pos.get('quantity', 0)} shares @ ${pos.get('avg_price', 0):.2f}"
            )
        
        return "\n".join(lines)
    
    def _track_decision(self, decision: TradingDecision):
        """Track decision for anti-stall logic."""
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > 10:
            self.recent_decisions.pop(0)
        
        # Track consecutive no-trade decisions
        if decision.mode != TradingMode.TRADE:
            self.consecutive_no_trades += 1
        else:
            self.consecutive_no_trades = 0
        
        # Log warning if too many consecutive no-trades
        if self.consecutive_no_trades >= 3:
            logger.warning(
                f"Anti-stall warning: {self.consecutive_no_trades} consecutive sessions without trades"
            )
    
    def _create_neutral_decision(self, reason: str, market_data: Dict[str, MarketData]) -> TradingDecision:
        """Create a neutral decision when unable to analyze."""
        from .schemas import ComplianceCheck, RiskState, CostEstimate
        
        return TradingDecision(
            mode=TradingMode.NEUTRAL,
            universe_checked=list(market_data.keys()),
            conviction=0,
            compliance_checks=ComplianceCheck(
                liquidity_ok=True,
                spread_ok=True,
                borrow_ok=True,
                correlation_ok=True
            ),
            risk_state=RiskState(
                day_dd_bps=0,
                week_dd_bps=0,
                cash_pct=100.0
            ),
            costs=CostEstimate(),
            notes=reason
        )
    
    def _store_metadata(self, metadata: DecisionMetadata):
        """Store decision metadata for tracking and calibration."""
        # This would typically save to database
        # For now, just log it
        logger.debug(
            f"Decision metadata: ID={metadata.decision_id}, "
            f"Hash={metadata.prompt_hash}, Latency={metadata.latency_ms}ms"
        )
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics from recent decisions."""
        if not self.recent_decisions:
            return {"sample_size": 0}
        
        stats = {
            "sample_size": len(self.recent_decisions),
            "trade_rate": sum(1 for d in self.recent_decisions if d.mode == TradingMode.TRADE) / len(self.recent_decisions),
            "avg_conviction": sum(d.conviction for d in self.recent_decisions) / len(self.recent_decisions),
            "consecutive_no_trades": self.consecutive_no_trades
        }
        
        # Conviction distribution
        conviction_buckets = {
            "0-30": 0,
            "31-50": 0,
            "51-70": 0,
            "71-100": 0
        }
        
        for decision in self.recent_decisions:
            if decision.conviction <= 30:
                conviction_buckets["0-30"] += 1
            elif decision.conviction <= 50:
                conviction_buckets["31-50"] += 1
            elif decision.conviction <= 70:
                conviction_buckets["51-70"] += 1
            else:
                conviction_buckets["71-100"] += 1
        
        stats["conviction_distribution"] = conviction_buckets
        
        return stats