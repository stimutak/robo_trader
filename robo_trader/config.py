import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    # IBKR Connection
    ibkr_host: str
    ibkr_port: int
    ibkr_client_id: int
    trading_mode: str  # "paper" | "live"
    
    # Risk Management - Core
    max_daily_loss: float
    max_position_risk_pct: float
    max_symbol_exposure_pct: float
    max_leverage: float
    default_cash: float
    
    # Risk Management - Enhanced
    per_trade_risk_bps: int  # Risk per trade in basis points (50 = 0.50%)
    max_weekly_loss_pct: float  # Weekly drawdown limit
    
    # Liquidity Requirements
    min_adv: float  # Minimum Average Daily Volume in dollars
    max_spread_pct: float  # Maximum bid-ask spread as percentage
    
    # Correlation Control
    max_bucket_exposure_pct: float  # Max exposure per correlation bucket
    
    # Edge Requirements
    min_ev_pct: float  # Minimum expected value percentage
    min_risk_reward: float  # Minimum risk:reward ratio
    min_p_win: float  # Minimum probability of win
    
    # LLM Settings
    aggressiveness_level: int  # 0-3, controls decisiveness
    conviction_threshold: int  # Minimum conviction to trade (55-60)
    
    # Execution
    max_order_notional: float  # Maximum order size in dollars
    max_daily_notional: float  # Maximum daily trading volume
    
    # Live Trading Safety
    live_allowed: bool  # Environment flag for live trading
    require_live_confirm: bool  # Require CLI confirmation for live
    
    # Symbols
    symbols: List[str]


def load_config() -> Config:
    """Load configuration from environment variables (.env supported).

    Returns:
        Config: Fully populated configuration dataclass.
    """
    load_dotenv()

    symbols_env = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").strip()
    symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]

    return Config(
        # IBKR Connection
        ibkr_host=os.getenv("IBKR_HOST", "127.0.0.1"),
        ibkr_port=int(os.getenv("IBKR_PORT", "7497")),  # TWS paper default
        ibkr_client_id=int(os.getenv("IBKR_CLIENT_ID", "123")),
        trading_mode=os.getenv("TRADING_MODE", "paper"),
        
        # Risk Management - Core (backwards compatible)
        max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "1000")),
        max_position_risk_pct=float(os.getenv("MAX_POSITION_RISK_PCT", "0.01")),
        max_symbol_exposure_pct=float(os.getenv("MAX_SYMBOL_EXPOSURE_PCT", "0.2")),
        max_leverage=float(os.getenv("MAX_LEVERAGE", "2.0")),
        default_cash=float(os.getenv("DEFAULT_CASH", "100000")),
        
        # Risk Management - Enhanced
        per_trade_risk_bps=int(os.getenv("PER_TRADE_RISK_BPS", "50")),  # 0.50% default
        max_weekly_loss_pct=float(os.getenv("MAX_WEEKLY_LOSS_PCT", "0.05")),  # 5% weekly
        
        # Liquidity Requirements
        min_adv=float(os.getenv("MIN_ADV", "3000000")),  # $3M minimum
        max_spread_pct=float(os.getenv("MAX_SPREAD_PCT", "0.01")),  # 1% max spread
        
        # Correlation Control
        max_bucket_exposure_pct=float(os.getenv("MAX_BUCKET_EXPOSURE_PCT", "0.35")),  # 35% max
        
        # Edge Requirements
        min_ev_pct=float(os.getenv("MIN_EV_PCT", "0")),  # 0% minimum EV
        min_risk_reward=float(os.getenv("MIN_RISK_REWARD", "1.8")),  # 1.8:1 minimum
        min_p_win=float(os.getenv("MIN_P_WIN", "0.45")),  # 45% minimum win probability
        
        # LLM Settings
        aggressiveness_level=int(os.getenv("AGGRESSIVENESS_LEVEL", "1")),  # Balanced default
        conviction_threshold=int(os.getenv("CONVICTION_THRESHOLD", "55")),  # 55% minimum
        
        # Execution
        max_order_notional=float(os.getenv("MAX_ORDER_NOTIONAL", "50000")),  # $50k max order
        max_daily_notional=float(os.getenv("MAX_DAILY_NOTIONAL", "500000")),  # $500k daily max
        
        # Live Trading Safety
        live_allowed=os.getenv("LIVE_ALLOWED", "false").lower() == "true",
        require_live_confirm=os.getenv("REQUIRE_LIVE_CONFIRM", "true").lower() == "true",
        
        # Symbols
        symbols=symbols,
    )


