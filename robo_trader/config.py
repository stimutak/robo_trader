import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    ibkr_host: str
    ibkr_port: int
    ibkr_client_id: int
    trading_mode: str  # "paper" | "live"
    max_daily_loss: float
    max_position_risk_pct: float
    max_symbol_exposure_pct: float
    max_leverage: float
    default_cash: float
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
        ibkr_host=os.getenv("IBKR_HOST", "127.0.0.1"),
        ibkr_port=int(os.getenv("IBKR_PORT", "7497")),  # TWS paper default
        ibkr_client_id=int(os.getenv("IBKR_CLIENT_ID", "123")),
        trading_mode=os.getenv("TRADING_MODE", "paper"),
        max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "1000")),
        max_position_risk_pct=float(os.getenv("MAX_POSITION_RISK_PCT", "0.01")),
        max_symbol_exposure_pct=float(os.getenv("MAX_SYMBOL_EXPOSURE_PCT", "0.2")),
        max_leverage=float(os.getenv("MAX_LEVERAGE", "2.0")),
        default_cash=float(os.getenv("DEFAULT_CASH", "100000")),
        symbols=symbols,
    )


