"""
Per-portfolio configuration for multiuser support.

Each portfolio has its own:
- Symbol watchlist
- Starting cash / current cash
- Risk parameters (overrides global defaults)
- Strategy settings (overrides global defaults)

If no PORTFOLIOS env var is set, a single 'default' portfolio is created
from the existing SYMBOLS and DEFAULT_CASH env vars for backward compatibility.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioConfig:
    """Configuration for a single portfolio."""

    id: str
    name: str
    starting_cash: float = 100_000.0
    symbols: List[str] = field(default_factory=list)
    active: bool = True

    # Risk overrides (None = use global default from Config.risk)
    max_position_pct: Optional[float] = None
    max_daily_loss_pct: Optional[float] = None
    max_open_positions: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    use_trailing_stop: Optional[bool] = None

    # Strategy overrides (None = use global default from Config.strategy)
    enabled_strategies: Optional[List[str]] = None
    min_confidence: Optional[float] = None

    def get_risk_param(self, param_name: str, global_default):
        """Get a risk parameter, falling back to global default if not overridden."""
        local_value = getattr(self, param_name, None)
        if local_value is not None:
            return local_value
        return global_default

    def to_dict(self) -> Dict:
        """Serialize to dict for database storage."""
        return {
            "id": self.id,
            "name": self.name,
            "starting_cash": self.starting_cash,
            "symbols": ",".join(self.symbols),
            "active": self.active,
            "max_position_pct": self.max_position_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_open_positions": self.max_open_positions,
            "stop_loss_pct": self.stop_loss_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "use_trailing_stop": self.use_trailing_stop,
            "enabled_strategies": (
                ",".join(self.enabled_strategies) if self.enabled_strategies else None
            ),
            "min_confidence": self.min_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PortfolioConfig":
        """Deserialize from dict (database row or JSON)."""
        symbols = data.get("symbols", "")
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]

        strategies = data.get("enabled_strategies")
        if isinstance(strategies, str) and strategies:
            strategies = [s.strip() for s in strategies.split(",") if s.strip()]
        elif not strategies:
            strategies = None

        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            starting_cash=float(data.get("starting_cash", 100_000)),
            symbols=symbols,
            active=bool(data.get("active", True)),
            max_position_pct=data.get("max_position_pct"),
            max_daily_loss_pct=data.get("max_daily_loss_pct"),
            max_open_positions=data.get("max_open_positions"),
            stop_loss_pct=data.get("stop_loss_pct"),
            trailing_stop_pct=data.get("trailing_stop_pct"),
            use_trailing_stop=data.get("use_trailing_stop"),
            enabled_strategies=strategies,
            min_confidence=data.get("min_confidence"),
        )


def load_portfolio_configs() -> List[PortfolioConfig]:
    """Load portfolio configurations from environment.

    Reads PORTFOLIOS env var (JSON array of portfolio objects).
    Falls back to creating a single 'default' portfolio from
    SYMBOLS and DEFAULT_CASH env vars for backward compatibility.

    Returns:
        List of PortfolioConfig objects
    """
    portfolios_json = os.getenv("PORTFOLIOS")

    if portfolios_json:
        try:
            raw_list = json.loads(portfolios_json)
            if not isinstance(raw_list, list) or len(raw_list) == 0:
                raise ValueError("PORTFOLIOS must be a non-empty JSON array")

            configs = []
            seen_ids = set()
            for item in raw_list:
                if not isinstance(item, dict):
                    raise ValueError(f"Each portfolio must be a JSON object, got: {type(item)}")
                if "id" not in item:
                    raise ValueError(f"Each portfolio must have an 'id' field: {item}")
                if item["id"] in seen_ids:
                    raise ValueError(f"Duplicate portfolio id: {item['id']}")
                seen_ids.add(item["id"])
                configs.append(PortfolioConfig.from_dict(item))

            active = [c for c in configs if c.active]
            logger.info(
                f"Loaded {len(configs)} portfolio configs ({len(active)} active): "
                f"{[c.id for c in configs]}"
            )
            return configs

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse PORTFOLIOS env var: {e}")
            raise ValueError(f"Invalid PORTFOLIOS JSON: {e}") from e

    # Backward compatibility: create single 'default' portfolio from existing env vars
    symbols_str = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    default_cash = float(os.getenv("DEFAULT_CASH", "100000"))

    logger.info(
        f"No PORTFOLIOS env var found. Creating default portfolio: "
        f"cash=${default_cash:,.0f}, symbols={symbols}"
    )

    return [
        PortfolioConfig(
            id="default",
            name="Default Portfolio",
            starting_cash=default_cash,
            symbols=symbols,
            active=True,
        )
    ]
