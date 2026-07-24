"""RoboTrader package metadata.

The package initializer must remain side-effect free. Broker-specific safety
guards are activated by the broker modules that explicitly import them, not by
unrelated imports such as ``robo_trader.safety``.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
