import logging
import os
from typing import Optional


_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    log_format = os.getenv("LOG_FORMAT", "text").lower()

    handler = logging.StreamHandler()
    if log_format == "json":
        # Simple JSON-like structure using default formatter
        fmt = "{" + "level=%(levelname)s, time=%(asctime)s, name=%(name)s, msg=%(message)s" + "}"
        formatter = logging.Formatter(fmt)
    else:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger configured once per process.

    Respects LOG_LEVEL and LOG_FORMAT env variables.
    """
    _configure_root_logger()
    return logging.getLogger(name)


