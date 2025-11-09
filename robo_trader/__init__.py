__all__ = [
    "__version__",
]

__version__ = "0.1.0"

# Importing ibkr_safe applies the disconnect monkey patch globally.
from .utils import ibkr_safe as _ibkr_safe  # noqa: F401
