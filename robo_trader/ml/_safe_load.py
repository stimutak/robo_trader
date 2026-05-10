"""HMAC-based integrity verification for serialized model artifacts.

The signing key is read from MODEL_SIGNING_KEY env var. If the env var is
unset, signing/verification are skipped (with a warning) unless
MODEL_SIGNING_REQUIRED=true, in which case a missing key raises.
"""

import hashlib
import hmac
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_key() -> bytes | None:
    key = os.environ.get("MODEL_SIGNING_KEY")
    if not key:
        if os.environ.get("MODEL_SIGNING_REQUIRED", "false").lower() == "true":
            raise RuntimeError(
                "MODEL_SIGNING_REQUIRED=true but MODEL_SIGNING_KEY is unset. "
                "Set MODEL_SIGNING_KEY to a 32+ char random string."
            )
        return None
    return key.encode("utf-8")


def sign_file(path: str | Path) -> Path | None:
    """Compute HMAC-SHA256 over file contents, write to <path>.sig.

    Returns the signature path, or None if signing is disabled.
    """
    key = _get_key()
    if key is None:
        return None
    p = Path(path)
    digest = hmac.new(key, p.read_bytes(), hashlib.sha256).hexdigest()
    sig_path = p.with_suffix(p.suffix + ".sig")
    sig_path.write_text(digest)
    return sig_path


def verify_file(path: str | Path) -> None:
    """Raise ValueError if the signature is invalid.

    If MODEL_SIGNING_KEY is unset and MODEL_SIGNING_REQUIRED is false,
    log a warning and pass.
    """
    key = _get_key()
    p = Path(path)
    sig_path = p.with_suffix(p.suffix + ".sig")
    if key is None:
        if os.environ.get("MODEL_SIGNING_REQUIRED", "false").lower() == "true":
            raise RuntimeError("MODEL_SIGNING_REQUIRED=true but no key available")
        logger.warning(
            "Loading %s without HMAC verification (MODEL_SIGNING_KEY unset).",
            p,
        )
        return
    if not sig_path.exists():
        if os.environ.get("MODEL_SIGNING_REQUIRED", "false").lower() == "true":
            raise ValueError(f"Missing signature file: {sig_path}")
        logger.warning("No .sig for %s; skipping verification (signing not required).", p)
        return
    expected = sig_path.read_text().strip()
    actual = hmac.new(key, p.read_bytes(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, actual):
        raise ValueError(f"HMAC mismatch for {p}: refusing to load")
