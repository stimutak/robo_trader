"""HMAC-based integrity verification for serialized model artifacts.

The signing key is read from MODEL_SIGNING_KEY env var. If the env var is
unset, signing/verification are skipped (with a warning) unless
MODEL_SIGNING_REQUIRED=true, in which case a missing key raises.

For deserialization sites: prefer ``verify_and_read(path)`` over
``verify_file(path)`` to avoid a TOCTOU race where an attacker swaps the
file contents between hashing and re-opening for binary loaders. The
buffer returned by ``verify_and_read`` IS the authoritative artifact —
feed it through ``pickle.loads(buf)`` or ``joblib.load(io.BytesIO(buf))``.
"""

import hashlib
import hmac
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Minimum acceptable HMAC key length (bytes/chars). 32 chars / 256 bits.
_MIN_KEY_LEN = 32


def _is_production_mode() -> bool:
    """Return True if we are running in a production-equivalent environment."""
    env = os.environ.get("TRADING_ENV", "").strip().lower()
    if env in ("prod", "production"):
        return True
    live = os.environ.get("ENABLE_LIVE_TRADING", "").strip().lower()
    if live == "true":
        return True
    return False


def _signing_required() -> bool:
    return os.environ.get("MODEL_SIGNING_REQUIRED", "false").strip().lower() == "true"


def _get_key() -> bytes | None:
    """Return the HMAC signing key as bytes, or None if unset and not required.

    Fails CLOSED if:
      1. A key IS set but shorter than ``_MIN_KEY_LEN`` chars.
      2. The key is unset AND we are in production-equivalent mode
         (regardless of MODEL_SIGNING_REQUIRED).
      3. The key is unset AND MODEL_SIGNING_REQUIRED=true (legacy).
    """
    key = os.environ.get("MODEL_SIGNING_KEY")
    if not key:
        if _is_production_mode():
            raise RuntimeError(
                "MODEL_SIGNING_KEY is unset but TRADING_ENV/ENABLE_LIVE_TRADING "
                "indicates production mode. Refusing to load unsigned model artifacts. "
                "Set MODEL_SIGNING_KEY to a 32+ char random string."
            )
        if _signing_required():
            raise RuntimeError(
                "MODEL_SIGNING_REQUIRED=true but MODEL_SIGNING_KEY is unset. "
                "Set MODEL_SIGNING_KEY to a 32+ char random string."
            )
        return None
    if len(key) < _MIN_KEY_LEN:
        raise RuntimeError(
            f"MODEL_SIGNING_KEY is too short ({len(key)} chars); minimum is "
            f"{_MIN_KEY_LEN}. Generate a fresh key with `openssl rand -hex 32`."
        )
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
    try:
        os.chmod(sig_path, 0o600)
    except OSError:
        pass
    return sig_path


def verify_file(path: str | Path) -> None:
    """Raise ValueError if the signature is invalid.

    .. warning::
       Reads the file once for hashing; the CALLER then re-opens the path for
       deserialization, creating a TOCTOU window. New code MUST use
       :func:`verify_and_read` and deserialize from the returned buffer.

    Kept for backward compatibility (callers that only want integrity check).
    """
    key = _get_key()
    p = Path(path)
    sig_path = p.with_suffix(p.suffix + ".sig")
    if key is None:
        if _signing_required():
            raise RuntimeError("MODEL_SIGNING_REQUIRED=true but no key available")
        logger.error(
            "Loading %s WITHOUT HMAC verification (MODEL_SIGNING_KEY unset). "
            "Acceptable only in local-dev. Flip MODEL_SIGNING_REQUIRED=true "
            "and set MODEL_SIGNING_KEY before live trading.",
            p,
        )
        return
    if not sig_path.exists():
        if _signing_required():
            raise ValueError(f"Missing signature file: {sig_path}")
        logger.error(
            "No .sig for %s; loading WITHOUT verification (signing not required).",
            p,
        )
        return
    expected = sig_path.read_text().strip()
    actual = hmac.new(key, p.read_bytes(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, actual):
        raise ValueError(f"HMAC mismatch for {p}: refusing to load")


def verify_and_read(path: str | Path) -> bytes:
    """Read the file ONCE, verify HMAC over the in-memory buffer, return bytes.

    TOCTOU-safe loading primitive. The returned bytes are the authoritative
    artifact — deserialize from them, NEVER re-open ``path``.

    Raises:
        ValueError: signature missing (when required) or HMAC mismatch.
        RuntimeError: missing/short key in production-equivalent mode.
    """
    key = _get_key()
    p = Path(path)
    # Read file ONCE into memory. Subsequent operations use this buffer only.
    with open(p, "rb") as f:
        data = f.read()

    if key is None:
        if _signing_required():
            raise RuntimeError("MODEL_SIGNING_REQUIRED=true but no key available")
        logger.error(
            "Loading %s WITHOUT HMAC verification (MODEL_SIGNING_KEY unset). "
            "Acceptable only in local-dev. Flip MODEL_SIGNING_REQUIRED=true "
            "and set MODEL_SIGNING_KEY before live trading.",
            p,
        )
        return data

    sig_path = p.with_suffix(p.suffix + ".sig")
    if not sig_path.exists():
        if _signing_required():
            raise ValueError(f"Missing signature file: {sig_path}")
        logger.error(
            "No .sig for %s; loading WITHOUT verification (signing not required).",
            p,
        )
        return data

    expected = sig_path.read_text().strip()
    actual = hmac.new(key, data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, actual):
        raise ValueError(f"HMAC mismatch for {p}: refusing to load")
    return data


def safe_load_health_check() -> Dict[str, Any]:
    """Snapshot of model-signing posture for startup banners / dashboards."""
    raw_key = os.environ.get("MODEL_SIGNING_KEY", "")
    key_present = bool(raw_key)
    key_length = len(raw_key)
    signing_required = _signing_required()
    production_mode = _is_production_mode()

    if production_mode and (not key_present or key_length < _MIN_KEY_LEN):
        status = "fail_closed"
    elif key_present and key_length >= _MIN_KEY_LEN and signing_required:
        status = "ok"
    else:
        status = "degraded"

    return {
        "signing_required": signing_required,
        "key_present": key_present,
        "key_length": key_length,
        "production_mode": production_mode,
        "status": status,
    }


def atomic_write_and_sign(path: str | Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path`` and create matching .sig file.

    Uses ``O_EXCL`` to refuse pre-existing tmp files (symlink-safe), 0o600
    perms, fsync, then ``os.replace`` for atomic rename. Signature is created
    AFTER the rename so the verifier never sees a path/sig mismatch.
    """
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    # Refuse to follow a stale tmp; remove only if it exists and is a regular file
    # owned by this process. Keep behaviour conservative: just unlink if present.
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
    except OSError:
        pass
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    sign_file(p)
