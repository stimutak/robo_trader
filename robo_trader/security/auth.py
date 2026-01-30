"""
Authentication and authorization system for RoboTrader.

Implements JWT-based authentication, role-based access control,
and API key management for secure access to trading operations.
"""

import hashlib
import hmac
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    import jwt
    from passlib.context import CryptContext

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("Warning: JWT/Passlib not available. Authentication features limited.")


class Role(Enum):
    """User roles with different permission levels."""

    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API = "api"


class Permission(Enum):
    """Granular permissions for access control."""

    # Trading permissions
    EXECUTE_TRADES = "execute_trades"
    MODIFY_STRATEGIES = "modify_strategies"
    OVERRIDE_RISK_LIMITS = "override_risk_limits"
    EMERGENCY_STOP = "emergency_stop"

    # Data permissions
    VIEW_POSITIONS = "view_positions"
    VIEW_PNL = "view_pnl"
    VIEW_ORDERS = "view_orders"
    EXPORT_DATA = "export_data"

    # Configuration permissions
    MODIFY_CONFIG = "modify_config"
    MANAGE_USERS = "manage_users"
    MANAGE_API_KEYS = "manage_api_keys"

    # Monitoring permissions
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_LOGS = "view_logs"
    VIEW_METRICS = "view_metrics"
    MANAGE_ALERTS = "manage_alerts"


# Role-permission mappings
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.TRADER: {
        Permission.EXECUTE_TRADES,
        Permission.MODIFY_STRATEGIES,
        Permission.VIEW_POSITIONS,
        Permission.VIEW_PNL,
        Permission.VIEW_ORDERS,
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_METRICS,
        Permission.EXPORT_DATA,
    },
    Role.ANALYST: {
        Permission.VIEW_POSITIONS,
        Permission.VIEW_PNL,
        Permission.VIEW_ORDERS,
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_METRICS,
        Permission.EXPORT_DATA,
    },
    Role.VIEWER: {
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_POSITIONS,
        Permission.VIEW_PNL,
    },
    Role.API: {
        Permission.VIEW_POSITIONS,
        Permission.VIEW_PNL,
        Permission.VIEW_ORDERS,
        Permission.VIEW_METRICS,
    },
}


@dataclass
class User:
    """User account information."""

    username: str
    email: str
    role: Role
    hashed_password: str
    is_active: bool = True
    is_2fa_enabled: bool = False
    totp_secret: Optional[str] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in ROLE_PERMISSIONS.get(self.role, set())

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
            "is_2fa_enabled": self.is_2fa_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


@dataclass
class APIKey:
    """API key for programmatic access."""

    key_id: str
    key_hash: str
    name: str
    user_id: str
    permissions: Set[Permission]
    rate_limit: int  # requests per minute
    expires_at: Optional[datetime] = None
    created_at: datetime = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    allowed_ips: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.allowed_ips is None:
            self.allowed_ips = []

    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP is allowed to use this key."""
        if not self.allowed_ips:
            return True  # No IP restrictions
        return ip in self.allowed_ips


class AuthManager:
    """Manages authentication and authorization."""

    def __init__(
        self,
        secret_key: str,
        token_expiry_hours: int = 24,
        max_login_attempts: int = 5,
        lockout_duration_minutes: int = 30,
        users_file: Optional[Path] = None,
        api_keys_file: Optional[Path] = None,
    ):
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.max_login_attempts = max_login_attempts
        self.lockout_duration_minutes = lockout_duration_minutes

        # Password hashing
        if JWT_AVAILABLE:
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        else:
            self.pwd_context = None

        # User and API key storage
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.active_sessions: Dict[str, Dict] = {}

        # Load users and API keys
        if users_file and users_file.exists():
            self.load_users(users_file)
        if api_keys_file and api_keys_file.exists():
            self.load_api_keys(api_keys_file)

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        if self.pwd_context:
            return self.pwd_context.hash(password)
        else:
            # Fallback to SHA256 if bcrypt not available
            return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if self.pwd_context:
            return self.pwd_context.verify(plain_password, hashed_password)
        else:
            # Fallback to SHA256
            return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: Role = Role.VIEWER,
        enable_2fa: bool = False,
    ) -> User:
        """Create a new user account."""
        if username in self.users:
            raise ValueError(f"User {username} already exists")

        user = User(
            username=username,
            email=email,
            role=role,
            hashed_password=self.hash_password(password),
            is_2fa_enabled=enable_2fa,
            totp_secret=secrets.token_urlsafe(32) if enable_2fa else None,
        )

        self.users[username] = user
        return user

    def authenticate(
        self, username: str, password: str, totp_code: Optional[str] = None
    ) -> Optional[User]:
        """Authenticate a user."""
        user = self.users.get(username)
        if not user:
            return None

        # Check if account is locked
        if user.locked_until and datetime.now() < user.locked_until:
            return None

        # Verify password
        if not self.verify_password(password, user.hashed_password):
            user.failed_login_attempts += 1

            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_login_attempts:
                user.locked_until = datetime.now() + timedelta(
                    minutes=self.lockout_duration_minutes
                )

            return None

        # Verify 2FA if enabled
        if user.is_2fa_enabled:
            if not totp_code or not self.verify_totp(user.totp_secret, totp_code):
                return None

        # Success - reset failed attempts
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()

        return user

    def verify_totp(self, secret: str, code: str) -> bool:
        """Verify TOTP code using pyotp library."""
        if not code or len(code) != 6 or not code.isdigit():
            return False

        try:
            import pyotp

            totp = pyotp.TOTP(secret)
            # valid_window=1 allows 1 period (30s) before/after for clock drift
            return totp.verify(code, valid_window=1)
        except ImportError:
            logger.error("pyotp not installed - TOTP verification unavailable")
            return False
        except Exception as e:
            logger.error(f"TOTP verification error: {e}")
            return False

    def create_token(self, user: User) -> str:
        """Create JWT token for user."""
        if not JWT_AVAILABLE:
            # Fallback to simple token
            token = secrets.token_urlsafe(32)
            self.active_sessions[token] = {
                "username": user.username,
                "role": user.role.value,
                "expires": datetime.now() + timedelta(hours=self.token_expiry_hours),
            }
            return token

        payload = {
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
        }

        token = jwt.encode(payload, self.secret_key, algorithm="HS256")

        # Track active session
        self.active_sessions[payload["jti"]] = {
            "username": user.username,
            "expires": payload["exp"],
        }

        return token

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token."""
        if not JWT_AVAILABLE:
            # Fallback to simple token
            session = self.active_sessions.get(token)
            if session and session["expires"] > datetime.now():
                return session
            return None

        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])

            # Check if token is revoked
            if payload.get("jti") not in self.active_sessions:
                return None

            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        if not JWT_AVAILABLE:
            if token in self.active_sessions:
                del self.active_sessions[token]
                return True
            return False

        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=["HS256"], options={"verify_exp": False}
            )
            jti = payload.get("jti")
            if jti in self.active_sessions:
                del self.active_sessions[jti]
                return True
        except Exception:
            pass

        return False

    def create_api_key(
        self,
        name: str,
        user_id: str,
        permissions: Set[Permission],
        rate_limit: int = 60,
        expires_days: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
    ) -> tuple[str, APIKey]:
        """Create a new API key."""
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        key_id = secrets.token_hex(8)

        # Hash the key for storage
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            permissions=permissions,
            rate_limit=rate_limit,
            expires_at=datetime.now() + timedelta(days=expires_days) if expires_days else None,
            allowed_ips=allowed_ips or [],
        )

        self.api_keys[key_id] = api_key

        # Return raw key (only shown once)
        return f"{key_id}.{raw_key}", api_key

    def verify_api_key(self, key: str, ip: Optional[str] = None) -> Optional[APIKey]:
        """Verify an API key."""
        try:
            key_id, raw_key = key.split(".", 1)
        except ValueError:
            return None

        api_key = self.api_keys.get(key_id)
        if not api_key:
            return None

        # Verify key hash
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        if not hmac.compare_digest(key_hash, api_key.key_hash):
            return None

        # Check if active
        if not api_key.is_active:
            return None

        # Check expiration
        if api_key.is_expired():
            return None

        # Check IP restrictions
        if ip and not api_key.is_ip_allowed(ip):
            return None

        # Update last used
        api_key.last_used = datetime.now()

        return api_key

    def check_permission(
        self,
        token_or_key: str,
        permission: Permission,
        ip: Optional[str] = None,
    ) -> bool:
        """Check if token/key has specific permission."""
        # Try as JWT token first
        token_data = self.verify_token(token_or_key)
        if token_data:
            username = token_data.get("username")
            user = self.users.get(username)
            return user and user.has_permission(permission)

        # Try as API key
        api_key = self.verify_api_key(token_or_key, ip)
        if api_key:
            return permission in api_key.permissions

        return False

    def save_users(self, filepath: Path) -> None:
        """Save users to file."""
        users_data = {
            username: {
                "email": user.email,
                "role": user.role.value,
                "hashed_password": user.hashed_password,
                "is_active": user.is_active,
                "is_2fa_enabled": user.is_2fa_enabled,
                "totp_secret": user.totp_secret,
                "created_at": user.created_at.isoformat() if user.created_at else None,
            }
            for username, user in self.users.items()
        }

        with open(filepath, "w") as f:
            json.dump(users_data, f, indent=2)

    def load_users(self, filepath: Path) -> None:
        """Load users from file."""
        with open(filepath, "r") as f:
            users_data = json.load(f)

        self.users = {}
        for username, data in users_data.items():
            self.users[username] = User(
                username=username,
                email=data["email"],
                role=Role(data["role"]),
                hashed_password=data["hashed_password"],
                is_active=data.get("is_active", True),
                is_2fa_enabled=data.get("is_2fa_enabled", False),
                totp_secret=data.get("totp_secret"),
                created_at=(
                    datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                ),
            )

    def save_api_keys(self, filepath: Path) -> None:
        """Save API keys to file."""
        keys_data = {
            key_id: {
                "key_hash": api_key.key_hash,
                "name": api_key.name,
                "user_id": api_key.user_id,
                "permissions": [p.value for p in api_key.permissions],
                "rate_limit": api_key.rate_limit,
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                "is_active": api_key.is_active,
                "allowed_ips": api_key.allowed_ips,
            }
            for key_id, api_key in self.api_keys.items()
        }

        with open(filepath, "w") as f:
            json.dump(keys_data, f, indent=2)

    def load_api_keys(self, filepath: Path) -> None:
        """Load API keys from file."""
        with open(filepath, "r") as f:
            keys_data = json.load(f)

        self.api_keys = {}
        for key_id, data in keys_data.items():
            self.api_keys[key_id] = APIKey(
                key_id=key_id,
                key_hash=data["key_hash"],
                name=data["name"],
                user_id=data["user_id"],
                permissions={Permission(p) for p in data["permissions"]},
                rate_limit=data["rate_limit"],
                expires_at=(
                    datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
                ),
                is_active=data.get("is_active", True),
                allowed_ips=data.get("allowed_ips", []),
            )


# Rate limiting
class RateLimiter:
    """Simple rate limiter for API endpoints."""

    def __init__(self):
        self.requests: Dict[str, List[datetime]] = {}

    def is_allowed(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """Check if request is allowed under rate limit."""
        now = datetime.now()

        if key not in self.requests:
            self.requests[key] = []

        # Remove old requests outside window
        cutoff = now - timedelta(seconds=window_seconds)
        self.requests[key] = [t for t in self.requests[key] if t > cutoff]

        # Check if under limit
        if len(self.requests[key]) < limit:
            self.requests[key].append(now)
            return True

        return False
