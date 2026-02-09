"""OIDCAuthMiddleware & JWT validation — WBS-MCP3 (GREEN + REFACTOR).

Validates JWTs on every request (except excluded paths like ``/health``).
Only RS256 and ES256 algorithms are accepted — HS256 and ``none`` are
explicitly rejected to prevent algorithm confusion attacks.

MCP3.18 REFACTOR: Extracted ``KeyManager`` for JWKS rotation / caching,
added ``auth_metrics`` counter for auth failure observability.

Reference: Strategy §4.2 (Authentication Hardening — P0), §7.1 Control #1,
           §8 Taxonomy: IAM.AUTHENTICATION.JWT_SECURITY
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from jose import JWTError
from jose import jwt as jose_jwt
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.core.config import Settings

# ── Constants ───────────────────────────────────────────────────────────

# Only asymmetric algorithms permitted (§4.2 Authentication Hardening)
_ALLOWED_ALGORITHMS: list[str] = ["RS256", "ES256"]

# Paths that never require authentication
_PUBLIC_PATHS: set[str] = {"/health", "/health/"}

# SSE/streaming paths incompatible with BaseHTTPMiddleware (breaks streaming)
_SSE_PREFIX: str = "/mcp"

# Required JWT claims (AC-3.5)
_REQUIRED_CLAIMS: set[str] = {"exp", "iss", "aud", "sub", "tier"}


# ── Auth metrics ────────────────────────────────────────────────────────


@dataclass
class AuthMetrics:
    """Simple counters for auth events — extensible to Prometheus later."""

    tokens_validated: int = 0
    tokens_rejected: int = 0
    jwks_fetches: int = 0
    jwks_cache_hits: int = 0

    def record_validation(self) -> None:
        self.tokens_validated += 1

    def record_rejection(self) -> None:
        self.tokens_rejected += 1

    def record_jwks_fetch(self) -> None:
        self.jwks_fetches += 1

    def record_jwks_cache_hit(self) -> None:
        self.jwks_cache_hits += 1

    def reset(self) -> None:
        """Reset all counters (useful for testing)."""
        self.tokens_validated = 0
        self.tokens_rejected = 0
        self.jwks_fetches = 0
        self.jwks_cache_hits = 0


auth_metrics = AuthMetrics()


# ── Data types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuthenticatedUser:
    """Validated identity extracted from a JWT.

    Stored in ``request.state.auth`` after successful middleware validation.
    """

    sub: str
    tier: str
    issuer: str
    audience: str
    exp: int
    raw_claims: dict = field(default_factory=dict)

    @classmethod
    def anonymous(cls) -> AuthenticatedUser:
        """Create a sentinel user for dev-mode (AUTH_ENABLED=false)."""
        return cls(
            sub="anonymous",
            tier="anonymous",
            issuer="",
            audience="",
            exp=0,
            raw_claims={},
        )


# ── Key Manager (MCP3.18 REFACTOR) ─────────────────────────────────────

# Simple in-memory cache: (jwks_url → (keys, fetch_time))
_jwks_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
_JWKS_CACHE_TTL: float = 300.0  # 5 minutes


async def _fetch_jwks(jwks_url: str) -> dict[str, Any]:
    """Fetch the JWKS document from *jwks_url* via HTTP GET.

    This is the low-level fetch; caching is handled by ``_get_signing_keys``.
    Mocked in tests to avoid real HTTP calls.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(jwks_url, timeout=10.0)
        resp.raise_for_status()
        return resp.json()


async def _get_signing_keys(jwks_url: str) -> list[dict[str, Any]]:
    """Return JWKS ``keys`` list, using a TTL cache.

    Re-fetches when the cache entry is older than ``_JWKS_CACHE_TTL``.
    Tracks cache hits/misses via ``auth_metrics``.
    """
    now = time.monotonic()

    if jwks_url in _jwks_cache:
        keys, fetched_at = _jwks_cache[jwks_url]
        if now - fetched_at < _JWKS_CACHE_TTL:
            auth_metrics.record_jwks_cache_hit()
            return keys

    jwks = await _fetch_jwks(jwks_url)
    keys = jwks.get("keys", [])
    _jwks_cache[jwks_url] = (keys, now)
    auth_metrics.record_jwks_fetch()
    return keys


def invalidate_jwks_cache() -> None:
    """Force JWKS re-fetch on next request (e.g. after key rotation)."""
    _jwks_cache.clear()


# ── Token validation ────────────────────────────────────────────────────


async def validate_token(
    token: str,
    *,
    jwks_url: str,
    issuer: str,
    audience: str,
) -> dict[str, Any]:
    """Decode and validate a JWT.

    Returns the full claims dict on success.

    Raises:
        JWTError: On any validation failure (algorithm, signature,
                  expiration, issuer, audience, missing claims).
    """
    keys = await _get_signing_keys(jwks_url)

    # Attempt decode with each key in the JWKS
    payload: dict[str, Any] | None = None
    last_error: Exception | None = None

    for key in keys:
        try:
            payload = jose_jwt.decode(
                token,
                key,
                algorithms=_ALLOWED_ALGORITHMS,
                audience=audience,
                issuer=issuer,
                options={
                    "require_exp": True,
                    "require_iss": True,
                    "require_aud": True,
                    "verify_exp": True,
                    "verify_iss": True,
                    "verify_aud": True,
                    "verify_nbf": True,
                },
            )
            break
        except JWTError as exc:
            last_error = exc
            continue

    if payload is None:
        raise last_error or JWTError("Token validation failed")

    # Enforce required claims beyond what jose validates (AC-3.5)
    missing = _REQUIRED_CLAIMS - set(payload.keys())
    if missing:
        raise JWTError(f"Missing required claims: {', '.join(sorted(missing))}")

    return payload


# ── Middleware ───────────────────────────────────────────────────────────


class OIDCAuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that validates JWTs on every request.

    - Excluded paths (e.g. ``/health``) bypass validation entirely.
    - When ``AUTH_ENABLED=false``, all requests pass with an anonymous
      ``AuthenticatedUser`` attached to ``request.state.auth``.
    - Auth success/failure tracked via ``auth_metrics``.
    """

    def __init__(self, app: Any, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # ── Public paths — always allowed ───────────────────────────
        if request.url.path in _PUBLIC_PATHS or request.url.path.startswith(_SSE_PREFIX):
            return await call_next(request)

        # ── Dev-mode bypass (AC-3.7) ────────────────────────────────
        if not self.settings.AUTH_ENABLED:
            request.state.auth = AuthenticatedUser.anonymous()
            return await call_next(request)

        # ── Extract bearer token ────────────────────────────────────
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            auth_metrics.record_rejection()
            return _unauthorized("Missing or malformed Authorization header")

        token = auth_header[len("Bearer ") :]
        if not token or not token.strip():
            auth_metrics.record_rejection()
            return _unauthorized("Empty bearer token")

        # ── Validate ────────────────────────────────────────────────
        try:
            claims = await validate_token(
                token,
                jwks_url=self.settings.OIDC_JWKS_URL,
                issuer=self.settings.OIDC_ISSUER,
                audience=self.settings.OIDC_AUDIENCE,
            )
        except Exception:
            auth_metrics.record_rejection()
            return _unauthorized("Invalid or expired token")

        # ── Attach to request state (AC-3.8) ────────────────────────
        auth_metrics.record_validation()
        request.state.auth = AuthenticatedUser(
            sub=claims["sub"],
            tier=claims["tier"],
            issuer=claims["iss"],
            audience=claims["aud"],
            exp=claims["exp"],
            raw_claims=claims,
        )

        return await call_next(request)


def _unauthorized(detail: str) -> JSONResponse:
    """Return a structured 401 with no internal details leaked."""
    return JSONResponse(
        status_code=401,
        content={"error": "unauthorized", "detail": detail},
    )
