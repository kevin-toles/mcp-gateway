"""Tests for OIDCAuthMiddleware & JWT validation — WBS-MCP3 (RED).

Covers all 8 Acceptance Criteria:
- AC-3.1: OIDCAuthMiddleware validates JWT on every request (except /health)
- AC-3.2: HS256 and alg:none rejected; RS256/ES256 accepted
- AC-3.3: JWKS key fetch with TTL cache
- AC-3.4: Claims validated: iss, aud, exp, nbf
- AC-3.5: Required claims enforced: exp, iss, aud, sub, tier
- AC-3.6: Invalid/missing/expired → 401 structured error (no leakage)
- AC-3.7: AUTH_ENABLED=false bypasses validation
- AC-3.8: Validated payload in request.state.auth

Technical debt: Self-signed RSA keys for test JWTs; resolved in WBS-MCP9.
"""

# NOTE: Do NOT use `from __future__ import annotations` here.
# It turns type hints into strings, which breaks FastAPI's detection
# of `request: Request` in endpoint functions defined inside fixtures.

import json
import time
from dataclasses import fields as dataclass_fields
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jose import jwt as jose_jwt
from starlette.testclient import TestClient

from src.core.config import Settings

# ═══════════════════════════════════════════════════════════════════════
# Test key fixtures — self-signed RSA/EC keys for JWT generation
# ═══════════════════════════════════════════════════════════════════════

# Generate RSA key pair for tests
_RSA_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_RSA_PUBLIC_KEY = _RSA_PRIVATE_KEY.public_key()

# PEM-encoded private key for signing tokens
RSA_PRIVATE_PEM = _RSA_PRIVATE_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)

# Public key numbers for JWKS response
_RSA_PUB_NUMBERS = _RSA_PUBLIC_KEY.public_numbers()


def _int_to_base64url(n: int, length: int | None = None) -> str:
    """Convert integer to base64url-encoded string (for JWK)."""
    import base64

    byte_length = length or (n.bit_length() + 7) // 8
    data = n.to_bytes(byte_length, byteorder="big")
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


JWKS_RESPONSE: dict[str, Any] = {
    "keys": [
        {
            "kty": "RSA",
            "kid": "test-key-1",
            "use": "sig",
            "alg": "RS256",
            "n": _int_to_base64url(_RSA_PUB_NUMBERS.n, 256),
            "e": _int_to_base64url(_RSA_PUB_NUMBERS.e, 3),
        }
    ]
}

ISSUER = "https://auth.example.com"
AUDIENCE = "ai-platform-tools"


def _make_token(
    claims: dict[str, Any] | None = None,
    algorithm: str = "RS256",
    key: Any = RSA_PRIVATE_PEM,
    headers: dict[str, str] | None = None,
) -> str:
    """Create a signed JWT with sensible defaults for testing."""
    now = int(time.time())
    default_claims = {
        "sub": "user-42",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": now + 3600,
        "nbf": now - 10,
        "iat": now,
        "tier": "gold",
    }
    if claims:
        default_claims.update(claims)

    hdr = {"kid": "test-key-1"}
    if headers:
        hdr.update(headers)

    return jose_jwt.encode(default_claims, key, algorithm=algorithm, headers=hdr)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_jwks_cache():
    """Clear the module-level JWKS cache before every test to avoid bleed."""
    import src.security.authn as _authn
    _authn._jwks_cache.clear()
    _authn.auth_metrics.reset()
    yield
    _authn._jwks_cache.clear()
    _authn.auth_metrics.reset()


@pytest.fixture
def auth_settings() -> Settings:
    """Settings with auth enabled and OIDC configured."""
    return Settings(
        AUTH_ENABLED=True,
        OIDC_JWKS_URL="https://auth.example.com/.well-known/jwks.json",
        OIDC_ISSUER=ISSUER,
        OIDC_AUDIENCE=AUDIENCE,
    )


@pytest.fixture
def noauth_settings() -> Settings:
    """Settings with auth disabled (dev mode)."""
    return Settings(AUTH_ENABLED=False)


@pytest.fixture
def mock_jwks():
    """Patch _fetch_jwks to return our test JWKS without HTTP call."""
    with patch(
        "src.security.authn._fetch_jwks",
        new_callable=AsyncMock,
        return_value=JWKS_RESPONSE,
    ) as m:
        yield m


@pytest.fixture
def app_with_auth(auth_settings, mock_jwks):
    """FastAPI app with OIDCAuthMiddleware enabled."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from src.security.authn import OIDCAuthMiddleware

    app = FastAPI()
    app.add_middleware(OIDCAuthMiddleware, settings=auth_settings)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/protected")
    async def protected(request: Request):
        auth = request.state.auth
        return {
            "sub": auth.sub,
            "tier": auth.tier,
            "issuer": auth.issuer,
        }

    return app


@pytest.fixture
def client(app_with_auth):
    """Starlette TestClient for middleware tests."""
    return TestClient(app_with_auth, raise_server_exceptions=False)


@pytest.fixture
def app_noauth(noauth_settings):
    """FastAPI app with auth disabled."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from src.security.authn import OIDCAuthMiddleware

    app = FastAPI()
    app.add_middleware(OIDCAuthMiddleware, settings=noauth_settings)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/protected")
    async def protected(request: Request):
        # When auth disabled, request.state.auth should still exist
        auth = request.state.auth
        return {"sub": auth.sub, "tier": auth.tier}

    return app


@pytest.fixture
def noauth_client(app_noauth):
    return TestClient(app_noauth, raise_server_exceptions=False)


# ═══════════════════════════════════════════════════════════════════════
# AC-3.2: Algorithm rejection — HS256 and alg:none rejected
# ═══════════════════════════════════════════════════════════════════════


class TestAlgorithmRestriction:
    """AC-3.2: Only RS256 and ES256 are accepted."""

    def test_alg_none_rejected(self, client):
        """Token with alg:none must be rejected with 401."""
        # Craft an unsigned token (alg=none)
        import base64

        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "none", "typ": "JWT"}).encode()
        ).rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(
            json.dumps({"sub": "attacker", "iss": ISSUER, "aud": AUDIENCE,
                        "exp": int(time.time()) + 3600, "tier": "enterprise"}).encode()
        ).rstrip(b"=").decode()
        token = f"{header}.{payload}."

        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_hs256_rejected(self, client):
        """Token signed with HS256 must be rejected with 401."""
        token = jose_jwt.encode(
            {
                "sub": "user-1",
                "iss": ISSUER,
                "aud": AUDIENCE,
                "exp": int(time.time()) + 3600,
                "tier": "gold",
            },
            "shared-secret",
            algorithm="HS256",
        )
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_rs256_accepted(self, client):
        """Valid RS256 token must be accepted."""
        token = _make_token()
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_error_does_not_leak_algorithm_details(self, client):
        """401 error body must not contain JWKS URL or key material."""
        token = jose_jwt.encode(
            {"sub": "x", "iss": ISSUER, "aud": AUDIENCE,
             "exp": int(time.time()) + 3600, "tier": "gold"},
            "secret", algorithm="HS256",
        )
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        body = resp.json()
        assert "jwks" not in json.dumps(body).lower()
        assert "secret" not in json.dumps(body).lower()
        assert "key" not in json.dumps(body).lower()


# ═══════════════════════════════════════════════════════════════════════
# AC-3.4: Claim validation — iss, aud, exp, nbf
# ═══════════════════════════════════════════════════════════════════════


class TestClaimValidation:
    """AC-3.4: iss, aud, exp, nbf claims must be validated."""

    def test_expired_token_returns_401(self, client):
        """Token with past exp must be rejected."""
        token = _make_token(claims={"exp": int(time.time()) - 600})
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_not_yet_valid_token_returns_401(self, client):
        """Token with future nbf must be rejected."""
        token = _make_token(claims={"nbf": int(time.time()) + 3600})
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_wrong_issuer_returns_401(self, client):
        """Token from untrusted issuer must be rejected."""
        token = _make_token(claims={"iss": "https://evil.example.com"})
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_wrong_audience_returns_401(self, client):
        """Token for wrong audience must be rejected."""
        token = _make_token(claims={"aud": "wrong-audience"})
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_valid_claims_accepted(self, client):
        """Token with all correct claims must be accepted."""
        token = _make_token()
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_error_body_is_structured_json(self, client):
        """AC-3.6: 401 responses must be structured JSON."""
        token = _make_token(claims={"exp": int(time.time()) - 600})
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        body = resp.json()
        assert "detail" in body or "error" in body


# ═══════════════════════════════════════════════════════════════════════
# AC-3.5: Required claims enforcement — exp, iss, aud, sub, tier
# ═══════════════════════════════════════════════════════════════════════


class TestRequiredClaims:
    """AC-3.5: Missing required claims must be rejected."""

    @pytest.mark.parametrize("missing_claim", ["sub", "tier", "iss", "aud", "exp"])
    def test_missing_required_claim_returns_401(self, client, missing_claim):
        """Each required claim must be present; omission → 401."""
        claims = {
            "sub": "user-1",
            "iss": ISSUER,
            "aud": AUDIENCE,
            "exp": int(time.time()) + 3600,
            "nbf": int(time.time()) - 10,
            "tier": "gold",
        }
        del claims[missing_claim]
        # Encode directly — do NOT use _make_token which adds defaults back
        token = jose_jwt.encode(
            claims, RSA_PRIVATE_PEM, algorithm="RS256",
            headers={"kid": "test-key-1"},
        )
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401, f"Missing '{missing_claim}' should return 401"

    def test_all_required_claims_present_accepted(self, client):
        """Token with all required claims must pass."""
        token = _make_token()
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# AC-3.6: Structured error responses — no secret leakage
# ═══════════════════════════════════════════════════════════════════════


class TestErrorResponses:
    """AC-3.6: 401 errors must be structured and leak nothing."""

    def test_missing_authorization_header_returns_401(self, client):
        """Request with no Authorization header → 401."""
        resp = client.get("/protected")
        assert resp.status_code == 401

    def test_malformed_bearer_returns_401(self, client):
        """Authorization header without 'Bearer ' prefix → 401."""
        resp = client.get("/protected", headers={"Authorization": "Token abc123"})
        assert resp.status_code == 401

    def test_empty_bearer_returns_401(self, client):
        """Authorization: Bearer (empty) → 401."""
        resp = client.get("/protected", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401

    def test_garbage_token_returns_401(self, client):
        """Random string as token → 401."""
        resp = client.get("/protected", headers={"Authorization": "Bearer not.a.jwt"})
        assert resp.status_code == 401

    def test_error_response_has_no_internal_details(self, client):
        """Error response must not contain JWKS URL, signing keys, or stack traces."""
        resp = client.get("/protected")
        body_str = json.dumps(resp.json()).lower()
        assert "jwks" not in body_str
        assert "traceback" not in body_str
        assert "signing" not in body_str
        assert "private" not in body_str

    def test_error_response_content_type_is_json(self, client):
        """401 must return application/json."""
        resp = client.get("/protected")
        assert "application/json" in resp.headers.get("content-type", "")


# ═══════════════════════════════════════════════════════════════════════
# AC-3.3: JWKS key fetch and caching
# ═══════════════════════════════════════════════════════════════════════


class TestJWKSCaching:
    """AC-3.3: Signing keys fetched from JWKS endpoint with TTL cache."""

    def test_jwks_fetched_on_first_request(self, client, mock_jwks):
        """JWKS should be fetched when first token is validated."""
        token = _make_token()
        client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        mock_jwks.assert_called()

    def test_jwks_cached_across_requests(self, client, mock_jwks):
        """Multiple requests should reuse cached JWKS, not re-fetch."""
        token = _make_token()
        client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        first_count = mock_jwks.call_count
        client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        # Second request uses cache — fetch count should not increase
        assert mock_jwks.call_count == first_count

    @pytest.mark.asyncio
    async def test_get_signing_keys_returns_key_set(self, auth_settings, mock_jwks):
        """_get_signing_keys returns the 'keys' list from JWKS."""
        from src.security.authn import _get_signing_keys

        keys = await _get_signing_keys(auth_settings.OIDC_JWKS_URL)
        assert isinstance(keys, list)
        assert len(keys) == 1
        assert keys[0]["kid"] == "test-key-1"

    @pytest.mark.asyncio
    async def test_fetch_jwks_called_with_correct_url(self, auth_settings, mock_jwks):
        """_get_signing_keys must pass the configured JWKS URL."""
        from src.security.authn import _get_signing_keys

        await _get_signing_keys(auth_settings.OIDC_JWKS_URL)
        mock_jwks.assert_called_with("https://auth.example.com/.well-known/jwks.json")


# ═══════════════════════════════════════════════════════════════════════
# AC-3.1: Middleware path exclusions
# ═══════════════════════════════════════════════════════════════════════


class TestMiddlewarePathExclusion:
    """AC-3.1: /health accessible without token."""

    def test_health_no_token_returns_200(self, client):
        """/health must be accessible without Authorization header."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_with_token_still_works(self, client):
        """/health with a valid token should also return 200."""
        token = _make_token()
        resp = client.get("/health", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_health_with_invalid_token_still_returns_200(self, client):
        """/health must not validate tokens — always accessible."""
        resp = client.get("/health", headers={"Authorization": "Bearer garbage"})
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# AC-3.7: Auth disabled (dev mode)
# ═══════════════════════════════════════════════════════════════════════


class TestAuthDisabled:
    """AC-3.7: AUTH_ENABLED=false bypasses all validation."""

    def test_protected_route_accessible_without_token(self, noauth_client):
        """When auth disabled, protected routes work without token."""
        resp = noauth_client.get("/protected")
        assert resp.status_code == 200

    def test_request_state_has_anonymous_user(self, noauth_client):
        """When auth disabled, request.state.auth has anonymous user."""
        resp = noauth_client.get("/protected")
        body = resp.json()
        assert body["sub"] == "anonymous"
        assert body["tier"] == "anonymous"

    def test_auth_disabled_ignores_invalid_token(self, noauth_client):
        """With auth disabled, even garbage tokens are ignored."""
        resp = noauth_client.get(
            "/protected",
            headers={"Authorization": "Bearer totally-invalid"},
        )
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# AC-3.8: Validated payload in request.state.auth
# ═══════════════════════════════════════════════════════════════════════


class TestRequestStateAuth:
    """AC-3.8: AuthenticatedUser stored in request.state.auth."""

    def test_auth_sub_populated(self, client):
        """request.state.auth.sub must be the JWT sub claim."""
        token = _make_token(claims={"sub": "service-account-99"})
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["sub"] == "service-account-99"

    def test_auth_tier_populated(self, client):
        """request.state.auth.tier must be the JWT tier claim."""
        token = _make_token(claims={"tier": "enterprise"})
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["tier"] == "enterprise"

    def test_auth_issuer_populated(self, client):
        """request.state.auth.issuer must be the JWT iss claim."""
        token = _make_token()
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["issuer"] == ISSUER


# ═══════════════════════════════════════════════════════════════════════
# AuthenticatedUser dataclass
# ═══════════════════════════════════════════════════════════════════════


class TestAuthenticatedUser:
    """AuthenticatedUser dataclass has all required fields."""

    def test_dataclass_fields(self):
        from src.security.authn import AuthenticatedUser

        names = {f.name for f in dataclass_fields(AuthenticatedUser)}
        assert names == {"sub", "tier", "issuer", "audience", "exp", "raw_claims"}

    def test_create_instance(self):
        from src.security.authn import AuthenticatedUser

        user = AuthenticatedUser(
            sub="user-1", tier="gold", issuer=ISSUER,
            audience=AUDIENCE, exp=9999999999, raw_claims={"sub": "user-1"},
        )
        assert user.sub == "user-1"
        assert user.tier == "gold"

    def test_anonymous_factory(self):
        """AuthenticatedUser.anonymous() creates an anonymous user."""
        from src.security.authn import AuthenticatedUser

        anon = AuthenticatedUser.anonymous()
        assert anon.sub == "anonymous"
        assert anon.tier == "anonymous"
        assert anon.raw_claims == {}


# ═══════════════════════════════════════════════════════════════════════
# MCP3.18 REFACTOR: Auth metrics & key rotation helpers
# ═══════════════════════════════════════════════════════════════════════


class TestAuthMetrics:
    """MCP3.18: Auth failure/success metrics for observability."""

    def test_metrics_increment_on_valid_token(self, client):
        """Successful auth should increment tokens_validated."""
        from src.security.authn import auth_metrics

        auth_metrics.reset()
        token = _make_token()
        client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert auth_metrics.tokens_validated == 1
        assert auth_metrics.tokens_rejected == 0

    def test_metrics_increment_on_rejected_token(self, client):
        """Failed auth should increment tokens_rejected."""
        from src.security.authn import auth_metrics

        auth_metrics.reset()
        client.get("/protected", headers={"Authorization": "Bearer garbage"})
        assert auth_metrics.tokens_rejected == 1
        assert auth_metrics.tokens_validated == 0

    def test_metrics_increment_on_missing_header(self, client):
        """Missing Authorization header should increment tokens_rejected."""
        from src.security.authn import auth_metrics

        auth_metrics.reset()
        client.get("/protected")
        assert auth_metrics.tokens_rejected == 1

    def test_metrics_jwks_fetch_tracked(self, client, mock_jwks):
        """JWKS fetch should be tracked in metrics."""
        from src.security.authn import auth_metrics

        auth_metrics.reset()
        token = _make_token()
        client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert auth_metrics.jwks_fetches == 1

    def test_metrics_jwks_cache_hit_tracked(self, client, mock_jwks):
        """Second request should hit JWKS cache and track it."""
        from src.security.authn import auth_metrics

        auth_metrics.reset()
        token = _make_token()
        client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert auth_metrics.jwks_cache_hits == 1
        assert auth_metrics.jwks_fetches == 1

    def test_metrics_reset(self):
        """reset() should zero all counters."""
        from src.security.authn import AuthMetrics

        m = AuthMetrics(tokens_validated=5, tokens_rejected=3, jwks_fetches=2, jwks_cache_hits=1)
        m.reset()
        assert m.tokens_validated == 0
        assert m.tokens_rejected == 0
        assert m.jwks_fetches == 0
        assert m.jwks_cache_hits == 0


class TestInvalidateJWKSCache:
    """MCP3.18: invalidate_jwks_cache() for key rotation support."""

    @pytest.mark.asyncio
    async def test_invalidate_forces_refetch(self, auth_settings, mock_jwks):
        """After invalidation, _get_signing_keys must re-fetch."""
        from src.security.authn import _get_signing_keys, invalidate_jwks_cache

        await _get_signing_keys(auth_settings.OIDC_JWKS_URL)
        assert mock_jwks.call_count == 1

        invalidate_jwks_cache()
        await _get_signing_keys(auth_settings.OIDC_JWKS_URL)
        assert mock_jwks.call_count == 2

    def test_invalidate_clears_cache_dict(self):
        """invalidate_jwks_cache should empty the cache dict."""
        from src.security.authn import _jwks_cache, invalidate_jwks_cache

        _jwks_cache["https://test"] = ([], 0.0)
        invalidate_jwks_cache()
        assert len(_jwks_cache) == 0
