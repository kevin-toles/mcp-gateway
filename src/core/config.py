"""Settings — WBS-MCP1.3 (GREEN), MCP1.8 (GREEN).

Centralized configuration for the mcp-gateway service.
All settings are loaded from environment variables with the MCP_GATEWAY_ prefix.

Reference: Strategy §4.1, §8.1 (Encryption.IN_TRANSIT), §10.1 (Week 1-2)
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """MCP Gateway configuration.

    All fields can be overridden by environment variables prefixed with
    ``MCP_GATEWAY_``.  For example, ``MCP_GATEWAY_PORT=9999`` overrides
    the default port.
    """

    # ── Service identity ────────────────────────────────────────────
    SERVICE_NAME: str = "mcp-gateway"
    SERVICE_VERSION: str = "0.1.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8087

    # ── Backend service URLs ────────────────────────────────────────
    LLM_GATEWAY_URL: str = "http://localhost:8080"
    SEMANTIC_SEARCH_URL: str = "http://localhost:8081"
    AI_AGENTS_URL: str = "http://localhost:8082"
    CODE_ORCHESTRATOR_URL: str = "http://localhost:8083"

    # ── Security / OIDC ─────────────────────────────────────────────
    OIDC_JWKS_URL: str = ""  # JWKS endpoint for JWT validation
    OIDC_ISSUER: str = ""  # Expected JWT issuer
    OIDC_AUDIENCE: str = "ai-platform-tools"
    AUTH_ENABLED: bool = False  # Disabled for dev, enabled for prod

    # ── Rate limiting ───────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"
    RATE_LIMIT_RPM: int = 100  # Default requests per minute

    # ── TLS (Strategy §8.1 — Encryption.IN_TRANSIT) ────────────────
    TLS_ENABLED: bool = False  # True in production
    TLS_CERT_PATH: str = ""  # Path to TLS certificate
    TLS_KEY_PATH: str = ""  # Path to TLS private key
    TLS_MIN_VERSION: str = "TLSv1.3"  # Minimum TLS version

    # ── Resilience (C-5: Circuit Breakers) ───────────────────────
    CIRCUIT_BREAKER_THRESHOLD: int = 5  # Consecutive failures before OPEN
    CIRCUIT_BREAKER_RECOVERY_SECONDS: float = 30.0  # Seconds before HALF_OPEN probe
    DISPATCH_MAX_RETRIES: int = 2  # Max retry attempts for transient failures
    DISPATCH_RETRY_BASE_DELAY: float = 0.5  # Base delay in seconds (exponential backoff)

    # ── Audit ───────────────────────────────────────────────────────
    AUDIT_LOG_PATH: str = "logs/audit.jsonl"

    model_config = {
        "env_prefix": "MCP_GATEWAY_",
    }


def get_ssl_config(settings: Settings) -> dict | None:
    """Build uvicorn SSL kwargs from Settings.

    Returns ``None`` when TLS is disabled (dev mode).
    Raises ``ValueError`` if paths are empty, or ``FileNotFoundError``
    if the referenced cert/key files do not exist on disk.
    """
    if not settings.TLS_ENABLED:
        return None

    if not settings.TLS_CERT_PATH or not settings.TLS_KEY_PATH:
        raise ValueError("TLS_CERT_PATH and TLS_KEY_PATH are required when TLS_ENABLED=true")

    cert_path = Path(settings.TLS_CERT_PATH)
    key_path = Path(settings.TLS_KEY_PATH)

    if not cert_path.exists():
        raise FileNotFoundError(f"TLS certificate not found: {cert_path}")
    if not key_path.exists():
        raise FileNotFoundError(f"TLS private key not found: {key_path}")

    return {
        "ssl_certfile": str(cert_path),
        "ssl_keyfile": str(key_path),
    }
