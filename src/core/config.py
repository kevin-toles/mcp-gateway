"""Settings — WBS-MCP1.3 (GREEN), MCP1.8 (GREEN).

Centralized configuration for the mcp-gateway service.
All settings are loaded from environment variables with the MCP_GATEWAY_ prefix.

Reference: Strategy §4.1, §8.1 (Encryption.IN_TRANSIT), §10.1 (Week 1-2)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import NewType

from pydantic_settings import BaseSettings


HEALTH_ENDPOINT = "/health"
READY_ENDPOINT = "/ready"

# ── SLA & Service Identity ─────────────────────────────────────────────
SLA_TIMEOUT: float = 30.0
ServiceKey = NewType("ServiceKey", str)


from src.core.keys import normalize_service_key  # re-export for back-compat

__all__ = ["normalize_service_key"]


def _restart_cmd(service_key: str) -> str:
    """Manifest-derived kill-then-start command (config/services.toml).

    Deferred import: health_config imports ServiceKey from this module, so
    importing it at module load would be circular. HEALTH_PROXY_SERVICE_CONFIG
    is a runtime property, so the import happens after both modules exist.
    """
    from src.config.health_config import restart_command_for

    cmd = restart_command_for(service_key)
    if cmd is None:
        raise KeyError(
            f"no manifest entry for service '{service_key}' — "
            "add it to config/services.toml"
        )
    return cmd


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
    UNIFIED_SEARCH_URL: str = "http://localhost:8081"
    UNIFIED_SEARCH_RS_URL: str = "http://localhost:8081"
    AI_AGENTS_URL: str = "http://localhost:8082"
    CODE_ORCHESTRATOR_URL: str = "http://localhost:8083"
    AUDIT_SERVICE_URL: str = "http://localhost:8084"
    AMVE_SERVICE_URL: str = "http://localhost:8092"
    STRUCT_ANALYZER_URL: str = "http://localhost:8088"
    CONTEXT_MANAGEMENT_URL: str = "http://localhost:8086"
    INFERENCE_SERVICE_URL: str = "http://localhost:8085"
    VALIDATION_SERVICE_URL: str = "http://localhost:8091"

    # ── Code-Orchestrator lifecycle policy (hot/warm/cold) ──────────
    # Central source of truth for CO startup behavior used by gateway restarts.
    CO_CODEBERT_START_MODE: str = "warm"  # hot|warm|cold
    CO_GRAPHCODEBERT_START_MODE: str = "cold"  # hot|warm|cold
    CO_CODET5_START_MODE: str = "cold"  # hot|warm|cold
    CO_READY_TIMEOUT_SECONDS: float = 90.0

    # ── Security / OIDC ─────────────────────────────────────────────
    OIDC_JWKS_URL: str = ""  # JWKS endpoint for JWT validation
    OIDC_ISSUER: str = ""  # Expected JWT issuer
    OIDC_AUDIENCE: str = "ai-platform-tools"
    AUTH_ENABLED: bool = False  # Disabled for dev, enabled for prod

    # ── Phase feature flags ──────────────────────────────────────────
    # All default False — flip via env var; existing behaviour is
    # byte-for-bit identical when every flag is False.
    # G1.2 (GREEN) — WBS Phase 1: Session Correlation
    CORRELATION_ENABLED: bool = False  # X-Session-ID generation + forwarding in dispatcher
    # G2.2 (GREEN) — WBS Phase 2: Content-Addressed Snapshot Store
    SNAPSHOT_STORE_ENABLED: bool = False  # SHA-256 snapshot hashing + amve:findings:anonymous stream key
    # G3.2 (GREEN) — WBS Phase 3: Multi-Tenant Identity Propagation
    IDENTITY_PROPAGATION: bool = False  # X-Tenant-ID + X-Agent-ID forwarding

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
    CIRCUIT_BREAKER_RECOVERY_SECONDS: float = 5.0   # Seconds before HALF_OPEN probe (reduced from 30s for SLA)
    DISPATCH_MAX_RETRIES: int = 2  # Max retry attempts for transient failures
    DISPATCH_RETRY_BASE_DELAY: float = 0.5  # Base delay in seconds (exponential backoff)

    # ── Audit ───────────────────────────────────────────────────────
    AUDIT_LOG_PATH: str = "logs/audit.jsonl"

    # ── Service shutdown / restart commands ──────────────────────────
    SERVICE_SHUTDOWN_COMMANDS: dict[str, str] = {
        "code-orchestrator": "lsof -ti:8083 | xargs kill -9 2>/dev/null || true",
        "llm-gateway": "lsof -ti:8080 | xargs kill -9 2>/dev/null || true",
        "ai-agents": "lsof -ti:8082 | xargs kill -9 2>/dev/null || true",
        "audit-service": "lsof -ti:8084 | xargs kill -9 2>/dev/null || true",
        "context-management-service": "lsof -ti:8086 | xargs kill -9 2>/dev/null || true",
        "amve": "lsof -ti:8092 | xargs kill -9 2>/dev/null || true",
        "struct-analyzer": "lsof -ti:8088 | xargs kill -9 2>/dev/null || true",
        "unified-search-rs": "lsof -ti:8081 | xargs kill -9 2>/dev/null || true",
        "inference-service-cpp": "pkill -f inference-service 2>/dev/null || true",
        "mcp-gateway": "lsof -ti:8087 | xargs kill -9 2>/dev/null || true",
        "validation-service": "lsof -ti:8091 | xargs kill -9 2>/dev/null || true",
    }

    model_config = {
        "env_prefix": "MCP_GATEWAY_",
    }

    # ── Backward-compat aliases ────────────────────────────────────
    @property
    def SEMANTIC_SEARCH_URL(self) -> str:
        """Primary search service — unified-search-rs (Rust :8081)."""
        return self.UNIFIED_SEARCH_RS_URL

    @property
    def CO_RESTART_COMMAND(self) -> str:
        """Manifest restart command with lifecycle start modes from settings."""
        import re as _re

        cmd = _restart_cmd("code-orchestrator")
        for var, value in (
            ("COS_CODEBERT_START_MODE", self.CO_CODEBERT_START_MODE),
            ("COS_GRAPHCODEBERT_START_MODE", self.CO_GRAPHCODEBERT_START_MODE),
            ("COS_CODET5_START_MODE", self.CO_CODET5_START_MODE),
        ):
            cmd = _re.sub(rf"{var}=\S+", f"{var}={value}", cmd)
        return cmd

    @property
    def HEALTH_PROXY_SERVICE_CONFIG(self) -> dict[str, dict[str, object]]:
        """Central service policy for managed service lifecycle.

        Single source of truth for: base URL, readiness/liveness endpoint,
        restart command, and timeout.
        """
        return {
            "semantic-search": {
                "name": "unified-search-rs",
                "url": self.UNIFIED_SEARCH_RS_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": _restart_cmd("unified-search-rs"),
                "timeout": 30.0,
                "sla_timeout": 2.0,
            },
            "code-analyze": {
                "name": "code-orchestrator",
                "url": self.CODE_ORCHESTRATOR_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": self.CO_RESTART_COMMAND,
                "timeout": self.CO_READY_TIMEOUT_SECONDS,
                "sla_timeout": 8.0,
            },
            "code-orchestrator": {
                "name": "code-orchestrator",
                "url": self.CODE_ORCHESTRATOR_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": self.CO_RESTART_COMMAND,
                "timeout": self.CO_READY_TIMEOUT_SECONDS,
                "sla_timeout": 8.0,
            },
            "llm-complete": {
                "name": "llm-gateway",
                "url": self.LLM_GATEWAY_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": _restart_cmd("llm-gateway"),
                "timeout": 5.0,
                "sla_timeout": 2.0,
            },
            "run-agent-function": {
                "name": "ai-agents",
                "url": self.AI_AGENTS_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": _restart_cmd("ai-agents"),
                "timeout": 8.0,
                "sla_timeout": 8.0,
            },
            "ai-agents": {
                "name": "ai-agents",
                "url": self.AI_AGENTS_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": _restart_cmd("ai-agents"),
                "timeout": 8.0,
                "sla_timeout": 8.0,
            },
            "audit-quality-scan": {
                "name": "audit-service",
                "url": self.AUDIT_SERVICE_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": _restart_cmd("audit-service"),
                "timeout": 8.0,
                "sla_timeout": 8.0,
            },
            "audit-service": {
                "name": "audit-service",
                "url": self.AUDIT_SERVICE_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": _restart_cmd("audit-service"),
                "timeout": 8.0,
                "sla_timeout": 8.0,
            },
            "context-management": {
                "name": "context-management-service",
                "url": self.CONTEXT_MANAGEMENT_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": _restart_cmd("context-management-service"),
                "timeout": 60.0,
                "sla_timeout": 15.0,
            },
            "amve": {
                "name": "amve",
                "url": self.AMVE_SERVICE_URL,
                "health_endpoint": "/v1/health",
                "restart_command": _restart_cmd("amve"),
                "timeout": 8.0,
                "sla_timeout": 8.0,
            },
            "struct-analyzer": {
                "name": "struct-analyzer",
                "url": self.STRUCT_ANALYZER_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": _restart_cmd("struct-analyzer"),
                "timeout": 8.0,
                "sla_timeout": 15.0,
            },
            "validation-service": {
                "name": "validation-service",
                "url": self.VALIDATION_SERVICE_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": _restart_cmd("validation-service"),
                "timeout": 8.0,
                "sla_timeout": 8.0,
            },
            "foundation-search": {
                "name": "unified-search-rs",
                "url": self.UNIFIED_SEARCH_RS_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": _restart_cmd("unified-search-rs"),
                "timeout": 30.0,
                "sla_timeout": 2.0,
            },
            "inference": {
                "name": "inference-service-cpp",
                "url": self.INFERENCE_SERVICE_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": _restart_cmd("inference-service-cpp"),
                "timeout": 60.0,
                "sla_timeout": 15.0,
            },
        }


# Module-level singleton — imported by idle_timeout and other subsystems.
settings = Settings()


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


_config_logger = logging.getLogger(__name__)


def validate_config(settings: Settings) -> list[str]:
    """Validate settings for the current deployment mode.

    In Docker mode, all service-to-service URLs must use container DNS names,
    not localhost. Misconfigured URLs silently fall through to Pydantic defaults
    when the env-prefixed var is missing — this catches that.

    Returns a list of warning strings (all are also logged at WARNING level).
    """
    mode = os.environ.get("DEPLOYMENT_MODE", "hybrid").lower()
    _config_logger.info("deployment_mode=%s service=%s", mode, settings.SERVICE_NAME)

    warnings: list[str] = []
    if mode != "docker":
        return warnings

    prefix = str(settings.model_config.get("env_prefix", ""))
    for field, value in settings.model_dump().items():
        if not field.upper().endswith("_URL"):
            continue
        if not isinstance(value, str):
            continue
        if "localhost" in value or "127.0.0.1" in value:
            warnings.append(
                f"DOCKER MODE: {field}={value!r} uses localhost — "
                f"set {prefix}{field.upper()} to a container DNS name"
            )

    for w in warnings:
        _config_logger.warning(w)
    return warnings
