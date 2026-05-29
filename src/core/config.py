"""Settings — WBS-MCP1.3 (GREEN), MCP1.8 (GREEN).

Centralized configuration for the mcp-gateway service.
All settings are loaded from environment variables with the MCP_GATEWAY_ prefix.

Reference: Strategy §4.1, §8.1 (Encryption.IN_TRANSIT), §10.1 (Week 1-2)
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


HEALTH_ENDPOINT = "/health"
READY_ENDPOINT = "/ready"


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
    UNIFIED_SEARCH_RS_URL: str = "http://localhost:8089"
    AI_AGENTS_URL: str = "http://localhost:8082"
    CODE_ORCHESTRATOR_URL: str = "http://localhost:8083"
    AUDIT_SERVICE_URL: str = "http://localhost:8084"
    AMVE_SERVICE_URL: str = "http://localhost:8088"

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
    CIRCUIT_BREAKER_RECOVERY_SECONDS: float = 30.0  # Seconds before HALF_OPEN probe
    DISPATCH_MAX_RETRIES: int = 2  # Max retry attempts for transient failures
    DISPATCH_RETRY_BASE_DELAY: float = 0.5  # Base delay in seconds (exponential backoff)

    # ── Audit ───────────────────────────────────────────────────────
    AUDIT_LOG_PATH: str = "logs/audit.jsonl"

    model_config = {
        "env_prefix": "MCP_GATEWAY_",
    }

    @property
    def CO_RESTART_COMMAND(self) -> str:
        """Build the canonical CO restart command with lifecycle env vars."""
        return (
            "lsof -ti:8083 | xargs kill -9 2>/dev/null || true; sleep 1; "
            "cd /Users/kevintoles/POC/Code-Orchestrator-Service && "
            f"COS_CODEBERT_START_MODE={self.CO_CODEBERT_START_MODE} "
            f"COS_GRAPHCODEBERT_START_MODE={self.CO_GRAPHCODEBERT_START_MODE} "
            f"COS_CODET5_START_MODE={self.CO_CODET5_START_MODE} "
            "/Users/kevintoles/POC/Code-Orchestrator-Service/.venv/bin/uvicorn "
            "src.main:app --host 0.0.0.0 --port 8083"
        )

    @property
    def HEALTH_PROXY_SERVICE_CONFIG(self) -> dict[str, dict[str, object]]:
        """Central service policy for HealthAwareProxy.

        This is the single source of truth for managed service lifecycle policy:
        base URL, readiness/liveness endpoint, restart command, and timeout.
        """
        return {
            "semantic_search": {
                "name": "unified-search-service",
                "url": self.UNIFIED_SEARCH_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": "lsof -ti:8081 | xargs kill -9 2>/dev/null || true; sleep 1; cd /Users/kevintoles/POC/unified-search-service && /Users/kevintoles/POC/unified-search-service/.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8081",
                "timeout": 45.0,
            },
            "code_analyze": {
                "name": "code-orchestrator",
                "url": self.CODE_ORCHESTRATOR_URL,
                "health_endpoint": READY_ENDPOINT,
                "restart_command": self.CO_RESTART_COMMAND,
                "timeout": self.CO_READY_TIMEOUT_SECONDS,
            },
            "code_orchestrator": {
                "name": "code-orchestrator",
                "url": self.CODE_ORCHESTRATOR_URL,
                "health_endpoint": READY_ENDPOINT,
                "restart_command": self.CO_RESTART_COMMAND,
                "timeout": self.CO_READY_TIMEOUT_SECONDS,
            },
            "llm_complete": {
                "name": "llm-gateway",
                "url": self.LLM_GATEWAY_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": "lsof -ti:8080 | xargs kill -9 2>/dev/null || true; sleep 1; cd /Users/kevintoles/POC/llm-gateway && /Users/kevintoles/POC/llm-gateway/.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8080",
                "timeout": 5.0,
            },
            "run_agent_function": {
                "name": "ai-agents",
                "url": self.AI_AGENTS_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": "lsof -ti:8082 | xargs kill -9 2>/dev/null || true; sleep 1; cd /Users/kevintoles/POC/ai-agents && /Users/kevintoles/POC/ai-agents/.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8082",
                "timeout": 8.0,
            },
            "ai_agents": {
                "name": "ai-agents",
                "url": self.AI_AGENTS_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": "lsof -ti:8082 | xargs kill -9 2>/dev/null || true; sleep 1; cd /Users/kevintoles/POC/ai-agents && /Users/kevintoles/POC/ai-agents/.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8082",
                "timeout": 8.0,
            },
            "audit_quality_scan": {
                "name": "audit-service",
                "url": self.AUDIT_SERVICE_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": "lsof -ti:8084 | xargs kill -9 2>/dev/null || true; sleep 1; cd /Users/kevintoles/POC/audit-service && /Users/kevintoles/POC/audit-service/.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8084",
                "timeout": 8.0,
            },
            "audit_service": {
                "name": "audit-service",
                "url": self.AUDIT_SERVICE_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": "lsof -ti:8084 | xargs kill -9 2>/dev/null || true; sleep 1; cd /Users/kevintoles/POC/audit-service && /Users/kevintoles/POC/audit-service/.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8084",
                "timeout": 8.0,
            },
            "context_management": {
                "name": "context-management-service",
                "url": "http://localhost:8086",
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": "cd /Users/kevintoles/POC/context-management-service && source .venv/bin/activate && uvicorn src.main:app --host 0.0.0.0 --port 8086",
                "timeout": 2.0,
            },
            "amve_evaluate_fitness": {
                "name": "amve",
                "url": self.AMVE_SERVICE_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": "cd /Users/kevintoles/POC/architecture-mapping-validation-engine && source .venv/bin/activate && python -m src.main",
                "timeout": 2.0,
            },
            "foundation_search": {
                "name": "unified-search-rs",
                "url": self.UNIFIED_SEARCH_RS_URL,
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": "cd /Users/kevintoles/POC/unified-search-rs && cargo run --release",
                "timeout": 2.0,
            },
            "inference": {
                "name": "inference-service-cpp",
                "url": "http://localhost:8085",
                "health_endpoint": HEALTH_ENDPOINT,
                "restart_command": "cd /Users/kevintoles/POC/inference-service-cpp && ./build/inference-service",
                "timeout": 2.0,
            },
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
