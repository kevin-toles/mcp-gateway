"""AuditMiddleware and audit logging — WBS-MCP7 (GREEN).

Provides ``AuditEntry`` dataclass with JSON serialization, SHA-256 input
hashing (no raw secrets), security event logging with severity levels,
``AuditServiceForwarder`` for dispatch to audit-service :8084 with
graceful JSONL fallback, and ``AuditMiddleware`` for JSONL request logging.

Reference: Strategy §7.1 Controls #9, #10, §7.4 AuditRecord,
           §8 COMPLIANCE.AUDIT_LOGGING.COMPREHENSIVE_LOGGING
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_security_logger = logging.getLogger("mcp_gateway.security")
_audit_logger = logging.getLogger("mcp_gateway.audit")

# Paths excluded from audit logging
_EXCLUDED_PATHS: set[str] = {"/health", "/health/"}

# SSE/streaming paths incompatible with BaseHTTPMiddleware (breaks streaming)
_SSE_PREFIX: str = "/mcp"

# Maximum length for input_summary
_MAX_SUMMARY_LEN: int = 200


# ── SecuritySeverity ────────────────────────────────────────────────────


class SecuritySeverity(enum.Enum):
    """Severity levels for security events (AC-7.5)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── hash_input ──────────────────────────────────────────────────────────


def hash_input(data: dict) -> str:
    """SHA-256 hex digest of *data* (AC-7.3 — no raw input in logs)."""
    canonical = json.dumps(data, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _truncate_summary(data: dict) -> str:
    """Produce a truncated human-readable summary of input data.

    Only includes keys (not values) to avoid leaking secrets (AC-7.3).
    """
    keys = list(data.keys())
    raw = f"keys={keys}"
    if len(raw) > _MAX_SUMMARY_LEN:
        return raw[: _MAX_SUMMARY_LEN - 3] + "..."
    return raw


# ── AuditEntry ──────────────────────────────────────────────────────────


@dataclass
class AuditEntry:
    """Structured audit record matching strategy §7.4 AuditRecord (AC-7.8)."""

    # Who
    tenant_id: str = ""
    actor_sub: str = ""
    tier: str = ""
    source_ip: str = ""

    # What
    tool: str = ""
    input_hash: str = ""
    input_summary: str = ""

    # When
    timestamp: str = ""
    latency_ms: float = 0.0

    # Outcome
    status: str = ""
    status_code: int = 0
    error_code: str | None = None
    tokens_consumed: int = 0

    # Provenance
    request_id: str = ""
    parent_request_id: str | None = None
    agent_depth: int = 0

    # Security
    security_flags: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to a single-line JSON string (JSONL-safe)."""
        return json.dumps(asdict(self), separators=(",", ":"), default=str)



@dataclass
class AuditContext:
    """Provenance and security context for audit entries.

    Groups tracing/provenance parameters to reduce function parameter count.
    """

    request_id: str = ""
    parent_request_id: str | None = None
    agent_depth: int = 0
    security_flags: list[str] = field(default_factory=list)


def create_audit_entry(
    *,
    tenant_id: str,
    actor_sub: str,
    tier: str,
    source_ip: str,
    tool: str,
    input_data: dict,
    status: str,
    status_code: int,
    latency_ms: float,
    error_code: str | None = None,
    tokens_consumed: int = 0,
    context: AuditContext | None = None,
) -> AuditEntry:
    """Factory for ``AuditEntry`` with auto-hashing and timestamp."""
    ctx = context or AuditContext()
    return AuditEntry(
        tenant_id=tenant_id,
        actor_sub=actor_sub,
        tier=tier,
        source_ip=source_ip,
        tool=tool,
        input_hash=hash_input(input_data),
        input_summary=_truncate_summary(input_data),
        timestamp=datetime.now(UTC).isoformat(),
        latency_ms=latency_ms,
        status=status,
        status_code=status_code,
        error_code=error_code,
        tokens_consumed=tokens_consumed,
        request_id=ctx.request_id,
        parent_request_id=ctx.parent_request_id,
        agent_depth=ctx.agent_depth,
        security_flags=ctx.security_flags,
    )


# ── Security event logging ──────────────────────────────────────────────


def log_security_event(
    event_type: str,
    severity: SecuritySeverity,
    detail: str,
    request_id: str = "",
) -> None:
    """Log a security event with severity (AC-7.5).

    CRITICAL severity logs at ERROR level; others at WARNING.
    """
    msg = f"SECURITY_EVENT event={event_type} severity={severity.value} detail='{detail}' request_id={request_id}"
    if severity == SecuritySeverity.CRITICAL:
        _security_logger.error(msg)
    else:
        _security_logger.warning(msg)


# ── AuditServiceForwarder ──────────────────────────────────────────────


class AuditServiceForwarder:
    """Forward audit entries to centralized audit-service (AC-7.7)."""

    def __init__(
        self,
        audit_service_url: str = "http://localhost:8084",
        fallback_path: str = "logs/audit_fallback.jsonl",
    ) -> None:
        self.audit_service_url = audit_service_url.rstrip("/")
        self.fallback_path = fallback_path

    async def forward(self, entry: AuditEntry) -> None:
        """POST entry to audit-service; fall back to JSONL on failure."""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.audit_service_url}/audit/entries",
                    content=entry.to_json(),
                    headers={"Content-Type": "application/json"},
                    timeout=5.0,
                )
        except Exception:
            _audit_logger.warning("Audit-service unavailable — writing to fallback JSONL")
            await self._write_fallback(entry)

    async def _write_fallback(self, entry: AuditEntry) -> None:
        """Append entry to local JSONL fallback file."""
        path = Path(self.fallback_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "a") as f:
            await f.write(entry.to_json() + "\n")


# ── AuditMiddleware ─────────────────────────────────────────────────────


class AuditMiddleware(BaseHTTPMiddleware):
    """Logs every non-health request as a JSONL audit entry (AC-7.1, AC-7.2)."""

    def __init__(self, app: Any, log_path: str = "logs/audit.jsonl") -> None:
        super().__init__(app)
        self.log_path = log_path

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.url.path in _EXCLUDED_PATHS or request.url.path.startswith(_SSE_PREFIX):
            return await call_next(request)

        start = time.monotonic()
        response = await call_next(request)
        latency_ms = (time.monotonic() - start) * 1000

        # Extract request context
        request_id = response.headers.get("x-request-id", "")
        source_ip = request.client.host if request.client else "unknown"

        entry = create_audit_entry(
            tenant_id=request.headers.get("X-Tenant-ID", "anonymous"),
            actor_sub=getattr(getattr(request.state, "auth", None), "sub", "anonymous"),
            tier=getattr(getattr(request.state, "auth", None), "tier", "unknown"),
            source_ip=source_ip,
            tool=request.url.path,
            input_data=dict(request.query_params),
            status="success" if response.status_code < 400 else "error",
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
            context=AuditContext(request_id=request_id),
        )

        # Write to JSONL
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "a") as f:
            await f.write(entry.to_json() + "\n")

        return response
