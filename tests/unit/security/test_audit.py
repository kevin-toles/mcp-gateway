"""Audit logging tests — WBS-MCP7 (RED).

Covers AC-7.1 (all fields), AC-7.2 (JSONL format), AC-7.3 (no secrets),
AC-7.5 (security events), AC-7.6 (request_id), AC-7.7 (audit-service forwarding),
AC-7.8 (source_ip, actor_sub, tokens_consumed, parent_request_id, agent_depth, security_flags).
"""

import hashlib
import json
from dataclasses import asdict
from unittest.mock import AsyncMock, patch

import pytest

from src.security.audit import (
    AuditEntry,
    AuditMiddleware,
    AuditServiceForwarder,
    SecuritySeverity,
    create_audit_entry,
    hash_input,
    log_security_event,
)


# ── AC-7.1: AuditEntry contains all required fields ─────────────────────


class TestAuditEntryFields:
    """AuditEntry has all 15+ required fields."""

    def test_all_who_fields_present(self) -> None:
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="user@example.com", tier="gold",
            source_ip="10.0.0.1", tool="semantic_search",
            input_data={"query": "test"}, status="success", status_code=200,
            latency_ms=42.0, request_id="req-123",
        )
        assert entry.tenant_id == "t1"
        assert entry.actor_sub == "user@example.com"
        assert entry.tier == "gold"
        assert entry.source_ip == "10.0.0.1"

    def test_all_what_fields_present(self) -> None:
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="graph_query", input_data={"cypher": "MATCH (n) RETURN n"},
            status="success", status_code=200, latency_ms=10.0,
            request_id="req-456",
        )
        assert entry.tool == "graph_query"
        assert entry.input_hash  # non-empty SHA-256
        assert len(entry.input_hash) == 64  # hex SHA-256
        assert entry.input_summary  # truncated summary

    def test_all_when_fields_present(self) -> None:
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="llm_complete", input_data={"prompt": "hello"},
            status="success", status_code=200, latency_ms=150.5,
            request_id="req-789",
        )
        assert entry.timestamp  # ISO 8601
        assert "T" in entry.timestamp  # ISO 8601 format
        assert entry.latency_ms == 150.5

    def test_all_outcome_fields_present(self) -> None:
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="code_analyze", input_data={"code": "x=1"},
            status="error", status_code=500, latency_ms=5.0,
            request_id="req-000", error_code="INTERNAL_ERROR",
            tokens_consumed=42,
        )
        assert entry.status == "error"
        assert entry.status_code == 500
        assert entry.error_code == "INTERNAL_ERROR"
        assert entry.tokens_consumed == 42

    def test_all_provenance_fields_present(self) -> None:
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="run_discussion", input_data={"topic": "arch"},
            status="success", status_code=200, latency_ms=1000.0,
            request_id="req-parent", parent_request_id="req-grandparent",
            agent_depth=2,
        )
        assert entry.request_id == "req-parent"
        assert entry.parent_request_id == "req-grandparent"
        assert entry.agent_depth == 2

    def test_security_flags_field(self) -> None:
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="graph_query", input_data={"cypher": "DROP TABLE"},
            status="denied", status_code=403, latency_ms=1.0,
            request_id="req-sec",
            security_flags=["injection_pattern_detected"],
        )
        assert entry.security_flags == ["injection_pattern_detected"]

    def test_defaults_for_optional_fields(self) -> None:
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="semantic_search", input_data={"query": "x"},
            status="success", status_code=200, latency_ms=5.0,
            request_id="req-def",
        )
        assert entry.error_code is None
        assert entry.parent_request_id is None
        assert entry.agent_depth == 0
        assert entry.tokens_consumed == 0
        assert entry.security_flags == []


# ── AC-7.2: JSONL format ─────────────────────────────────────────────────


class TestAuditJSONL:
    """Audit entries serialize to valid JSONL."""

    def test_to_json_returns_valid_json(self) -> None:
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="semantic_search", input_data={"query": "hello"},
            status="success", status_code=200, latency_ms=10.0,
            request_id="req-json",
        )
        line = entry.to_json()
        parsed = json.loads(line)
        assert parsed["tool"] == "semantic_search"

    def test_to_json_single_line(self) -> None:
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="llm_complete", input_data={"prompt": "multi\nline"},
            status="success", status_code=200, latency_ms=5.0,
            request_id="req-sl",
        )
        line = entry.to_json()
        assert "\n" not in line

    def test_asdict_roundtrip(self) -> None:
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="hybrid_search", input_data={"query": "test"},
            status="success", status_code=200, latency_ms=8.0,
            request_id="req-rt",
        )
        d = asdict(entry)
        assert isinstance(d, dict)
        assert d["tool"] == "hybrid_search"


# ── AC-7.3: No secrets in audit ──────────────────────────────────────────


class TestNoSecrets:
    """Audit entries never contain raw secrets."""

    def test_input_hashed_not_raw(self) -> None:
        secret_input = {"api_key": "sk-super-secret-key-12345"}
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="llm_complete", input_data=secret_input,
            status="success", status_code=200, latency_ms=5.0,
            request_id="req-sec",
        )
        line = entry.to_json()
        assert "sk-super-secret-key-12345" not in line
        assert entry.input_hash == hashlib.sha256(
            json.dumps(secret_input, sort_keys=True).encode()
        ).hexdigest()

    def test_jwt_not_in_json(self) -> None:
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="semantic_search",
            input_data={"token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"},
            status="success", status_code=200, latency_ms=5.0,
            request_id="req-jwt",
        )
        line = entry.to_json()
        assert "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9" not in line

    def test_input_summary_truncated(self) -> None:
        long_input = {"prompt": "x" * 10000}
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="llm_complete", input_data=long_input,
            status="success", status_code=200, latency_ms=5.0,
            request_id="req-trunc",
        )
        assert len(entry.input_summary) <= 200

    def test_input_summary_truncated_many_keys(self) -> None:
        """Summary truncates when there are many keys."""
        big_input = {f"key_{i:04d}": "v" for i in range(100)}
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="llm_complete", input_data=big_input,
            status="success", status_code=200, latency_ms=5.0,
            request_id="req-trunc2",
        )
        assert len(entry.input_summary) <= 200


# ── AC-7.3 (continued): hash_input utility ────────────────────────────


class TestHashInput:
    """SHA-256 hashing of input data."""

    def test_deterministic_hash(self) -> None:
        data = {"query": "hello world"}
        assert hash_input(data) == hash_input(data)

    def test_different_inputs_different_hashes(self) -> None:
        assert hash_input({"a": 1}) != hash_input({"a": 2})

    def test_returns_hex_sha256(self) -> None:
        h = hash_input({"x": "y"})
        assert len(h) == 64
        int(h, 16)  # valid hex


# ── AC-7.5: Security events with severity ───────────────────────────────


class TestSecurityEvents:
    """Security events logged with severity levels."""

    def test_severity_enum_values(self) -> None:
        assert SecuritySeverity.LOW.value == "low"
        assert SecuritySeverity.MEDIUM.value == "medium"
        assert SecuritySeverity.HIGH.value == "high"
        assert SecuritySeverity.CRITICAL.value == "critical"

    def test_log_security_event_emits_warning(self, caplog) -> None:
        import logging
        with caplog.at_level(logging.WARNING, logger="mcp_gateway.security"):
            log_security_event(
                event_type="cypher_injection",
                severity=SecuritySeverity.HIGH,
                detail="CREATE keyword blocked",
                request_id="req-001",
            )
        assert any("SECURITY_EVENT" in r.message for r in caplog.records)
        assert any("cypher_injection" in r.message for r in caplog.records)
        assert any("high" in r.message for r in caplog.records)

    def test_log_security_event_critical_uses_error_level(self, caplog) -> None:
        import logging
        with caplog.at_level(logging.ERROR, logger="mcp_gateway.security"):
            log_security_event(
                event_type="brute_force",
                severity=SecuritySeverity.CRITICAL,
                detail="multiple auth failures",
                request_id="req-002",
            )
        assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ── AC-7.7: AuditServiceForwarder ──────────────────────────────────────


class TestAuditServiceForwarder:
    """Forwarding audit entries to audit-service :8084."""

    @pytest.mark.asyncio
    async def test_forwards_entry_via_http_post(self) -> None:
        forwarder = AuditServiceForwarder(audit_service_url="http://localhost:8084")
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="semantic_search", input_data={"query": "test"},
            status="success", status_code=200, latency_ms=10.0,
            request_id="req-fwd",
        )
        with patch("src.security.audit.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=AsyncMock(status_code=201))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client
            await forwarder.forward(entry)
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_fallback_on_connection_error(self, tmp_path) -> None:
        forwarder = AuditServiceForwarder(
            audit_service_url="http://localhost:8084",
            fallback_path=str(tmp_path / "fallback.jsonl"),
        )
        entry = create_audit_entry(
            tenant_id="t1", actor_sub="u1", tier="free", source_ip="1.2.3.4",
            tool="graph_query", input_data={"cypher": "MATCH (n) RETURN n"},
            status="success", status_code=200, latency_ms=5.0,
            request_id="req-fb",
        )
        with patch("src.security.audit.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=ConnectionError("no audit svc"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client
            await forwarder.forward(entry)  # should not raise
        # Verify fallback JSONL was written
        fallback_file = tmp_path / "fallback.jsonl"
        assert fallback_file.exists()
        line = fallback_file.read_text().strip()
        parsed = json.loads(line)
        assert parsed["tool"] == "graph_query"


# ── AC-7.2: AuditMiddleware writes JSONL ───────────────────────────────


class TestAuditMiddleware:
    """AuditMiddleware integration behavior."""

    def test_middleware_importable(self) -> None:
        assert AuditMiddleware is not None

    def test_writes_jsonl_on_request(self, tmp_path) -> None:
        from fastapi import FastAPI, Request
        from starlette.testclient import TestClient

        log_path = tmp_path / "audit.jsonl"
        app = FastAPI()
        app.add_middleware(AuditMiddleware, log_path=str(log_path))

        @app.get("/test")
        async def _test(request: Request):
            return {"ok": True}

        client = TestClient(app)
        client.get("/test")
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) >= 1
        parsed = json.loads(lines[0])
        assert "timestamp" in parsed
        assert "latency_ms" in parsed

    def test_health_not_audited(self, tmp_path) -> None:
        from fastapi import FastAPI, Request
        from starlette.testclient import TestClient

        log_path = tmp_path / "audit.jsonl"
        app = FastAPI()
        app.add_middleware(AuditMiddleware, log_path=str(log_path))

        @app.get("/health")
        async def _health(request: Request):
            return {"status": "ok"}

        client = TestClient(app)
        client.get("/health")
        # Health should not produce audit entries
        if log_path.exists():
            assert log_path.read_text().strip() == ""
