"""ASCP.INT.1 — mcp-gateway → validation-service live round-trip.

Covers AC-ASCP.1 (validate_wbs) and AC-ASCP.2 (validate_design_doc):
the MCP tool call traverses the full path:
  in-process MCP server → ToolDispatcher → validation-service HTTP → /validate/wbs

Run with:
    INTEGRATION=1 pytest tests/integration/test_validate_wbs_mcp_roundtrip.py

Both services must be reachable:
  mcp-gateway  — any host, auto-wired via ToolDispatcher
  validation-service — http://localhost:8091 (MCP_GATEWAY_VALIDATION_SERVICE_URL)
"""

from __future__ import annotations

import pytest
from fastmcp import Client

from tests.integration.conftest import _check_backend, _extract_body

pytestmark = pytest.mark.integration

_VALIDATION_SERVICE_HEALTH = "http://localhost:8091/health"

# ── Lifecycle fixture: one server shared across all lifecycle test methods ────
# Intentionally class-scoped so the dispatcher and HTTP client survive across
# multiple requests — tests production-like connection reuse.  Normal functional
# tests use the function-scoped mcp_server from conftest.py instead.


@pytest.fixture(scope="class")
def lifecycle_mcp_server():
    """Shared MCP server for lifecycle tests — same dispatcher across requests."""
    from pathlib import Path
    from src.core.config import Settings
    from src.security.output_sanitizer import OutputSanitizer
    from src.server import create_mcp_server
    from src.tool_dispatcher import ToolDispatcher
    from src.tool_registry import ToolRegistry

    config_path = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"
    if not config_path.exists():
        pytest.skip("config/tools.yaml not found")
    registry = ToolRegistry(config_path)
    dispatcher = ToolDispatcher(Settings())
    sanitizer = OutputSanitizer()
    return create_mcp_server(registry, dispatcher, sanitizer)

# Canonical conformant WBS fixture — mirrors the structure in
# validation-service/rust/src/features/validate_wbs/test_validate_wbs.rs CONFORMANT_WBS.
# Required elements: Risk Profile, Type column in AC table, Requirement Verification,
# Test Matrix, Compliance Controls, Network & Distributed Systems Validation,
# PERRY-MOCK-VALIDATION task, Dockerfile task, RED task for every AC.
_CONFORMANT_WBS = """\
## WBS-INT1: Integration Test Capability
**Dependencies:** none
**Reference:** ASCP-1.0 §4.1
**Risk Profile:** LOW
**Status:** IN PROGRESS

### Acceptance Criteria
| AC ID | Description | Type |
| --- | --- | --- |
| AC-INT1.1 | validate_wbs MCP tool call returns a ValidationResult with verdict field | A |
| AC-INT1.2 | Conformant WBS fixture produces score=100 from validation-service | B |
| AC-INT1.3 | validate_wbs round-trip completes within p95 < 500ms under 10 req/s | C |

### Requirement Verification
N/A - all ACs are independently testable with standard integration tests

### Test Matrix
N/A - single service boundary, covered by integration tests

### Compliance Controls
N/A - internal service, no regulatory requirements

### Network & Distributed Systems Validation
N/A - no external network dependencies in this feature

### WBS Tasks
| ID | Task | AC | File(s) |
| --- | --- | --- | --- |
| INT1.1 | RED: Write failing round-trip test | AC-INT1.1 | tests/integration/test_validate_wbs_mcp_roundtrip.py |
| INT1.2 | GREEN: Wire validate_wbs tool in mcp-gateway | AC-INT1.1 | config/tools.yaml |
| INT1.3 | RED: Write failing score assertion test | AC-INT1.2 | tests/integration/test_validate_wbs_mcp_roundtrip.py |
| INT1.4 | GREEN: Confirm conformant fixture scores 100 | AC-INT1.2 | config/tools.yaml |
| INT1.5 | REFACTOR: Verify schema compliance | AC-INT1.1, AC-INT1.2 | config/tools.yaml |
| INT1.6 | RED: Write failing latency benchmark test | AC-INT1.3 | tests/integration/test_validate_wbs_mcp_roundtrip.py |
| INT1.7 | GREEN: Implement p95 latency assertion | AC-INT1.3 | tests/integration/test_validate_wbs_mcp_roundtrip.py |
| INT1.8 | PERRY-MOCK-VALIDATION: Verify mock validation response matches live service schema | AC-INT1.1 | tests/integration/ |
| INT1.9 | INTEGRATION: Confirm live PASS verdict end-to-end | AC-INT1.1, AC-INT1.2 | tests/integration/test_validate_wbs_mcp_roundtrip.py |
| INT1.10 | INTEGRATION: Add service health check to CI | AC-INT1.1 | src/Dockerfile |

### Exit Criteria
- [x] All validation tests pass
- [x] Code reviewed

### Technical Debt
None.
"""

# Minimal conformant design-doc fixture — structure that validate_design_doc accepts as PASS.
_CONFORMANT_DESIGN_DOC = """\
##### 5.1.9.1a Directory Tree
```text
src/
  validate_wbs.py
  test_validate_wbs.py
```

##### 5.1.9.2 Validator
| Target File(s) | validate_wbs.py → validate_wbs |
| Responsibility | Accept the request |

| AAF Zone | Z1 |
| Anti-Responsibilities | Must not import from another feature folder |

| AAF Zone | Z2 |
| Anti-Responsibilities | Must not import from sibling sub-process or peer folder |

| AAF Zone | Z3 Coordinator |
| Anti-Responsibilities | no logic / no conditions — routing only |

AAF Enforcement Notes
Feature silo maintained. Only shared/ is accessible across features.

Atomicity Validation
| Independently unit-testable | ✓ |

##### 5.1.10.1 Implementation
```pseudo
call check_c1
```

## §4.7 REVIEW Readiness Gate
All checks must pass.

## 5.2.7 Container Specifications
| Dockerfile Path | validation-service/Dockerfile |
"""


# ── AC-ASCP.1: validate_wbs end-to-end ──────────────────────────────────────


class TestValidateWBSMCPRoundTrip:
    """ASCP.INT.1 — AC-ASCP.1: validate_wbs tool returns PASS via live service."""

    async def test_validate_wbs_returns_pass_verdict(self, mcp_server):
        """Full path: MCP client → dispatcher → validation-service → /validate/wbs."""
        if not await _check_backend(_VALIDATION_SERVICE_HEALTH):
            pytest.skip("validation-service not running on :8091")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "validate_wbs",
                {"content": _CONFORMANT_WBS},
            )

        body = _extract_body(result)
        assert body.get("verdict") == "PASS", (
            f"Expected verdict=PASS from /validate/wbs, got: {body}"
        )

    async def test_validate_wbs_response_has_score_100(self, mcp_server):
        """Response body must include score=100 for the conformant WBS fixture.

        This is the live-service half of the ASCP.INT.1 exit milestone:
        POST /validate/wbs returns {"verdict": "PASS", "score": 100}.
        The static half (plumbing correctness) is covered by
        tests/unit/test_validate_wbs_contract.py.
        """
        if not await _check_backend(_VALIDATION_SERVICE_HEALTH):
            pytest.skip("validation-service not running on :8091")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "validate_wbs",
                {"content": _CONFORMANT_WBS},
            )

        body = _extract_body(result)
        assert "score" in body, f"score field missing from response: {body}"
        assert body["score"] == 100, (
            f"score must be 100 for conformant WBS (exit milestone ASCP.INT.1), "
            f"got {body['score']}; failures: {body.get('failures', [])}"
        )

    async def test_validate_wbs_response_has_failures_list(self, mcp_server):
        """Response body must include a failures list (empty for conformant WBS)."""
        if not await _check_backend(_VALIDATION_SERVICE_HEALTH):
            pytest.skip("validation-service not running on :8091")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "validate_wbs",
                {"content": _CONFORMANT_WBS},
            )

        body = _extract_body(result)
        assert "failures" in body, f"failures field missing from response: {body}"
        assert body["failures"] == [], (
            f"Expected no failures for conformant WBS, got: {body['failures']}"
        )


# ── AC-ASCP.2: validate_design_doc end-to-end ───────────────────────────────


class TestValidateDesignDocMCPRoundTrip:
    """ASCP.INT.1 — AC-ASCP.2: validate_design_doc tool returns PASS via live service."""

    async def test_validate_design_doc_returns_pass_verdict(self, mcp_server):
        """Full path: MCP client → dispatcher → validation-service → /validate/design_document."""
        if not await _check_backend(_VALIDATION_SERVICE_HEALTH):
            pytest.skip("validation-service not running on :8091")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "validate_design_doc",
                {"content": _CONFORMANT_DESIGN_DOC, "phase": "DRAFT"},
            )

        body = _extract_body(result)
        assert body.get("verdict") == "PASS", (
            f"Expected verdict=PASS from /validate/design_document, got: {body}"
        )

    async def test_validate_design_doc_response_has_score(self, mcp_server):
        """Response body must include a numeric score field."""
        if not await _check_backend(_VALIDATION_SERVICE_HEALTH):
            pytest.skip("validation-service not running on :8091")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "validate_design_doc",
                {"content": _CONFORMANT_DESIGN_DOC, "phase": "DRAFT"},
            )

        body = _extract_body(result)
        assert "score" in body, f"score field missing from response: {body}"


# ── Production-lifecycle tests: shared dispatcher, shared event loop ──────────
# These tests intentionally share one server/dispatcher/HTTP client across all
# requests to verify production-like connection reuse and sequential throughput.
# They are isolated from the functional tests above by living in their own class
# with a class-scoped fixture.


@pytest.mark.asyncio(loop_scope="class")
class TestValidationServiceLifecycle:
    """Verify the dispatcher handles repeated requests without degradation.

    Uses lifecycle_mcp_server (class-scoped) so the same ToolDispatcher and
    its underlying HTTP client survive across all four requests.  This catches
    connection-pool exhaustion, state leakage, and reuse regressions that
    per-test isolation would hide.
    """

    async def test_repeated_validate_wbs_requests_all_succeed(
        self, lifecycle_mcp_server
    ):
        """Four sequential validate_wbs calls through the same dispatcher all return PASS."""
        if not await _check_backend(_VALIDATION_SERVICE_HEALTH):
            pytest.skip("validation-service not running on :8091")

        results = []
        async with Client(lifecycle_mcp_server) as client:
            for i in range(4):
                result = await client.call_tool(
                    "validate_wbs",
                    {"content": _CONFORMANT_WBS},
                )
                body = _extract_body(result)
                results.append(body)

        assert len(results) == 4, "Expected 4 results"
        for i, body in enumerate(results):
            assert body.get("verdict") == "PASS", (
                f"Request {i + 1}/4 returned non-PASS verdict: {body}"
            )

    async def test_interleaved_wbs_and_design_doc_requests_succeed(
        self, lifecycle_mcp_server
    ):
        """Alternating validate_wbs / validate_design_doc calls through one dispatcher."""
        if not await _check_backend(_VALIDATION_SERVICE_HEALTH):
            pytest.skip("validation-service not running on :8091")

        async with Client(lifecycle_mcp_server) as client:
            wbs_result = await client.call_tool(
                "validate_wbs", {"content": _CONFORMANT_WBS}
            )
            design_result = await client.call_tool(
                "validate_design_doc",
                {"content": _CONFORMANT_DESIGN_DOC, "phase": "DRAFT"},
            )
            wbs_result2 = await client.call_tool(
                "validate_wbs", {"content": _CONFORMANT_WBS}
            )

        assert _extract_body(wbs_result).get("verdict") == "PASS"
        assert _extract_body(design_result).get("verdict") == "PASS"
        assert _extract_body(wbs_result2).get("verdict") == "PASS"
