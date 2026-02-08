# DEP1 Migration Log — Legacy MCP Server Removal

**WBS Reference:** WBS-DEP1 — Legacy MCP Server Migration & Deprecation  
**Date:** 2025-02-08  
**Status:** ✅ COMPLETE

---

## Summary

Removed the legacy MCP server module from `ai-agents` (:8082).  
All MCP tool capabilities are now served by `mcp-gateway` (:8087).

---

## Changes Made

### ai-agents (`:8082`)

| Step | Action | Details |
|------|--------|---------|
| DEP1.4 | **Modified** `src/main.py` | Removed `from src.mcp.agent_functions_server import create_agent_functions_mcp_server`, removed `ProtocolFeatureFlags` import, removed `_init_mcp_server()` and `_close_mcp_server()` functions, removed MCP init/shutdown from lifespan |
| DEP1.5 | **Deleted** `src/mcp/` | 5 files (~1,348 lines): `server.py` (820), `agent_functions_server.py` (233), `semantic_search_wrapper.py` (140), `toolbox_manager.py` (145), `__init__.py` (10) |
| DEP1.6 | **Archived** MCP tests | 7 test files (67 tests) moved to `tests/_archived_mcp/`: `test_server.py`, `test_agent_functions_server.py`, `test_mcp_server_unit.py`, `test_mcp_lifecycle.py`, `test_mcp_server.py`, `test_mcp_client.py`, `test_kitchen_brigade_e2e.py` |
| DEP1.9 | **Deleted** `AGENTS_MCP_SERVER_ENABLED:n` | Stale MCP config flag file |

### VS Code MCP Config (`~/.vscode/mcp.json`)

| Step | Action | Details |
|------|--------|---------|
| DEP1.8 | **Added** `mcp-gateway` entry | SSE transport pointing to `http://localhost:8087/mcp`. Existing `ai-kitchen-br` entry preserved. |

### mcp-gateway (`:8087`)

| Step | Action | Details |
|------|--------|---------|
| DEP1.1 | **Created** `tests/migration/test_tool_parity.py` | 14 passing unit tests + 4 skipped live tests verifying all legacy tools mapped to gateway |
| DEP1.2 | **Created** `tests/migration/test_rest_endpoints.py` | 5 passing unit tests + 9 skipped live tests verifying REST endpoint independence |
| DEP1.3 | **Created** `docs/migration/GENAI_TOOLBOX_DECISION.md` | Decision doc: DEPRECATE genai-toolbox (never deployed) |
| DEP1.12 | **Created** `docs/migration/MIGRATION_LOG.md` | This document |

---

## Files Removed

```
ai-agents/
├── AGENTS_MCP_SERVER_ENABLED:n          (deleted)
└── src/mcp/
    ├── __init__.py                       (deleted)
    ├── agent_functions_server.py         (deleted)
    ├── semantic_search_wrapper.py        (deleted)
    ├── server.py                         (deleted)
    └── toolbox_manager.py                (deleted)
```

## Files Archived

```
ai-agents/tests/_archived_mcp/
├── test_kitchen_brigade_e2e.py
├── test_mcp_client.py
├── test_mcp_lifecycle.py
├── test_mcp_server_unit.py
├── test_mcp_server.py
└── unit_mcp/
    ├── __init__.py
    ├── test_agent_functions_server.py
    └── test_server.py
```

## Files Modified

| File | Changes |
|------|---------|
| `ai-agents/src/main.py` | Removed MCP import, init, close functions, lifespan calls |
| `~/.vscode/mcp.json` | Added `mcp-gateway` SSE server entry |

## Files Created

| File | Purpose |
|------|---------|
| `mcp-gateway/tests/migration/__init__.py` | Migration test package |
| `mcp-gateway/tests/migration/test_tool_parity.py` | Tool coverage verification (18 tests) |
| `mcp-gateway/tests/migration/test_rest_endpoints.py` | REST endpoint independence (14 tests) |
| `mcp-gateway/docs/migration/GENAI_TOOLBOX_DECISION.md` | genai-toolbox deprecation decision |
| `mcp-gateway/docs/migration/MIGRATION_LOG.md` | This migration log |

---

## Tool Mapping

| Legacy Tool (ai-agents MCP) | Gateway Tool (mcp-gateway) | Route |
|------------------------------|---------------------------|-------|
| `cross_reference` | `run_agent_function` | → POST :8082/v1/functions/cross-reference/run |
| `analyze_code` | `code_analyze` | → POST :8083/v1/analyze |
| `generate_code` | `run_agent_function` | → POST :8082/v1/functions/generate-code/run |
| `explain_code` | `run_agent_function` | → POST :8082/v1/functions/* |
| `extract_structure` | `run_agent_function` | → POST :8082/v1/functions/extract-structure/run |
| `summarize_content` | `run_agent_function` | → POST :8082/v1/functions/summarize-content/run |
| `analyze_artifact` | `run_agent_function` | → POST :8082/v1/functions/analyze-artifact/run |
| `validate_against_spec` | `run_agent_function` | → POST :8082/v1/functions/validate-against-spec/run |
| `decompose_task` | `run_agent_function` | → POST :8082/v1/functions/decompose-task/run |
| `synthesize_outputs` | `run_agent_function` | → POST :8082/v1/functions/synthesize-outputs/run |

**Additional gateway tools** (no legacy equivalent):
- `semantic_search` — Direct search via :8081
- `hybrid_search` — Hybrid search via :8081
- `code_pattern_audit` — Anti-pattern detection via :8083
- `graph_query` — Neo4j Cypher via :8082
- `llm_complete` — LLM completion via :8080
- `run_discussion` — Kitchen Brigade protocol via :8082
- `agent_execute` — Autonomous agent execution via :8082

---

## Verification Results

### mcp-gateway tests
```
512 passed, 13 skipped — 98.64% coverage
```

### ai-agents tests (post-removal)
```
2478 passed, 39 failed (pre-existing), 8 skipped
No MCP-related failures — all 39 failures are pre-existing issues.
```

### What Was NOT Changed
- `ai-agents/src/config/feature_flags.py` — MCP flags kept (become no-ops); cleanup deferred to Phase 2
- `ai-agents/pyproject.toml` — No MCP-specific dependencies to remove (server was dict-based)
- `ai-kitchen-br` MCP server in VS Code config — preserved (production Kitchen Brigade tools)

---

## Rollback Plan

If rollback is needed:
1. `git checkout -- src/mcp/` to restore the module
2. `git checkout -- src/main.py` to restore MCP lifecycle
3. Move `tests/_archived_mcp/*` back to original locations
4. Restore `AGENTS_MCP_SERVER_ENABLED:n` from git
