# genai-toolbox Evaluation & Decision

**WBS Reference:** WBS-DEP1.3 — Evaluate toolbox_manager.py  
**AC-D1.9:** Evaluate genai-toolbox and document decision  
**Date:** 2025-02-08  
**Status:** ✅ DECISION: DEPRECATE (not migrate)

---

## Summary

`ai-agents/src/mcp/toolbox_manager.py` wraps the
[genai-toolbox](https://github.com/googleapis/genai-toolbox) Go server
(`:5000`) as an MCP client for Neo4j and Redis access.  
After evaluation, **this component is deprecated** — its capabilities are
fully superseded by direct REST routing through `mcp-gateway`.

---

## Component Analysis

### What toolbox_manager.py Does

| Capability | Implementation |
|-----------|---------------|
| Neo4j Cypher queries | `McpToolboxManager.get_neo4j_toolset()` → genai-toolbox `:5000` |
| Redis key-value ops | `McpToolboxManager.get_redis_toolset()` → genai-toolbox `:5000` |
| Feature-flag gated | `ProtocolFeatureFlags.mcp_toolbox_neo4j/redis` |
| SDK dependency | `toolbox-core` (pip) |

### Why It's Redundant

1. **`graph_query` tool in mcp-gateway** — Routes directly to
   `ai-agents :8082`, which has its own Neo4j client (`src/clients/neo4j_client.py`).
   No genai-toolbox intermediary needed.

2. **Redis access** — ai-agents uses Redis directly via `redis-py`.
   The toolbox Redis wrapper adds latency and an extra Go process.

3. **genai-toolbox server not running** — The Go binary on `:5000` was never
   deployed in the hybrid architecture. The feature flags
   (`mcp_toolbox_neo4j`, `mcp_toolbox_redis`) default to `True` but the
   server doesn't exist, so calls would fail.

4. **Architectural simplification** — Removing the MCP client layer
   (`toolbox-core` → Go server → Neo4j) in favor of direct REST
   (`mcp-gateway` → ai-agents → Neo4j) reduces latency and moving parts.

### Migration Path

| Legacy Path | New Path |
|-------------|----------|
| VS Code → ai-agents MCP → toolbox_manager → genai-toolbox:5000 → Neo4j | VS Code → mcp-gateway:8087 → ai-agents:8082/graph_query → Neo4j |
| VS Code → ai-agents MCP → toolbox_manager → genai-toolbox:5000 → Redis | Direct Redis access via ai-agents internal client |

---

## Decision

**DEPRECATE `toolbox_manager.py`** along with the rest of `src/mcp/`.

- ❌ Do NOT migrate to mcp-gateway
- ❌ Do NOT install genai-toolbox Go binary
- ✅ Remove `toolbox-core` SDK dependency (if present in pyproject.toml)
- ✅ Remove feature flags: `mcp_toolbox_neo4j`, `mcp_toolbox_redis`
  (deferred to Phase 2 — feature flag cleanup is separate work)
- ✅ Graph queries handled by `graph_query` tool via REST
- ✅ Redis handled by ai-agents internal Redis client

---

## Files Affected

| File | Action |
|------|--------|
| `ai-agents/src/mcp/toolbox_manager.py` | DELETE (with entire `src/mcp/`) |
| `ai-agents/src/mcp/__init__.py` | DELETE (exports `McpToolboxManager`) |
| `ai-agents/src/config/feature_flags.py` | Keep for now; flags become no-ops |
| `toolbox-core` in dependencies | Not present in current pyproject.toml |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| genai-toolbox needed later | Low | Medium | Re-add as mcp-gateway tool if needed |
| Neo4j access regression | None | N/A | `graph_query` tool already tested in MCP9 |
| Redis access regression | None | N/A | ai-agents uses redis-py directly |
