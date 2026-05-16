# MCP Session Resilience Design

## Problem Statement

**Current Issue**: When mcp-gateway restarts (code reload, crash, deployment), FastMCP's in-memory session store is lost. Clients (VS Code, Claude Desktop, etc.) retain stale session IDs and all subsequent tool calls return 404, appearing to "hang" indefinitely.

**Impact**:
- External users cannot recover without manual intervention (reload IDE)
- Poor user experience - tools appear broken
- No visibility into session state issues
- Breaks production deployments (zero-downtime impossible)

**Root Cause**: FastMCP's SSE transport stores sessions in-memory with no persistence or recovery mechanism.

---

## Solution Architecture: Defense in Depth

### Layer 1: Unknown Session Handler (IMMEDIATE - 2 hours)

**Goal**: Gracefully handle unknown session IDs instead of returning 404.

**Implementation**: Custom middleware that intercepts session 404s and returns actionable error:

```python
# src/middleware/session_recovery.py
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class SessionRecoveryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Intercept 404 on /mcp/messages with session_id
        if (response.status_code == 404
            and request.url.path.startswith("/mcp/messages")
            and "session_id" in request.query_params):

            session_id = request.query_params["session_id"]
            return Response(
                status_code=410,  # Gone - more semantic than 404
                content={
                    "error": "session_expired",
                    "message": f"Session {session_id} no longer exists. Please reconnect via SSE at /mcp/sse",
                    "recovery_endpoint": "/mcp/sse",
                    "session_id": session_id
                },
                media_type="application/json"
            )

        return response
```

**Benefits**:
- ✅ Clear error messaging instead of silent hang
- ✅ Client knows to reconnect
- ✅ No changes to FastMCP internals
- ✅ Works with all MCP clients

**Limitations**:
- ❌ Requires client-side reconnection logic
- ❌ In-flight operations still lost

---

### Layer 2: Session Persistence (SHORT-TERM - 1 day)

**Goal**: Survive server restarts by persisting session state to Redis.

**Implementation**: Custom session store that wraps FastMCP:

```python
# src/middleware/persistent_sessions.py
import json
from typing import Optional
import redis.asyncio as aioredis

class RedisSessionStore:
    """Persist MCP sessions to Redis with 24h TTL."""

    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
        self.ttl = 86400  # 24 hours

    async def save_session(self, session_id: str, session_data: dict):
        """Store session state in Redis."""
        key = f"mcp:session:{session_id}"
        await self.redis.setex(key, self.ttl, json.dumps(session_data))

    async def load_session(self, session_id: str) -> Optional[dict]:
        """Load session state from Redis."""
        key = f"mcp:session:{session_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def delete_session(self, session_id: str):
        """Remove session from Redis."""
        key = f"mcp:session:{session_id}"
        await self.redis.delete(key)

    async def touch_session(self, session_id: str):
        """Extend session TTL on activity."""
        key = f"mcp:session:{session_id}"
        await self.redis.expire(key, self.ttl)
```

**Integration**: Hook into FastMCP's session lifecycle (may require FastMCP patch or fork).

**Benefits**:
- ✅ Sessions survive restarts
- ✅ Multi-instance support (load balancing)
- ✅ Automatic expiration (TTL)

**Limitations**:
- ❌ Requires Redis dependency
- ❌ May need FastMCP modification
- ❌ Adds latency to session operations

---

### Layer 3: Health Check + Auto-Recovery (LONG-TERM - 2 days)

**Goal**: Detect stale sessions and auto-recover without user action.

**Implementation**:

1. **Server-side health endpoint**:
```python
@app.get("/mcp/sessions/{session_id}/health")
async def check_session_health(session_id: str):
    """Check if a session ID is valid."""
    # Query FastMCP's session store
    exists = await mcp_server.session_exists(session_id)
    return {
        "session_id": session_id,
        "valid": exists,
        "server_version": settings.SERVICE_VERSION,
        "restart_count": _restart_counter  # Track restarts
    }
```

2. **Client-side heartbeat** (VS Code extension or Claude config):
```javascript
// Pseudo-code for MCP client
setInterval(async () => {
    const health = await fetch(`/mcp/sessions/${sessionId}/health`);
    if (!health.valid) {
        console.warn("Session expired, reconnecting...");
        await reconnectSSE();
    }
}, 30000);  // Check every 30 seconds
```

**Benefits**:
- ✅ Automatic recovery
- ✅ No user intervention needed
- ✅ Detects server restarts

**Limitations**:
- ❌ Requires client-side implementation
- ❌ 30s window of potential failure
- ❌ Not all MCP clients support custom heartbeat

---

### Layer 4: Stateless Tool Execution (IDEAL - 1 week)

**Goal**: Eliminate session dependency entirely for tool execution.

**Design**: Each tool call is stateless with embedded authentication:

```python
# Before: Session-based (stateful)
POST /mcp/messages/?session_id=abc123
{ "tool": "mirror_cre_repos", "args": {...} }

# After: Token-based (stateless)
POST /mcp/tools/call
Authorization: Bearer <jwt_token>
{ "tool": "mirror_cre_repos", "args": {...} }
```

**Benefits**:
- ✅ No session state to lose
- ✅ Zero-downtime deployments
- ✅ Horizontal scaling trivial
- ✅ Works with any client

**Limitations**:
- ❌ Major architectural change
- ❌ Breaks MCP protocol spec
- ❌ Requires custom client integration

---

### Layer 5: Graceful Shutdown + Session Drain (PRODUCTION - 3 days)

**Goal**: Give sessions time to complete before server shutdown.

**Implementation**:

```python
# src/core/lifecycle.py
import asyncio
import signal
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle graceful startup and shutdown."""

    # Startup
    _restart_counter = await _load_restart_counter()
    yield

    # Shutdown
    logger.info("Received shutdown signal, draining sessions...")

    # 1. Stop accepting new sessions
    await mcp_server.stop_accepting_connections()

    # 2. Wait for active tool calls to complete (max 60s)
    deadline = asyncio.get_event_loop().time() + 60
    while await mcp_server.has_active_calls():
        if asyncio.get_event_loop().time() > deadline:
            logger.warning("Shutdown timeout, forcing termination")
            break
        await asyncio.sleep(1)

    # 3. Persist remaining sessions to Redis
    await mcp_server.persist_all_sessions(_session_store)

    logger.info("Graceful shutdown complete")

app = FastAPI(lifespan=lifespan)
```

**Deployment**:
```bash
# systemd service with proper signals
ExecStop=/bin/kill -SIGTERM $MAINPID
TimeoutStopSec=90
```

**Benefits**:
- ✅ Zero tool call failures during deployment
- ✅ Production-grade reliability
- ✅ Works with rolling updates

**Limitations**:
- ❌ Delays deployments (60s drain time)
- ❌ Requires orchestration support
- ❌ Complex to test

---

## Recommended Implementation Plan

### Phase 1: Immediate (Today)
- ✅ Implement SessionRecoveryMiddleware (Layer 1)
- ✅ Add `/mcp/sessions/{id}/health` endpoint
- ✅ Update error messages to guide users

**Effort**: 2 hours
**Impact**: Users get clear error instead of hang

### Phase 2: Short-term (This Week)
- ✅ Implement RedisSessionStore (Layer 2)
- ✅ Add session persistence hooks
- ✅ Test restart scenarios

**Effort**: 1 day
**Impact**: Sessions survive restarts

### Phase 3: Long-term (Next Sprint)
- ✅ Implement graceful shutdown (Layer 5)
- ✅ Add client heartbeat example
- ✅ Document client integration patterns

**Effort**: 3 days
**Impact**: Production-grade reliability

### Phase 4: Future (Backlog)
- ⏸️ Evaluate stateless tool execution (Layer 4)
- ⏸️ Contribute fixes upstream to FastMCP
- ⏸️ Benchmark session persistence overhead

---

## Testing Strategy

### Unit Tests
```python
# tests/middleware/test_session_recovery.py
async def test_unknown_session_returns_410():
    response = await client.post(
        "/mcp/messages/?session_id=unknown",
        json={"tool": "ask", "args": {}}
    )
    assert response.status_code == 410
    assert response.json()["error"] == "session_expired"
```

### Integration Tests
```python
# tests/integration/test_session_persistence.py
async def test_session_survives_restart():
    # Create session
    session_id = await create_session()

    # Restart server
    await restart_mcp_gateway()

    # Session should still be valid
    response = await client.get(f"/mcp/sessions/{session_id}/health")
    assert response.json()["valid"] is True
```

### Load Tests
```python
# tests/load/test_session_churn.py
async def test_1000_concurrent_sessions():
    """Verify Redis can handle production session load."""
    sessions = [create_session() for _ in range(1000)]
    await asyncio.gather(*sessions)
    # Verify all sessions in Redis
```

---

## Metrics & Monitoring

### Key Metrics
```python
# src/middleware/session_recovery.py
SESSION_RECOVERY_COUNTER = Counter(
    "mcp_session_recovery_total",
    "Number of session recovery attempts",
    ["outcome"]  # success, expired, not_found
)

SESSION_DURATION_HISTOGRAM = Histogram(
    "mcp_session_duration_seconds",
    "Session lifetime from creation to expiry"
)

ACTIVE_SESSIONS_GAUGE = Gauge(
    "mcp_sessions_active",
    "Current number of active sessions"
)
```

### Alerts
- `mcp_session_recovery_total{outcome="expired"} > 10/min` → Server restarting too frequently
- `mcp_sessions_active > 1000` → Potential session leak
- `mcp_session_duration_seconds < 60` → Sessions expiring too quickly

---

## Security Considerations

1. **Session Hijacking**: Use cryptographically secure session IDs (UUID4)
2. **Session Fixation**: Regenerate session ID after authentication changes
3. **Session Expiry**: Enforce TTL even with Redis persistence (24h max)
4. **Redis Security**: Use TLS for Redis connections, ACLs for key access
5. **Audit Logging**: Log all session creation/expiry events

---

## References

- FastMCP docs: https://github.com/jlowin/fastmcp
- MCP Protocol spec: https://spec.modelcontextprotocol.io/
- Redis session patterns: https://redis.io/docs/manual/patterns/session-store/
- Graceful shutdown patterns: https://cloud.google.com/blog/products/containers-kubernetes/kubernetes-best-practices-terminating-with-grace

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-21 | Use Redis for session persistence | Already in platform stack, proven pattern |
| 2026-04-21 | Implement all 5 layers | Defense in depth - each layer catches different failure modes |
| 2026-04-21 | Return 410 instead of 404 | More semantic - 410 = "Gone", 404 = "Never existed" |
| 2026-04-21 | 24h session TTL | Balances UX (long sessions) with security (auto-expiry) |
