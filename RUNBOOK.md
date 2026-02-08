# mcp-gateway Operational Runbook

## Table of Contents

1. [Service Overview](#service-overview)
2. [Startup](#startup)
3. [Health Check](#health-check)
4. [Configuration](#configuration)
5. [Dependencies](#dependencies)
6. [Monitoring & Observability](#monitoring--observability)
7. [Troubleshooting](#troubleshooting)
8. [Security Operations](#security-operations)
9. [Testing](#testing)
10. [Shutdown](#shutdown)

---

## Service Overview

| Field | Value |
|-------|-------|
| **Service** | mcp-gateway |
| **Port** | 8087 |
| **Protocol** | MCP (SSE/HTTP transport) + REST |
| **Runtime** | Python 3.11+ / FastAPI / uvicorn |
| **Tools** | 9 (semantic_search, hybrid_search, code_analyze, code_pattern_audit, graph_query, llm_complete, run_agent_function, run_discussion, agent_execute) |

---

## Startup

### Prerequisites

1. Python 3.11+ with virtual environment
2. Redis on `localhost:6379` (rate limiting)
3. Backend services running on `:8080-:8083`
4. `config/tools.yaml` present (MCP tool definitions)

### Standard Startup

```bash
cd /Users/kevintoles/POC/mcp-gateway
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8087
```

### Startup with TLS

```bash
MCP_GATEWAY_TLS_ENABLED=true \
MCP_GATEWAY_TLS_CERT_PATH=/path/to/cert.pem \
MCP_GATEWAY_TLS_KEY_PATH=/path/to/key.pem \
uvicorn src.main:app --host 0.0.0.0 --port 8087
```

### Startup with Authentication

```bash
MCP_GATEWAY_AUTH_ENABLED=true \
MCP_GATEWAY_OIDC_JWKS_URL=https://auth.example.com/.well-known/jwks.json \
MCP_GATEWAY_OIDC_ISSUER=https://auth.example.com \
MCP_GATEWAY_OIDC_AUDIENCE=ai-platform-tools \
uvicorn src.main:app --host 0.0.0.0 --port 8087
```

### Verify Startup

```bash
curl -s http://localhost:8087/health | python3 -m json.tool
```

Expected:
```json
{
    "service": "mcp-gateway",
    "version": "0.1.0",
    "status": "healthy",
    "uptime_seconds": 1.23
}
```

---

## Health Check

### Gateway Health

```bash
curl -sf http://localhost:8087/health && echo "✓ Gateway healthy" || echo "✗ Gateway down"
```

### Backend Service Health

```bash
echo "=== Backend Services ==="
for port in 8080 8081 8082 8083; do
    echo -n "  Port $port: "
    curl -sf http://localhost:$port/health > /dev/null && echo "✓" || echo "✗"
done
```

### Redis Health

```bash
redis-cli -h localhost -p 6379 ping
```

### Full Platform Health

```bash
echo "=== mcp-gateway ==="
curl -sf http://localhost:8087/health && echo " ✓" || echo " ✗"

echo "=== Redis ==="
redis-cli -h localhost -p 6379 ping 2>/dev/null && echo "  ✓" || echo "  ✗"

echo "=== Backends ==="
for svc in "llm-gateway:8080" "semantic-search:8081" "ai-agents:8082" "code-orchestrator:8083" "audit-service:8084"; do
    name=${svc%%:*}
    port=${svc##*:}
    echo -n "  $name ($port): "
    curl -sf http://localhost:$port/health > /dev/null && echo "✓" || echo "✗"
done
```

---

## Configuration

See `README.md` for full configuration reference. Key environment variables:

| Variable | Description | When to Change |
|----------|-------------|----------------|
| `MCP_GATEWAY_AUTH_ENABLED` | JWT authentication | Enable for production |
| `MCP_GATEWAY_RATE_LIMIT_RPM` | Rate limit (req/min) | Tune for workload |
| `MCP_GATEWAY_REDIS_URL` | Redis connection | Non-default Redis |
| `MCP_GATEWAY_TLS_ENABLED` | TLS termination | Production deployment |
| `MCP_GATEWAY_AUDIT_LOG_PATH` | Audit log location | Custom log directory |

---

## Dependencies

### Infrastructure Dependencies

| Service | Port | Purpose | Impact if Down |
|---------|------|---------|----------------|
| Redis | 6379 | Rate limiting | Rate limiting disabled (allows all) |
| Neo4j | 7474/7687 | Knowledge graph (via ai-agents) | graph_query fails |
| Qdrant | 6333 | Vector store (via semantic-search) | semantic_search/hybrid_search fail |

### Service Dependencies

| Service | Port | Tools Routed | Impact if Down |
|---------|------|-------------|----------------|
| llm-gateway | 8080 | llm_complete | LLM completions fail |
| semantic-search | 8081 | semantic_search, hybrid_search | Search tools fail |
| ai-agents | 8082 | graph_query, run_agent_function, run_discussion, agent_execute | Agent tools fail |
| code-orchestrator | 8083 | code_analyze, code_pattern_audit | Code analysis fails |
| audit-service | 8084 | (audit forwarding) | Falls back to local JSONL |

---

## Monitoring & Observability

### Response Headers

Every response includes:
- `X-Request-ID` — unique request identifier (UUID v4)
- `X-RateLimit-Limit` — configured RPM
- `X-RateLimit-Remaining` — remaining requests in window
- `X-RateLimit-Reset` — window reset timestamp (epoch)

### Audit Log

Audit entries are written to JSONL at `MCP_GATEWAY_AUDIT_LOG_PATH` (default: `logs/audit.jsonl`).

```bash
# Tail audit log
tail -f logs/audit.jsonl | python3 -m json.tool

# Count requests by tool
cat logs/audit.jsonl | python3 -c "
import json, sys, collections
tools = collections.Counter()
for line in sys.stdin:
    entry = json.loads(line)
    tools[entry.get('tool', 'unknown')] += 1
for tool, count in tools.most_common():
    print(f'  {tool}: {count}')
"

# Find slow requests (>1s)
cat logs/audit.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    entry = json.loads(line)
    if entry.get('latency_ms', 0) > 1000:
        print(f\"{entry['tool']}: {entry['latency_ms']:.0f}ms ({entry['request_id']})\")
"
```

### Structured Logs

Application logs use Python's `logging` module:
- `mcp_gateway.security` — rate limiting, auth events
- `mcp_gateway.audit` — audit forwarding failures

---

## Troubleshooting

### Gateway Won't Start

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Address already in use` | Port 8087 occupied | `lsof -ti:8087 \| xargs kill -9` |
| `ModuleNotFoundError` | Venv not activated | `source .venv/bin/activate` |
| `config/tools.yaml not found` | Missing config | MCP server won't mount (warning logged) |
| `ImportError: fastmcp` | Missing dependency | `pip install -e ".[dev]"` |

### Tools Not Working

| Symptom | Cause | Fix |
|---------|-------|-----|
| `BackendUnavailableError` | Backend service down | Start the missing service |
| `ToolTimeoutError` | Backend too slow | Check backend health, increase timeout |
| `404 Not Found` | Wrong API path | Verify backend routes match dispatcher |
| `CypherValidationError` | Injection detected | Review Cypher query (write ops blocked) |
| `PathValidationError` | Path traversal detected | File path is outside allowed roots |

### Rate Limiting Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| All requests get 429 | Redis has stale keys | `redis-cli FLUSHDB` or wait for window reset |
| Rate limiting not working | Redis unavailable | Check Redis connection, restart Redis |
| Wrong limit applied | Misconfigured RPM | Set `MCP_GATEWAY_RATE_LIMIT_RPM` |

### Authentication Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `401 Unauthorized` | Missing/invalid JWT | Check Bearer token in Authorization header |
| `Invalid algorithm` | Using HS256/HS384 | Only RS256 and ES256 are accepted |
| `Missing required claims` | Incomplete JWT | Ensure exp, iss, aud, sub, tier are present |
| Auth works in dev, fails in prod | `AUTH_ENABLED=false` in dev | Enable auth for production |

### Audit Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| No audit log | Directory doesn't exist | Middleware creates it automatically |
| Audit forwarding fails | audit-service down | Falls back to local JSONL (check logs) |
| Missing entries | `/health` excluded by design | Health endpoints are not audited |

---

## Security Operations

### Rotate JWT Keys

1. Update JWKS endpoint with new keys
2. Keep old keys active for token expiry period
3. Verify with: `curl -s $MCP_GATEWAY_OIDC_JWKS_URL | python3 -m json.tool`

### Review Audit Trail

```bash
# Last 10 audit entries
tail -10 logs/audit.jsonl | python3 -m json.tool

# Search by request ID
grep "req-12345" logs/audit.jsonl | python3 -m json.tool

# Security events only
grep "security_flags" logs/audit.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    e = json.loads(line)
    if e.get('security_flags'):
        print(json.dumps(e, indent=2))
"
```

### Clear Rate Limit State

```bash
# Clear all rate limit keys
redis-cli KEYS "ratelimit:*" | xargs redis-cli DEL

# Clear for specific tenant
redis-cli KEYS "ratelimit:tenant-123:*" | xargs redis-cli DEL
```

---

## Testing

### Run Unit Tests

```bash
pytest tests/unit/
# Expected: 493 tests, 98.64% coverage
```

### Run Integration Tests

```bash
# Requires: Redis + backend services running
INTEGRATION=1 pytest tests/integration/ --no-cov -v
# Expected: 112 tests (some may skip if backends unavailable)
```

### Run Security Tests Only

```bash
INTEGRATION=1 pytest tests/integration/test_security_suite.py --no-cov -v
```

### Run Performance Benchmarks

```bash
# Requires: mcp-gateway running on :8087
INTEGRATION=1 pytest tests/integration/test_performance.py --no-cov -v -s
```

---

## Shutdown

### Graceful Shutdown

```bash
# Send SIGTERM to uvicorn
kill $(lsof -ti:8087) 2>/dev/null

# Or Ctrl+C in the terminal running uvicorn
```

### Force Shutdown

```bash
lsof -ti:8087 | xargs kill -9
```
