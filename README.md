# mcp-gateway

Standalone MCP-compliant API gateway for the AI Platform.

Exposes 9 platform tools (semantic search, code analysis, LLM completion, agent execution, etc.) to VS Code/Copilot, external LLMs (Claude, GPT, Gemini), and in Phase 3, external tenant agents — all through the Model Context Protocol (MCP).

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run unit tests (no services required)
pytest tests/unit/

# Run integration tests (requires live services + Redis)
INTEGRATION=1 pytest tests/integration/ --no-cov

# Start the gateway
uvicorn src.main:app --host 0.0.0.0 --port 8087
```

## Architecture

```
External Client (VS Code / Claude / GPT)
    │
    ▼
mcp-gateway :8087 — FastMCP SSE/HTTP
    │
    ├── RequestID middleware (UUID v4)
    ├── OIDCAuthMiddleware (JWT RS256/ES256)
    ├── RateLimitMiddleware (Redis token bucket)
    ├── ToolRegistry (YAML-driven, 9 tools)
    ├── Input Validation (Pydantic per tool)
    ├── Injection Prevention (path traversal + Cypher)
    ├── ToolDispatcher (HTTP POST to backends, connection pooling)
    ├── OutputSanitizer (Phase 1: passthrough)
    └── AuditMiddleware (JSONL provenance + :8084 forwarding)
    │
    ▼
Backend Services
    ├── llm-gateway         :8080
    ├── semantic-search      :8081
    ├── ai-agents            :8082
    ├── code-orchestrator    :8083
    └── audit-service        :8084
```

## MCP Protocol

The gateway implements the [Model Context Protocol](https://modelcontextprotocol.io/) via SSE/HTTP transport at `/mcp`.

**Supported operations:**
- `tools/list` — returns all 9 registered tools with schemas
- `tools/call` — validates input → dispatches → sanitizes → returns result

**Connecting from a client:**
```python
from fastmcp import Client

async with Client("http://localhost:8087/mcp") as client:
    tools = await client.list_tools()
    result = await client.call_tool("semantic_search", {
        "query": "error handling patterns",
        "collection": "code",
        "top_k": 5,
    })
```

## Tools

| Tool | Backend | Tier | Description |
|------|---------|------|-------------|
| `semantic_search` | semantic-search :8081 | bronze | Semantic similarity search across code, docs, textbooks |
| `hybrid_search` | semantic-search :8081 | bronze | Combined semantic + keyword search |
| `code_analyze` | code-orchestrator :8083 | silver | Code complexity, dependency, and quality analysis |
| `code_pattern_audit` | code-orchestrator :8083 | silver | Detect anti-patterns with dual-net detection |
| `graph_query` | ai-agents :8082 | gold | Neo4j Cypher queries (read-only, injection-protected) |
| `llm_complete` | llm-gateway :8080 | gold | LLM completion with tiered fallback |
| `run_agent_function` | ai-agents :8082 | gold | Execute single-purpose agent functions |
| `run_discussion` | ai-agents :8082 | enterprise | Multi-LLM Kitchen Brigade discussions |
| `agent_execute` | ai-agents :8082 | enterprise | Execute autonomous agent tasks |

Tool definitions are loaded from `config/tools.yaml`.

## Configuration

All settings use environment variables with the `MCP_GATEWAY_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_GATEWAY_HOST` | `0.0.0.0` | Bind address |
| `MCP_GATEWAY_PORT` | `8087` | Bind port |
| `MCP_GATEWAY_SERVICE_NAME` | `mcp-gateway` | Service identity |
| `MCP_GATEWAY_SERVICE_VERSION` | `0.1.0` | Service version |
| `MCP_GATEWAY_LLM_GATEWAY_URL` | `http://localhost:8080` | LLM gateway backend |
| `MCP_GATEWAY_SEMANTIC_SEARCH_URL` | `http://localhost:8081` | Semantic search backend |
| `MCP_GATEWAY_AI_AGENTS_URL` | `http://localhost:8082` | AI agents backend |
| `MCP_GATEWAY_CODE_ORCHESTRATOR_URL` | `http://localhost:8083` | Code orchestrator backend |
| `MCP_GATEWAY_AUTH_ENABLED` | `false` | Enable JWT authentication |
| `MCP_GATEWAY_OIDC_JWKS_URL` | (empty) | JWKS endpoint for JWT validation |
| `MCP_GATEWAY_OIDC_ISSUER` | (empty) | Expected JWT issuer |
| `MCP_GATEWAY_OIDC_AUDIENCE` | `ai-platform-tools` | Expected JWT audience |
| `MCP_GATEWAY_TLS_ENABLED` | `false` | Enable TLS 1.3 |
| `MCP_GATEWAY_TLS_CERT_PATH` | (empty) | Path to TLS certificate |
| `MCP_GATEWAY_TLS_KEY_PATH` | (empty) | Path to TLS private key |
| `MCP_GATEWAY_REDIS_URL` | `redis://localhost:6379` | Redis for rate limiting |
| `MCP_GATEWAY_RATE_LIMIT_RPM` | `100` | Requests per minute per tenant |
| `MCP_GATEWAY_AUDIT_LOG_PATH` | `logs/audit.jsonl` | Local audit log path |

## Security

### Authentication
- JWT validation via OIDC (RS256 / ES256 only — no symmetric algorithms)
- Required claims: `exp`, `iss`, `aud`, `sub`, `tier`
- Dev mode bypass when `AUTH_ENABLED=false`

### Rate Limiting
- Per-tenant Redis token bucket (INCR + EXPIRE)
- Configurable RPM with `X-RateLimit-*` headers
- Graceful degradation when Redis is unavailable

### Input Validation
- Pydantic models per tool with field validators
- OWASP path traversal protection (CWE-22): double-decode, null byte, symlink detection
- Cypher injection prevention: keyword blocking, admin command filtering

### Audit
- Every non-health request logged to JSONL with SHA-256 input hash
- Forwarding to centralized audit-service on :8084
- JSONL fallback on audit-service unavailability

## Testing

```bash
# Unit tests (493 tests, 98.64% coverage)
pytest tests/unit/

# Integration tests (requires live services + Redis)
INTEGRATION=1 pytest tests/integration/ --no-cov

# Full suite
INTEGRATION=1 pytest --no-cov
```

### Test Structure
```
tests/
├── unit/                           # 493 tests, no external dependencies
│   ├── core/                       # config, errors
│   ├── models/                     # schemas, budget
│   ├── security/                   # authn, rate_limiter, audit, validators
│   ├── test_main.py               # app startup, middleware chain
│   ├── test_tool_dispatcher.py    # dispatch routing, error handling
│   ├── test_tool_registry.py      # YAML loading, validation
│   └── test_server.py             # MCP server creation
├── integration/                    # 112 tests, requires live services
│   ├── test_tool_dispatch_e2e.py  # 9 tools → live backends
│   ├── test_security_suite.py     # OWASP, JWT, rate limiting, audit
│   ├── test_performance.py        # latency benchmarks
│   └── test_mcp_protocol.py       # FastMCP protocol verification
└── conftest.py
```

## API Reference

### `GET /health`
Returns service health status.

```json
{
  "service": "mcp-gateway",
  "version": "0.1.0",
  "status": "healthy",
  "uptime_seconds": 123.45
}
```

### `GET /mcp/sse` (MCP SSE Transport)
Server-Sent Events endpoint for MCP protocol communication.

## Development

```bash
# Lint
ruff check src/ tests/

# Type check
mypy src/

# Format
ruff format src/ tests/
```

## License

Proprietary — AI Platform internal service.
