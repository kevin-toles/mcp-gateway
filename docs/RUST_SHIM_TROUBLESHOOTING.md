# Rust Shim Troubleshooting Guide

A comprehensive guide to debugging and resolving issues with the MCP Lifecycle Proxy (Rust shim).

---

## Table of Contents

1. [Startup & Lifecycle Issues](#startup--lifecycle-issues)
2. [Cold-Start Spawn Problems](#cold-start-spawn-problems)
3. [H/W/C Tier Management Issues](#hwc-tier-management-issues)
4. [Docker-Mode Compatibility Issues](#docker-mode-compatibility-issues)
5. [Test Suite Execution Problems](#test-suite-execution-problems)
6. [Diagnostic Commands](#diagnostic-commands)

---

## Startup & Lifecycle Issues

### Issue 1: LaunchAgent Not Triggering at Startup

**Symptom:**
- Shim binary exists but doesn't start when machine boots
- Manual startup works fine
- Shim is not listening on :8090 after restart

**Root Causes Found:**

#### A. XML Encoding in LaunchAgent Plist
**What happened:**
- Plist file had embedded shell script with `&` characters
- XML parser saw `&` as entity start (should be `&amp;`)
- Error: `Encountered unknown ampersand-escape sequence at line 18`
- LaunchAgent failed to load silently

**Symptoms:**
```bash
$ plutil -lint ~/Library/LaunchAgents/com.kevintoles.mcp-gateway-hwc-startup-test.plist
/Users/kevintoles/Library/LaunchAgents/...: Encountered unknown ampersand-escape sequence at line 18

$ launchctl list | grep hwc-startup-test
-    0    com.kevintoles.mcp-gateway-hwc-startup-test  # "-" means not loaded
```

**Solution:**
```xml
<!-- BEFORE (broken) -->
<string>if (echo > /dev/tcp/127.0.0.1/8090) 2>/dev/null; then</string>
<!--                    ^ XML parser breaks here -->

<!-- AFTER (fixed) -->
<!-- Move shell logic to external script, call directly -->
<array>
  <string>/bin/bash</string>
  <string>/path/to/test_runner.sh</string>
</array>
```

**Key Lesson:** Avoid embedding shell scripts with special characters in plist files. Call external scripts instead.

---

#### B. LaunchAgent Load Failure (I/O Error)
**What happened:**
- Plist was valid but `launchctl load` returned "Input/output error"
- Likely due to cached state or file lock

**Solution:**
```bash
# Force reload sequence
launchctl unload ~/Library/LaunchAgents/com.kevintoles.mcp-gateway-hwc-startup-test.plist
launchctl remove com.kevintoles.mcp-gateway-hwc-startup-test
sleep 2
launchctl load ~/Library/LaunchAgents/com.kevintoles.mcp-gateway-hwc-startup-test.plist

# Verify
launchctl list | grep hwc-startup-test
# Should show PID (not "-")
```

**Verification:**
```bash
# Check if LaunchAgent is loaded
launchctl list | grep -i mcp

# Expected output:
# 771    0    com.kevintoles.mcp-gateway-shim  ← PID 771 means loaded
# 3910   0    com.kevintoles.mcp-gateway-hwc-startup-test  ← Loaded
```

---

### Issue 2: Shim Process Dies on Startup

**Symptom:**
- Shim starts but exits immediately
- No error messages
- Checking logs shows silence

**Root Causes Found:**

#### A. Missing Dependencies in Start Command
**What happened:**
- `src/main.rs` called functions that weren't imported
- `platform_services::seed_platform_services()` called but module not exported

**Error:**
```
error[E0433]: cannot find function `seed_platform_services` in module `platform_services`
```

**Solution:**
```rust
// src/lib.rs
pub mod platform_services;  // ← Export the module

// src/main.rs
use shim_mcp_gateway::platform_services;

// In main():
let registry = Arc::new(registry::ServiceRegistry::new());
platform_services::seed_platform_services(&registry);  // ← Now available
```

**Verification:**
```bash
cargo build --release
cargo test  # Run all unit tests
```

---

#### B. Platform Services Not Seeded at Startup
**What happened:**
- Shim started but `registry` was empty (no services registered)
- `startup_scan()` tried to health-check non-existent services
- No services transitioned to COLD tier
- First request timeout when trying to spawn mcp-gateway

**Symptom:**
```bash
$ curl http://localhost:8090/health
# Timeout or connection refused
```

**Root Cause:**
```rust
// BEFORE (broken):
let registry = Arc::new(registry::ServiceRegistry::new());
lifecycle::startup_scan(&registry).await;  // ← No services in registry!

// AFTER (fixed):
let registry = Arc::new(registry::ServiceRegistry::new());
platform_services::seed_platform_services(&registry);  // ← Seed first!
lifecycle::startup_scan(&registry).await;  // ← Now has 9 services
```

**Verification:**
```bash
# Check that seed function registers 9 services
grep -n "seed_platform_services\|len()" src/platform_services.rs | head -5

# Run tests
cargo test --lib platform_services
```

---

## Cold-Start Spawn Problems

### Issue 3: start_mcp_gateway() Using Kevin-Specific Paths

**Symptom:**
- Shim won't start Python mcp-gateway on first request
- Error: `spawn: failed to find uvicorn binary`
- Proxy returns 502 Bad Gateway
- Works on Kevin's machine but fails elsewhere (dev machines, CI)

**Root Cause:**
```rust
// BEFORE (broken - hardcoded paths):
fn start_mcp_gateway() -> Option<Child> {
    let uvicorn_bin = "/Users/kevintoles/POC/mcp-gateway/.venv/bin/uvicorn";
    //                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Kevin's home directory
    
    Command::new(&uvicorn_bin)
        .args(["src.main:app", "--host", "127.0.0.1", "--port", "8087"])
        .spawn()  // ← Fails on any other machine
        .ok()
}
```

**Impact:**
- Shim runs but can't spawn Python gateway
- :8090 listens, but first request gets 502
- H/W/C system completely broken for cold-start

**Solution:**
```rust
// AFTER (fixed - env-var driven):
fn start_mcp_gateway() -> Option<Child> {
    let home = std::env::var("MCP_GATEWAY_HOME")
        .unwrap_or_else(|_| {
            std::env::current_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| ".".to_string())
        });
    
    let uvicorn_bin = std::env::var("MCP_GATEWAY_UVICORN_BIN")
        .unwrap_or_else(|_| format!("{}/.venv/bin/uvicorn", home));
    
    // ... use uvicorn_bin from env vars, not hardcoded
}
```

**Configuration:**
```bash
# In ~/.zshrc or for LaunchAgent:
export MCP_GATEWAY_HOME="/Users/kevintoles/POC/mcp-gateway"
export MCP_GATEWAY_UVICORN_BIN="/Users/kevintoles/POC/mcp-gateway/.venv/bin/uvicorn"

# Or in LaunchAgent plist:
<key>EnvironmentVariables</key>
<dict>
  <key>MCP_GATEWAY_HOME</key>
  <string>/Users/kevintoles/POC/mcp-gateway</string>
  <key>MCP_GATEWAY_UVICORN_BIN</key>
  <string>/Users/kevintoles/POC/mcp-gateway/.venv/bin/uvicorn</string>
</dict>
```

**Verification:**
```bash
# Kill existing shim
pkill -f shim-mcp-gateway

# Restart without requesting gateway
/Users/kevintoles/POC/mcp-gateway/target/release/shim-mcp-gateway &

# Make first request (should spawn gateway in ~2-4 seconds)
curl -v http://localhost:8090/health

# Verify gateway is running
lsof -i :8087 | grep uvicorn  # Should show running process
```

---

## H/W/C Tier Management Issues

### Issue 4: Services Not Transitioning Between Tiers

**Symptom:**
- All services stay in same tier (usually COLD)
- Idle monitor not triggering HOT→WARM transitions
- Services don't respond to health checks
- Startup scan completes but services remain unhealthy

**Root Causes Found:**

#### A. Registry Empty During startup_scan()
**What happened:**
- `startup_scan()` tries to health-check services
- But registry has no services (not seeded yet)
- Loop completes with zero transitions
- All services remain in BOOT tier

**Solution:**
See Issue 2B above — seed before calling startup_scan()

---

#### B. Idle Monitor Not Running
**Symptom:**
- Services never transition HOT→WARM
- Idle timeout disabled implicitly

**Verification:**
```bash
# Check if idle monitor is spawned
ps aux | grep -i idle | grep -v grep

# Check logs
grep -i "idle\|warm\|cold" /tmp/shim-mcp-gateway.log
```

**Solution:**
Verify that `lifecycle::shim_idle_monitor()` is spawned as background task:

```rust
// In main.rs:
tokio::spawn(async move {
    lifecycle::shim_idle_monitor(reg_for_idle).await;
});
```

---

### Issue 5: Health Checks Failing for Services

**Symptom:**
- startup_scan() reports all services unreachable
- Even when services ARE running
- Services stay in BOOT tier

**Root Cause:**
Wrong health endpoint URL or wrong port

**Verification:**
```bash
# Check each service individually
curl http://localhost:8080/health  # llm-gateway
curl http://localhost:8081/health  # unified-search
curl http://localhost:8083/health  # code-orchestrator

# Check what startup_scan is using
grep -A10 "health_endpoint\|SERVICE_HEALTH" src/lifecycle.rs
```

---

## Docker-Mode Compatibility Issues

### Issue 6: Docker Compose Services Can't Reach Each Other

**Symptom:**
- Services in Docker containers show "degraded"
- Each service says dependencies are "unreachable"
- unified-search: `qdrant: unreachable`, `neo4j: unreachable`
- code-orchestrator can't reach inference-service

**Root Causes Found:**

#### A. Hardcoded localhost URLs in Python Config
**What happened:**
- Services configured with `http://localhost:8083` (code-orchestrator)
- In Docker mode, services are on different containers
- `localhost` in container = container itself, not the host
- Connection refused

**Solution:**
Use container DNS names (Docker internal naming):
```python
# BEFORE (broken in Docker):
CODE_ORCHESTRATOR_URL = "http://localhost:8083"

# AFTER (works in Docker):
# From docker-compose.yml service name:
CODE_ORCHESTRATOR_URL = "http://code-orchestrator:8083"
```

**Configuration in docker-compose.docker.yml:**
```yaml
services:
  mcp-gateway:
    environment:
      - MCP_GATEWAY_CODE_ORCHESTRATOR_URL=http://code-orchestrator:8083
      - MCP_GATEWAY_LLM_GATEWAY_URL=http://llm-gateway:8080
      - MCP_GATEWAY_AUDIT_SERVICE_URL=http://audit-service:8084
```

---

#### B. Pydantic env_prefix Mismatch in docker-compose.yml
**What happened:**
- docker-compose.yml had bare env var names: `REDIS_URL=redis://redis:6379`
- Services use Pydantic BaseSettings with `env_prefix="AUDIT_"`
- Bare `REDIS_URL` doesn't match `AUDIT_REDIS_URL` pattern
- Settings silently fall back to localhost defaults
- In Docker: `localhost` = container itself, not host
- Connection fails

**Symptom:**
```json
{
  "status": "degraded",
  "redis": "unreachable at localhost:6379"  // Should be redis:6379
}
```

**Solution:**
Use prefixed env var names in compose:
```yaml
# BEFORE (broken):
audit-service:
  environment:
    - REDIS_URL=redis://redis:6379  # Ignored by Pydantic!
    - QDRANT_URL=http://qdrant-knowledge:6333  # Ignored!

# AFTER (fixed):
audit-service:
  environment:
    - AUDIT_REDIS_URL=redis://redis:6379  # Matches env_prefix
    - AUDIT_QDRANT_URL=http://qdrant-knowledge:6333  # Matches env_prefix
```

**Verification:**
```bash
# Check service's Pydantic env_prefix
grep -n "class Settings\|env_prefix" src/core/config.py | head -5

# Example output:
# class Settings(BaseSettings):
#     model_config = ConfigDict(env_prefix="AUDIT_")

# So all env vars must start with AUDIT_
```

---

#### C. Tool Handlers Reading localhost Defaults
**What happened:**
- batch_extract_metadata.py, batch_enrich_metadata.py, convert_pdf_to_json.py had hardcoded CO URLs
- Used localhost:8083 as constant
- In Docker mode, localhost = container, not host
- Tool calls failed with "connection refused"

**Solution:**
Read CO URL from Settings at call time:
```python
# BEFORE (broken):
CO_RESTART_COMMAND = "cd /Users/kevintoles/POC/Code-Orchestrator-Service && ..."

# AFTER (fixed):
def _check_co_health(co_url: str | None = None) -> str | None:
    """Resolve CO URL from Settings in Docker mode."""
    if co_url is None:
        from src.core.config import Settings
        co_url = Settings().CODE_ORCHESTRATOR_URL  # Reads MCP_GATEWAY_CODE_ORCHESTRATOR_URL
    # ... health check using resolved URL
```

---

## Test Suite Execution Problems

### Issue 7: E2E Test Suite Crashes with pytest-postgresql Error

**Symptom:**
- Test runner crashes at startup
- Error: `ImportError: no pq wrapper available`
- Trace: `psycopg.pq: libpq library not found`
- No tests run
- Exit code 1

**Root Cause:**
System Python used instead of venv Python
- System Python has `pytest-postgresql` plugin
- pytest-postgresql depends on libpq (PostgreSQL C library)
- libpq not installed on macOS
- pytest fails to load plugin before running tests

**What Happened:**
```bash
# Test runner ran:
python -m pytest tests/integration/test_hwc_e2e_startup.py

# Resolved to:
/Library/Frameworks/Python.framework/Versions/3.13/bin/python

# Which has pytest-postgresql:
/Library/Frameworks/Python.framework/.../site-packages/pytest_postgresql/

# Which needs libpq:
ImportError: libpq library not found  # ← CRASH
```

**Solution:**
Use venv Python which doesn't have pytest-postgresql:
```bash
# BEFORE (broken):
python -m pytest ...

# AFTER (fixed):
.venv/bin/python -m pytest ...
```

**Implementation in Test Runner:**
```bash
# In hwc_e2e_test_runner.sh:
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
  VENV_PYTHON="python3"  # Fallback
fi

"$VENV_PYTHON" -m pytest "$TESTS_DIR/$TEST_FILE" \
  -v --tb=short -m integration
```

**Verification:**
```bash
# Check venv has pytest without pytest-postgresql
.venv/bin/python -m pip list | grep pytest

# Should NOT show pytest-postgresql

# Try collecting tests
.venv/bin/python -m pytest tests/integration/test_hwc_e2e_startup.py --collect-only

# Should show:
# collected 22 items (no errors)
```

---

## Diagnostic Commands

### Check Shim Status

```bash
# Is shim running?
lsof -i :8090

# Is it listening?
curl http://localhost:8090/health

# What process?
ps aux | grep shim-mcp-gateway

# Is LaunchAgent loaded?
launchctl list | grep mcp-gateway-shim

# Check logs
tail -50 /tmp/shim-mcp-gateway.log
```

### Check Gateway Status

```bash
# Is gateway running?
lsof -i :8087 | grep uvicorn

# Is it healthy?
curl http://localhost:8087/health

# Check logs
tail -50 /tmp/mcp-gateway.log
```

### Check Platform Services

```bash
# Check registry seeding
grep -n "seed_platform_services" src/main.rs

# Verify module export
grep -n "pub mod platform_services" src/lib.rs

# Test service health
for port in 8080 8081 8082 8083 8084 8086 8088; do
  echo -n "Port $port: "
  curl -sf http://localhost:$port/health | jq .status 2>/dev/null || echo "OFFLINE"
done
```

### Verify Tier Management

```bash
# Check idle monitor is running
ps aux | grep idle | grep -v grep

# Monitor tier transitions
tail -f /tmp/mcp-registry.log | grep -i "hot\|warm\|cold"

# Check idle timeout config
grep -i "IDLE_TIMEOUT\|HOT_IDLE\|WARM_IDLE" src/lifecycle.rs
```

### Rebuild and Redeploy

```bash
# Full rebuild
cd /Users/kevintoles/POC/mcp-gateway
cargo clean
cargo build --release

# Verify binary
file target/release/shim-mcp-gateway
# Should output: Mach-O 64-bit executable arm64

# Stop existing shim
pkill -f shim-mcp-gateway

# Start new shim
./auto_start_shim.sh

# Verify
lsof -i :8090
```

---

## Summary: Lessons Learned

| Issue | Category | Impact | Prevention |
|-------|----------|--------|-----------|
| XML encoding in plist | Configuration | LaunchAgent won't load | Avoid embedding shell scripts in plist files |
| Hardcoded paths | Portability | Only works on one machine | Use env vars for all paths |
| Missing module exports | Compilation | Shim crashes on startup | Run cargo test before deployment |
| Registry not seeded | Logic | All services stay BOOT tier | Seed registry before startup_scan() |
| localhost URLs in Docker | Architecture | Services can't reach each other | Use container DNS names (docker-compose service names) |
| Pydantic env_prefix mismatch | Configuration | Settings silently fall back to defaults | Prefix all env vars in docker-compose.yml |
| System pytest conflicts | Environment | Tests crash before running | Use venv Python for test execution |

---

## References

- [Rust Shim Source](../src/main.rs)
- [Platform Services Seeding](../src/platform_services.rs)
- [H/W/C Lifecycle](../src/lifecycle.rs)
- [E2E Test Suite](../tests/integration/test_hwc_e2e_startup.py)
- [Test Runner Script](../scripts/hwc_e2e_test_runner.sh)
- [Docker Compose Configuration](../../ai-platform-data/docker/docker-compose.docker.yml)

