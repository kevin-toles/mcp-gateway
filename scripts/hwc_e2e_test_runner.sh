#!/bin/bash
set -e

# ═════════════════════════════════════════════════════════════════════════════
# H/W/C E2E Startup Test Runner
# ═════════════════════════════════════════════════════════════════════════════
# 
# Opens an external terminal and runs comprehensive H/W/C endpoint tests.
# After tests, transitions services back to WARM tier (ready for active dev).
#
# Usage:
#   ./scripts/hwc_e2e_test_runner.sh [--headless]
#   
# Options:
#   --headless    Run tests without opening external terminal (CI/batch mode)
#
# Called from: LaunchAgent plist at shim startup
# ═════════════════════════════════════════════════════════════════════════════

HEADLESS=false
if [ "$1" = "--headless" ]; then
  HEADLESS=true
fi

PROJECT_ROOT="/Users/kevintoles/POC/mcp-gateway"
TESTS_DIR="$PROJECT_ROOT/tests/integration"
TEST_FILE="test_hwc_e2e_startup.py"
LOG_FILE="/tmp/hwc_e2e_test_$(date +%Y%m%d_%H%M%S).log"
TIER_RECOVERY_SCRIPT="/tmp/hwc_tier_recovery.sh"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Wait for Proxy & Gateway
# ─────────────────────────────────────────────────────────────────────────────

wait_for_proxy() {
  echo "Waiting for MCP Lifecycle Proxy (:8090)..."
  for i in {1..30}; do
    if (echo > /dev/tcp/127.0.0.1/8090) 2>/dev/null; then
      echo "✓ Proxy listening"
      return 0
    fi
    sleep 1
  done
  echo "✗ Proxy did not start within 30 seconds"
  return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Run Test Suite
# ─────────────────────────────────────────────────────────────────────────────

run_tests() {
  cd "$PROJECT_ROOT"
  
  echo ""
  echo "╔════════════════════════════════════════════════════════════════╗"
  echo "║           H/W/C E2E Startup Test Suite                        ║"
  echo "║                                                                ║"
  echo "║  Testing: Proxy, Gateway, Platform Services                   ║"
  echo "║  Coverage: All endpoints, tier state, stability               ║"
  echo "╚════════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Timestamp: $(date)"
  echo "Log file: $LOG_FILE"
  echo ""
  
  # Use venv Python to avoid system pytest plugin conflicts (libpq/psycopg issues)
  VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
  if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"  # Fallback to system Python
  fi

  # Run pytest with output to both screen and log
  "$VENV_PYTHON" -m pytest "$TESTS_DIR/$TEST_FILE" \
    -v \
    --tb=short \
    -m integration \
    2>&1 | tee "$LOG_FILE"
  
  TEST_EXIT_CODE=${PIPESTATUS[0]}
  return $TEST_EXIT_CODE
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Service Tier Recovery (Transition to WARM)
# ─────────────────────────────────────────────────────────────────────────────

create_tier_recovery_script() {
  cat > "$TIER_RECOVERY_SCRIPT" << 'EOFRECOVERY'
#!/bin/bash

echo ""
echo "Transitioning services from test state to production tier..."
echo ""

# WARM tier services (stay running, but will idle timeout eventually)
# These are the core services that benefit from being pre-warmed
WARM_SERVICES=(
  "mcp-gateway:8087"
  "unified-search:8081"
  "llm-gateway:8080"
)

# COLD tier services (stop after tests, start on-demand)
COLD_SERVICES=(
  "ai-agents:8082"
  "code-orchestrator:8083"
  "audit-service:8084"
  "context-management:8086"
  "struct-analyzer:8088"
)

echo "Services to keep WARM (will idle timeout eventually):"
for svc in "${WARM_SERVICES[@]}"; do
  name="${svc%:*}"
  port="${svc#*:}"
  if curl -sf http://localhost:$port/health >/dev/null 2>&1; then
    echo "  ✓ $name (:$port) — WARM"
  else
    echo "  • $name (:$port) — COLD (not running)"
  fi
done

echo ""
echo "Services to transition to COLD (will restart on-demand):"
for svc in "${COLD_SERVICES[@]}"; do
  name="${svc%:*}"
  port="${svc#*:}"
  
  if curl -sf http://localhost:$port/health >/dev/null 2>&1; then
    # Could add logic here to actually stop services
    # For now, just report they'll timeout to COLD
    echo "  • $name (:$port) — Will transition to COLD on idle timeout"
  else
    echo "  ✓ $name (:$port) — Already COLD"
  fi
done

echo ""
echo "✓ Tier recovery complete. Services ready for development."
echo ""
EOFRECOVERY

  chmod +x "$TIER_RECOVERY_SCRIPT"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Open Terminal & Run (or run headless)
# ─────────────────────────────────────────────────────────────────────────────

main() {
  # Ensure proxy is running
  wait_for_proxy || {
    echo "✗ Proxy startup failed. Check LaunchAgent configuration."
    exit 1
  }
  
  # Create tier recovery script
  create_tier_recovery_script
  
  if [ "$HEADLESS" = true ]; then
    # Headless mode: just run tests
    run_tests
    TEST_EXIT=$?
    bash "$TIER_RECOVERY_SCRIPT"
    exit $TEST_EXIT
  else
    # Interactive mode: open Terminal and run tests
    # This allows user to see real-time output and interact if needed
    
    # Create a wrapper script that runs tests and stays open
    WRAPPER_SCRIPT="/tmp/hwc_test_wrapper_$$.sh"
    cat > "$WRAPPER_SCRIPT" << 'EOFWRAPPER'
#!/bin/bash
cd "PROJECT_ROOT_PLACEHOLDER"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           H/W/C E2E Startup Test Suite                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Use venv Python to avoid pytest-postgresql plugin conflicts
VENV_PYTHON="PROJECT_ROOT_PLACEHOLDER/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
  VENV_PYTHON="python3"
fi

"$VENV_PYTHON" -m pytest "TESTS_DIR_PLACEHOLDER/TEST_FILE_PLACEHOLDER" \
  -v --tb=short -m integration

TEST_EXIT=$?

echo ""
echo "Running service tier recovery..."
bash "TIER_RECOVERY_SCRIPT_PLACEHOLDER"

echo ""
echo "Test suite complete. Press Enter to close this window..."
read -r

exit $TEST_EXIT
EOFWRAPPER

    # Replace placeholders
    sed -i '' "s|PROJECT_ROOT_PLACEHOLDER|$PROJECT_ROOT|g" "$WRAPPER_SCRIPT"
    sed -i '' "s|TESTS_DIR_PLACEHOLDER|$TESTS_DIR|g" "$WRAPPER_SCRIPT"
    sed -i '' "s|TEST_FILE_PLACEHOLDER|$TEST_FILE|g" "$WRAPPER_SCRIPT"
    sed -i '' "s|TIER_RECOVERY_SCRIPT_PLACEHOLDER|$TIER_RECOVERY_SCRIPT|g" "$WRAPPER_SCRIPT"
    
    chmod +x "$WRAPPER_SCRIPT"
    
    # Open in Terminal.app
    open -a Terminal "$WRAPPER_SCRIPT"
    
    # Clean up wrapper script after terminal opens (Terminal has own process)
    sleep 1
    rm -f "$WRAPPER_SCRIPT"
  fi
}

main "$@"
