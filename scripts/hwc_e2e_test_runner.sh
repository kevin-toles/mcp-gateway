#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# H/W/C E2E Startup Test Runner
# ═════════════════════════════════════════════════════════════════════════════
#
# Validates the HWC lifecycle on every startup. Design invariants:
#
#   1. Tests ALWAYS run — headless, in this process, with no dependency on
#      Terminal.app, /tmp wrapper scripts, or anything a reboot can delete.
#   2. Results are ALWAYS captured — logs go to ~/Library/Logs/hwc/ which
#      persists across reboots (unlike /tmp, which macOS wipes).
#   3. Visibility is best-effort and NEVER gates execution — a Terminal
#      viewer window and a macOS notification are attempted, but failure
#      to display never prevents or interrupts the test run.
#
# Usage:
#   ./scripts/hwc_e2e_test_runner.sh              # full run (viewer + tests)
#   ./scripts/hwc_e2e_test_runner.sh --no-viewer  # tests only (CI/manual)
#   --headless is accepted as an alias for --no-viewer (back-compat)
#
# Called from: LaunchAgent com.kevintoles.mcp-gateway-hwc-startup-test
# ═════════════════════════════════════════════════════════════════════════════

VIEWER=true
case "$1" in
  --no-viewer|--headless) VIEWER=false ;;
esac

PROJECT_ROOT="/Users/kevintoles/POC/mcp-gateway"
TESTS_DIR="$PROJECT_ROOT/tests/integration"
TEST_FILE="test_hwc_e2e_startup.py"

# Persistent log location — /tmp gets wiped on reboot, this does not.
LOG_DIR="$HOME/Library/Logs/hwc"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/startup_test_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG_FILE"
ln -sf "$LOG_FILE" "$LOG_DIR/latest.log"

# Keep only the 20 most recent logs
ls -t "$LOG_DIR"/startup_test_*.log 2>/dev/null | tail -n +21 | xargs rm -f 2>/dev/null

# From here on, everything we print goes to the persistent log.
# The viewer window (if open) tails this file live.
exec >>"$LOG_FILE" 2>&1

# ─────────────────────────────────────────────────────────────────────────────
# Best-effort display channels — never gate execution
# ─────────────────────────────────────────────────────────────────────────────

open_viewer() {
  [ "$VIEWER" = true ] || return 0
  # Permanent script in the repo — nothing to lose on reboot, nothing to
  # clean up. If Terminal isn't available yet (early login), open fails
  # harmlessly and the run continues; results are in the log either way.
  open -a Terminal "$PROJECT_ROOT/scripts/hwc_test_viewer.sh" 2>/dev/null || true
}

notify() {
  # $1 = title, $2 = message
  osascript -e "display notification \"$2\" with title \"$1\"" 2>/dev/null || true
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Wait for core infrastructure
# ─────────────────────────────────────────────────────────────────────────────

wait_for_services() {
  echo "Waiting for platform lifecycle daemon (:8079) and mcp-gateway (:8087)..."

  local lifecycle_up=false
  local gateway_up=false

  for _ in $(seq 1 60); do
    if [ "$lifecycle_up" = false ] && (echo > /dev/tcp/127.0.0.1/8079) 2>/dev/null; then
      echo "✓ Platform lifecycle daemon listening (:8079)"
      lifecycle_up=true
    fi
    if [ "$gateway_up" = false ] && (echo > /dev/tcp/127.0.0.1/8087) 2>/dev/null; then
      echo "✓ MCP gateway listening (:8087)"
      gateway_up=true
    fi
    if [ "$lifecycle_up" = true ] && [ "$gateway_up" = true ]; then
      return 0
    fi
    sleep 1
  done

  [ "$lifecycle_up" = false ] && echo "✗ Platform lifecycle daemon (:8079) did not start within 60 seconds"
  [ "$gateway_up" = false ] && echo "✗ MCP gateway (:8087) did not start within 60 seconds"
  return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Run test suite
# ─────────────────────────────────────────────────────────────────────────────

run_tests() {
  cd "$PROJECT_ROOT" || return 1

  echo ""
  echo "╔════════════════════════════════════════════════════════════════╗"
  echo "║           H/W/C E2E Startup Test Suite                        ║"
  echo "║  Testing: lifecycle daemon, gateway, auto-start, SLO, tiers   ║"
  echo "╚════════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Timestamp: $(date)"
  echo "Log file:  $LOG_FILE"
  echo ""

  # Use venv Python to avoid system pytest plugin conflicts
  VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
  [ -f "$VENV_PYTHON" ] || VENV_PYTHON="python3"

  INTEGRATION=1 "$VENV_PYTHON" -m pytest "$TESTS_DIR/$TEST_FILE" \
    -v \
    --tb=short \
    -m integration
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Tier status report (informational)
# ─────────────────────────────────────────────────────────────────────────────

tier_report() {
  echo ""
  echo "Current tier state (:8079/slo):"
  curl -sf --max-time 5 http://localhost:8079/slo 2>/dev/null \
    | "$VENV_PYTHON" -c "
import json, sys
try:
    for e in sorted(json.load(sys.stdin), key=lambda x: x['service']):
        mark = '✓' if e.get('within_slo') else '✗'
        print(f\"  {mark} {e['service']}: tier={e.get('tier','?')} uptime={e.get('uptime_ratio',0):.3f}\")
except Exception:
    print('  (SLO endpoint unavailable)')
" 2>/dev/null || echo "  (SLO endpoint unavailable)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

main() {
  echo "H/W/C startup validation — $(date)"

  # Open the viewer FIRST so the user watches the whole run, including the
  # wait phase. Display failure never affects the run.
  open_viewer

  if ! wait_for_services; then
    echo ""
    echo "✗ STARTUP FAILED: core infrastructure did not come up."
    echo "  Check LaunchAgents: com.kevintoles.mcp-gateway-shim, com.kevintoles.mcp-gateway"
    notify "HWC Startup Tests" "FAILED: core infrastructure did not start"
    echo "HWC-RUN-COMPLETE status=infra-failure"
    exit 1
  fi

  run_tests
  TEST_EXIT=$?

  VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
  [ -f "$VENV_PYTHON" ] || VENV_PYTHON="python3"
  tier_report

  echo ""
  if [ "$TEST_EXIT" -eq 0 ]; then
    echo "✓ All startup tests passed. Platform validated."
    notify "HWC Startup Tests" "PASSED — platform validated"
    echo "HWC-RUN-COMPLETE status=pass"
  else
    echo "✗ Startup tests FAILED (exit $TEST_EXIT). See log above."
    notify "HWC Startup Tests" "FAILED — check $LOG_FILE"
    echo "HWC-RUN-COMPLETE status=fail exit=$TEST_EXIT"
  fi

  exit "$TEST_EXIT"
}

main "$@"
