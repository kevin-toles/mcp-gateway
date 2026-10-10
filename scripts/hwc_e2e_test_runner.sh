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
# Phase 1b: Sequenced on-demand startup
# ─────────────────────────────────────────────────────────────────────────────
#
# On-demand services are NOT started by launchd at login (RunAtLoad=false).
# They are brought up here one group at a time so login never boots every
# model-loading service at once (boot storm → memory exhaustion). A group
# starts only after every service in the previous group is healthy or has
# timed out. Group membership is defined here; ports and health paths come
# from config/services.toml (single source of truth).
#
# Group 0 (launchd KeepAlive): shim, mcp-gateway, llm-gateway, unified-search-rs
STARTUP_GROUPS=(
  "validation-service struct-analyzer amve"                      # light Rust/Go/Python
  "ai-agents audit-service context-management-service"           # medium Python
  "code-orchestrator"                                            # model-heavy, alone
  "inference-service-cpp"                                        # model-heavy, alone
)
GROUP_TIMEOUT_SECS=150

# Print "<port> <health_path>" for a service from the manifest.
manifest_endpoint() {
  "$VENV_PYTHON" - "$1" "$PROJECT_ROOT/config/services.toml" <<'PY'
import sys, tomllib
svc = tomllib.load(open(sys.argv[2], "rb"))["services"][sys.argv[1]]
print(svc["port"], svc.get("health_path", "/health"))
PY
}

service_healthy() {
  curl -sf --max-time 2 "http://localhost:$1$2" >/dev/null 2>&1
}

start_group() {
  local group_no="$1"; shift
  local started_at=$SECONDS
  echo ""
  echo "── Startup group $group_no: $* ──"

  local svc port path
  for svc in "$@"; do
    read -r port path < <(manifest_endpoint "$svc")
    if service_healthy "$port" "$path"; then
      echo "  ✓ $svc (:$port) already healthy"
    else
      echo "  → $svc (:$port) requesting start"
      curl -sf --max-time 5 -X POST "http://localhost:8079/activity/$svc" >/dev/null 2>&1 \
        || echo "  ✗ $svc: activity request to :8079 failed"
    fi
  done

  local pending=("$@")
  while [ "${#pending[@]}" -gt 0 ] && [ $((SECONDS - started_at)) -lt "$GROUP_TIMEOUT_SECS" ]; do
    local still=()
    for svc in "${pending[@]}"; do
      read -r port path < <(manifest_endpoint "$svc")
      if service_healthy "$port" "$path"; then
        echo "  ✓ $svc (:$port) healthy after $((SECONDS - started_at))s"
      else
        still+=("$svc")
      fi
    done
    pending=("${still[@]}")
    [ "${#pending[@]}" -gt 0 ] && sleep 2
  done

  for svc in "${pending[@]}"; do
    echo "  ✗ $svc not healthy after ${GROUP_TIMEOUT_SECS}s — continuing; test suite will report it"
  done
}

sequenced_startup() {
  echo ""
  echo "Sequenced startup of on-demand services (${#STARTUP_GROUPS[@]} groups)..."
  local i=1 group
  for group in "${STARTUP_GROUPS[@]}"; do
    # shellcheck disable=SC2086  # word-split the group into service names
    start_group "$i" $group
    i=$((i + 1))
  done
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

  VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
  [ -f "$VENV_PYTHON" ] || VENV_PYTHON="python3"

  sequenced_startup

  run_tests
  TEST_EXIT=$?

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
