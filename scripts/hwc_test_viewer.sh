#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
# H/W/C Startup Test Viewer
# ═════════════════════════════════════════════════════════════════════════════
# Pure display: live-tails the current startup test log. The tests run
# headless in the LaunchAgent regardless of whether this window opens —
# closing or killing this viewer never affects the test run.
#
# Opened by: scripts/hwc_e2e_test_runner.sh via `open -a Terminal`
# ═════════════════════════════════════════════════════════════════════════════

LOG_DIR="$HOME/Library/Logs/hwc"
LOG="$LOG_DIR/latest.log"

clear
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           H/W/C E2E Startup Test Suite (live view)             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Wait up to 30s for the log to appear (runner creates it before opening us)
for _ in $(seq 1 30); do
  [ -f "$LOG" ] && break
  sleep 1
done

if [ ! -f "$LOG" ]; then
  echo "✗ No test log found at $LOG after 30s."
  echo "  The test runner may have failed before creating it."
  echo "  Check: /tmp/hwc-startup-test-launch-error.log"
  echo ""
  echo "Press Enter to close..."
  read -r
  exit 1
fi

# Live-tail until the runner writes its completion sentinel
tail -n +1 -f "$LOG" 2>/dev/null | while IFS= read -r line; do
  printf '%s\n' "$line"
  case "$line" in
    *HWC-RUN-COMPLETE*) pkill -P $$ -x tail 2>/dev/null; break ;;
  esac
done

echo ""
echo "Run complete. Full log: $LOG"
echo "Press Enter to close this window..."
read -r
