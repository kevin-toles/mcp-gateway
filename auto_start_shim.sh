#!/bin/bash
# Auto-start shim for mcp-gateway
# Checks if shim is already running, starts it if not

SHIM_BIN="/Users/kevintoles/POC/mcp-gateway/target/release/shim-mcp-gateway"
PID_FILE="/tmp/shim-mcp-gateway.pid"

# Check if shim is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "shim-mcp-gateway already running (PID: $PID)"
        exit 0
    fi
fi

# Start the shim in background
cd /Users/kevintoles/POC/mcp-gateway
nohup "$SHIM_BIN" > /tmp/shim-mcp-gateway.log 2>&1 &
SHIM_PID=$!

# Save PID
echo "$SHIM_PID" > "$PID_FILE"

echo "shim-mcp-gateway started (PID: $SHIM_PID)"
echo "Logs: /tmp/shim-mcp-gateway.log"
