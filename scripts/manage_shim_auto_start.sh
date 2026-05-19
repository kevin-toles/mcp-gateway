#!/bin/bash
# MCP Gateway Shim - System Auto-Start Manager
LABEL="com.kevintoles.mcp-gateway-shim"
PLIST="$HOME/Library/LaunchAgents/com.kevintoles.mcp-gateway-shim.plist"

show_status() {
    echo "=== MCP Gateway Shim Status ==="
    if launchctl list | grep -q "$LABEL"; then
        echo "✓ LaunchAgent is loaded"
        launchctl list | grep "$LABEL" | awk '{print "  PID: " $1}'
    else
        echo "✗ LaunchAgent is NOT loaded"
    fi
    if pgrep -f "shim-mcp-gateway" > /dev/null 2>&1; then
        echo "✓ Shim is running"
        pgrep -f "shim-mcp-gateway" | while read pid; do echo "  PID: $pid"; done
    else
        echo "✗ Shim is NOT running"
    fi
    if lsof -i :8087 > /dev/null 2>&1; then
        echo "✓ Port 8087 is in use"
    else
        echo "✗ Port 8087 is NOT in use"
    fi
    echo ""
}

case "${1:-status}" in
    status) show_status ;;
    start)  launchctl load "$PLIST"; sleep 2; show_status ;;
    stop)   launchctl unload "$PLIST"; pkill -f "shim-mcp-gateway" 2>/dev/null || true; echo "✓ Stopped" ;;
    restart) "$0" stop; sleep 1; "$0" start ;;
    *) echo "Usage: $0 {status|start|stop|restart}" ;;
esac
