#!/bin/bash
# MCP Gateway + Shim - System Auto-Start Manager (Option B)
SHIM_LABEL="com.kevintoles.mcp-gateway-shim"
GATEWAY_LABEL="com.kevintoles.mcp-gateway"
SHIM_PLIST="$HOME/Library/LaunchAgents/com.kevintoles.mcp-gateway-shim.plist"
GATEWAY_PLIST="$HOME/Library/LaunchAgents/com.kevintoles.mcp-gateway.plist"
MCP_JSON="$HOME/.vscode/mcp.json"
MCP_SERVER_NAME="ai-kitchen-br"
MCP_EXPECTED_URL="http://localhost:8090/mcp/sse"

is_launchctl_loaded() {
    local label="$1"
    launchctl list | awk -v target="$label" '$3 == target {found=1} END {exit found ? 0 : 1}'
}

ensure_mcp_config() {
    mkdir -p "$HOME/.vscode"

    python3 - "$MCP_JSON" "$MCP_SERVER_NAME" "$MCP_EXPECTED_URL" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
server_name = sys.argv[2]
expected_url = sys.argv[3]

data = {}
if path.exists():
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}

servers = data.get("servers")
if not isinstance(servers, dict):
    servers = {}

server = servers.get(server_name)
if not isinstance(server, dict):
    server = {}

server["type"] = "sse"
server["url"] = expected_url
if "env" not in server or not isinstance(server["env"], dict):
    server["env"] = {}

servers[server_name] = server
data["servers"] = servers

path.write_text(json.dumps(data, indent=2) + "\n")
print(f"✓ Enforced MCP endpoint for '{server_name}': {expected_url}")
PY
}

show_status() {
    echo "=== MCP Gateway + Shim Status (Option B) ==="

    if is_launchctl_loaded "$SHIM_LABEL"; then
        echo "✓ Shim LaunchAgent is loaded"
        launchctl list | grep "$SHIM_LABEL" | awk '{print "  PID: " $1}'
    else
        echo "✗ Shim LaunchAgent is NOT loaded"
    fi

    if is_launchctl_loaded "$GATEWAY_LABEL"; then
        echo "! Gateway LaunchAgent is loaded (should be disabled for Option B)"
        launchctl list | grep "$GATEWAY_LABEL" | awk '{print "  PID: " $1}'
    else
        echo "✓ Gateway LaunchAgent is NOT loaded (correct for Option B)"
    fi

    if pgrep -f "shim-mcp-gateway" > /dev/null 2>&1; then
        echo "✓ Shim is running"
        pgrep -f "shim-mcp-gateway" | while read pid; do echo "  PID: $pid"; done
    else
        echo "✗ Shim is NOT running"
    fi

    if pgrep -f "uvicorn src.main:app --host 127.0.0.1 --port 8087" > /dev/null 2>&1; then
        echo "✓ MCP gateway is running"
        pgrep -f "uvicorn src.main:app --host 127.0.0.1 --port 8087" | while read pid; do echo "  PID: $pid"; done
    else
        echo "✗ MCP gateway is NOT running"
    fi

    if lsof -i :8090 > /dev/null 2>&1; then
        echo "✓ Port 8090 (shim) is in use"
    else
        echo "✗ Port 8090 (shim) is NOT in use"
    fi

    if lsof -i :8087 > /dev/null 2>&1; then
        echo "✓ Port 8087 (gateway) is in use"
    else
        echo "✗ Port 8087 (gateway) is NOT in use"
    fi

    echo ""
}

verify_option_b() {
    local failed=0

    echo "=== Verifying Option B (shim-only startup) ==="

    if [ ! -f "$MCP_JSON" ]; then
        echo "✗ Missing $MCP_JSON"
        failed=1
    elif ! grep -q "$MCP_EXPECTED_URL" "$MCP_JSON"; then
        echo "✗ VS Code MCP URL is not set to $MCP_EXPECTED_URL"
        failed=1
    else
        echo "✓ VS Code MCP URL points to shim on :8090"
    fi

    if is_launchctl_loaded "$GATEWAY_LABEL"; then
        echo "✗ Gateway LaunchAgent is loaded (must be disabled for Option B)"
        failed=1
    else
        echo "✓ Gateway LaunchAgent is not loaded"
    fi

    if ! is_launchctl_loaded "$SHIM_LABEL"; then
        echo "✗ Shim LaunchAgent is not loaded"
        failed=1
    else
        echo "✓ Shim LaunchAgent is loaded"
    fi

    if ! lsof -i :8090 > /dev/null 2>&1; then
        echo "✗ Shim port 8090 is not listening"
        failed=1
    else
        echo "✓ Shim port 8090 is listening"
    fi

    if [ "$failed" -ne 0 ]; then
        echo ""
        echo "Option B verification FAILED"
        return 1
    fi

    echo ""
    echo "Option B verification PASSED"
    return 0
}

case "${1:-status}" in
    status) show_status ;;
    start)
        ensure_mcp_config
        launchctl load "$SHIM_PLIST" 2>/dev/null || true
        launchctl unload "$GATEWAY_PLIST" 2>/dev/null || true
        sleep 2
        show_status
        ;;
    stop)
        launchctl unload "$SHIM_PLIST" 2>/dev/null || true
        launchctl unload "$GATEWAY_PLIST" 2>/dev/null || true
        pkill -f "shim-mcp-gateway" 2>/dev/null || true
        pkill -f "uvicorn src.main:app --host 127.0.0.1 --port 8087" 2>/dev/null || true
        echo "✓ Stopped"
        ;;
    restart) "$0" stop; sleep 1; "$0" start ;;
    verify) verify_option_b ;;
    enforce-mcp)
        ensure_mcp_config
        ;;
    *) echo "Usage: $0 {status|start|stop|restart|verify|enforce-mcp}" ;;
esac
