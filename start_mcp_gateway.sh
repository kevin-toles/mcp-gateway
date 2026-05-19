#!/bin/bash
# Start mcp-gateway natively on its traditional port :8087
# (Shim on :8088 proxies to this port)
export PATH="/Users/kevintoles/POC/mcp-gateway/.venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
cd /Users/kevintoles/POC/mcp-gateway
exec /Users/kevintoles/POC/mcp-gateway/.venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 8087
