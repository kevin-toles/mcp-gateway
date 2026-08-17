#!/bin/bash
# Launchd startup script for mcp-gateway.
# Reclaims port 8087 if a stale process holds it (prevents crash loop).
/usr/sbin/lsof -ti:8087 | xargs kill -9 2>/dev/null || true
sleep 1
cd "$(dirname "$0")"
exec .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8087
