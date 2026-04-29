#!/usr/bin/env python3
"""Test if mirror_cre_repos hangs when called via MCP SSE."""

import asyncio
import json

import httpx


async def test_mirror_via_sse():
    """Call mirror_cre_repos via SSE and see if it hangs."""
    url = "http://localhost:8087/mcp/sse"

    # MCP SSE request format
    request_data = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "mirror_cre_repos",
            "arguments": {
                "repo_ids": ["deepspeed"],
                "source_url": "https://github.com/deepspeed-ai/DeepSpeed",
                "domain": "ml-training-infrastructure",
                "auto_continue": True,
                "dry_run": True,
            },
        },
    }

    print("Calling mirror_cre_repos via SSE...")
    print(f"Request: {json.dumps(request_data, indent=2)}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # SSE uses EventSource, but we can test with a simple POST
            response = await client.post(url, json=request_data)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
    except httpx.TimeoutException:
        print("TIMEOUT - tool hung for 30+ seconds")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_mirror_via_sse())
