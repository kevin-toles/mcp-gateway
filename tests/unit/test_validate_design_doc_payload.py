"""validate_design_doc payload contract — empty phase must be omitted.

The Rust validation-service deserializes `phase` into a strict enum
(DRAFT | REVIEW | APPROVED). A missing key is accepted; an empty string is
a 422 ("unknown variant ``"). The gateway handler must therefore OMIT the
key when no phase is given, and reject values the service would 422 on —
never forward a payload guaranteed to fail.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools import validate_design_doc as vdd


def _handler(dispatcher):
    sanitizer = MagicMock()
    sanitizer.sanitize.side_effect = lambda body: body
    return vdd.create_handler(dispatcher, sanitizer)


def _dispatcher(body=None):
    d = MagicMock()
    result = MagicMock()
    result.body = body if body is not None else {"verdict": "PASS"}
    d.dispatch = AsyncMock(return_value=result)
    return d


@pytest.mark.asyncio
async def test_empty_phase_is_omitted_from_payload():
    d = _dispatcher()
    await _handler(d)("# Doc")
    payload = d.dispatch.call_args.args[1]
    assert "phase" not in payload, (
        f"empty phase must be omitted (Rust 422s on ''): {payload}"
    )
    assert payload["content"] == "# Doc"


@pytest.mark.asyncio
async def test_valid_phase_is_forwarded():
    d = _dispatcher()
    await _handler(d)("# Doc", phase="DRAFT")
    payload = d.dispatch.call_args.args[1]
    assert payload["phase"] == "DRAFT"


@pytest.mark.asyncio
async def test_phase_is_case_normalized():
    d = _dispatcher()
    await _handler(d)("# Doc", phase="draft")
    assert d.dispatch.call_args.args[1]["phase"] == "DRAFT"


@pytest.mark.asyncio
async def test_invalid_phase_is_rejected_before_dispatch():
    d = _dispatcher()
    with pytest.raises(ValueError, match="DRAFT, REVIEW, APPROVED"):
        await _handler(d)("# Doc", phase="ARCHITECT")
    d.dispatch.assert_not_called()
