"""Manifest-driven config — health_config must derive from services.toml.

The manifest (config/services.toml) is the single source of truth shared
with the Rust daemon. These tests specify that the Python side reads the
same file instead of maintaining its own duplicated command/tier tables
(assessment finding #1: config duplication is the primary fragility driver).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from src.config.health_config import (
    SERVICE_STARTUP_COMMANDS,
    SERVICE_TIERS,
)

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "config" / "services.toml"


@pytest.fixture(scope="module")
def manifest() -> dict:
    with open(MANIFEST_PATH, "rb") as f:
        return tomllib.load(f)["services"]


def test_manifest_file_exists():
    assert MANIFEST_PATH.is_file(), f"manifest missing: {MANIFEST_PATH}"


def test_every_manifest_service_has_startup_command(manifest):
    for name in manifest:
        assert name in SERVICE_STARTUP_COMMANDS, (
            f"{name} in manifest but missing from SERVICE_STARTUP_COMMANDS — "
            "health_config must derive commands from the manifest"
        )


def test_aliases_resolve_to_same_command(manifest):
    for name, svc in manifest.items():
        for alias in svc.get("aliases", []):
            assert alias in SERVICE_STARTUP_COMMANDS, f"alias {alias} missing"
            assert SERVICE_STARTUP_COMMANDS[alias] == SERVICE_STARTUP_COMMANDS[name]


def test_startup_commands_have_no_unexpanded_placeholders():
    for name, cmd in SERVICE_STARTUP_COMMANDS.items():
        assert "${POC_ROOT}" not in cmd, f"{name}: unexpanded POC_ROOT in {cmd!r}"


def test_validation_service_command_uses_rust_binary():
    cmd = SERVICE_STARTUP_COMMANDS["validation-service"]
    assert "validation-service/rust" in cmd, cmd
    assert "PORT=8091" in cmd, cmd
    assert "uvicorn" not in cmd, f"stale Python command: {cmd!r}"


def test_service_tiers_match_manifest(manifest):
    for name, svc in manifest.items():
        assert SERVICE_TIERS.get(name) == svc["tier"], (
            f"{name}: SERVICE_TIERS says {SERVICE_TIERS.get(name)!r}, "
            f"manifest says {svc['tier']!r}"
        )
        for alias in svc.get("aliases", []):
            assert SERVICE_TIERS.get(alias) == svc["tier"]


def test_no_python_side_services_missing_from_manifest(manifest):
    """Every command/tier key must trace back to the manifest — no orphans."""
    manifest_keys = set(manifest)
    for svc in manifest.values():
        manifest_keys.update(svc.get("aliases", []))
    orphans = set(SERVICE_STARTUP_COMMANDS) - manifest_keys
    assert not orphans, f"commands not in manifest: {orphans}"
    orphans = set(SERVICE_TIERS) - manifest_keys
    assert not orphans, f"tiers not in manifest: {orphans}"


# ── config.py HEALTH_PROXY_SERVICE_CONFIG derivation ─────────────────────


def test_hpsc_restart_commands_derive_from_manifest(manifest):
    """Every HPSC restart_command must come from the manifest, not hand
    duplication — a manifest edit must propagate to restart behavior."""
    from src.core.config import settings

    for key, entry in settings.HEALTH_PROXY_SERVICE_CONFIG.items():
        cmd = entry["restart_command"]
        name = entry["name"]
        # unified-search-rs is an alias of semantic-search in the manifest
        svc = manifest.get(name) or next(
            (s for s in manifest.values() if name in s.get("aliases", [])), None
        )
        assert svc is not None, f"HPSC entry {key} -> {name} not in manifest"
        assert f"lsof -ti:{svc['port']}" in cmd, (
            f"{key}: restart must kill manifest port {svc['port']}: {cmd!r}"
        )
        assert "${POC_ROOT}" not in cmd, f"{key}: unexpanded placeholder"


def test_hpsc_validation_service_restart_uses_rust(manifest):
    from src.core.config import settings

    cmd = settings.HEALTH_PROXY_SERVICE_CONFIG["validation-service"]["restart_command"]
    assert "validation-service/rust" in cmd
    assert "uvicorn" not in cmd
