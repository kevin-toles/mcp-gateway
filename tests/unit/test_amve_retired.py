"""AMVE is retired (user, 2026-11-26: "SAS replaced the AMVE ... AMVE is no longer used").

No amve_* tool may be registered, routed, or mapped to a service; HWC must not be able to spawn the AMVE service; and
preflight must not probe it. Every former amve_* capability has a struct-analyzer (sa_*) equivalent, which stays.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

from src import tool_dispatcher
from src.core.config import Settings
from src.tool_registry import _INPUT_MODELS

ROOT = Path(__file__).resolve().parents[2]
SA_REPLACEMENTS = {
    "sa_detect_patterns", "sa_detect_boundaries", "sa_build_call_graph", "sa_detect_dead_code", "sa_detect_drift",
    "sa_evaluate_fitness", "sa_extract_architecture", "sa_detect_messaging", "sa_architecture_mapping_log",
}


def test_no_amve_tool_is_in_the_registry_or_tools_yaml() -> None:
    assert not [n for n in _INPUT_MODELS if n.startswith("amve_")]
    tools = yaml.safe_load((ROOT / "config" / "tools.yaml").read_text())["tools"]
    assert not [t["name"] for t in tools if t["name"].startswith("amve_")]


def test_no_amve_tool_is_routed_or_mapped_to_a_service() -> None:
    routes = tool_dispatcher._build_routes(Settings())
    assert not [n for n in routes if n.startswith("amve_")]
    assert not [n for n, svc in tool_dispatcher._TOOL_SERVICE_NAMES.items() if n.startswith("amve_") or "amve" in svc]


def test_hwc_has_no_amve_service_and_settings_no_amve_url() -> None:
    services = tomllib.loads((ROOT / "config" / "services.toml").read_text())["services"]
    assert "amve" not in services
    assert not [k for k in Settings.model_fields if "AMVE" in k]


def test_preflight_does_not_probe_amve() -> None:
    assert not re.search(r'"amve"', (ROOT / "src" / "preflight.py").read_text())


def test_every_replacement_sa_tool_stays_routed() -> None:
    routes = tool_dispatcher._build_routes(Settings())
    assert SA_REPLACEMENTS <= set(routes)


def test_no_core_service_map_names_amve() -> None:
    """Health tiers, idle-timeout settings and warm-promotion URL maps carry no amve entry."""
    for rel in ("src/core/health_config.py", "src/core/idle_timeout.py", "src/core/cold_warm_promoter.py"):
        assert not re.search(r'"amve"\s*:', (ROOT / rel).read_text()), rel
    assert not [k for k in Settings.model_fields if "AMVE" in k]
