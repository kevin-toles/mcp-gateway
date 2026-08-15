"""BEH-04.01 Step 2 — Tests for build_model_map() (AC-ASCP.18 routing half).

Covers:
  - Returns empty dict when no BYOK env vars set (no tier-3 activation)
  - Returns all pipeline roles mapped to Anthropic default when ANTHROPIC_API_KEY set
  - Returns all pipeline roles mapped to OpenAI default when OPENAI_API_KEY set
  - PLATFORM_PREFERRED_MODEL overrides provider default model name
  - PLATFORM_PREFERRED_MODEL alone (without an API key) still builds map
  - Priority: PLATFORM_PREFERRED_MODEL > ANTHROPIC_API_KEY > OPENAI_API_KEY
  - All PIPELINE_PROTOCOL roles are present in the returned map
"""

from __future__ import annotations

from unittest.mock import patch


_ALL_ROLES = {"ARCHITECT", "ENGINEER", "REVIEWER", "FINALIZER", "VALIDATOR"}


class TestBuildModelMapNoByok:
    def test_returns_empty_dict_when_no_env_vars(self):
        """No BYOK env vars → empty dict (tier-3 not activated)."""
        from src.byok.model_map_builder import build_model_map

        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PLATFORM_PREFERRED_MODEL")}
        with patch.dict("os.environ", env, clear=True):
            assert build_model_map() == {}

    def test_returns_empty_dict_when_all_vars_empty_string(self):
        """Explicitly empty env vars are treated as unset."""
        from src.byok.model_map_builder import build_model_map

        with patch.dict("os.environ", {
            "ANTHROPIC_API_KEY": "",
            "OPENAI_API_KEY": "",
            "PLATFORM_PREFERRED_MODEL": "",
        }):
            assert build_model_map() == {}


class TestBuildModelMapAnthropicKey:
    def test_all_roles_mapped_to_anthropic_default(self):
        """ANTHROPIC_API_KEY → all roles map to claude-sonnet-4-20250514."""
        from src.byok.model_map_builder import build_model_map, _ANTHROPIC_DEFAULT_MODEL

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test123"}):
            result = build_model_map()

        assert set(result.keys()) == _ALL_ROLES
        assert all(v == _ANTHROPIC_DEFAULT_MODEL for v in result.values())

    def test_all_pipeline_roles_present(self):
        """Every PIPELINE_PROTOCOL role must appear in the map."""
        from src.byok.model_map_builder import build_model_map

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test123"}):
            result = build_model_map()

        assert result.keys() == _ALL_ROLES


class TestBuildModelMapOpenAIKey:
    def test_all_roles_mapped_to_openai_default(self):
        """OPENAI_API_KEY alone → all roles map to gpt-4o."""
        from src.byok.model_map_builder import build_model_map, _OPENAI_DEFAULT_MODEL

        env = {k: v for k, v in __import__("os").environ.items()
               if k != "ANTHROPIC_API_KEY"}
        env["OPENAI_API_KEY"] = "sk-openai-test123"
        with patch.dict("os.environ", env, clear=True):
            result = build_model_map()

        assert set(result.keys()) == _ALL_ROLES
        assert all(v == _OPENAI_DEFAULT_MODEL for v in result.values())


class TestBuildModelMapPreferredModel:
    def test_platform_preferred_model_overrides_anthropic_default(self):
        """PLATFORM_PREFERRED_MODEL beats provider default when both set."""
        from src.byok.model_map_builder import build_model_map

        with patch.dict("os.environ", {
            "ANTHROPIC_API_KEY": "sk-ant-test123",
            "PLATFORM_PREFERRED_MODEL": "claude-opus-4-7",
        }):
            result = build_model_map()

        assert all(v == "claude-opus-4-7" for v in result.values())

    def test_platform_preferred_model_without_api_key_still_builds_map(self):
        """PLATFORM_PREFERRED_MODEL alone activates tier-3 override."""
        from src.byok.model_map_builder import build_model_map

        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
        env["PLATFORM_PREFERRED_MODEL"] = "qwen3.5-9b"
        with patch.dict("os.environ", env, clear=True):
            result = build_model_map()

        assert set(result.keys()) == _ALL_ROLES
        assert all(v == "qwen3.5-9b" for v in result.values())

    def test_platform_preferred_model_overrides_openai_default(self):
        """PLATFORM_PREFERRED_MODEL beats gpt-4o when OPENAI_API_KEY set."""
        from src.byok.model_map_builder import build_model_map

        env = {k: v for k, v in __import__("os").environ.items()
               if k != "ANTHROPIC_API_KEY"}
        env.update({
            "OPENAI_API_KEY": "sk-openai-test",
            "PLATFORM_PREFERRED_MODEL": "gpt-4o-mini",
        })
        with patch.dict("os.environ", env, clear=True):
            result = build_model_map()

        assert all(v == "gpt-4o-mini" for v in result.values())


class TestBuildModelMapAll:
    def test_exports_only_build_model_map(self):
        import src.byok.model_map_builder as mod

        assert mod.__all__ == ["build_model_map"]

    def test_returned_map_is_independent_copy(self):
        """Mutating the returned dict must not affect subsequent calls."""
        from src.byok.model_map_builder import build_model_map

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-x"}):
            first = build_model_map()
            first["ARCHITECT"] = "mutated"
            second = build_model_map()

        assert second["ARCHITECT"] != "mutated"
