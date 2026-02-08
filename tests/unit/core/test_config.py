"""Tests for Settings — WBS-MCP1.2 (RED).

Verifies that Settings:
- Loads typed defaults for all fields
- Reads overrides from MCP_GATEWAY_ prefixed env vars
- Provides correct backend service URLs
- Handles TLS configuration fields
- Handles OIDC configuration fields
"""

import os

import pytest


class TestSettingsDefaults:
    """AC-1.3: Settings loaded from environment variables with typed defaults."""

    def test_service_name_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.SERVICE_NAME == "mcp-gateway"

    def test_service_version_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.SERVICE_VERSION == "0.1.0"

    def test_host_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.HOST == "0.0.0.0"

    def test_port_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.PORT == 8087

    def test_port_is_int(self):
        from src.core.config import Settings

        settings = Settings()
        assert isinstance(settings.PORT, int)


class TestSettingsBackendURLs:
    """AC-1.5: Service runs independently — URLs point to platform services."""

    def test_llm_gateway_url_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.LLM_GATEWAY_URL == "http://localhost:8080"

    def test_semantic_search_url_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.SEMANTIC_SEARCH_URL == "http://localhost:8081"

    def test_ai_agents_url_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.AI_AGENTS_URL == "http://localhost:8082"

    def test_code_orchestrator_url_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.CODE_ORCHESTRATOR_URL == "http://localhost:8083"


class TestSettingsEnvOverrides:
    """AC-1.3: Settings loaded from environment variables with MCP_GATEWAY_ prefix."""

    def test_port_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_GATEWAY_PORT", "9999")
        from src.core.config import Settings

        settings = Settings()
        assert settings.PORT == 9999

    def test_service_name_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_GATEWAY_SERVICE_NAME", "custom-gateway")
        from src.core.config import Settings

        settings = Settings()
        assert settings.SERVICE_NAME == "custom-gateway"

    def test_llm_gateway_url_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_GATEWAY_LLM_GATEWAY_URL", "http://remote:8080")
        from src.core.config import Settings

        settings = Settings()
        assert settings.LLM_GATEWAY_URL == "http://remote:8080"

    def test_auth_enabled_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_GATEWAY_AUTH_ENABLED", "true")
        from src.core.config import Settings

        settings = Settings()
        assert settings.AUTH_ENABLED is True

    def test_redis_url_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_GATEWAY_REDIS_URL", "redis://remote:6379/1")
        from src.core.config import Settings

        settings = Settings()
        assert settings.REDIS_URL == "redis://remote:6379/1"


class TestSettingsOIDC:
    """AC-1.3: OIDC configuration fields present with defaults."""

    def test_oidc_jwks_url_default_empty(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.OIDC_JWKS_URL == ""

    def test_oidc_issuer_default_empty(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.OIDC_ISSUER == ""

    def test_oidc_audience_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.OIDC_AUDIENCE == "ai-platform-tools"

    def test_auth_enabled_default_false(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.AUTH_ENABLED is False


class TestSettingsRateLimiting:
    """AC-1.3: Rate limiting configuration fields."""

    def test_redis_url_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.REDIS_URL == "redis://localhost:6379"

    def test_rate_limit_rpm_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.RATE_LIMIT_RPM == 100

    def test_rate_limit_rpm_is_int(self):
        from src.core.config import Settings

        settings = Settings()
        assert isinstance(settings.RATE_LIMIT_RPM, int)


class TestSettingsTLS:
    """AC-1.6: TLS configuration fields for production HTTPS."""

    def test_tls_enabled_default_false(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.TLS_ENABLED is False

    def test_tls_cert_path_default_empty(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.TLS_CERT_PATH == ""

    def test_tls_key_path_default_empty(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.TLS_KEY_PATH == ""

    def test_tls_min_version_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.TLS_MIN_VERSION == "TLSv1.3"

    def test_tls_enabled_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_GATEWAY_TLS_ENABLED", "true")
        from src.core.config import Settings

        settings = Settings()
        assert settings.TLS_ENABLED is True

    def test_tls_cert_path_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_GATEWAY_TLS_CERT_PATH", "/etc/ssl/cert.pem")
        from src.core.config import Settings

        settings = Settings()
        assert settings.TLS_CERT_PATH == "/etc/ssl/cert.pem"

    def test_tls_key_path_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_GATEWAY_TLS_KEY_PATH", "/etc/ssl/key.pem")
        from src.core.config import Settings

        settings = Settings()
        assert settings.TLS_KEY_PATH == "/etc/ssl/key.pem"


class TestSettingsAudit:
    """AC-1.3: Audit log configuration."""

    def test_audit_log_path_default(self):
        from src.core.config import Settings

        settings = Settings()
        assert settings.AUDIT_LOG_PATH == "logs/audit.jsonl"

    def test_audit_log_path_override_from_env(self, monkeypatch):
        monkeypatch.setenv("MCP_GATEWAY_AUDIT_LOG_PATH", "/var/log/audit.jsonl")
        from src.core.config import Settings

        settings = Settings()
        assert settings.AUDIT_LOG_PATH == "/var/log/audit.jsonl"


class TestSettingsEnvPrefix:
    """AC-1.3: Verify MCP_GATEWAY_ prefix is enforced."""

    def test_unprefixed_env_var_ignored(self, monkeypatch):
        """Settings should NOT load from unprefixed env vars."""
        monkeypatch.setenv("PORT", "1111")
        from src.core.config import Settings

        settings = Settings()
        # PORT without prefix should NOT override the default
        assert settings.PORT == 8087


class TestTLSValidation:
    """AC-1.6: TLS configuration validation and ssl_config helper."""

    def test_get_ssl_config_returns_none_when_disabled(self):
        """When TLS_ENABLED=false, ssl_config should return None."""
        from src.core.config import Settings, get_ssl_config

        settings = Settings()
        assert settings.TLS_ENABLED is False
        assert get_ssl_config(settings) is None

    def test_get_ssl_config_raises_when_cert_missing(self, monkeypatch, tmp_path):
        """When TLS_ENABLED=true but cert file doesn't exist, should raise."""
        monkeypatch.setenv("MCP_GATEWAY_TLS_ENABLED", "true")
        monkeypatch.setenv("MCP_GATEWAY_TLS_CERT_PATH", str(tmp_path / "nonexistent.pem"))
        monkeypatch.setenv("MCP_GATEWAY_TLS_KEY_PATH", str(tmp_path / "key.pem"))
        from src.core.config import Settings, get_ssl_config

        settings = Settings()
        with pytest.raises(FileNotFoundError, match="TLS certificate"):
            get_ssl_config(settings)

    def test_get_ssl_config_raises_when_key_missing(self, monkeypatch, tmp_path):
        """When TLS_ENABLED=true but key file doesn't exist, should raise."""
        cert_file = tmp_path / "cert.pem"
        cert_file.write_text("CERT")
        monkeypatch.setenv("MCP_GATEWAY_TLS_ENABLED", "true")
        monkeypatch.setenv("MCP_GATEWAY_TLS_CERT_PATH", str(cert_file))
        monkeypatch.setenv("MCP_GATEWAY_TLS_KEY_PATH", str(tmp_path / "nonexistent.pem"))
        from src.core.config import Settings, get_ssl_config

        settings = Settings()
        with pytest.raises(FileNotFoundError, match="TLS private key"):
            get_ssl_config(settings)

    def test_get_ssl_config_returns_dict_when_valid(self, monkeypatch, tmp_path):
        """When TLS_ENABLED=true and files exist, return ssl config dict."""
        cert_file = tmp_path / "cert.pem"
        key_file = tmp_path / "key.pem"
        cert_file.write_text("CERT")
        key_file.write_text("KEY")
        monkeypatch.setenv("MCP_GATEWAY_TLS_ENABLED", "true")
        monkeypatch.setenv("MCP_GATEWAY_TLS_CERT_PATH", str(cert_file))
        monkeypatch.setenv("MCP_GATEWAY_TLS_KEY_PATH", str(key_file))
        from src.core.config import Settings, get_ssl_config

        settings = Settings()
        ssl_cfg = get_ssl_config(settings)
        assert ssl_cfg is not None
        assert ssl_cfg["ssl_certfile"] == str(cert_file)
        assert ssl_cfg["ssl_keyfile"] == str(key_file)

    def test_get_ssl_config_raises_when_paths_empty(self, monkeypatch):
        """When TLS_ENABLED=true but paths are empty strings, should raise."""
        monkeypatch.setenv("MCP_GATEWAY_TLS_ENABLED", "true")
        monkeypatch.setenv("MCP_GATEWAY_TLS_CERT_PATH", "")
        monkeypatch.setenv("MCP_GATEWAY_TLS_KEY_PATH", "")
        from src.core.config import Settings, get_ssl_config

        settings = Settings()
        with pytest.raises(ValueError, match="TLS_CERT_PATH.*required"):
            get_ssl_config(settings)
