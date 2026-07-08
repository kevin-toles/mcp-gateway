// config.rs
// Deployment mode detection and configuration validation for the mcp-gateway shim.
//
// Supports four modes:
//   - Docker:   Everything runs in containers. Python gateway managed by Docker.
//               Validation checks /var/run/docker.sock exists.
//   - Hybrid:   Infrastructure in Docker, services run natively (default).
//               Validation checks POC_ROOT env var is set.
//   - Native:   Everything runs natively.
//               Validation checks POC_ROOT env var is set.
//   - Auto:     Auto-detect Docker vs local. Detects Docker by checking
//               /var/run/docker.sock; falls back to Hybrid.

use std::path::Path;
use std::str::FromStr;
use serde::Serialize;

/// Deployment mode for the mcp-gateway shim.
///
/// Determines how the shim discovers and manages the Python gateway process:
/// - `Docker`: Gateway is managed by Docker (always running), shim should NOT
///   attempt to spawn it.
/// - `Hybrid`: Gateway is spawned natively. Infrastructure runs in Docker.
/// - `Native`: Everything runs natively.
/// - `Auto`:   Auto-detect based on environment (Docker if /var/run/docker.sock
///             exists, otherwise Hybrid).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum DeploymentMode {
    Docker,
    Hybrid,
    Native,
    Auto,
}

impl std::fmt::Display for DeploymentMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeploymentMode::Docker => write!(f, "docker"),
            DeploymentMode::Hybrid => write!(f, "hybrid"),
            DeploymentMode::Native => write!(f, "native"),
            DeploymentMode::Auto => write!(f, "auto"),
        }
    }
}

impl FromStr for DeploymentMode {
    type Err = String;

    /// Parse a deployment mode string (case-insensitive).
    ///
    /// Accepts: `"docker"`, `"hybrid"`, `"native"`, `"auto"`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "docker" => Ok(DeploymentMode::Docker),
            "hybrid" => Ok(DeploymentMode::Hybrid),
            "native" => Ok(DeploymentMode::Native),
            "auto" => Ok(DeploymentMode::Auto),
            other => Err(format!(
                "unknown deployment mode '{}': expected one of docker, hybrid, native, auto",
                other
            )),
        }
    }
}

impl DeploymentMode {
    /// Detect deployment mode from the `DEPLOYMENT_MODE` environment variable.
    ///
    /// Defaults to `Hybrid` when the env var is unset or empty.
    /// Acceptable values (case-insensitive): `"docker"`, `"hybrid"`, `"native"`, `"auto"`.
    pub fn from_env() -> Self {
        match std::env::var("DEPLOYMENT_MODE")
            .ok()
            .as_deref()
        {
            Some("docker") => DeploymentMode::Docker,
            Some("hybrid") => DeploymentMode::Hybrid,
            Some("native") => DeploymentMode::Native,
            Some("auto") => DeploymentMode::Auto,
            _ => DeploymentMode::Hybrid, // default
        }
    }

    /// Return the concrete `ServiceRuntime` string for `Auto` resolution.
    ///
    /// This maps `DeploymentMode` back to the string that `spawn::resolve_runtime`
    /// expects: `"docker"`, `"hybrid"`, or `"native"`.
    ///
    /// For `Auto`, this resolves to the detected mode's string.
    pub fn as_deployment_str(&self) -> &'static str {
        match self {
            DeploymentMode::Docker => "docker",
            DeploymentMode::Hybrid => "hybrid",
            DeploymentMode::Native => "native",
            DeploymentMode::Auto => Self::detect_from_env().as_deployment_str(),
        }
    }

    /// Resolve `Auto` to a concrete mode by probing the environment.
    ///
    /// Detection logic:
    /// 1. If `DEPLOYMENT_MODE` is explicitly set to a concrete value, use it.
    /// 2. If `/var/run/docker.sock` exists → `Docker`.
    /// 3. Otherwise → `Hybrid` (default).
    pub fn detect_from_env() -> Self {
        // Check if DEPLOYMENT_MODE is explicitly set to a concrete (non-auto) value
        match std::env::var("DEPLOYMENT_MODE")
            .ok()
            .as_deref()
        {
            Some("docker") => return DeploymentMode::Docker,
            Some("hybrid") => return DeploymentMode::Hybrid,
            Some("native") => return DeploymentMode::Native,
            _ => {}
        }

        // Probe for Docker environment
        if Path::new("/var/run/docker.sock").exists() {
            DeploymentMode::Docker
        } else {
            DeploymentMode::Hybrid
        }
    }
}

/// Validate the environment is consistent with the detected deployment mode.
///
/// Returns `Ok(())` if validation passes, or `Err(String)` with a description
/// of what's missing or misconfigured.
///
/// # Checks
///
/// - **Docker**: `/var/run/docker.sock` must exist.
/// - **Hybrid / Native**: `POC_ROOT` environment variable must be set.
/// - **Auto**: Resolved concrete mode's checks apply (POC_ROOT for Hybrid,
///   docker.sock for Docker).
pub fn validate_config(mode: &DeploymentMode) -> Result<(), String> {
    // Resolve Auto before validating
    let concrete = match mode {
        DeploymentMode::Auto => DeploymentMode::detect_from_env(),
        other => other.clone(),
    };

    match concrete {
        DeploymentMode::Docker => {
            if !Path::new("/var/run/docker.sock").exists() {
                return Err("Docker mode requires /var/run/docker.sock".to_string());
            }
        }
        DeploymentMode::Hybrid | DeploymentMode::Native => {
            if std::env::var("POC_ROOT").is_err() {
                return Err("POC_ROOT env var required in hybrid/native mode".to_string());
            }
        }
        DeploymentMode::Auto => {
            // Should not reach here since Auto is resolved above
            unreachable!("Auto mode should have been resolved before validation");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Display tests (no env races) ─────────────────────────────────────

    #[test]
    fn test_display_docker() {
        assert_eq!(format!("{}", DeploymentMode::Docker), "docker");
    }

    #[test]
    fn test_display_hybrid() {
        assert_eq!(format!("{}", DeploymentMode::Hybrid), "hybrid");
    }

    #[test]
    fn test_display_native() {
        assert_eq!(format!("{}", DeploymentMode::Native), "native");
    }

    #[test]
    fn test_display_auto() {
        assert_eq!(format!("{}", DeploymentMode::Auto), "auto");
    }

    // ── FromStr tests (no env races) ─────────────────────────────────────

    #[test]
    fn test_from_str_docker() {
        assert_eq!("docker".parse::<DeploymentMode>().unwrap(), DeploymentMode::Docker);
    }

    #[test]
    fn test_from_str_hybrid() {
        assert_eq!("hybrid".parse::<DeploymentMode>().unwrap(), DeploymentMode::Hybrid);
    }

    #[test]
    fn test_from_str_native() {
        assert_eq!("native".parse::<DeploymentMode>().unwrap(), DeploymentMode::Native);
    }

    #[test]
    fn test_from_str_auto() {
        assert_eq!("auto".parse::<DeploymentMode>().unwrap(), DeploymentMode::Auto);
    }

    #[test]
    fn test_from_str_case_insensitive() {
        assert_eq!("DOCKER".parse::<DeploymentMode>().unwrap(), DeploymentMode::Docker);
        assert_eq!("Hybrid".parse::<DeploymentMode>().unwrap(), DeploymentMode::Hybrid);
        assert_eq!("NATIVE".parse::<DeploymentMode>().unwrap(), DeploymentMode::Native);
        assert_eq!("AuTo".parse::<DeploymentMode>().unwrap(), DeploymentMode::Auto);
    }

    #[test]
    fn test_from_str_invalid() {
        let result = "unknown".parse::<DeploymentMode>();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown deployment mode"));
    }

    #[test]
    fn test_from_str_empty() {
        let result = "".parse::<DeploymentMode>();
        assert!(result.is_err());
    }

    // ── Env-dependent tests ──────────────────────────────────────────────
    // These mutate global env vars and must run sequentially (serial_test).

    #[test]
    #[serial_test::serial]
    fn test_from_env_default_to_hybrid() {
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
        assert_eq!(DeploymentMode::from_env(), DeploymentMode::Hybrid);
    }

    #[test]
    #[serial_test::serial]
    fn test_from_env_docker() {
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "docker"); }
        assert_eq!(DeploymentMode::from_env(), DeploymentMode::Docker);
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
    }

    #[test]
    #[serial_test::serial]
    fn test_from_env_hybrid() {
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "hybrid"); }
        assert_eq!(DeploymentMode::from_env(), DeploymentMode::Hybrid);
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
    }

    #[test]
    #[serial_test::serial]
    fn test_from_env_native() {
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "native"); }
        assert_eq!(DeploymentMode::from_env(), DeploymentMode::Native);
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
    }

    #[test]
    #[serial_test::serial]
    fn test_from_env_auto() {
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "auto"); }
        assert_eq!(DeploymentMode::from_env(), DeploymentMode::Auto);
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
    }

    #[test]
    #[serial_test::serial]
    fn test_as_deployment_str_docker() {
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "docker"); }
        assert_eq!(DeploymentMode::Docker.as_deployment_str(), "docker");
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
    }

    #[test]
    #[serial_test::serial]
    fn test_as_deployment_str_hybrid() {
        assert_eq!(DeploymentMode::Hybrid.as_deployment_str(), "hybrid");
    }

    #[test]
    #[serial_test::serial]
    fn test_as_deployment_str_native() {
        assert_eq!(DeploymentMode::Native.as_deployment_str(), "native");
    }

    #[test]
    #[serial_test::serial]
    fn test_as_deployment_str_auto_resolves() {
        // Auto resolves; on this machine docker.sock exists → Docker
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
        assert_eq!(DeploymentMode::Auto.as_deployment_str(), "docker");
    }

    // ── validate_config tests ────────────────────────────────────────────

    #[test]
    #[serial_test::serial]
    fn test_validate_config_fails_without_poc_root_in_hybrid() {
        unsafe { std::env::remove_var("POC_ROOT"); }
        let result = validate_config(&DeploymentMode::Hybrid);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("POC_ROOT"));
    }

    #[test]
    #[serial_test::serial]
    fn test_validate_config_fails_without_poc_root_in_native() {
        unsafe { std::env::remove_var("POC_ROOT"); }
        let result = validate_config(&DeploymentMode::Native);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("POC_ROOT"));
    }

    #[test]
    #[serial_test::serial]
    fn test_validate_config_succeeds_with_poc_root_in_hybrid() {
        unsafe { std::env::set_var("POC_ROOT", "/tmp"); }
        let result = validate_config(&DeploymentMode::Hybrid);
        assert!(result.is_ok());
        unsafe { std::env::remove_var("POC_ROOT"); }
    }

    #[test]
    #[serial_test::serial]
    fn test_validate_config_auto_no_poc_root() {
        // Auto → detect_from_env → docker.sock exists → Docker → no POC_ROOT check
        // So this should succeed (Docker mode doesn't check POC_ROOT)
        unsafe {
            std::env::remove_var("DEPLOYMENT_MODE");
            std::env::remove_var("POC_ROOT");
        }
        assert!(validate_config(&DeploymentMode::Auto).is_ok());
    }

    #[test]
    #[serial_test::serial]
    fn test_validate_config_docker_succeeds() {
        // Docker mode checks docker.sock only (exists on this machine)
        unsafe {
            std::env::remove_var("POC_ROOT");
            std::env::remove_var("DEPLOYMENT_MODE");
        }
        assert!(validate_config(&DeploymentMode::Docker).is_ok());
    }

    // ── detect_from_env tests (sequential env var) ──────────────────────

    #[test]
    #[serial_test::serial]
    fn test_detect_from_env_explicit_docker() {
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "docker"); }
        assert_eq!(DeploymentMode::detect_from_env(), DeploymentMode::Docker);
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
    }

    #[test]
    #[serial_test::serial]
    fn test_detect_from_env_explicit_hybrid() {
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "hybrid"); }
        assert_eq!(DeploymentMode::detect_from_env(), DeploymentMode::Hybrid);
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
    }

    #[test]
    #[serial_test::serial]
    fn test_detect_from_env_explicit_native() {
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "native"); }
        assert_eq!(DeploymentMode::detect_from_env(), DeploymentMode::Native);
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
    }

    #[test]
    #[serial_test::serial]
    fn test_detect_from_env_auto_falls_back() {
        // Auto env → falls through to docker.sock probe → Docker (exists on this machine)
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "auto"); }
        assert_eq!(DeploymentMode::detect_from_env(), DeploymentMode::Docker);
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
    }

    #[test]
    #[serial_test::serial]
    fn test_detect_from_env_no_env_falls_back() {
        // No env → docker.sock probe → Docker (exists on this machine)
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
        assert_eq!(DeploymentMode::detect_from_env(), DeploymentMode::Docker);
    }
}
