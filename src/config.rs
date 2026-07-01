// config.rs
// Deployment mode detection and configuration validation for the mcp-gateway shim.
//
// Supports three modes:
//   - Docker:   Everything runs in containers. Python gateway managed by Docker.
//               Validation checks /var/run/docker.sock exists.
//   - Hybrid:   Infrastructure in Docker, services run natively (default).
//               Validation checks POC_ROOT env var is set.
//   - Native:   Everything runs natively.
//               Validation checks POC_ROOT env var is set.

use std::path::Path;
use serde::Serialize;

/// Deployment mode for the mcp-gateway shim.
///
/// Determines how the shim discovers and manages the Python gateway process:
/// - `Docker`: Gateway is managed by Docker (always running), shim should NOT
///   attempt to spawn it.
/// - `Hybrid`: Gateway is spawned natively. Infrastructure runs in Docker.
/// - `Native`: Everything runs natively.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum DeploymentMode {
    Docker,
    Hybrid,
    Native,
}

impl DeploymentMode {
    /// Detect deployment mode from the `DEPLOYMENT_MODE` environment variable.
    ///
    /// Defaults to `Hybrid` when the env var is unset or empty.
    /// Acceptable values (case-insensitive): `"docker"`, `"hybrid"`, `"native"`.
    pub fn from_env() -> Self {
        match std::env::var("DEPLOYMENT_MODE")
            .ok()
            .as_deref()
        {
            Some("docker") => DeploymentMode::Docker,
            Some("hybrid") => DeploymentMode::Hybrid,
            Some("native") => DeploymentMode::Native,
            _ => DeploymentMode::Hybrid, // default
        }
    }

    /// Return the concrete `ServiceRuntime` string for `Auto` resolution.
    ///
    /// This maps `DeploymentMode` back to the string that `spawn::resolve_runtime`
    /// expects: `"docker"`, `"hybrid"`, or `"native"`.
    pub fn as_deployment_str(&self) -> &'static str {
        match self {
            DeploymentMode::Docker => "docker",
            DeploymentMode::Hybrid => "hybrid",
            DeploymentMode::Native => "native",
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
pub fn validate_config(mode: &DeploymentMode) -> Result<(), String> {
    match mode {
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
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Env-dependent tests ──────────────────────────────────────────────
    // Rust tests run in parallel by default, and cargo test does not
    // guarantee that --test-threads=1 is used.  Since DEPLOYMENT_MODE is a
    // global env var, tests that read it would race with each other.
    // We avoid the race by running all env-dependent assertions inside a
    // single sequential #[test].

    #[test]
    fn test_all_deployment_modes_sequentially() {
        // 1. Default (no env) → Hybrid
        unsafe { std::env::remove_var("DEPLOYMENT_MODE"); }
        assert_eq!(DeploymentMode::from_env(), DeploymentMode::Hybrid);
        assert_eq!(DeploymentMode::Hybrid.as_deployment_str(), "hybrid");

        // 2. docker env → Docker
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "docker"); }
        let dm = DeploymentMode::from_env();
        assert_eq!(dm, DeploymentMode::Docker);
        assert_eq!(dm.as_deployment_str(), "docker");

        // 3. native env → Native
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "native"); }
        let dm = DeploymentMode::from_env();
        assert_eq!(dm, DeploymentMode::Native);
        assert_eq!(dm.as_deployment_str(), "native");

        // 4. hybrid env → Hybrid
        unsafe { std::env::set_var("DEPLOYMENT_MODE", "hybrid"); }
        let dm = DeploymentMode::from_env();
        assert_eq!(dm, DeploymentMode::Hybrid);
        assert_eq!(dm.as_deployment_str(), "hybrid");
    }

    // ── validate_config tests (no env races: we pass mode directly) ──────

    #[test]
    fn test_validate_config_fails_without_poc_root_in_hybrid() {
        unsafe { std::env::remove_var("POC_ROOT"); }
        let result = validate_config(&DeploymentMode::Hybrid);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("POC_ROOT"));
    }

    #[test]
    fn test_validate_config_fails_without_poc_root_in_native() {
        unsafe { std::env::remove_var("POC_ROOT"); }
        let result = validate_config(&DeploymentMode::Native);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("POC_ROOT"));
    }
}
