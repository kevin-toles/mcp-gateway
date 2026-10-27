//! Service manifest loader — single source of truth for service config.
//!
//! Parses `config/services.toml`, the one file that defines every platform
//! service's port, tier, runtime, health/ready paths, and spawn command.
//! Both this Rust daemon and the Python gateway read the same file, which
//! eliminates the config-duplication drift class (assessment finding #1:
//! four independent config locations silently diverging).
//!
//! Resolution order:
//!   1. `$POC_ROOT/mcp-gateway/config/services.toml` on disk (authoritative)
//!   2. The copy embedded at compile time via `include_str!` (fallback —
//!      logged as an error, because it means the on-disk manifest is missing
//!      or corrupt and edits to it are NOT taking effect)
//!
//! Spawn commands are stored raw (no `sh -c` wrapper) with `${POC_ROOT}`
//! placeholders; [`ManifestService::spawn_cmd`] expands the placeholder and
//! adds the wrapper expected by `spawn::spawn_native_process`.

use serde::Deserialize;
use std::collections::BTreeMap;

use crate::registry::ActivationTier;

/// Compile-time fallback copy of the manifest.
const EMBEDDED_MANIFEST: &str = include_str!("../config/services.toml");

pub const DEFAULT_POC_ROOT: &str = "/Users/kevintoles/POC";

#[derive(Debug, Deserialize)]
pub struct Manifest {
    /// BTreeMap for deterministic iteration order (stable registry seeding).
    pub services: BTreeMap<String, ManifestService>,
}

#[derive(Debug, Deserialize)]
pub struct ManifestService {
    pub port: u16,
    /// Python health-timeout tier classification: "hot" | "warm" | "cold".
    pub tier: String,
    /// "auto" (launchd KeepAlive owns restart) | "native" (daemon respawns).
    pub runtime: String,
    #[serde(default = "default_health_path")]
    pub health_path: String,
    #[serde(default)]
    pub ready_path: Option<String>,
    /// Raw shell command with `${POC_ROOT}` placeholders, no `sh -c` wrapper.
    pub spawn: String,
    /// Extra Python-side lookup keys; Rust registers the primary name only.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Idle-demotion floor: "warm" = Hot→Warm only (never Cold); absent = Cold.
    #[serde(default)]
    pub min_tier: Option<String>,
    /// Seconds idle before Hot→Warm (default `HOT_IDLE_TIMEOUT_SECS`).
    #[serde(default)]
    pub hot_idle_secs: Option<u64>,
    /// Seconds idle before Warm→Cold (default `WARM_IDLE_TIMEOUT_SECS`; 0 = no Warm hold).
    #[serde(default)]
    pub warm_idle_secs: Option<u64>,
}

fn default_health_path() -> String {
    "/health".to_string()
}

pub fn poc_root() -> String {
    std::env::var("POC_ROOT").unwrap_or_else(|_| DEFAULT_POC_ROOT.to_string())
}

impl ManifestService {
    /// The spawn command ready for `spawn::spawn_native_process`:
    /// `${POC_ROOT}` expanded and wrapped in `sh -c '...'`.
    pub fn spawn_cmd(&self) -> String {
        let expanded = self.spawn.replace("${POC_ROOT}", &poc_root());
        format!("sh -c '{}'", expanded)
    }

    pub fn is_native(&self) -> bool {
        self.runtime == "native"
    }

    /// `(hot, warm)` idle timeouts in seconds, manifest values over the defaults.
    pub fn idle_timeouts(&self) -> (u64, u64) {
        (
            self.hot_idle_secs.unwrap_or(crate::lifecycle::HOT_IDLE_TIMEOUT_SECS),
            self.warm_idle_secs.unwrap_or(crate::lifecycle::WARM_IDLE_TIMEOUT_SECS),
        )
    }

    /// The registry idle floor for this service.
    pub fn min_tier(&self) -> ActivationTier {
        match self.min_tier.as_deref() {
            Some("warm") => ActivationTier::Warm,
            _ => ActivationTier::Cold,
        }
    }
}

fn manifest_path() -> String {
    format!("{}/mcp-gateway/config/services.toml", poc_root())
}

fn parse(source: &str) -> Result<Manifest, toml::de::Error> {
    toml::from_str(source)
}

/// Load the manifest: disk first, embedded copy as loud fallback.
pub fn load_manifest() -> Manifest {
    let path = manifest_path();
    match std::fs::read_to_string(&path) {
        Ok(content) => match parse(&content) {
            Ok(m) => {
                tracing::info!(path = %path, services = m.services.len(), "loaded service manifest");
                m
            }
            Err(e) => {
                tracing::error!(path = %path, error = %e,
                    "service manifest on disk is UNPARSABLE — falling back to embedded copy; \
                     edits to the on-disk manifest are NOT taking effect");
                parse(EMBEDDED_MANIFEST).expect("embedded manifest must parse")
            }
        },
        Err(e) => {
            tracing::error!(path = %path, error = %e,
                "service manifest MISSING on disk — falling back to embedded copy; \
                 edits to the on-disk manifest are NOT taking effect");
            parse(EMBEDDED_MANIFEST).expect("embedded manifest must parse")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedded() -> Manifest {
        parse(EMBEDDED_MANIFEST).expect("embedded manifest must parse")
    }

    #[test]
    fn manifest_parses_all_eleven_services() {
        // M5: Python unified-search-service retired — no H/W/C entry.
        let m = embedded();
        assert_eq!(m.services.len(), 11, "expected 11 services in manifest");
    }

    #[test]
    fn manifest_ports_match_platform_assignments() {
        let m = embedded();
        let expected = [
            ("llm-gateway", 8080),
            ("ai-agents", 8082),
            ("code-orchestrator", 8083),
            ("audit-service", 8084),
            ("inference-service-cpp", 8085),
            ("context-management-service", 8086),
            ("mcp-gateway", 8087),
            ("struct-analyzer", 8088),
            ("validation-service", 8091),
            ("amve", 8092),
            ("semantic-search", 8081),
        ];
        for (name, port) in expected {
            let svc = m
                .services
                .get(name)
                .unwrap_or_else(|| panic!("missing service: {}", name));
            assert_eq!(svc.port, port, "{} port mismatch", name);
        }
    }

    #[test]
    fn manifest_runtimes_split_auto_and_native() {
        let m = embedded();
        for name in ["llm-gateway", "semantic-search", "mcp-gateway"] {
            assert_eq!(m.services[name].runtime, "auto", "{} should be auto", name);
        }
        let native: Vec<_> = m
            .services
            .iter()
            .filter(|(_, s)| s.is_native())
            .map(|(n, _)| n.clone())
            .collect();
        assert_eq!(native.len(), 8, "expected 8 native services, got {:?}", native);
    }

    #[test]
    fn spawn_cmd_expands_poc_root_and_wraps() {
        let m = embedded();
        let cmd = m.services["validation-service"].spawn_cmd();
        assert!(
            !cmd.contains("${POC_ROOT}"),
            "placeholder must be expanded; got: {}",
            cmd
        );
        assert!(cmd.starts_with("sh -c '"), "must wrap in sh -c: {}", cmd);
        assert!(cmd.contains("PORT=8091"), "must set PORT env: {}", cmd);
        assert!(
            cmd.contains("/validation-service/rust"),
            "must point at the Rust service dir: {}",
            cmd
        );
    }

    #[test]
    fn no_absolute_paths_in_raw_spawn_commands() {
        let m = embedded();
        for (name, svc) in &m.services {
            assert!(
                !svc.spawn.contains("/Users/"),
                "{} spawn has a hardcoded absolute path — use ${{POC_ROOT}}: {}",
                name,
                svc.spawn
            );
        }
    }

    #[test]
    fn amve_overrides_health_path() {
        let m = embedded();
        assert_eq!(m.services["amve"].health_path, "/v1/health");
        assert_eq!(m.services["ai-agents"].health_path, "/health");
    }

    #[test]
    fn launchd_services_declare_warm_floor() {
        // Option A: "Hot→Warm only" is explicit in the manifest, not inferred
        // from runtime = "auto".
        let m = parse(EMBEDDED_MANIFEST).unwrap();
        for (name, svc) in &m.services {
            if svc.is_native() {
                assert_eq!(svc.min_tier, None, "{name}: native services idle to Cold");
                assert_eq!(svc.min_tier(), ActivationTier::Cold);
            } else {
                assert_eq!(svc.min_tier.as_deref(), Some("warm"), "{name}: launchd services floor at Warm");
                assert_eq!(svc.min_tier(), ActivationTier::Warm);
            }
        }
    }

    #[test]
    fn model_heavy_services_release_memory_fast() {
        // 2026-10-20: llama-server (~4.8 GB) held 30 min Hot + 10 min Warm after
        // every wake. Model-heavy services idle 10 min Hot → straight to Cold;
        // every other service keeps the 30 + 10 min defaults.
        let m = parse(EMBEDDED_MANIFEST).unwrap();
        for (name, svc) in &m.services {
            let heavy = matches!(name.as_str(), "inference-service-cpp" | "code-orchestrator");
            let expected = if heavy {
                (600, 0)
            } else {
                (crate::lifecycle::HOT_IDLE_TIMEOUT_SECS, crate::lifecycle::WARM_IDLE_TIMEOUT_SECS)
            };
            assert_eq!(svc.idle_timeouts(), expected, "{name}");
        }
    }

    #[test]
    fn spawn_commands_do_not_keep_binaries_in_tmp() {
        // macOS periodically deletes /tmp files: struct-analyzer's /tmp binary
        // vanished, forcing a rebuild the shim's PATH could not run (2026-10-27).
        let m = parse(EMBEDDED_MANIFEST).unwrap();
        for (name, svc) in &m.services {
            assert!(!svc.spawn.contains("/tmp/"), "{name}: spawn must not use /tmp: {}", svc.spawn);
        }
    }

    #[test]
    fn spawn_commands_run_prebuilt_artifacts_without_compiling() {
        // Every service starts a prebuilt artifact (venv, target/release, repo
        // binary). Compiling at spawn needs a toolchain on the shim's launchd PATH
        // and ~40 s per cold start — struct-analyzer failed on both (2026-10-27).
        let m = parse(EMBEDDED_MANIFEST).unwrap();
        for (name, svc) in &m.services {
            for compiler in ["go build", "cargo build", "rustc ", "cmake ", "make "] {
                assert!(!svc.spawn.contains(compiler), "{name}: spawn must not compile ({compiler}): {}", svc.spawn);
            }
        }
    }

    #[test]
    fn slow_start_services_declare_ready_path() {
        let m = embedded();
        // inference-service-cpp runs stock llama.cpp `llama-server` (run_native.sh):
        // no /ready route (404); /health is 503 while loading, 200 when ready.
        for (name, path) in [("code-orchestrator", "/ready"), ("inference-service-cpp", "/health")] {
            assert_eq!(
                m.services[name].ready_path.as_deref(),
                Some(path),
                "{} must declare ready_path",
                name
            );
        }
        assert!(m.services["ai-agents"].ready_path.is_none());
    }

    #[test]
    fn semantic_search_carries_unified_search_rs_alias() {
        let m = embedded();
        assert_eq!(m.services["semantic-search"].aliases, vec!["unified-search-rs"]);
    }
}
