//! Platform service catalog for the mcp-gateway shim.
//!
//! Populates the `ServiceRegistry` with all known platform services and their
//! per-service tier assignments (HYBRID_ARCHITECTURE_RECOMMENDATION.md §4.6.2).
//!
//! Each entry carries a real `spawn_cmd` so `health_failure_monitor` can respawn
//! a dead service regardless of whether the call came from the shim proxy path,
//! a direct port connection, or an MCP tool call.
//!
//! Called once at shim startup by `main.rs` before `lifecycle::startup_scan()`
//! so that scan actually finds services to promote.

use crate::manifest::load_manifest;
use crate::registry::{
    ActivationTier, ServiceEntry, ServiceRegistry, ServiceRuntime,
};
use std::collections::VecDeque;

/// Populate `registry` from the service manifest (`config/services.toml`).
///
/// The manifest is the single source of truth for ports, tiers, runtimes,
/// health/ready paths, and spawn commands — shared with the Python gateway.
/// All entries start in `Boot` so `startup_scan()` and `BootColdMonitor`
/// can promote them appropriately.
///
/// **KeepAlive rule**: services with `runtime = "auto"` in the manifest have
/// `KeepAlive=true` LaunchAgent plists — launchd is their sole restart
/// authority (H-4/A-7/D-2). They are registered with `ServiceRuntime::Auto`
/// for tier tracking only; `health_failure_monitor` never respawns them
/// (double-management race).
///
/// `spawn_cmd` uses `sh -c '...'` syntax so shell pipelines and `&&` chains
/// are handled correctly by `spawn::spawn_native_process` (which calls
/// `shlex::split` then `Command::new`).
pub fn seed_platform_services(registry: &ServiceRegistry) {
    // The manifest is the single source of truth (config/services.toml).
    // runtime = "auto"   → launchd (KeepAlive=true) is the sole restart
    //                      authority; registered for tier tracking only so
    //                      record_request() / POST /activity/{service} work.
    //                      health_failure_monitor never respawns (Auto →
    //                      RuntimeUnresolved).
    // runtime = "native" → health_failure_monitor respawns via spawn_cmd
    //                      when the port is unbound and the service is due.
    let manifest = load_manifest();

    for (name, svc) in &manifest.services {
        let runtime = if svc.is_native() {
            ServiceRuntime::Native {
                spawn_cmd: svc.spawn_cmd(),
                pid_file: None,
            }
        } else {
            ServiceRuntime::Auto
        };
        registry.register(ServiceEntry {
            name: name.clone(),
            port: svc.port,
            health_port: 0,
            tier: ActivationTier::Boot,
            runtime,
            health_path: svc.health_path.clone(),
            last_health: None,
            failure_count: 0,
            hot_idle_timeout_secs: 1800,
            warm_idle_timeout_secs: 600,
            last_request: None,
            request_timestamps: VecDeque::new(),
            circuit_open_since: None,
            last_half_open_probe: None,
            first_failure_at: None,
            ready_path: svc.ready_path.clone(),
            health_checks_total: 0,
            health_checks_passed: 0,
            last_transition_reason: None,
            last_transition_at: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_populates_expected_services() {
        let registry = ServiceRegistry::new();
        assert!(registry.is_empty());

        seed_platform_services(&registry);

        assert_eq!(registry.len(), 12);
        // Managed services (Native runtime, health_failure_monitor respawns)
        for name in [
            "ai-agents",
            "code-orchestrator",
            "audit-service",
            "inference-service-cpp",
            "context-management-service",
            "struct-analyzer",
            "validation-service",
            "amve",
        ] {
            let entry = registry
                .get(name)
                .unwrap_or_else(|| panic!("missing service: {}", name));
            assert_eq!(entry.tier, ActivationTier::Boot);
            assert!(
                matches!(entry.runtime, ServiceRuntime::Native { .. }),
                "{} should have Native runtime, not Auto",
                name
            );
            let expected_health = if name == "amve" { "/v1/health" } else { "/health" };
            assert_eq!(
                entry.health_path, expected_health,
                "{} health_path mismatch", name
            );
        }
        // KeepAlive services (Auto runtime, tier tracking only)
        for name in ["llm-gateway", "semantic-search", "unified-search-service", "mcp-gateway"] {
            let entry = registry
                .get(name)
                .unwrap_or_else(|| panic!("missing KeepAlive service: {}", name));
            assert_eq!(entry.tier, ActivationTier::Boot);
            assert!(
                matches!(entry.runtime, ServiceRuntime::Auto),
                "{} should have Auto runtime (KeepAlive, tier tracking only)",
                name
            );
        }
    }

    #[test]
    fn seed_ports_match_hybrid_recommendation() {
        let registry = ServiceRegistry::new();
        seed_platform_services(&registry);

        let expected = [
            ("ai-agents", 8082),
            ("code-orchestrator", 8083),
            ("audit-service", 8084),
            ("inference-service-cpp", 8085),
            ("context-management-service", 8086),
            ("struct-analyzer", 8088),
            ("validation-service", 8091),
            ("amve", 8092),
        ];
        for (name, port) in expected {
            let entry = registry.get(name).unwrap();
            assert_eq!(entry.port, port, "port mismatch for {}", name);
        }
    }

    #[test]
    fn seed_is_idempotent() {
        let registry = ServiceRegistry::new();
        seed_platform_services(&registry);
        let first_len = registry.len();

        seed_platform_services(&registry);
        assert_eq!(registry.len(), first_len);
    }

    #[test]
    fn managed_entries_have_real_spawn_commands() {
        let registry = ServiceRegistry::new();
        seed_platform_services(&registry);

        for entry in registry.all() {
            match &entry.runtime {
                ServiceRuntime::Native { spawn_cmd, .. } => {
                    assert!(
                        !spawn_cmd.is_empty(),
                        "{} has empty spawn_cmd",
                        entry.name
                    );
                    assert!(
                        spawn_cmd.starts_with("sh -c '"),
                        "{} spawn_cmd should start with 'sh -c ' for shell pipeline support, got: {}",
                        entry.name,
                        spawn_cmd
                    );
                }
                ServiceRuntime::Auto => {
                    // KeepAlive services — launchd restarts, no spawn_cmd needed
                }
                other => panic!(
                    "{} has unexpected runtime {:?}",
                    entry.name, other
                ),
            }
        }
    }

    #[test]
    fn struct_analyzer_spawn_cmd_includes_serve_subcommand() {
        let registry = ServiceRegistry::new();
        seed_platform_services(&registry);

        let entry = registry.get("struct-analyzer").unwrap();
        let ServiceRuntime::Native { spawn_cmd, .. } = &entry.runtime else {
            panic!("struct-analyzer must have Native runtime");
        };
        assert!(
            spawn_cmd.contains("serve"),
            "struct-analyzer spawn_cmd must include 'serve' subcommand to start the HTTP server; got: {}",
            spawn_cmd
        );
    }

    #[test]
    fn validation_service_spawn_cmd_sets_port_env() {
        let registry = ServiceRegistry::new();
        seed_platform_services(&registry);

        let entry = registry.get("validation-service").unwrap();
        assert_eq!(entry.port, 8091);
        let ServiceRuntime::Native { spawn_cmd, .. } = &entry.runtime else {
            panic!("validation-service must have Native runtime");
        };
        assert!(
            spawn_cmd.contains("PORT=8091"),
            "validation-service spawn_cmd must set PORT=8091 (service defaults to 8090 otherwise); got: {}",
            spawn_cmd
        );
    }

    #[test]
    fn amve_uses_v1_health_path() {
        let registry = ServiceRegistry::new();
        seed_platform_services(&registry);

        let entry = registry.get("amve").unwrap();
        assert_eq!(entry.port, 8092);
        assert_eq!(
            entry.health_path, "/v1/health",
            "amve health_path must be /v1/health (mounted with prefix=/v1)"
        );
        let ServiceRuntime::Native { spawn_cmd, .. } = &entry.runtime else {
            panic!("amve must have Native runtime");
        };
        assert!(
            spawn_cmd.contains("PORT=8092"),
            "amve spawn_cmd must set PORT=8092; got: {}",
            spawn_cmd
        );
    }

    #[test]
    fn slow_start_services_have_ready_path() {
        let registry = ServiceRegistry::new();
        seed_platform_services(&registry);

        for name in ["code-orchestrator", "inference-service-cpp"] {
            let entry = registry.get(name).unwrap();
            assert_eq!(
                entry.ready_path,
                Some("/ready".to_string()),
                "D-3: {} must have ready_path for startup readiness",
                name
            );
        }
        for name in ["ai-agents", "audit-service", "context-management-service", "amve"] {
            let entry = registry.get(name).unwrap();
            assert_eq!(
                entry.ready_path, None,
                "{} should not have ready_path",
                name
            );
        }
    }

    #[test]
    fn keepalive_services_registered_with_auto_runtime() {
        let registry = ServiceRegistry::new();
        seed_platform_services(&registry);

        // KeepAlive=true services are registered for tier tracking only.
        // Auto runtime ensures health_failure_monitor skips respawn.
        for (name, port) in [("llm-gateway", 8080), ("semantic-search", 8093), ("unified-search-service", 8081), ("mcp-gateway", 8087)] {
            let entry = registry.get(name).unwrap_or_else(|| panic!("{} should be registered", name));
            assert_eq!(entry.port, port, "{} port mismatch", name);
            assert!(matches!(entry.runtime, ServiceRuntime::Auto), "{} must have Auto runtime", name);
        }
    }
}
