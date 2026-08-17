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

use crate::registry::{
    ActivationTier, ServiceEntry, ServiceRegistry, ServiceRuntime,
};
use std::collections::VecDeque;

/// Populate `registry` with the platform's known services.
///
/// Ports match the services' actual listen addresses. Tiers follow
/// HYBRID_ARCHITECTURE_RECOMMENDATION.md §4.6.2 per-service assignments.
/// All entries start in `Boot` so `startup_scan()` and `BootColdMonitor`
/// can promote them appropriately.
///
/// **KeepAlive rule**: services whose LaunchAgent plist sets `KeepAlive=true`
/// are NOT registered here — launchd is their sole restart authority. The
/// shim still routes to them and health-checks them, but `health_failure_monitor`
/// must not respawn them (double-management race). Excluded:
///   - unified-search-service (Python, :8081) — per A-7
///   - llm-gateway (:8080) — per H-4
///   - unified-search-rs (:8093) — per H-4
///   - mcp-gateway (:8087) — per D-2 (dual restart authority resolved)
///
/// `spawn_cmd` uses `sh -c '...'` syntax so shell pipelines and `&&` chains
/// are handled correctly by `spawn::spawn_native_process` (which calls
/// `shlex::split` then `Command::new`).
pub fn seed_platform_services(registry: &ServiceRegistry) {
    // (name, port, tier, spawn_cmd)
    // Spawn commands are only invoked when the port is not bound and the
    // health_failure_monitor determines the service needs to be respawned.
    let services: &[(&str, u16, ActivationTier, &str)] = &[
        // ── Hot tier: always-on ────────────────────────────────────────────
        // llm-gateway (:8080) has KeepAlive=true in its LaunchAgent plist.
        // NOT registered here — launchd is the sole restart authority (H-4).
        //
        // mcp-gateway has KeepAlive=true in its LaunchAgent plist.
        // NOT registered here — launchd is the sole restart authority (D-2).
        //
        // unified-search-rs (:8093) has KeepAlive=true in its LaunchAgent plist.
        // NOT registered here — launchd is the sole restart authority (H-4).
        // The legacy Python unified-search-service (:8081) is also excluded
        // for the same reason (A-7).

        // ── Warm tier: on-demand backends ─────────────────────────────────
        (
            "ai-agents",
            8082,
            ActivationTier::Boot,
            "sh -c 'cd /Users/kevintoles/POC/ai-agents && (test -x .venv/bin/python || (python3 -m venv .venv && .venv/bin/pip install -q -e .)) && .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8082'",
        ),
        (
            "code-orchestrator",
            8083,
            ActivationTier::Boot,
            "sh -c 'cd /Users/kevintoles/POC/Code-Orchestrator-Service && (test -x .venv/bin/python || (python3 -m venv .venv && .venv/bin/pip install -q -e .)) && COS_CODEBERT_START_MODE=warm COS_GRAPHCODEBERT_START_MODE=cold COS_CODET5_START_MODE=cold .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8083'",
        ),
        (
            "audit-service",
            8084,
            ActivationTier::Boot,
            "sh -c 'cd /Users/kevintoles/POC/audit-service && (test -x .venv/bin/python || (python3 -m venv .venv && .venv/bin/pip install -q -e .)) && .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8084'",
        ),
        // validation-service: on-demand document/WBS/arch-decision validation.
        // Requires .venv in /Users/kevintoles/POC/validation-service/python.
        // Port from config.py VALIDATION_SERVICE_URL; service defaults to PORT env var.
        (
            "validation-service",
            8091,
            ActivationTier::Boot,
            "sh -c 'cd /Users/kevintoles/POC/validation-service/python && (test -x .venv/bin/python || (python3 -m venv .venv && .venv/bin/pip install -q -e .)) && PORT=8091 .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8091'",
        ),

        // ── Cold tier: heavy resource, manual/GPU ─────────────────────────
        // inference-service-cpp: actual port is INFERENCE_PORT env (default 8085).
        // Was previously registered as 8089 — corrected to match run_native.sh.
        (
            "inference-service-cpp",
            8085,
            ActivationTier::Boot,
            "sh -c 'cd /Users/kevintoles/POC/inference-service-cpp && ./run_native.sh'",
        ),
        (
            "context-management-service",
            8086,
            ActivationTier::Boot,
            "sh -c 'cd /Users/kevintoles/POC/context-management-service && (test -x .venv/bin/python || (python3 -m venv .venv && .venv/bin/pip install -q -e .)) && .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8086'",
        ),
        // struct-analyzer: Go binary, must be built before first spawn.
        // `serve` subcommand is required — running without it prints usage and exits.
        (
            "struct-analyzer",
            8088,
            ActivationTier::Boot,
            "sh -c 'cd /Users/kevintoles/POC/struct-analyzer-service && go build -o /tmp/struct-analyzer ./cmd/struct-analyzer && STRUCT_ANALYZER_PORT=:8088 /tmp/struct-analyzer serve'",
        ),
    ];

    for (name, port, tier, spawn_cmd) in services {
        let ready_path = match *name {
            "code-orchestrator" | "inference-service-cpp" => Some("/ready".to_string()),
            _ => None,
        };
        registry.register(ServiceEntry {
            name: name.to_string(),
            port: *port,
            health_port: 0,
            tier: tier.clone(),
            runtime: ServiceRuntime::Native {
                spawn_cmd: spawn_cmd.to_string(),
                pid_file: None,
            },
            health_path: "/health".to_string(),
            last_health: None,
            failure_count: 0,
            hot_idle_timeout_secs: 1800,
            warm_idle_timeout_secs: 600,
            last_request: None,
            request_timestamps: VecDeque::new(),
            circuit_open_since: None,
            last_half_open_probe: None,
            first_failure_at: None,
            ready_path,
            health_checks_total: 0,
            health_checks_passed: 0,
        });
    }

    // AMVE uses /v1/health (mounted with prefix="/v1"), not /health.
    // Registered separately to set a custom health_path.
    registry.register(ServiceEntry {
        name: "amve".to_string(),
        port: 8092,
        health_port: 0,
        tier: ActivationTier::Boot,
        runtime: ServiceRuntime::Native {
            spawn_cmd: "sh -c 'cd /Users/kevintoles/POC/architecture-mapping-validation-engine && (test -x .venv/bin/python || (python3 -m venv .venv && .venv/bin/pip install -q -e .)) && PORT=8092 .venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8092'".to_string(),
            pid_file: None,
        },
        health_path: "/v1/health".to_string(),
        last_health: None,
        failure_count: 0,
        hot_idle_timeout_secs: 1800,
        warm_idle_timeout_secs: 600,
        last_request: None,
        request_timestamps: VecDeque::new(),
        circuit_open_since: None,
        last_half_open_probe: None,
        first_failure_at: None,
        ready_path: None,
        health_checks_total: 0,
        health_checks_passed: 0,
    });

    // ── KeepAlive=true services: tier tracking only ──────────────────────
    // These services are restarted by launchd (KeepAlive=true), NOT by
    // health_failure_monitor. They are registered here solely so that
    // record_request() and tier transitions work when activity is reported
    // via POST /activity/{service}. ServiceRuntime::Auto ensures
    // health_failure_monitor skips respawn (RuntimeUnresolved).
    let keepalive_services: &[(&str, u16, &str)] = &[
        ("llm-gateway", 8080, "/health"),
        ("semantic-search", 8093, "/health"),
        ("unified-search-service", 8081, "/health"),
        ("mcp-gateway", 8087, "/health"),
    ];
    for (name, port, health_path) in keepalive_services {
        registry.register(ServiceEntry {
            name: name.to_string(),
            port: *port,
            health_port: 0,
            tier: ActivationTier::Boot,
            runtime: ServiceRuntime::Auto,
            health_path: health_path.to_string(),
            last_health: None,
            failure_count: 0,
            hot_idle_timeout_secs: 1800,
            warm_idle_timeout_secs: 600,
            last_request: None,
            request_timestamps: VecDeque::new(),
            circuit_open_since: None,
            last_half_open_probe: None,
            first_failure_at: None,
            ready_path: None,
            health_checks_total: 0,
            health_checks_passed: 0,
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
