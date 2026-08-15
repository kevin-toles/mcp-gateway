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
/// `spawn_cmd` uses `sh -c '...'` syntax so shell pipelines and `&&` chains
/// are handled correctly by `spawn::spawn_native_process` (which calls
/// `shlex::split` then `Command::new`).
pub fn seed_platform_services(registry: &ServiceRegistry) {
    // (name, port, tier, spawn_cmd)
    // Spawn commands are only invoked when the port is not bound and the
    // health_failure_monitor determines the service needs to be respawned.
    let services: &[(&str, u16, ActivationTier, &str)] = &[
        // ── Hot tier: always-on ────────────────────────────────────────────
        (
            "llm-gateway",
            8080,
            ActivationTier::Boot,
            "sh -c 'cd /Users/kevintoles/POC/llm-gateway && .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8080'",
        ),
        // mcp-gateway is respawned by start_mcp_gateway() in the shim connection
        // handler in addition to the health_failure_monitor — both paths use
        // is_port_bound as a gate so double-spawn is prevented at the OS level.
        (
            "mcp-gateway",
            8087,
            ActivationTier::Boot,
            "sh -c 'cd /Users/kevintoles/POC/mcp-gateway && .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8087'",
        ),
        // unified-search-rs Rust binary (primary semantic search, :8093).
        // The legacy Python unified-search-service (:8081) is managed by its
        // own launchd KeepAlive plist and is NOT tracked here to avoid
        // double-management of port 8081.
        (
            "semantic-search",
            8093,
            ActivationTier::Boot,
            "sh -c 'cd /Users/kevintoles/POC/unified-search-rs && PORT=8093 cargo run --release'",
        ),

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
            "sh -c 'cd /Users/kevintoles/POC/validation-service/python && PORT=8091 .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8091'",
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
            "sh -c 'cd /Users/kevintoles/POC/struct-analyzer-service && go build -o /tmp/struct-analyzer ./cmd/struct-analyzer && /tmp/struct-analyzer serve'",
        ),
    ];

    for (name, port, tier, spawn_cmd) in services {
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
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_populates_expected_services() {
        let registry = ServiceRegistry::new();
        assert!(registry.is_empty());

        seed_platform_services(&registry);

        assert_eq!(registry.len(), 11);
        for name in [
            "llm-gateway",
            "mcp-gateway",
            "semantic-search",
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
    }

    #[test]
    fn seed_ports_match_hybrid_recommendation() {
        let registry = ServiceRegistry::new();
        seed_platform_services(&registry);

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
            ("semantic-search", 8093),
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
    fn all_entries_have_real_spawn_commands() {
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
                other => panic!(
                    "{} has non-Native runtime {:?} — all services must have real spawn commands",
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
    fn semantic_search_spawn_cmd_sets_port_8093() {
        let registry = ServiceRegistry::new();
        seed_platform_services(&registry);

        let entry = registry.get("semantic-search").unwrap();
        assert_eq!(entry.port, 8093);
        let ServiceRuntime::Native { spawn_cmd, .. } = &entry.runtime else {
            panic!("semantic-search must have Native runtime");
        };
        assert!(
            spawn_cmd.contains("PORT=8093"),
            "semantic-search spawn_cmd must set PORT=8093 (uss-server defaults to 8081); got: {}",
            spawn_cmd
        );
    }
}
