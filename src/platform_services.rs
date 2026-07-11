//! Platform service catalog for the mcp-gateway shim.
//!
//! Populates the `ServiceRegistry` with all known platform services and their
//! per-service tier assignments (HYBRID_ARCHITECTURE_RECOMMENDATION.md §4.6.2).
//!
//! Runtime is left as `ServiceRuntime::Auto` — resolved at spawn time in
//! `spawn.rs:resolve_runtime()` based on `DEPLOYMENT_MODE`. This means the same
//! seed works for hybrid, docker, and native modes without duplication.
//!
//! Called once at shim startup by `main.rs` before `lifecycle::startup_scan()`
//! so that scan actually finds services to promote.

use crate::registry::{
    ActivationTier, ServiceEntry, ServiceRegistry, ServiceRuntime,
};
use std::collections::VecDeque;

/// Populate `registry` with the platform's known services.
///
/// Ports match HYBRID_ARCHITECTURE_RECOMMENDATION.md §9.1. Tiers match
/// §4.6.2 per-service assignments. All entries start in `Boot` so
/// `startup_scan()` and `BootColdMonitor` can promote them appropriately.
pub fn seed_platform_services(registry: &ServiceRegistry) {
    let services = [
        // Always-on Hot tier (§4.6.2)
        ("llm-gateway", 8080, ActivationTier::Boot),
        ("mcp-gateway", 8087, ActivationTier::Boot),
        ("semantic-search", 8081, ActivationTier::Boot),

        // Warm tier — on-demand backends (§4.6.2)
        ("ai-agents", 8082, ActivationTier::Boot),
        ("code-orchestrator", 8083, ActivationTier::Boot),
        ("audit-service", 8084, ActivationTier::Boot),

        // Cold tier — heavy resource, manual/GPU (§4.6.2)
        ("inference-service-cpp", 8089, ActivationTier::Boot),
        ("context-management-service", 8086, ActivationTier::Boot),

        // Struct-analyzer — replaces amve. Port 8088 in native/hybrid;
        // remapped to 8092 in Docker mode via STRUCT_ANALYZER_PORT.
        ("struct-analyzer", 8088, ActivationTier::Boot),
    ];

    for (name, port, tier) in services {
        registry.register(ServiceEntry {
            name: name.to_string(),
            port,
            health_port: 0, // 0 → use `port`
            tier,
            runtime: ServiceRuntime::Auto,
            health_path: "/health".to_string(),
            last_health: None,
            failure_count: 0,
            hot_idle_timeout_secs: 1800, // 30min per HYBRID §4.6.3
            warm_idle_timeout_secs: 600, // 10min per HYBRID §4.6.3
            last_request: None,
            request_timestamps: VecDeque::new(),
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

        assert_eq!(registry.len(), 9);
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
        ] {
            let entry = registry
                .get(name)
                .unwrap_or_else(|| panic!("missing service: {}", name));
            assert_eq!(entry.tier, ActivationTier::Boot);
            assert!(matches!(entry.runtime, ServiceRuntime::Auto));
            assert_eq!(entry.health_path, "/health");
        }
    }

    #[test]
    fn seed_ports_match_hybrid_recommendation() {
        let registry = ServiceRegistry::new();
        seed_platform_services(&registry);

        let expected = [
            ("llm-gateway", 8080),
            ("semantic-search", 8081),
            ("ai-agents", 8082),
            ("code-orchestrator", 8083),
            ("audit-service", 8084),
            ("context-management-service", 8086),
            ("mcp-gateway", 8087),
            ("struct-analyzer", 8088),
            ("inference-service-cpp", 8089),
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

        // Second call overwrites entries with same name — count stays constant.
        seed_platform_services(&registry);
        assert_eq!(registry.len(), first_len);
    }
}
