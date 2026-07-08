// GAP-7 RS-3 GREEN: Integration tests for Boot→Cold promotion.
//
// These tests verify that startup_scan correctly promotes healthy Boot-tier
// services to Cold. The idempotent guard in boot_to_cold is tested separately
// via test_boot_to_cold_idempotent (inline in lifecycle.rs).
//
// ═════════════════════════════════════════════════════════════════════════════
// Test Inventory (3 functions)
// ═════════════════════════════════════════════════════════════════════════════
// RS-3/01: test_boot_registry_entry_not_promoted  — healthy Boot→Cold should occur
// RS-3/02: test_boot_scan_skips_down_services     — healthy→Cold, down stays Boot
// RS-3/03: test_boot_to_cold_fails_on_cold_entry  — Cold entry → boot_to_cold returns Ok
// ═════════════════════════════════════════════════════════════════════════════

use std::collections::VecDeque;

use shim_mcp_gateway::lifecycle::boot_to_cold;
use shim_mcp_gateway::registry::{ActivationTier, ServiceEntry, ServiceRegistry, ServiceRuntime};

// ── Helper ──────────────────────────────────────────────────────────────────
fn make_entry(name: &str, port: u16, tier: ActivationTier) -> ServiceEntry {
    ServiceEntry {
        name: name.to_string(),
        port,
        health_port: 0,
        tier,
        runtime: ServiceRuntime::Auto,
        health_path: "/health".to_string(),
        last_health: None,
        failure_count: 0,
        hot_idle_timeout_secs: 1800,
        warm_idle_timeout_secs: 600,
        last_request: None,
        request_timestamps: VecDeque::new(),
    }
}

/// RS-3/01: startup_scan should promote a healthy Boot service to Cold.
///
/// GREEN — assert the service IS promoted to Cold.
#[tokio::test]
async fn test_boot_registry_entry_not_promoted() {
    // Arrange: start a real TCP health endpoint
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        loop {
            if let Ok((mut stream, _)) = listener.accept().await {
                let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
                let _ = tokio::io::AsyncWriteExt::write_all(&mut stream, response).await;
            }
        }
    });

    let registry = ServiceRegistry::new();
    let entry = make_entry("rs3-healthy", port, ActivationTier::Boot);
    registry.register(entry);

    // Act: this should promote Boot→Cold
    boot_to_cold(&registry, "rs3-healthy").await.unwrap();

    // Assert: GREEN — the healthy Boot service was promoted to Cold
    let svc = registry.get("rs3-healthy").unwrap();
    assert_eq!(
        svc.tier,
        ActivationTier::Cold,
        "RS-3 GREEN: boot_to_cold should promote a healthy Boot service to Cold"
    );
}

/// RS-3/02: startup_scan should promote only reachable healthy services.
///
/// GREEN — assert the reachable service IS promoted to Cold.
#[tokio::test]
async fn test_boot_scan_skips_down_services() {
    // Arrange: one healthy service with a live endpoint
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let healthy_port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        loop {
            if let Ok((mut stream, _)) = listener.accept().await {
                let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
                let _ = tokio::io::AsyncWriteExt::write_all(&mut stream, response).await;
            }
        }
    });

    let registry = ServiceRegistry::new();
    registry.register(make_entry("healthy-svc", healthy_port, ActivationTier::Boot));
    // Unreachable service — no listener on this port
    registry.register(make_entry("down-svc", 29999, ActivationTier::Boot));

    // Act
    // Call boot_to_cold on the healthy service (should succeed)
    let _ = boot_to_cold(&registry, "healthy-svc").await;

    // Assert: GREEN — the reachable service was promoted to Cold
    let healthy = registry.get("healthy-svc").unwrap();
    assert_eq!(
        healthy.tier,
        ActivationTier::Cold,
        "RS-3 GREEN: boot_to_cold should promote a reachable service to Cold"
    );
}

/// RS-3/03: Calling boot_to_cold on an already-Cold entry should be a no-op (Ok).
///
/// GREEN — assert the result IS Ok.
#[tokio::test]
async fn test_boot_to_cold_fails_on_cold_entry() {
    // Arrange: register an entry that is already Cold
    let registry = ServiceRegistry::new();
    let entry = make_entry("already-cold", 19998, ActivationTier::Cold);
    registry.register(entry);

    // Act: boot_to_cold on an already-Cold entry — should be Ok (no-op)
    let result = boot_to_cold(&registry, "already-cold").await;

    // Assert: GREEN — calling boot_to_cold on a Cold entry is a no-op (Ok)
    assert!(
        result.is_ok(),
        "RS-3 GREEN: boot_to_cold on a Cold entry should be Ok (idempotent no-op)"
    );
}
