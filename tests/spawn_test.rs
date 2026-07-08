// GAP-6: Integration tests for mcp-gateway spawn module.
//
// These tests target `shim_mcp_gateway::spawn` functions using real OS commands
// (echo, sleep) and TCP listeners for health endpoints. Since no lib.rs exists,
// integration tests import via the binary crate name `shim_mcp_gateway`.
//
// ═════════════════════════════════════════════════════════════════════════════
// Test Inventory (9 functions)
// ═════════════════════════════════════════════════════════════════════════════
// G6-01: test_spawn_native_uvicorn          — Native spawn with echo mock
// G6-02: test_spawn_native_binary           — Native spawn, verify Ok(pid)
// G6-03: test_spawn_docker_container        — Docker spawn (or #[ignore])
// G6-04: test_spawn_auto_resolves_native    — Auto → Native in hybrid mode
// G6-05: test_spawn_auto_resolves_docker    — Auto → Docker in docker mode
// G6-06: test_spawn_already_running         — Port bound → AlreadyRunning
// G6-07: test_spawn_updates_registry_tier   — Successful spawn → tier Hot
// G6-08: test_spawn_health_poll_timeout     — Unreachable → HealthTimeout
// G6-09: test_spawn_health_poll_success     — Mock + TCP health → Ok(pid)
// ═════════════════════════════════════════════════════════════════════════════

use std::sync::Arc;
use std::collections::VecDeque;

use serial_test::serial;
use shim_mcp_gateway::registry::{ActivationTier, ServiceEntry, ServiceRegistry, ServiceRuntime};
use shim_mcp_gateway::spawn::SpawnError;

// ── Helper ──────────────────────────────────────────────────────────────────
// Build a minimal ServiceEntry for testing. Mirrors the make_entry() helper in
// lifecycle.rs tests but accessible from integration tests.
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

// Helper: bind a TCP listener on a random port and return (listener, port).
// The caller keeps the listener alive to keep the port bound.
async fn bind_random_port() -> (tokio::net::TcpListener, u16) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    (listener, port)
}

// Helper: spawn a TCP health endpoint that returns HTTP 200 OK on every connection.
// Returns the port it's listening on.
async fn spawn_health_endpoint() -> u16 {
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
    port
}

// ═════════════════════════════════════════════════════════════════════════════
// G6-01: Native spawn with echo command (simulating a uvicorn-like command)
// ═════════════════════════════════════════════════════════════════════════════
#[tokio::test]
#[serial]
async fn test_spawn_native_uvicorn() {
    // Use /bin/sleep as a stand-in for a long-running native process
    // (analogous to "uvicorn src.main:app"). The echo command is insufficient
    // because poll_health needs the process to stay alive while checking.
    let port = spawn_health_endpoint().await;

    let entry = ServiceEntry {
        name: "g6-01-uvicorn".into(),
        port,
        health_port: port,
        tier: ActivationTier::Warm,
        runtime: ServiceRuntime::Native {
            spawn_cmd: "/bin/sleep".into(),
            pid_file: None,
        },
        health_path: "/health".into(),
        ..make_entry("", 0, ActivationTier::Warm) // fields overwritten above
    };

    let result = shim_mcp_gateway::spawn::spawn_service(&entry, "hybrid").await;
    match result {
        Ok(pid) => assert!(pid > 0, "should return a valid PID > 0"),
        Err(e) => panic!("expected Ok(pid), got Err({:?})", e),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// G6-02: Native spawn, verify Ok(pid) with echo
// ═════════════════════════════════════════════════════════════════════════════
#[tokio::test]
#[serial]
async fn test_spawn_native_binary() {
    let port = spawn_health_endpoint().await;

    let entry = ServiceEntry {
        name: "g6-02-binary".into(),
        port,
        health_port: port,
        tier: ActivationTier::Warm,
        runtime: ServiceRuntime::Native {
            spawn_cmd: "/bin/sleep".into(),
            pid_file: None,
        },
        health_path: "/health".into(),
        ..make_entry("", 0, ActivationTier::Warm)
    };

    let pid = shim_mcp_gateway::spawn::spawn_service(&entry, "hybrid")
        .await
        .expect("spawn_service should succeed with health endpoint reachable");
    assert!(pid > 0, "PID must be positive, got {}", pid);
}

// ═════════════════════════════════════════════════════════════════════════════
// G6-03: Docker container spawn
// ═════════════════════════════════════════════════════════════════════════════
// Marked #[ignore] by default because it requires Docker and a pre-existing
// container named "test-mcp-health". Remove #[ignore] and create the container
// with: docker run -d --name test-mcp-health -p 19001:80 nginx:alpine
#[tokio::test]
#[ignore = "requires Docker and pre-existing 'test-mcp-health' container"]
async fn test_spawn_docker_container() {
    let entry = ServiceEntry {
        name: "g6-03-docker".into(),
        port: 19001,
        health_port: 0,
        tier: ActivationTier::Warm,
        runtime: ServiceRuntime::Docker {
            container_name: "test-mcp-health".into(),
            compose_file: std::path::PathBuf::new(),
        },
        health_path: "/health".into(),
        ..make_entry("", 0, ActivationTier::Warm)
    };

    let result = shim_mcp_gateway::spawn::spawn_service(&entry, "docker").await;
    match result {
        Ok(pid) => assert_eq!(pid, 0, "Docker spawn returns pid=0"),
        Err(e) => panic!("expected Ok(0), got Err({:?})", e),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// G6-04: Auto runtime resolves to Native in hybrid mode
// ═════════════════════════════════════════════════════════════════════════════
#[tokio::test]
#[serial]
async fn test_spawn_auto_resolves_native() {
    let port = spawn_health_endpoint().await;

    let entry = ServiceEntry {
        name: "g6-04-auto-native".into(),
        port,
        health_port: port,
        tier: ActivationTier::Warm,
        runtime: ServiceRuntime::Auto,
        health_path: "/health".into(),
        ..make_entry("", 0, ActivationTier::Warm)
    };

    let result = shim_mcp_gateway::spawn::spawn_service(&entry, "hybrid").await;
    match result {
        Ok(pid) => assert!(pid > 0, "Auto+hybrid should spawn native and return PID > 0"),
        Err(e) => panic!("expected Ok(pid), got Err({:?})", e),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// G6-05: Auto runtime resolves to Docker in docker mode
// ═════════════════════════════════════════════════════════════════════════════
#[tokio::test]
async fn test_spawn_auto_resolves_docker() {
    let entry = ServiceEntry {
        name: "g6-05-auto-docker".into(),
        port: 19002,
        health_port: 0,
        tier: ActivationTier::Warm,
        runtime: ServiceRuntime::Auto,
        health_path: "/health".into(),
        ..make_entry("", 0, ActivationTier::Warm)
    };

    let result = shim_mcp_gateway::spawn::spawn_service(&entry, "docker").await;
    match result {
        // Auto + "docker" → resolve to Docker → docker_start on non-existent
        // container → DockerFailed. We can't test success without a real
        // container, so we verify it resolved to Docker (not RuntimeUnresolved).
        Err(SpawnError::DockerFailed(_)) => {
            // Expected: Docker resolution attempted, container doesn't exist.
        }
        Err(e) => panic!("expected DockerFailed (container doesn't exist), got {:?}", e),
        Ok(_) => panic!("should not succeed without a real Docker container"),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// G6-06: Port already bound → AlreadyRunning
// ═════════════════════════════════════════════════════════════════════════════
#[tokio::test]
async fn test_spawn_already_running() {
    let (listener, port) = bind_random_port().await;

    let entry = ServiceEntry {
        name: "g6-06-already".into(),
        port,
        health_port: 0,
        tier: ActivationTier::Warm,
        runtime: ServiceRuntime::Native {
            spawn_cmd: "/bin/echo".into(),
            pid_file: None,
        },
        health_path: "/health".into(),
        ..make_entry("", 0, ActivationTier::Warm)
    };

    let result = shim_mcp_gateway::spawn::spawn_service(&entry, "hybrid").await;
    match result {
        Err(SpawnError::AlreadyRunning) => { /* expected */ }
        other => panic!("expected AlreadyRunning, got {:?}", other),
    }

    drop(listener);
}

// ═════════════════════════════════════════════════════════════════════════════
// G6-07: Successful spawn updates registry tier to Hot
// ═════════════════════════════════════════════════════════════════════════════
#[tokio::test]
#[serial]
async fn test_spawn_updates_registry_tier() {
    let port = spawn_health_endpoint().await;

    let registry = Arc::new(ServiceRegistry::new());
    let entry = ServiceEntry {
        name: "g6-07-tier-hot".into(),
        port,
        health_port: port,
        tier: ActivationTier::Warm,
        runtime: ServiceRuntime::Native {
            spawn_cmd: "/bin/sleep".into(),
            pid_file: None,
        },
        health_path: "/health".into(),
        ..make_entry("", 0, ActivationTier::Warm)
    };
    registry.register(entry);

    let result = shim_mcp_gateway::spawn::spawn_service_and_promote(
        registry.get("g6-07-tier-hot").as_ref().unwrap(),
        "hybrid",
        &registry,
    )
    .await;

    match result {
        Ok(pid) => {
            assert!(pid > 0, "should return a valid PID");
            let svc = registry.get("g6-07-tier-hot").unwrap();
            assert_eq!(
                svc.tier,
                ActivationTier::Hot,
                "RS-2: after successful spawn + health, tier must be Hot"
            );
        }
        Err(e) => panic!("spawn_service_and_promote should succeed, got {:?}", e),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// G6-08: Unreachable health endpoint → HealthTimeout
// ═════════════════════════════════════════════════════════════════════════════
#[tokio::test]
#[serial]
async fn test_spawn_health_poll_timeout() {
    // Use a port that accepts TCP but returns nothing useful, plus a health_port
    // on an unreachable port. We bind the entry port to pass the port check,
    // but set health_port to a different unreachable port.
    let (listener, port) = bind_random_port().await;

    // A second listener on health_port that accepts but never sends HTTP 200.
    // This avoids the TcpStream fast-fail path so poll_health actually retries.
    let health_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let health_port = health_listener.local_addr().unwrap().port();

    // Accept connections but never respond (no write_all) → poll_health gets
    // TCP success but reqwest times out → exponential backoff until HealthTimeout.
    tokio::spawn(async move {
        loop {
            let _ = health_listener.accept().await;
            // Intentionally drop without sending HTTP response
        }
    });

    let registry = Arc::new(ServiceRegistry::new());
    let entry = ServiceEntry {
        name: "g6-08-timeout".into(),
        port,
        health_port,
        tier: ActivationTier::Warm,
        runtime: ServiceRuntime::Native {
            spawn_cmd: "/bin/sleep".into(),
            pid_file: None,
        },
        health_path: "/health".into(),
        ..make_entry("", 0, ActivationTier::Warm)
    };
    registry.register(entry);

    // Re-fetch the entry from registry (cloned by get()) for spawn_service.
    let entry = registry.get("g6-08-timeout").unwrap();

    // This calls spawn_service which goes through spawn_service_inner.
    // Port check passes (listener holds the entry port), native spawn succeeds
    // (/bin/sleep), then poll_health hits the health_port which accepts TCP
    // but never responds with HTTP → HealthTimeout.
    let result = shim_mcp_gateway::spawn::spawn_service(&entry, "hybrid").await;
    match result {
        Err(SpawnError::HealthTimeout { elapsed_secs }) => {
            assert!(
                elapsed_secs > 0,
                "HealthTimeout should report positive elapsed seconds"
            );
        }
        other => panic!("expected HealthTimeout, got {:?}", other),
    }

    drop(listener);
    // health_listener was moved into tokio::spawn above; no need to drop here
}

// ═════════════════════════════════════════════════════════════════════════════
// G6-09: Health poll succeeds with TCP health endpoint
// ═════════════════════════════════════════════════════════════════════════════
#[tokio::test]
#[serial]
async fn test_spawn_health_poll_success() {
    let port = spawn_health_endpoint().await;

    let entry = ServiceEntry {
        name: "g6-09-health-ok".into(),
        port,
        health_port: port,
        tier: ActivationTier::Warm,
        runtime: ServiceRuntime::Native {
            spawn_cmd: "/bin/sleep".into(),
            pid_file: None,
        },
        health_path: "/health".into(),
        ..make_entry("", 0, ActivationTier::Warm)
    };

    let result = shim_mcp_gateway::spawn::spawn_service(&entry, "hybrid").await;
    match result {
        Ok(pid) => assert!(pid > 0, "should return valid PID when health endpoint responds 200"),
        Err(e) => panic!("expected Ok(pid), got Err({:?})", e),
    }
}
