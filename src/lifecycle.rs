use crate::registry::{ServiceRegistry, ActivationTier};
use crate::spawn::poll_health;
use std::sync::Arc;
use std::time::Duration;

pub const BOOT_HEALTH_TIMEOUT_SECS: u64 = 120;

// ── RS-5: HOT→WARM Shim Connection Idle Management ─────────────────────────
// HYBRID §4.6.4 — after 30min idle, transition shim connection from Hot back to Warm.
// The shim stays alive but the upstream Python gateway connection is released.

pub const HOT_IDLE_TIMEOUT_SECS: u64 = 1800;   // 30 minutes
pub const IDLE_CHECK_INTERVAL_SECS: u64 = 60;  // check every 1 minute

// ── RS-6: WARM→COLD Shim Connection Fallback ─────────────────────────────────
// HYBRID §4.6.4 — after 10min idle in Warm, transition to Cold (registry only).
// The Python mcp-gateway process is NOT stopped; re-promotion from Cold→Hot
// happens on the next MCP request via RS-3 health check.

pub const WARM_IDLE_TIMEOUT_SECS: u64 = 600;  // 10 minutes

// ── RS-4: COLD→WARM Auto-Promotion ─────────────────────────────────────────
// HYBRID §4.6.3 — N requests in M minutes (N=5, M=10min)
// Env var overrides: COLD_PROMOTION_REQUESTS, COLD_PROMOTION_WINDOW_SECS

pub const COLD_PROMOTION_WINDOW_SECS: u64 = 600; // 10 minutes

#[derive(Debug)]
pub enum LifecycleError {
    BootTimeout { service: String },
    InvalidTransition { from: ActivationTier, to: ActivationTier },
}

/// BOOT→COLD: Called once at shim startup for each service.
/// Polls health for up to 120s. On success, transitions to Cold.
/// Per HYBRID §4.6.3 transition table.
pub async fn boot_to_cold(
    registry: &ServiceRegistry,
    service_name: &str,
) -> Result<(), LifecycleError> {
    let entry = registry.get(service_name)
        .ok_or_else(|| LifecycleError::BootTimeout { service: service_name.to_string() })?;

    if entry.tier != ActivationTier::Boot {
        return Ok(()); // idempotent — already past Boot
    }

    let health_url = format!("http://localhost:{}{}", entry.port, entry.health_path);
    poll_health(
        &health_url,
        Duration::from_secs(BOOT_HEALTH_TIMEOUT_SECS),
        Duration::from_millis(500),
    ).await.map_err(|_| LifecycleError::BootTimeout {
        service: service_name.to_string()
    })?;

    registry.update_tier(service_name, ActivationTier::Cold);
    Ok(())
}

/// HOT→WARM + WARM→COLD idle monitor — spawned as a background task at shim startup.
/// Checks `mcp-gateway` entry every `IDLE_CHECK_INTERVAL_SECS`.
///
/// Two-stage detection:
///   RS-5: If Hot and idle >= hot_idle_timeout_secs → transitions to Warm (continue loop)
///   RS-6: If Warm and idle >= WARM_IDLE_TIMEOUT_SECS → transitions to Cold (break, terminal)
///
/// The upstream gateway process is NEVER stopped; this is a registry-only state change.
/// Per HYBRID §4.6.4: demoted registration means the next request triggers reconnection
/// rather than live forwarding. Re-promotion from Cold→Hot happens on the next MCP
/// request via RS-3 health check.
pub async fn shim_idle_monitor(registry: Arc<ServiceRegistry>) {
    let mut interval = tokio::time::interval(Duration::from_secs(IDLE_CHECK_INTERVAL_SECS));
    loop {
        interval.tick().await;

        let entry = match registry.get("mcp-gateway") {
            Some(e) => e,
            None => continue,
        };

        if entry.tier == ActivationTier::Cold {
            // Already Cold — nothing more to do.
            break;
        }

        // RS-5: Hot→Warm check
        if entry.tier == ActivationTier::Hot {
            let idle_secs = entry
                .last_request
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(u64::MAX);

            if idle_secs >= entry.hot_idle_timeout_secs {
                tracing::info!(
                    service = "mcp-gateway",
                    idle_secs = idle_secs,
                    timeout = entry.hot_idle_timeout_secs,
                    "hot→warm: connection idle timeout reached"
                );
                registry.update_tier("mcp-gateway", ActivationTier::Warm);
                // Continue loop — let the next tick check Warm→Cold
                continue;
            }
        }

        // RS-6: Warm→Cold check
        if entry.tier == ActivationTier::Warm {
            let warm_timeout = std::env::var("WARM_IDLE_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(WARM_IDLE_TIMEOUT_SECS);

            let idle_secs = entry
                .last_request
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(u64::MAX);

            if idle_secs >= warm_timeout {
                tracing::info!(
                    service = "mcp-gateway",
                    idle_secs = idle_secs,
                    timeout = warm_timeout,
                    "warm→cold: shim connection idle timeout reached (registry only)"
                );
                registry.update_tier("mcp-gateway", ActivationTier::Cold);
                break; // Cold is terminal for the monitor
            }
        }
    }
}

/// Startup scan — called once when shim starts.
/// Transitions all Boot services that are already healthy to Cold.
/// Implements HYBRID §4.6.4 "auto-detect running services on startup."
pub async fn startup_scan(registry: &ServiceRegistry) {
    for entry in registry.all() {
        if entry.tier == ActivationTier::Boot {
            // Non-blocking check — if already responding, promote immediately
            let health_url = format!(
                "http://localhost:{}{}",
                entry.port, entry.health_path
            );
            if reqwest::get(&health_url).await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
            {
                registry.update_tier(&entry.name, ActivationTier::Cold);
                tracing::info!(
                    service = %entry.name,
                    "startup scan: service already healthy, promoted Boot→Cold"
                );
            }
        }
    }
}
