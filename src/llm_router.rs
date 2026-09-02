//! LLM Router — mesh layer for VSCode → Anthropic API routing.
//!
//! Listens on LLM_ROUTER_PORT (:8079). Routing decision at call time (CMS pattern):
//!
//!   1. TCP probe llm-gateway (:8080) — fast, ~1ms on localhost
//!   2. Healthy  → forward via HTTP to localhost:8080 (normal path)
//!   3. Unhealthy → attempt spawn via registry machinery (H/W/C on-demand start)
//!   4. Still down → forward directly to https://api.anthropic.com (graceful degradation)
//!
//! Port 8079 never goes dark — either the gateway enriches the call or Anthropic receives
//! it directly. ANTHROPIC_BASE_URL should point to http://localhost:8079.

use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::registry::{ActivationTier, ServiceRegistry, TierEvent};
use crate::spawn;

pub const LLM_ROUTER_PORT: u16 = 8079;
const LLM_GATEWAY_ADDR: &str = "127.0.0.1:8080";
const LLM_GATEWAY_BASE: &str = "http://127.0.0.1:8080";
const ANTHROPIC_BASE: &str = "https://api.anthropic.com";

/// TCP probe timeout — intentionally tight. localhost should respond in <5ms.
const PROBE_TIMEOUT_MS: u64 = 300;
/// How long to wait for llm-gateway to become reachable after a spawn attempt.
const SPAWN_SETTLE_MS: u64 = 2_000;
/// Total budget for the spawn + re-probe path before degrading to Anthropic direct.
const SPAWN_BUDGET_SECS: u64 = 8;

pub async fn run(registry: Arc<ServiceRegistry>) {
    let addr = format!("0.0.0.0:{}", LLM_ROUTER_PORT);
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(addr = %addr, error = %e, "llm-router: bind failed");
            return;
        }
    };
    tracing::info!(
        port = LLM_ROUTER_PORT,
        "llm-router ready (gateway :8080 | anthropic fallback)"
    );

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let reg = Arc::clone(&registry);
                tokio::spawn(handle(stream, reg));
            }
            Err(e) => tracing::error!(error = %e, "llm-router: accept failed"),
        }
    }
}

// ── Request parsing ──────────────────────────────────────────────────────────

struct Req {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

async fn read_req(stream: &mut TcpStream) -> Option<Req> {
    let mut raw: Vec<u8> = Vec::with_capacity(4096);
    let mut tmp = [0u8; 4096];

    let header_end = loop {
        let n = stream.read(&mut tmp).await.ok()?;
        if n == 0 {
            return None;
        }
        raw.extend_from_slice(&tmp[..n]);
        if let Some(p) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if raw.len() > 131_072 {
            return None; // 128 KB header limit
        }
    };

    let header_str = std::str::from_utf8(&raw[..header_end]).ok()?;
    let mut lines = header_str.lines();

    let request_line = lines.next()?;
    let mut parts = request_line.splitn(3, ' ');
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut headers: Vec<(String, String)> = Vec::new();
    let mut content_length: usize = 0;
    for line in lines {
        if let Some(colon) = line.find(':') {
            let k = line[..colon].trim().to_string();
            let v = line[colon + 1..].trim().to_string();
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }

    // Body bytes already pulled in beyond the header block
    let body_start = header_end + 4; // skip \r\n\r\n
    let mut body: Vec<u8> = if raw.len() > body_start {
        raw[body_start..].to_vec()
    } else {
        Vec::new()
    };

    while body.len() < content_length {
        let want = (content_length - body.len()).min(65_536);
        let mut chunk = vec![0u8; want];
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }

    Some(Req { method, path, headers, body })
}

// ── Connection handler ───────────────────────────────────────────────────────

async fn handle(mut client: TcpStream, registry: Arc<ServiceRegistry>) {
    let req = match read_req(&mut client).await {
        Some(r) => r,
        None => return,
    };

    if req.path.starts_with("/health") {
        let body = b"{\"status\":\"ok\",\"mode\":\"llm-router\"}";
        let _ = simple_response(&mut client, 200, "OK", body).await;
        return;
    }

    if req.path == "/status" && req.method == "GET" {
        let body = serde_json::to_string(&status_snapshot(&registry))
            .unwrap_or_else(|_| "{\"services\":[]}".to_string());
        let _ = simple_response(&mut client, 200, "OK", body.as_bytes()).await;
        return;
    }

    if req.path == "/slo" && req.method == "GET" {
        let mut slo_data = Vec::new();
        for entry in registry.all() {
            let ratio = crate::slo::uptime_ratio(
                entry.health_checks_passed,
                entry.health_checks_total,
            );
            let in_slo = crate::slo::within_slo(ratio, &entry.tier);
            let budget = crate::slo::error_budget_remaining(ratio, &entry.tier);
            slo_data.push(serde_json::json!({
                "service": entry.name,
                "tier": format!("{:?}", entry.tier),
                "health_checks_total": entry.health_checks_total,
                "health_checks_passed": entry.health_checks_passed,
                "uptime_ratio": (ratio * 10000.0).round() / 10000.0,
                "slo_target": crate::slo::slo_for_tier(&entry.tier).uptime_target,
                "within_slo": in_slo,
                "error_budget_remaining": (budget * 10000.0).round() / 10000.0,
            }));
        }
        let body = serde_json::to_string(&slo_data).unwrap_or_else(|_| "[]".to_string());
        let _ = simple_response(&mut client, 200, "OK", body.as_bytes()).await;
        return;
    }

    // POST /activity/{service} — request-driven lifecycle transition.
    // Any caller (Python gateway, direct HTTP, service-to-service) can
    // report that a service handled a valid request. This is the primary
    // mechanism for Cold/Warm → Hot promotion after the proxy removal.
    if req.path.starts_with("/activity/") && req.method == "POST" {
        let service = &req.path["/activity/".len()..];
        if service.is_empty() {
            let _ = simple_response(&mut client, 400, "Bad Request", b"{\"error\":\"missing service name\"}").await;
        } else {
            registry.record_request(service);
            // The reducer logs the transition (with from/to tiers) itself.
            let promoted = registry.apply_event(service, TierEvent::Activity).is_some();
            // Request-driven lifecycle: if the service is dead (Native
            // runtime, port unbound), spawn it NOW in the background —
            // don't make the caller wait for health-monitor cadence.
            {
                let reg = Arc::clone(&registry);
                let name = service.to_string();
                tokio::spawn(async move {
                    crate::lifecycle::ensure_running_after_activity(&reg, &name).await;
                });
            }
            let body = if promoted {
                format!("{{\"service\":\"{}\",\"promoted\":true,\"tier\":\"Hot\"}}", service)
            } else {
                format!("{{\"service\":\"{}\",\"recorded\":true}}", service)
            };
            let _ = simple_response(&mut client, 200, "OK", body.as_bytes()).await;
        }
        return;
    }

    let base = if gateway_reachable().await {
        LLM_GATEWAY_BASE
    } else {
        let started = tokio::time::timeout(
            Duration::from_secs(SPAWN_BUDGET_SECS),
            try_start_gateway(&registry),
        )
        .await
        .unwrap_or(false);

        if started {
            LLM_GATEWAY_BASE
        } else {
            // CMS-style graceful degradation — go direct to Anthropic
            ANTHROPIC_BASE
        }
    };

    forward(&mut client, &req, base).await;
}

// ── Health probe ─────────────────────────────────────────────────────────────

async fn gateway_reachable() -> bool {
    tokio::time::timeout(
        Duration::from_millis(PROBE_TIMEOUT_MS),
        TcpStream::connect(LLM_GATEWAY_ADDR),
    )
    .await
    .map(|r| r.is_ok())
    .unwrap_or(false)
}

async fn try_start_gateway(registry: &Arc<ServiceRegistry>) -> bool {
    let entry = match registry.get("llm-gateway") {
        Some(e) => e,
        None => return false,
    };
    if spawn::spawn_service_and_promote(&entry, "hybrid", registry)
        .await
        .is_ok()
    {
        tokio::time::sleep(Duration::from_millis(SPAWN_SETTLE_MS)).await;
        gateway_reachable().await
    } else {
        false
    }
}

// ── Upstream forwarding ──────────────────────────────────────────────────────

async fn forward(client: &mut TcpStream, req: &Req, base: &str) {
    let url = format!("{}{}", base, req.path);
    let to_anthropic = base.starts_with("https://api.anthropic");

    let http = match reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("{{\"error\":\"reqwest build: {}\"}}", e);
            let _ = simple_response(client, 502, "Bad Gateway", msg.as_bytes()).await;
            return;
        }
    };

    let method = reqwest::Method::from_bytes(req.method.as_bytes())
        .unwrap_or(reqwest::Method::POST);

    let mut builder = http.request(method, &url);
    for (k, v) in &req.headers {
        let kl = k.to_lowercase();
        // Drop hop-by-hop and host — reqwest sets host from the URL
        if matches!(
            kl.as_str(),
            "host" | "content-length" | "transfer-encoding" | "connection"
        ) {
            continue;
        }
        builder = builder.header(k.as_str(), v.as_str());
    }
    if to_anthropic {
        builder = builder.header("host", "api.anthropic.com");
    }
    if !req.body.is_empty() {
        builder = builder.body(req.body.clone());
    }

    let mut resp = match builder.send().await {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("{{\"error\":\"upstream: {}\"}}", e);
            let _ = simple_response(client, 502, "Bad Gateway", msg.as_bytes()).await;
            return;
        }
    };

    // Write status line
    let status = resp.status();
    let status_line = format!(
        "HTTP/1.1 {} {}\r\n",
        status.as_u16(),
        status.canonical_reason().unwrap_or("Unknown")
    );
    if client.write_all(status_line.as_bytes()).await.is_err() {
        return;
    }

    // Forward response headers, stripping hop-by-hop
    for (k, v) in resp.headers() {
        let kl = k.as_str().to_lowercase();
        if matches!(
            kl.as_str(),
            "transfer-encoding" | "connection" | "content-length"
        ) {
            continue;
        }
        if let Ok(vs) = v.to_str() {
            let hdr = format!("{}: {}\r\n", k.as_str(), vs);
            if client.write_all(hdr.as_bytes()).await.is_err() {
                return;
            }
        }
    }
    // Connection: close so the client knows body ends when we close
    if client.write_all(b"connection: close\r\n\r\n").await.is_err() {
        return;
    }

    // Stream body — each chunk forwarded immediately (SSE-safe)
    loop {
        match resp.chunk().await {
            Ok(Some(bytes)) => {
                if client.write_all(&bytes).await.is_err() {
                    break;
                }
            }
            Ok(None) | Err(_) => break,
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

async fn simple_response(
    client: &mut TcpStream,
    code: u16,
    reason: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        code, reason, body.len()
    );
    client.write_all(head.as_bytes()).await?;
    client.write_all(body).await
}

/// Full platform-state snapshot for GET /status (assessment finding #2).
/// One JSON object per service with everything needed to answer
/// "why is X stuck?" without log diving: tier, runtime, health, failure
/// streaks, circuit state, last request, and the last tier transition.
pub fn status_snapshot(registry: &ServiceRegistry) -> serde_json::Value {
    let mut services = Vec::new();
    for entry in registry.all() {
        let runtime = match &entry.runtime {
            crate::registry::ServiceRuntime::Auto => "auto",
            crate::registry::ServiceRuntime::Native { .. } => "native",
            crate::registry::ServiceRuntime::Docker { .. } => "docker",
        };
        let last_health = entry.last_health.as_ref().map(|h| format!("{:?}", h));
        services.push(serde_json::json!({
            "service": entry.name,
            "port": entry.port,
            "tier": format!("{:?}", entry.tier),
            "runtime": runtime,
            "last_health": last_health,
            "failure_count": entry.failure_count,
            "first_failure_elapsed_secs": entry.first_failure_at.map(|t| t.elapsed().as_secs()),
            "circuit_open": entry.circuit_open_since.is_some(),
            "circuit_open_elapsed_secs": entry.circuit_open_since.map(|t| t.elapsed().as_secs()),
            "last_request_elapsed_secs": entry.last_request_elapsed_secs().map(|s| s.round()),
            "last_transition": entry.last_transition_reason,
            "last_transition_elapsed_secs": entry.last_transition_at.map(|t| t.elapsed().as_secs()),
            "health_checks_total": entry.health_checks_total,
            "health_checks_passed": entry.health_checks_passed,
        }));
    }
    serde_json::json!({ "services": services })
}

#[cfg(test)]
mod status_tests {
    use super::*;
    use crate::registry::{ActivationTier, ServiceEntry, ServiceRuntime, TierEvent};
    use std::collections::VecDeque;

    fn make_entry(name: &str, tier: ActivationTier) -> ServiceEntry {
        ServiceEntry {
            name: name.to_string(),
            port: 9000,
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
            circuit_open_since: None,
            last_half_open_probe: None,
            first_failure_at: None,
            ready_path: None,
            health_checks_total: 0,
            health_checks_passed: 0,
            last_transition_reason: None,
            last_transition_at: None,
        }
    }

    #[test]
    fn status_snapshot_reports_all_diagnostic_fields() {
        let registry = ServiceRegistry::new();
        registry.register(make_entry("svc", ActivationTier::Boot));
        registry.apply_event("svc", TierEvent::ScanFoundReady);

        let snap = status_snapshot(&registry);
        let services = snap["services"].as_array().expect("services array");
        assert_eq!(services.len(), 1);
        let svc = &services[0];
        assert_eq!(svc["service"], "svc");
        assert_eq!(svc["port"], 9000);
        assert_eq!(svc["tier"], "Cold");
        assert_eq!(svc["runtime"], "auto");
        assert_eq!(svc["failure_count"], 0);
        assert_eq!(svc["circuit_open"], false);
        assert_eq!(svc["last_transition"], "Boot→Cold (scan-found-ready)");
        assert!(svc["last_transition_elapsed_secs"].is_number());
    }

    #[test]
    fn status_snapshot_reports_failure_diagnostics() {
        use crate::registry::HealthState;
        let registry = ServiceRegistry::new();
        registry.register(make_entry("sick", ActivationTier::Hot));
        for _ in 0..3 {
            registry.update_health("sick", HealthState::Unreachable);
        }
        let snap = status_snapshot(&registry);
        let svc = &snap["services"][0];
        assert_eq!(svc["failure_count"], 3);
        assert!(svc["first_failure_elapsed_secs"].is_number());
        assert!(svc["last_health"].as_str().unwrap().contains("Unreachable"));
    }
}
