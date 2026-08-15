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

use crate::registry::ServiceRegistry;
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
            eprintln!("llm-router: bind {} failed: {}", addr, e);
            return;
        }
    };
    println!(
        "llm-router: :{} ready  (gateway :8080 | anthropic fallback)",
        LLM_ROUTER_PORT
    );

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let reg = Arc::clone(&registry);
                tokio::spawn(handle(stream, reg));
            }
            Err(e) => eprintln!("llm-router: accept: {}", e),
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
