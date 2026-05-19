// shim-mcp-gateway.rs
// Minimal cold-start shim for mcp-gateway (Rust)
// Listens on :8088, proxies to mcp-gateway on :8087.
// Starts mcp-gateway if not running on first request.

use std::net::{TcpListener, TcpStream};
use std::process::{Command, Child};
use std::io::{Read, Write};
use std::thread;
use std::time::{Duration, Instant};

// Public port that clients (VS Code) connect to
const SHIM_ADDR: &str = "127.0.0.1:8088";
// Internal port where mcp-gateway actually listens
const MCP_ADDR: &str = "127.0.0.1:8087";
const MCP_STARTUP_TIMEOUT_MS: u64 = 4000;

fn is_mcp_running() -> bool {
    match TcpStream::connect(MCP_ADDR) {
        Ok(_) => true,
        Err(_) => false,
    }
}

fn start_mcp_gateway() -> Option<Child> {
    Command::new("/Users/kevintoles/POC/mcp-gateway/.venv/bin/uvicorn")
        .args(&["src.main:app", "--host", "127.0.0.1", "--port", "8087"])
        .current_dir("/Users/kevintoles/POC/mcp-gateway")
        .env("PATH", "/Users/kevintoles/POC/mcp-gateway/.venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
        .spawn()
        .ok()
}

fn wait_for_mcp_ready(timeout_ms: u64) -> bool {
    let start = Instant::now();
    while start.elapsed().as_millis() < timeout_ms as u128 {
        if is_mcp_running() {
            return true;
        }
        thread::sleep(Duration::from_millis(50));
    }
    false
}

fn proxy(mut client: TcpStream) {
    match TcpStream::connect(MCP_ADDR) {
        Ok(mut backend) => {
            let mut buf = [0u8; 8192];
            let n = client.read(&mut buf).unwrap_or(0);
            if n > 0 {
                backend.write_all(&buf[..n]).ok();
            }
            let mut backend_clone = backend.try_clone().unwrap();
            let mut client_clone = client.try_clone().unwrap();
            thread::spawn(move || std::io::copy(&mut backend, &mut client_clone).ok());
            std::io::copy(&mut client, &mut backend_clone).ok();
        }
        Err(_) => {
            let _ = client.write_all(b"HTTP/1.1 502 Bad Gateway\r\n\r\nMCP Gateway failed to start\n");
        }
    }
}

fn main() {
    if is_mcp_running() {
        println!("mcp-gateway already running on {}", MCP_ADDR);
        return;
    }
    let listener = TcpListener::bind(SHIM_ADDR).expect("Failed to bind shim to port 8088");
    println!("shim-mcp-gateway: listening on {} (proxying to {} on demand)", SHIM_ADDR, MCP_ADDR);
    for stream in listener.incoming() {
        match stream {
            Ok(client) => {
                if !is_mcp_running() {
                    println!("shim: launching mcp-gateway...");
                    let _child = start_mcp_gateway();
                    if !wait_for_mcp_ready(MCP_STARTUP_TIMEOUT_MS) {
                        let _ = client.try_clone().unwrap().write_all(b"HTTP/1.1 502 Bad Gateway\r\n\r\nMCP Gateway failed to start\n");
                        continue;
                    }
                }
                thread::spawn(move || proxy(client));
            }
            Err(e) => eprintln!("shim: connection failed: {}", e),
        }
    }
}
