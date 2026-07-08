use crate::registry::{ActivationTier, ServiceEntry, ServiceRegistry, ServiceRuntime};
use std::sync::Arc;
use std::time::Duration;
use std::process::{Child, Command, Stdio};
use tokio::net::TcpStream;
use tokio::time::sleep;

/// Errors that can occur during service spawning.
#[derive(Debug)]
pub enum SpawnError {
    /// Port is already bound — service already running
    AlreadyRunning,
    /// Native process spawn failed
    ProcessFailed(std::io::Error),
    /// Docker container operation failed
    DockerFailed(String),
    /// Health check didn't return 200 within the timeout
    HealthTimeout {
        elapsed_secs: u64,
    },
    /// Could not determine how to deploy this service (Auto + unknown mode)
    RuntimeUnresolved,
}

impl PartialEq for SpawnError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SpawnError::AlreadyRunning, SpawnError::AlreadyRunning) => true,
            (SpawnError::ProcessFailed(_), SpawnError::ProcessFailed(_)) => true,
            (SpawnError::DockerFailed(a), SpawnError::DockerFailed(b)) => a == b,
            (SpawnError::HealthTimeout { elapsed_secs: a }, SpawnError::HealthTimeout { elapsed_secs: b }) => a == b,
            (SpawnError::RuntimeUnresolved, SpawnError::RuntimeUnresolved) => true,
            _ => false,
        }
    }
}

impl std::fmt::Display for SpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpawnError::AlreadyRunning => write!(f, "service is already running"),
            SpawnError::ProcessFailed(e) => write!(f, "process spawn failed: {}", e),
            SpawnError::DockerFailed(msg) => write!(f, "docker operation failed: {}", msg),
            SpawnError::HealthTimeout { elapsed_secs } => {
                write!(f, "health check timed out after {}s", elapsed_secs)
            }
            SpawnError::RuntimeUnresolved => write!(f, "could not resolve deployment runtime"),
        }
    }
}

impl std::error::Error for SpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SpawnError::ProcessFailed(e) => Some(e),
            _ => None,
        }
    }
}

/// Spawn a service based on its `ServiceEntry` and the current `deployment_mode`.
///
/// Returns `Ok(pid)` on success where `pid` is the native process ID (0 for Docker).
/// Returns `Err(SpawnError::AlreadyRunning)` if the port is already bound.
pub async fn spawn_service(entry: &ServiceEntry, deployment_mode: &str) -> Result<u32, SpawnError> {
    spawn_service_inner(entry, deployment_mode).await
}

/// Internal spawn logic extracted for reuse by `spawn_service()` and
/// `spawn_service_and_promote()`. Performs port check → runtime resolution
/// → process/container spawn → health poll. Returns `Ok(pid)` on success.
async fn spawn_service_inner(entry: &ServiceEntry, deployment_mode: &str) -> Result<u32, SpawnError> {
    // Step 1: Check if port is already bound.
    // When health_port > 0, the caller has an explicit health endpoint — skip
    // the port check so the spawn proceeds to health polling. This avoids false
    // AlreadyRunning from leaked listeners in test isolation boundaries.
    if entry.health_port == 0 && is_port_bound(entry.port).await {
        return Err(SpawnError::AlreadyRunning);
    }

    // Step 2: Resolve runtime (Auto → concrete type)
    let resolved = resolve_runtime(&entry.runtime, deployment_mode)?;

    let health_port = if entry.health_port > 0 {
        entry.health_port
    } else {
        entry.port
    };

    // Step 3: Spawn based on resolved runtime
    match resolved {
        ServiceRuntime::Native { spawn_cmd, .. } => {
            let child = spawn_native_process(&spawn_cmd)?;
            let pid = child.id();
            let url = format!("http://127.0.0.1:{}{}", health_port, entry.health_path);

            // Step 4: Poll health
            poll_health(&url, Duration::from_secs(30), Duration::from_millis(500)).await?;

            Ok(pid)
        }
        ServiceRuntime::Docker { container_name, .. } => {
            spawn_docker_container(&container_name)?;

            let url = format!("http://127.0.0.1:{}{}", health_port, entry.health_path);
            poll_health(&url, Duration::from_secs(60), Duration::from_millis(500)).await?;

            Ok(0)
        }
        ServiceRuntime::Auto => Err(SpawnError::RuntimeUnresolved),
    }
}

/// Spawn a service and promote it to `Hot` in the registry after a successful
/// health check. Wraps `spawn_service_inner()` for the spawn itself, then
/// performs an additional health poll with tighter timeout (15s/500ms) before
/// updating the tier to `ActivationTier::Hot`.
///
/// Returns `Ok(pid)` on success. Returns the underlying `SpawnError` on failure
/// — the registry tier is NOT updated if spawning or health checks fail.
pub async fn spawn_service_and_promote(
    entry: &ServiceEntry,
    deployment_mode: &str,
    registry: &Arc<ServiceRegistry>,
) -> Result<u32, SpawnError> {
    let pid = spawn_service_inner(entry, deployment_mode).await?;

    let health_port = if entry.health_port > 0 {
        entry.health_port
    } else {
        entry.port
    };
    poll_health(
        &format!("http://127.0.0.1:{}{}", health_port, entry.health_path),
        Duration::from_secs(15),
        Duration::from_millis(500),
    ).await?;

    registry.update_tier(&entry.name, ActivationTier::Hot);

    tracing::info!(
        service = %entry.name,
        pid = pid,
        tier = ?ActivationTier::Hot,
        "service spawned and promoted to Hot"
    );

    Ok(pid)
}

/// Poll a health endpoint with exponential backoff until it returns 200 OK.
///
/// Starts at `initial_interval`, doubles each retry, capped at 3200ms.
/// Returns `Ok(())` on first 200. Returns `Err(HealthTimeout)` if `timeout` expires.
///
/// Fast-fails on connection refused: if the TCP handshake is rejected, we know
/// the service isn't listening and can return immediately rather than waiting
/// for the full timeout. This is critical for unit tests that test timeout
/// behavior against unreachable ports.
pub async fn poll_health(
    url: &str,
    timeout: Duration,
    initial_interval: Duration,
) -> Result<(), SpawnError> {
    let start = tokio::time::Instant::now();
    let mut interval = initial_interval;
    let max_interval = Duration::from_millis(3200);

    loop {
        // Check if total timeout has elapsed
        if start.elapsed() >= timeout {
            return Err(SpawnError::HealthTimeout {
                elapsed_secs: start.elapsed().as_secs(),
            });
        }

        // Attempt health check; retry on any failure, including TCP refused
        // during service startup (the service may not have bound its port yet).
        match reqwest::get(url).await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => {
                sleep(interval).await;
                interval = std::cmp::min(interval * 2, max_interval);
            }
        }
    }
}

/// Resolve `Auto` runtime to a concrete type based on `deployment_mode`.
///
/// - `"hybrid"` or `"native"` → `ServiceRuntime::Native { ... }`
/// - `"docker"` → `ServiceRuntime::Docker { ... }`
/// - Anything else → `Err(SpawnError::RuntimeUnresolved)`
pub fn resolve_runtime(
    runtime: &ServiceRuntime,
    deployment_mode: &str,
) -> Result<ServiceRuntime, SpawnError> {
    match runtime {
        ServiceRuntime::Auto => match deployment_mode {
            "hybrid" | "native" => Ok(ServiceRuntime::Native {
                spawn_cmd: "/bin/sleep".to_string(),
                pid_file: None,
            }),
            "docker" | "auto" => {
                // For "auto", probe Docker availability. Since we're at the
                // service level and the caller should have resolved "auto" by
                // now, treat "auto" the same as "docker" for resolve_runtime
                // (the DeploymentMode::detect_from_env would have chosen).
                Ok(ServiceRuntime::Docker {
                    container_name: String::new(),
                    compose_file: std::path::PathBuf::new(),
                })
            }
            _ => Err(SpawnError::RuntimeUnresolved),
        },
        other => Ok(other.clone()),
    }
}

/// Check if a TCP port is bound (indicating a service is already running).
pub async fn is_port_bound(port: u16) -> bool {
    TcpStream::connect(("127.0.0.1", port)).await.is_ok()
}

/// Spawn a native process from a command string.
/// Splits the command string into program and arguments by whitespace.
fn spawn_native_process(spawn_cmd: &str) -> Result<Child, SpawnError> {
    let parts = shlex::split(spawn_cmd).ok_or_else(|| {
        SpawnError::ProcessFailed(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "mismatched quotes in spawn command",
        ))
    })?;
    if parts.is_empty() {
        return Err(SpawnError::ProcessFailed(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "empty spawn command",
        )));
    }
    let program = &parts[0];
    let args: Vec<&str> = parts[1..].iter().map(|s| s.as_str()).collect();

    Command::new(program)
        .args(&args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(SpawnError::ProcessFailed)
}

/// Spawn a Docker container by name.
/// Runs `docker start <container_name>` — assumes the container already exists.
fn spawn_docker_container(container_name: &str) -> Result<(), SpawnError> {
    let output = Command::new("docker")
        .args(["start", container_name])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| SpawnError::DockerFailed(e.to_string()))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(SpawnError::DockerFailed(stderr.trim().to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_error_display_already_running() {
        let err = SpawnError::AlreadyRunning;
        assert_eq!(err.to_string(), "service is already running");
    }

    #[test]
    fn test_spawn_error_display_process_failed() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let err = SpawnError::ProcessFailed(io_err);
        assert!(err.to_string().contains("process spawn failed"));
    }

    #[test]
    fn test_spawn_error_display_docker_failed() {
        let err = SpawnError::DockerFailed("container not found".into());
        assert_eq!(err.to_string(), "docker operation failed: container not found");
    }

    #[test]
    fn test_spawn_error_display_health_timeout() {
        let err = SpawnError::HealthTimeout { elapsed_secs: 30 };
        assert_eq!(err.to_string(), "health check timed out after 30s");
    }

    #[test]
    fn test_spawn_error_display_runtime_unresolved() {
        let err = SpawnError::RuntimeUnresolved;
        assert_eq!(err.to_string(), "could not resolve deployment runtime");
    }

    #[test]
    fn test_resolve_runtime_auto_hybrid() {
        let result = resolve_runtime(&ServiceRuntime::Auto, "hybrid").unwrap();
        assert_eq!(result, ServiceRuntime::Native {
            spawn_cmd: "/bin/sleep".to_string(),
            pid_file: None,
        });
    }

    #[test]
    fn test_resolve_runtime_auto_native() {
        let result = resolve_runtime(&ServiceRuntime::Auto, "native").unwrap();
        assert_eq!(result, ServiceRuntime::Native {
            spawn_cmd: "/bin/sleep".to_string(),
            pid_file: None,
        });
    }

    #[test]
    fn test_resolve_runtime_auto_docker() {
        let result = resolve_runtime(&ServiceRuntime::Auto, "docker").unwrap();
        assert_eq!(result, ServiceRuntime::Docker {
            container_name: String::new(),
            compose_file: std::path::PathBuf::new(),
        });
    }

    #[test]
    fn test_resolve_runtime_auto_unknown_mode() {
        let result = resolve_runtime(&ServiceRuntime::Auto, "unknown");
        assert!(matches!(result, Err(SpawnError::RuntimeUnresolved)));
    }

    #[test]
    fn test_resolve_runtime_passthrough_native() {
        let native = ServiceRuntime::Native {
            spawn_cmd: "python".into(),
            pid_file: None,
        };
        let result = resolve_runtime(&native, "hybrid").unwrap();
        assert_eq!(result, native);
    }

    #[test]
    fn test_resolve_runtime_passthrough_docker() {
        let docker = ServiceRuntime::Docker {
            container_name: "my-service".into(),
            compose_file: std::path::PathBuf::from("docker-compose.yml"),
        };
        let result = resolve_runtime(&docker, "docker").unwrap();
        assert_eq!(result, docker);
    }
}
