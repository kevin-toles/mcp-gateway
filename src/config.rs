// config.rs
// Deployment mode resolution and configuration validation for the lifecycle shim.
//
// `DEPLOYMENT_MODE` is required (twelve-factor config, fail fast):
//   - docker | hybrid | native  → used as configured.
//   - auto                      → opt-in detection: Docker if the Docker API answers a
//                                 ping, otherwise Hybrid. The result and how it was
//                                 decided are logged and exposed in `/status`.
//   - unset / empty / unknown   → startup error (no silent default).
//
// The decision logic is pure (`resolve_mode`, `validate`); the shell (`load`) reads the
// environment and pings the Docker API. The ping is a startup check only — it is never
// part of the health/respawn path (a dependency check there causes restart cascades).

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use serde::Serialize;

/// Docker API socket pinged for `docker` / `auto` modes.
pub const DOCKER_SOCK: &str = "/var/run/docker.sock";
/// Upper bound for the startup Docker API ping.
pub const DOCKER_PING_TIMEOUT: Duration = Duration::from_secs(2);

/// How the shim runs and manages the Python gateway process:
/// - `Docker`: gateway managed by Docker; the shim does not spawn it.
/// - `Hybrid`: services spawned natively; infrastructure runs in Docker.
/// - `Native`: everything runs natively.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum DeploymentMode {
    Docker,
    Hybrid,
    Native,
}

impl std::fmt::Display for DeploymentMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_deployment_str())
    }
}

impl DeploymentMode {
    /// The string `spawn::resolve_runtime` expects: `docker`, `hybrid` or `native`.
    pub fn as_deployment_str(&self) -> &'static str {
        match self {
            DeploymentMode::Docker => "docker",
            DeploymentMode::Hybrid => "hybrid",
            DeploymentMode::Native => "native",
        }
    }
}

/// Whether the mode was configured explicitly or detected (`DEPLOYMENT_MODE=auto`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModeSource {
    Configured,
    AutoDetected,
}

/// The resolved deployment mode and how it was decided.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Resolution {
    pub mode: DeploymentMode,
    pub source: ModeSource,
}

/// Resolve `DEPLOYMENT_MODE` (pure). `docker_reachable` is called only for `auto`.
///
/// # Errors
///
/// Unset, empty or unknown values — there is no silent default.
pub fn resolve_mode(
    configured: Option<&str>,
    docker_reachable: impl FnOnce() -> bool,
) -> Result<Resolution, String> {
    let value = configured.map(str::trim).filter(|v| !v.is_empty()).ok_or_else(|| {
        "DEPLOYMENT_MODE is not set: expected one of docker, hybrid, native, auto".to_string()
    })?;
    let configured_mode = |mode| Ok(Resolution { mode, source: ModeSource::Configured });
    match value.to_ascii_lowercase().as_str() {
        "docker" => configured_mode(DeploymentMode::Docker),
        "hybrid" => configured_mode(DeploymentMode::Hybrid),
        "native" => configured_mode(DeploymentMode::Native),
        "auto" => Ok(Resolution {
            mode: if docker_reachable() { DeploymentMode::Docker } else { DeploymentMode::Hybrid },
            source: ModeSource::AutoDetected,
        }),
        other => Err(format!(
            "unknown DEPLOYMENT_MODE '{other}': expected one of docker, hybrid, native, auto"
        )),
    }
}

/// Check what `mode` actually needs (pure).
///
/// # Errors
///
/// Docker without a reachable Docker API; Hybrid/Native without `POC_ROOT`.
pub fn validate(mode: &DeploymentMode, docker_reachable: bool, poc_root: Option<&str>) -> Result<(), String> {
    match mode {
        DeploymentMode::Docker if !docker_reachable => Err(format!(
            "docker mode requires the Docker API, but {DOCKER_SOCK} did not answer a ping"
        )),
        DeploymentMode::Hybrid | DeploymentMode::Native if poc_root.is_none_or(str::is_empty) => {
            Err("POC_ROOT env var required in hybrid/native mode".to_string())
        }
        _ => Ok(()),
    }
}

/// `GET /_ping` over the Docker socket; true only for an HTTP 200 within `timeout`.
pub fn docker_api_reachable(sock: &Path, timeout: Duration) -> bool {
    let ping = || -> std::io::Result<bool> {
        let mut stream = UnixStream::connect(sock)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        stream.write_all(b"GET /_ping HTTP/1.0\r\nHost: docker\r\n\r\n")?;
        let mut buf = [0u8; 64];
        let n = stream.read(&mut buf)?;
        let status_line = String::from_utf8_lossy(&buf[..n]);
        Ok(status_line.starts_with("HTTP/1.") && status_line.split_whitespace().nth(1) == Some("200"))
    };
    ping().unwrap_or(false)
}

/// Shell: read the environment, ping Docker only when the mode needs it, validate.
///
/// # Errors
///
/// Any `resolve_mode` / `validate` error.
pub fn load() -> Result<Resolution, String> {
    let ping = || docker_api_reachable(Path::new(DOCKER_SOCK), DOCKER_PING_TIMEOUT);
    let resolution = resolve_mode(std::env::var("DEPLOYMENT_MODE").ok().as_deref(), ping)?;
    // Auto only resolves to Docker after a successful ping; a configured Docker mode is pinged now.
    let docker_reachable = match (&resolution.mode, resolution.source) {
        (DeploymentMode::Docker, ModeSource::AutoDetected) => true,
        (DeploymentMode::Docker, ModeSource::Configured) => ping(),
        _ => false,
    };
    validate(&resolution.mode, docker_reachable, std::env::var("POC_ROOT").ok().as_deref())?;
    Ok(resolution)
}

static RESOLVED: OnceLock<Resolution> = OnceLock::new();

/// Record the startup resolution (first call wins) so `/status` can expose it.
pub fn record(resolution: Resolution) {
    let _ = RESOLVED.set(resolution);
}

/// The resolution recorded at startup, if any.
pub fn resolved() -> Option<&'static Resolution> {
    RESOLVED.get()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::os::unix::net::UnixListener;
    use std::time::Duration;

    fn never_probed() -> bool {
        panic!("the Docker probe must only run for DEPLOYMENT_MODE=auto")
    }

    // ── resolve_mode: explicit by default, fail fast, `auto` opt-in ──────

    #[test]
    fn unset_or_empty_mode_fails_fast() {
        for configured in [None, Some(""), Some("  ")] {
            let err = resolve_mode(configured, never_probed).unwrap_err();
            assert!(err.contains("DEPLOYMENT_MODE"), "{err}");
        }
    }

    #[test]
    fn invalid_mode_fails_fast_instead_of_defaulting() {
        let err = resolve_mode(Some("dokcer"), never_probed).unwrap_err();
        assert!(err.contains("dokcer") && err.contains("docker, hybrid, native, auto"), "{err}");
    }

    #[test]
    fn explicit_modes_resolve_without_probing_docker() {
        for (raw, mode) in [
            ("docker", DeploymentMode::Docker),
            ("hybrid", DeploymentMode::Hybrid),
            ("NATIVE", DeploymentMode::Native),
        ] {
            assert_eq!(
                resolve_mode(Some(raw), never_probed).unwrap(),
                Resolution { mode, source: ModeSource::Configured }
            );
        }
    }

    #[test]
    fn auto_is_opt_in_and_uses_the_docker_probe() {
        assert_eq!(
            resolve_mode(Some("auto"), || true).unwrap(),
            Resolution { mode: DeploymentMode::Docker, source: ModeSource::AutoDetected }
        );
        assert_eq!(
            resolve_mode(Some("auto"), || false).unwrap(),
            Resolution { mode: DeploymentMode::Hybrid, source: ModeSource::AutoDetected }
        );
    }

    // ── validate: what each mode actually needs ─────────────────────────

    #[test]
    fn docker_mode_requires_a_reachable_docker_api() {
        assert!(validate(&DeploymentMode::Docker, true, None).is_ok());
        let err = validate(&DeploymentMode::Docker, false, None).unwrap_err();
        assert!(err.contains("Docker API"), "{err}");
    }

    #[test]
    fn hybrid_and_native_require_poc_root() {
        for mode in [DeploymentMode::Hybrid, DeploymentMode::Native] {
            assert!(validate(&mode, false, Some("/tmp")).is_ok());
            assert!(validate(&mode, false, None).unwrap_err().contains("POC_ROOT"));
            assert!(validate(&mode, false, Some("")).unwrap_err().contains("POC_ROOT"));
        }
    }

    // ── docker_api_reachable: a real ping, not a file-exists check ──────

    fn fake_docker(reply: Option<&'static [u8]>) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("hwc-fake-docker-{}-{}.sock", std::process::id(), reply.map_or(0, <[u8]>::len)));
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 256];
                let _ = stream.read(&mut buf);
                match reply {
                    Some(bytes) => { let _ = stream.write_all(bytes); }
                    None => std::thread::sleep(Duration::from_secs(5)), // accepts, never answers
                }
            }
        });
        path
    }

    #[test]
    fn docker_ping_ok_is_reachable() {
        let sock = fake_docker(Some(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"));
        assert!(docker_api_reachable(&sock, Duration::from_secs(2)));
    }

    #[test]
    fn missing_socket_is_unreachable() {
        assert!(!docker_api_reachable(std::path::Path::new("/tmp/hwc-no-such-docker.sock"), Duration::from_secs(1)));
    }

    #[test]
    fn socket_that_never_answers_times_out_as_unreachable() {
        let sock = fake_docker(None);
        let started = std::time::Instant::now();
        assert!(!docker_api_reachable(&sock, Duration::from_millis(300)));
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn non_ok_ping_is_unreachable() {
        let sock = fake_docker(Some(b"HTTP/1.1 500 Internal Server Error\r\n\r\n"));
        assert!(!docker_api_reachable(&sock, Duration::from_secs(2)));
    }

    // ── concrete-mode helpers ───────────────────────────────────────────

    #[test]
    fn deployment_strings_match_spawn_resolve_runtime() {
        assert_eq!(DeploymentMode::Docker.as_deployment_str(), "docker");
        assert_eq!(DeploymentMode::Hybrid.as_deployment_str(), "hybrid");
        assert_eq!(DeploymentMode::Native.as_deployment_str(), "native");
        assert_eq!(DeploymentMode::Hybrid.to_string(), "hybrid");
    }
}
