use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Activation tier per HYBRID §4.6.1 4-tier model
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationTier {
    Boot,  // First-start; single-use; transitions to Cold after first start
    Cold,  // Not running; requires manual start
    Warm,  // Not running; can auto-start on demand via shim
    Hot,   // Running and available; zero startup cost
}

impl ActivationTier {
    pub fn is_active(&self) -> bool {
        matches!(self, ActivationTier::Hot)
    }

    pub fn can_auto_start(&self) -> bool {
        matches!(self, ActivationTier::Warm)
    }
}

/// How the service is deployed in the current mode
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceRuntime {
    Docker {
        container_name: String,
        compose_file: PathBuf,
    },
    Native {
        spawn_cmd: String,
        pid_file: Option<PathBuf>,
    },
    Auto, // Detected from DEPLOYMENT_MODE at runtime
}

#[derive(Debug, Clone)]
pub enum HealthState {
    Healthy,
    Degraded { reason: String },
    Unreachable,
}

/// Failure threshold at which the registry demotes a service tier.
/// Hot → Warm at 5 consecutive failures per HYBRID §4.5 TODO annotation.
pub const FAILURE_DEMOTION_THRESHOLD: u32 = 5;

/// D-7: Circuit breaker opens after this many consecutive failures.
pub const CIRCUIT_OPEN_THRESHOLD: u32 = 10;

/// D-7: Seconds to wait in open state before allowing a half-open probe.
pub const CIRCUIT_HALF_OPEN_INTERVAL_SECS: u64 = 300;

#[derive(Debug, Clone)]
pub struct ServiceEntry {
    pub name: String,
    pub port: u16,
    pub health_port: u16, // 0 means use port
    pub tier: ActivationTier,
    pub runtime: ServiceRuntime,
    pub health_path: String,
    pub last_health: Option<HealthState>,
    pub failure_count: u32,
    /// Idle timeout config per HYBRID §4.6.3 transition rules
    pub hot_idle_timeout_secs: u64,  // default: 1800 (30min)
    pub warm_idle_timeout_secs: u64, // default: 600  (10min)
    pub last_request: Option<Instant>,
    /// Timestamps of recent requests for COLD→WARM auto-promotion (RS-4).
    /// Pruned to retain only entries within COLD_PROMOTION_WINDOW_SECS.
    pub request_timestamps: VecDeque<Instant>,
    /// D-7: When the circuit breaker opened (None = closed).
    pub circuit_open_since: Option<Instant>,
    /// D-7: Last time a half-open probe was attempted.
    pub last_half_open_probe: Option<Instant>,
    /// D-8: When consecutive failures started (None = no active failure streak).
    pub first_failure_at: Option<Instant>,
    /// D-3: Readiness endpoint path (None = use health_path for readiness too).
    pub ready_path: Option<String>,
    /// D-5: Total health checks performed against this service.
    pub health_checks_total: u64,
    /// D-5: Health checks that returned healthy.
    pub health_checks_passed: u64,
}

impl ServiceEntry {
    /// Returns the number of seconds since the last request, or `None` if
    /// no request has ever been recorded.
    pub fn last_request_elapsed_secs(&self) -> Option<f64> {
        self.last_request.map(|t| t.elapsed().as_secs_f64())
    }
}

/// Thread-safe in-memory registry for all platform service entries.
/// Single source of truth for service lifecycle state in the shim.
pub struct ServiceRegistry {
    entries: Arc<RwLock<HashMap<String, ServiceEntry>>>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        ServiceRegistry {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn register(&self, entry: ServiceEntry) {
        let mut entries = self.entries.write().unwrap();
        entries.insert(entry.name.clone(), entry);
    }

    pub fn get(&self, name: &str) -> Option<ServiceEntry> {
        self.entries.read().unwrap().get(name).cloned()
    }

    pub fn update_tier(&self, name: &str, tier: ActivationTier) {
        let mut entries = self.entries.write().unwrap();
        if let Some(entry) = entries.get_mut(name) {
            entry.tier = tier;
        }
    }

    /// Record a request timestamp for the given service.
    /// Updates `last_request` and appends `Instant::now()` to `request_timestamps`,
    /// pruning entries older than `COLD_PROMOTION_WINDOW_SECS` (600s).
    pub fn record_request(&self, name: &str) {
        let mut entries = self.entries.write().unwrap();
        if let Some(entry) = entries.get_mut(name) {
            let now = Instant::now();
            entry.last_request = Some(now);
            let window = Duration::from_secs(super::lifecycle::COLD_PROMOTION_WINDOW_SECS);
            entry.request_timestamps.push_back(now);
            // Prune expired entries older than the promotion window
            while let Some(front) = entry.request_timestamps.front() {
                if now.duration_since(*front) > window {
                    entry.request_timestamps.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    /// Count request timestamps within the given sliding window.
    pub fn recent_request_count(&self, name: &str, window: Duration) -> usize {
        let entries = self.entries.read().unwrap();
        if let Some(entry) = entries.get(name) {
            let now = Instant::now();
            entry
                .request_timestamps
                .iter()
                .filter(|&&ts| now.duration_since(ts) <= window)
                .count()
        } else {
            0
        }
    }

    pub fn update_health(&self, name: &str, state: HealthState) {
        // Record last_request when healthy so idle timers measure from
        // the most recent health-check success, not from registration.
        if matches!(state, HealthState::Healthy) {
            self.record_request(name);
        }

        let mut entries = self.entries.write().unwrap();
        if let Some(entry) = entries.get_mut(name) {
            // Reset failure count on healthy; apply threshold demotion on failure
            match &state {
                HealthState::Healthy => {
                    entry.failure_count = 0;
                    entry.circuit_open_since = None;
                    entry.last_half_open_probe = None;
                    entry.first_failure_at = None;
                }
                _ => {
                    if entry.first_failure_at.is_none() {
                        entry.first_failure_at = Some(Instant::now());
                    }
                    entry.failure_count += 1;
                    if entry.failure_count >= FAILURE_DEMOTION_THRESHOLD {
                        if entry.tier == ActivationTier::Hot {
                            entry.tier = ActivationTier::Warm;
                        }
                    }
                    if entry.failure_count >= CIRCUIT_OPEN_THRESHOLD
                        && entry.circuit_open_since.is_none()
                    {
                        entry.circuit_open_since = Some(Instant::now());
                    }
                }
            }
            entry.last_health = Some(state);
        }
    }

    pub fn open_circuit(&self, name: &str) {
        let mut entries = self.entries.write().unwrap();
        if let Some(entry) = entries.get_mut(name) {
            entry.circuit_open_since = Some(Instant::now());
        }
    }

    pub fn close_circuit(&self, name: &str) {
        let mut entries = self.entries.write().unwrap();
        if let Some(entry) = entries.get_mut(name) {
            entry.circuit_open_since = None;
            entry.last_half_open_probe = None;
            entry.failure_count = 0;
        }
    }

    pub fn try_half_open_probe(&self, name: &str) -> bool {
        let mut entries = self.entries.write().unwrap();
        if let Some(entry) = entries.get_mut(name) {
            let now = Instant::now();
            if let Some(opened) = entry.circuit_open_since {
                if now.duration_since(opened).as_secs() < CIRCUIT_HALF_OPEN_INTERVAL_SECS {
                    return false;
                }
                if let Some(last_probe) = entry.last_half_open_probe {
                    if now.duration_since(last_probe).as_secs() < CIRCUIT_HALF_OPEN_INTERVAL_SECS {
                        return false;
                    }
                }
                entry.last_half_open_probe = Some(now);
                return true;
            }
        }
        false
    }

    pub fn increment_health_check(&self, name: &str, passed: bool) {
        let mut entries = self.entries.write().unwrap();
        if let Some(entry) = entries.get_mut(name) {
            entry.health_checks_total += 1;
            if passed {
                entry.health_checks_passed += 1;
            }
        }
    }

    pub fn all(&self) -> Vec<ServiceEntry> {
        self.entries.read().unwrap().values().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().unwrap().is_empty()
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            circuit_open_since: None,
            last_half_open_probe: None,
            first_failure_at: None,
            ready_path: None,
            health_checks_total: 0,
            health_checks_passed: 0,
        }
    }

    #[test]
    fn test_register_service() {
        let registry = ServiceRegistry::new();
        let entry = make_entry("llm-gateway", 8080, ActivationTier::Hot);
        registry.register(entry.clone());
        let got = registry.get("llm-gateway");
        assert!(got.is_some());
        assert_eq!(got.unwrap().port, 8080);
    }

    #[test]
    fn test_get_missing() {
        let registry = ServiceRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_update_tier() {
        let registry = ServiceRegistry::new();
        let entry = make_entry("test-svc", 9000, ActivationTier::Cold);
        registry.register(entry);
        registry.update_tier("test-svc", ActivationTier::Hot);
        let got = registry.get("test-svc").unwrap();
        assert_eq!(got.tier, ActivationTier::Hot);
    }

    #[test]
    fn test_update_health() {
        let registry = ServiceRegistry::new();
        let entry = make_entry("test-svc", 9000, ActivationTier::Hot);
        registry.register(entry);
        registry.update_health("test-svc", HealthState::Degraded {
            reason: "timeout".into(),
        });
        let got = registry.get("test-svc").unwrap();
        match got.last_health.unwrap() {
            HealthState::Degraded { reason } => assert_eq!(reason, "timeout"),
            _ => panic!("expected Degraded"),
        }
    }

    #[test]
    fn test_increment_failure_count() {
        let registry = ServiceRegistry::new();
        let mut entry = make_entry("test-svc", 9000, ActivationTier::Hot);
        entry.failure_count = 0;
        registry.register(entry);

        // Each non-Healthy update increments failure_count
        registry.update_health("test-svc", HealthState::Unreachable);
        registry.update_health("test-svc", HealthState::Unreachable);
        let got = registry.get("test-svc").unwrap();
        assert_eq!(got.failure_count, 2);
    }

    #[test]
    fn test_reset_failures_on_healthy() {
        let registry = ServiceRegistry::new();
        let mut entry = make_entry("test-svc", 9000, ActivationTier::Hot);
        entry.failure_count = 3;
        registry.register(entry);

        registry.update_health("test-svc", HealthState::Healthy);
        let got = registry.get("test-svc").unwrap();
        assert_eq!(got.failure_count, 0);
    }

    #[test]
    fn test_failure_threshold_demotes() {
        let registry = ServiceRegistry::new();
        let mut entry = make_entry("test-svc", 9000, ActivationTier::Hot);
        entry.failure_count = 0;
        registry.register(entry);

        // 5 failures triggers demotion Hot → Warm
        for _ in 0..5 {
            registry.update_health("test-svc", HealthState::Unreachable);
        }
        let got = registry.get("test-svc").unwrap();
        assert_eq!(got.tier, ActivationTier::Warm);
        assert_eq!(got.failure_count, 5);
    }

    #[test]
    fn test_all_services() {
        let registry = ServiceRegistry::new();
        registry.register(make_entry("svc-a", 8001, ActivationTier::Hot));
        registry.register(make_entry("svc-b", 8002, ActivationTier::Warm));
        registry.register(make_entry("svc-c", 8003, ActivationTier::Cold));
        assert_eq!(registry.all().len(), 3);
    }

    #[test]
    fn test_concurrent_access() {
        let registry = ServiceRegistry::new();
        registry.register(make_entry("shared", 9000, ActivationTier::Hot));

        let mut handles = vec![];
        for i in 0..10 {
            let reg_ref = &registry as *const ServiceRegistry;
            unsafe {
                let reg = &*reg_ref;
                handles.push(std::thread::spawn(move || {
                    if i % 2 == 0 {
                        reg.update_health("shared", HealthState::Unreachable);
                    } else {
                        let _ = reg.get("shared");
                    }
                }));
            }
        }
        for h in handles {
            h.join().unwrap();
        }
        // No crash = no data race
        let final_count = registry.get("shared").unwrap().failure_count;
        assert!(final_count <= 10);
    }
}
