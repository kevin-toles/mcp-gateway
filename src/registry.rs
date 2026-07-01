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

#[derive(Debug, Clone)]
pub struct ServiceEntry {
    pub name: String,
    pub port: u16,
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
        let mut entries = self.entries.write().unwrap();
        if let Some(entry) = entries.get_mut(name) {
            // Reset failure count on healthy; apply threshold demotion on failure
            match &state {
                HealthState::Healthy => {
                    entry.failure_count = 0;
                }
                _ => {
                    entry.failure_count += 1;
                    if entry.failure_count >= FAILURE_DEMOTION_THRESHOLD {
                        if entry.tier == ActivationTier::Hot {
                            entry.tier = ActivationTier::Warm;
                        }
                    }
                }
            }
            entry.last_health = Some(state);
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
