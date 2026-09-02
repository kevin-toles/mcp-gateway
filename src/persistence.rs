use crate::registry::{ActivationTier, ServiceRegistry};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

const STATE_FILE: &str = "/tmp/hwc-tier-state.json";
const MAX_AGE_SECS: u64 = 3600;

#[derive(Serialize, Deserialize)]
struct TierSnapshot {
    services: HashMap<String, String>,
}

fn tier_to_str(tier: &ActivationTier) -> &'static str {
    match tier {
        ActivationTier::Boot => "Boot",
        ActivationTier::Cold => "Cold",
        ActivationTier::Warm => "Warm",
        ActivationTier::Hot => "Hot",
    }
}

fn str_to_tier(s: &str) -> Option<ActivationTier> {
    match s {
        "Boot" => Some(ActivationTier::Boot),
        "Cold" => Some(ActivationTier::Cold),
        "Warm" => Some(ActivationTier::Warm),
        "Hot" => Some(ActivationTier::Hot),
        _ => None,
    }
}

pub fn save_tier_state(registry: &ServiceRegistry) {
    save_tier_state_to(registry, Path::new(STATE_FILE));
}

pub fn load_tier_state(registry: &ServiceRegistry) {
    load_tier_state_from(registry, Path::new(STATE_FILE));
}

pub fn save_tier_state_to(registry: &ServiceRegistry, path: &Path) {
    let mut services = HashMap::new();
    for entry in registry.all() {
        services.insert(entry.name.clone(), tier_to_str(&entry.tier).to_string());
    }
    let snapshot = TierSnapshot { services };
    let json = match serde_json::to_string_pretty(&snapshot) {
        Ok(j) => j,
        Err(e) => {
            tracing::error!(error = %e, "failed to serialize tier state");
            return;
        }
    };
    let tmp_path = path.with_extension("tmp");
    if let Err(e) = std::fs::write(&tmp_path, &json) {
        tracing::error!(error = %e, path = %tmp_path.display(), "failed to write tier state tmp file");
        return;
    }
    if let Err(e) = std::fs::rename(&tmp_path, path) {
        tracing::error!(error = %e, "failed to rename tier state file");
    }
}

pub fn load_tier_state_from(registry: &ServiceRegistry, path: &Path) {
    let metadata = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return,
    };
    if let Some(age) = metadata.modified().ok().and_then(|m| m.elapsed().ok()) {
        if age.as_secs() > MAX_AGE_SECS {
            tracing::info!(path = %path.display(), age_secs = age.as_secs(), "tier state file too old, ignoring");
            return;
        }
    }
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!(error = %e, "failed to read tier state file");
            return;
        }
    };
    let snapshot: TierSnapshot = match serde_json::from_str(&data) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "failed to parse tier state file");
            return;
        }
    };
    for (name, tier_str) in &snapshot.services {
        if let Some(tier) = str_to_tier(tier_str) {
            if registry.get(name).is_some() {
                registry.update_tier(name, tier);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{ServiceEntry, ServiceRuntime};
    use std::collections::VecDeque;

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
            last_transition_reason: None,
            last_transition_at: None,
        }
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tier-state.json");

        let registry = ServiceRegistry::new();
        registry.register(make_entry("svc-a", 8001, ActivationTier::Warm));
        registry.register(make_entry("svc-b", 8002, ActivationTier::Hot));
        registry.register(make_entry("svc-c", 8003, ActivationTier::Cold));

        save_tier_state_to(&registry, &path);

        let registry2 = ServiceRegistry::new();
        registry2.register(make_entry("svc-a", 8001, ActivationTier::Boot));
        registry2.register(make_entry("svc-b", 8002, ActivationTier::Boot));
        registry2.register(make_entry("svc-c", 8003, ActivationTier::Boot));

        load_tier_state_from(&registry2, &path);

        assert_eq!(registry2.get("svc-a").unwrap().tier, ActivationTier::Warm);
        assert_eq!(registry2.get("svc-b").unwrap().tier, ActivationTier::Hot);
        assert_eq!(registry2.get("svc-c").unwrap().tier, ActivationTier::Cold);
    }

    #[test]
    fn test_load_handles_missing_file() {
        let registry = ServiceRegistry::new();
        registry.register(make_entry("svc-a", 8001, ActivationTier::Boot));

        load_tier_state_from(&registry, Path::new("/tmp/nonexistent-hwc-test.json"));

        assert_eq!(registry.get("svc-a").unwrap().tier, ActivationTier::Boot);
    }

    #[test]
    fn test_load_skips_stale_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tier-state.json");

        let registry = ServiceRegistry::new();
        registry.register(make_entry("svc-a", 8001, ActivationTier::Warm));
        save_tier_state_to(&registry, &path);

        // Backdate the file modification time by 2 hours
        let two_hours_ago = std::time::SystemTime::now()
            - std::time::Duration::from_secs(7200);
        filetime::FileTime::from_system_time(two_hours_ago);
        filetime::set_file_mtime(
            &path,
            filetime::FileTime::from_system_time(two_hours_ago),
        )
        .unwrap();

        let registry2 = ServiceRegistry::new();
        registry2.register(make_entry("svc-a", 8001, ActivationTier::Boot));
        load_tier_state_from(&registry2, &path);

        assert_eq!(
            registry2.get("svc-a").unwrap().tier,
            ActivationTier::Boot,
            "D-6: stale file (>1h) should be ignored"
        );
    }
}
