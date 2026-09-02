use crate::registry::ActivationTier;

pub struct TierSlo {
    pub uptime_target: f64,
}

pub const HOT_SLO: TierSlo = TierSlo { uptime_target: 0.999 };
pub const WARM_SLO: TierSlo = TierSlo { uptime_target: 0.99 };
pub const COLD_SLO: TierSlo = TierSlo { uptime_target: 0.95 };
pub const BOOT_SLO: TierSlo = TierSlo { uptime_target: 0.0 };

pub fn uptime_ratio(passed: u64, total: u64) -> f64 {
    if total == 0 {
        return 1.0;
    }
    passed as f64 / total as f64
}

pub fn slo_for_tier(tier: &ActivationTier) -> &'static TierSlo {
    match tier {
        ActivationTier::Hot => &HOT_SLO,
        ActivationTier::Warm => &WARM_SLO,
        ActivationTier::Cold => &COLD_SLO,
        ActivationTier::Boot => &BOOT_SLO,
    }
}

pub fn within_slo(ratio: f64, tier: &ActivationTier) -> bool {
    ratio >= slo_for_tier(tier).uptime_target
}

pub fn error_budget_remaining(ratio: f64, tier: &ActivationTier) -> f64 {
    let target = slo_for_tier(tier).uptime_target;
    let budget = 1.0 - target;
    if budget <= 0.0 {
        return 0.0;
    }
    let consumed = 1.0 - ratio;
    ((budget - consumed) / budget).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uptime_ratio() {
        assert!((uptime_ratio(990, 1000) - 0.99).abs() < f64::EPSILON);
        assert!((uptime_ratio(0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((uptime_ratio(500, 1000) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_within_slo() {
        assert!(!within_slo(0.99, &ActivationTier::Hot));
        assert!(within_slo(0.999, &ActivationTier::Hot));
        assert!(within_slo(0.99, &ActivationTier::Warm));
        assert!(!within_slo(0.98, &ActivationTier::Warm));
        assert!(within_slo(0.95, &ActivationTier::Cold));
    }

    #[test]
    fn test_error_budget_remaining() {
        let remaining = error_budget_remaining(0.999, &ActivationTier::Hot);
        assert!((remaining - 0.0).abs() < 0.01);

        let remaining = error_budget_remaining(1.0, &ActivationTier::Hot);
        assert!((remaining - 1.0).abs() < 0.01);

        let remaining = error_budget_remaining(0.99, &ActivationTier::Warm);
        assert!((remaining - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_sli_counters_increment() {
        use crate::registry::{ServiceEntry, ServiceRegistry, ServiceRuntime};
        use std::collections::VecDeque;

        let registry = ServiceRegistry::new();
        registry.register(ServiceEntry {
            name: "test-svc".to_string(),
            port: 9000,
            health_port: 0,
            tier: ActivationTier::Hot,
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
        });

        registry.increment_health_check("test-svc", true);
        registry.increment_health_check("test-svc", true);
        registry.increment_health_check("test-svc", false);
        registry.increment_health_check("test-svc", true);
        registry.increment_health_check("test-svc", false);

        let entry = registry.get("test-svc").unwrap();
        assert_eq!(entry.health_checks_total, 5);
        assert_eq!(entry.health_checks_passed, 3);

        let ratio = uptime_ratio(entry.health_checks_passed, entry.health_checks_total);
        assert!((ratio - 0.6).abs() < f64::EPSILON);
    }
}
