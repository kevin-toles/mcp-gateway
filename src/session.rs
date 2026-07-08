// session.rs
// Session management for the Rust shim layer (RS-2).
// Maps inbound TCP connections to sessions with lifecycle tracking
// (Active→Idle→Suspended→Expired) and integration with the registry
// tier system (Hot→Warm→Cold).
//
// Architecture:
//   SessionManager wraps a SessionPool + PoolConfig, providing
//   a thread-safe API for the proxy loop. Each inbound connection
//   gets a session. Idle/session-ttl expiry drives cleanup.
//
// SessionState maps to ActivationTier as follows:
//   Active  → Hot
//   Idle    → Warm
//   Expired → Cold

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use uuid::Uuid;

use crate::registry::{ActivationTier, ServiceRegistry};

/// Newtype wrapper for session identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(Uuid);

impl SessionId {
    pub fn new() -> Self {
        SessionId(Uuid::new_v4())
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Session lifecycle states per RS-2 design.
///
/// Active  = connection is live, being proxied → maps to Hot
/// Idle    = connection still open but no activity → maps to Warm
/// Suspended = session preserved but connection closed → could auto-resume
/// Expired = removed from pool → maps to Cold
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SessionState {
    Active,
    Idle,
    Suspended,
    Expired,
}

impl SessionState {
    pub fn is_active(&self) -> bool {
        matches!(self, SessionState::Active)
    }

    pub fn can_auto_resume(&self) -> bool {
        matches!(self, SessionState::Suspended)
    }

    pub fn to_activation_tier(&self) -> ActivationTier {
        match self {
            SessionState::Active => ActivationTier::Hot,
            SessionState::Idle | SessionState::Suspended => ActivationTier::Warm,
            SessionState::Expired => ActivationTier::Cold,
        }
    }
}

/// Configuration for session pool behaviour.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of concurrent sessions (0 = unlimited).
    pub max_sessions: u32,
    /// Time of inactivity before Active→Idle transition.
    pub idle_timeout: Duration,
    /// How often the cleanup sweep runs.
    pub cleanup_interval: Duration,
    /// Maximum lifetime for any session regardless of activity.
    pub session_ttl: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            max_sessions: 256,
            idle_timeout: Duration::from_secs(300),   // 5 min
            cleanup_interval: Duration::from_secs(60), // 1 min
            session_ttl: Duration::from_secs(3600),    // 1 hour
        }
    }
}

/// A single session assigned to an inbound TCP proxy connection.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: SessionId,
    pub state: SessionState,
    pub created_at: Instant,
    pub updated_at: Instant,
    pub last_active: Instant,
    pub metadata: HashMap<String, String>,
    pub service_name: Option<String>,
}

impl Session {
    pub fn new() -> Self {
        let now = Instant::now();
        Session {
            id: SessionId::new(),
            state: SessionState::Active,
            created_at: now,
            updated_at: now,
            last_active: now,
            metadata: HashMap::new(),
            service_name: None,
        }
    }

    /// Time since last activity.
    pub fn idle_duration(&self) -> Duration {
        self.last_active.elapsed()
    }

    /// Age of the session since creation.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Update last_active and updated_at to now.
    pub fn touch(&mut self) {
        let now = Instant::now();
        self.last_active = now;
        self.updated_at = now;
    }

    /// Transition to a new state, recording the time.
    pub fn transition_to(&mut self, new_state: SessionState) {
        self.state = new_state;
        self.updated_at = Instant::now();
    }
}

/// Pool management errors.
#[derive(Debug, Clone, PartialEq)]
pub enum PoolError {
    SessionNotFound(SessionId),
    PoolExhausted { max: u32 },
    InvalidTransition { from: SessionState, to: SessionState },
    SessionExpired(SessionId),
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::SessionNotFound(id) => write!(f, "session {} not found", id),
            PoolError::PoolExhausted { max } => {
                write!(f, "pool exhausted (max {})", max)
            }
            PoolError::InvalidTransition { from, to } => {
                write!(f, "invalid state transition {:?} -> {:?}", from, to)
            }
            PoolError::SessionExpired(id) => write!(f, "session {} has expired", id),
        }
    }
}

impl std::error::Error for PoolError {}

/// Thread-safe session lifecycle API.
///
/// Provides the primary public interface for the proxy loop:
/// - `create()` — assign a new session for an incoming connection
/// - `activate()` — mark a session as active (on data)
/// - `suspend()` — mark session as suspended (connection close with retain)
/// - `expire()` — mark session as expired (removal from pool)
/// - `destroy()` — remove a session immediately
pub trait SessionLifecycle: Send + Sync {
    fn create(&self, service_name: Option<&str>) -> Result<SessionId, PoolError>;
    fn activate(&self, id: SessionId) -> Result<(), PoolError>;
    fn suspend(&self, id: SessionId) -> Result<(), PoolError>;
    fn expire(&self, id: SessionId) -> Result<(), PoolError>;
    fn destroy(&self, id: SessionId) -> Result<(), PoolError>;
    fn get(&self, id: SessionId) -> Option<Session>;
    fn active_count(&self) -> usize;
    fn total_count(&self) -> usize;
}

/// Core session pool with partitioned lookup sets.
///
/// Uses `Arc<RwLock<HashMap>>` for session storage (matching `registry.rs`)
/// and separate `RwLock<HashSet>` pools for fast iteration by state.
struct SessionPool {
    sessions: Arc<RwLock<HashMap<SessionId, Session>>>,
    active_ids: RwLock<HashSet<SessionId>>,
    idle_ids: RwLock<HashSet<SessionId>>,
    suspended_ids: RwLock<HashSet<SessionId>>,
    expired_ids: RwLock<HashSet<SessionId>>,
}

impl SessionPool {
    fn new() -> Self {
        SessionPool {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            active_ids: RwLock::new(HashSet::new()),
            idle_ids: RwLock::new(HashSet::new()),
            suspended_ids: RwLock::new(HashSet::new()),
            expired_ids: RwLock::new(HashSet::new()),
        }
    }

    /// Track a session id in the appropriate set for its state.
    fn track_in_state_set(&self, id: SessionId, state: SessionState) {
        // Remove from all sets first, then add to the correct one
        let _ = self.active_ids.write().unwrap().remove(&id);
        let _ = self.idle_ids.write().unwrap().remove(&id);
        let _ = self.suspended_ids.write().unwrap().remove(&id);
        let _ = self.expired_ids.write().unwrap().remove(&id);

        match state {
            SessionState::Active => { self.active_ids.write().unwrap().insert(id); }
            SessionState::Idle => { self.idle_ids.write().unwrap().insert(id); }
            SessionState::Suspended => { self.suspended_ids.write().unwrap().insert(id); }
            SessionState::Expired => { self.expired_ids.write().unwrap().insert(id); }
        }
    }

    /// Remove a session id from all state sets.
    fn untrack_completely(&self, id: SessionId) {
        let _ = self.active_ids.write().unwrap().remove(&id);
        let _ = self.idle_ids.write().unwrap().remove(&id);
        let _ = self.suspended_ids.write().unwrap().remove(&id);
        let _ = self.expired_ids.write().unwrap().remove(&id);
    }
}

/// Public manager wrapping SessionPool + PoolConfig + optional registry link.
pub struct SessionManager {
    pool: SessionPool,
    config: PoolConfig,
    /// Optional link to the service registry for tier cascade.
    registry: Option<Arc<ServiceRegistry>>,
}

impl SessionManager {
    pub fn new(config: PoolConfig) -> Self {
        SessionManager {
            pool: SessionPool::new(),
            config,
            registry: None,
        }
    }

    /// Link this manager to a service registry for tier cascading.
    /// When set, session state transitions also trigger registry tier updates.
    pub fn with_registry(mut self, registry: Arc<ServiceRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Cascade session state to the registry tier for a given service name.
    fn cascade_to_registry(&self, service_name: Option<&str>, session_state: SessionState) {
        if let Some(name) = service_name {
            if let Some(ref reg) = self.registry {
                reg.update_tier(name, session_state.to_activation_tier());
                // GAP-3: Record a request timestamp alongside the tier update
                // so the idle monitor measures from last actual traffic (RS-5).
                reg.record_request(name);
            }
        }
    }

    /// Run one cleanup pass: expire idle sessions past idle_timeout
    /// and any session past session_ttl.
    pub fn cleanup_expired(&self) -> Vec<SessionId> {
        let mut expired = Vec::new();
        let config = self.config.clone();

        // Check idle sessions for timeout
        let idle_ids: Vec<SessionId> = self.pool.idle_ids.read().unwrap().iter().copied().collect();
        for id in &idle_ids {
            if let Some(session) = self.pool.sessions.read().unwrap().get(id) {
                if session.idle_duration() >= config.idle_timeout
                    || session.age() >= config.session_ttl
                {
                    expired.push(*id);
                }
            }
        }

        // Also check active sessions for session_ttl
        let active_ids: Vec<SessionId> = self.pool.active_ids.read().unwrap().iter().copied().collect();
        for id in &active_ids {
            if let Some(session) = self.pool.sessions.read().unwrap().get(id) {
                if session.age() >= config.session_ttl {
                    expired.push(*id);
                }
            }
        }

        // Expire all collected ids
        for id in &expired {
            if let Some(session) = self.pool.sessions.write().unwrap().get_mut(id) {
                session.transition_to(SessionState::Expired);
                let service = session.service_name.clone();
                self.pool.track_in_state_set(*id, SessionState::Expired);
                if let Some(ref name) = service {
                    if let Some(ref reg) = self.registry {
                        reg.update_tier(name, ActivationTier::Cold);
                        // GAP-3: Record a request timestamp so the idle monitor
                        // measures from last actual traffic (RS-5).
                        reg.record_request(name);
                    }
                }
            }
        }

        expired
    }

    /// Move active→idle for sessions past idle_timeout.
    pub fn check_idle_timeouts(&self) -> Vec<SessionId> {
        let mut idled = Vec::new();
        let timeout = self.config.idle_timeout;

        let active_ids: Vec<SessionId> = self.pool.active_ids.read().unwrap().iter().copied().collect();
        for id in &active_ids {
            if let Some(session) = self.pool.sessions.read().unwrap().get(id) {
                if session.idle_duration() >= timeout {
                    idled.push(*id);
                }
            }
        }

        for id in &idled {
            if let Some(session) = self.pool.sessions.write().unwrap().get_mut(id) {
                session.transition_to(SessionState::Idle);
                let service = session.service_name.clone();
                self.pool.track_in_state_set(*id, SessionState::Idle);
                self.cascade_to_registry(service.as_deref(), SessionState::Idle);
            }
        }

        idled
    }
}

impl SessionLifecycle for SessionManager {
    fn create(&self, service_name: Option<&str>) -> Result<SessionId, PoolError> {
        // Check pool capacity
        if self.config.max_sessions > 0 {
            let current = self.pool.sessions.read().unwrap().len() as u32;
            if current >= self.config.max_sessions {
                return Err(PoolError::PoolExhausted { max: self.config.max_sessions });
            }
        }

        let mut session = Session::new();
        session.service_name = service_name.map(|s| s.to_string());
        let id = session.id;

        self.pool.sessions.write().unwrap().insert(id, session);
        self.pool.track_in_state_set(id, SessionState::Active);
        self.cascade_to_registry(service_name, SessionState::Active);

        Ok(id)
    }

    fn activate(&self, id: SessionId) -> Result<(), PoolError> {
        let mut sessions = self.pool.sessions.write().unwrap();
        let session = sessions.get_mut(&id).ok_or(PoolError::SessionNotFound(id))?;

        if matches!(session.state, SessionState::Expired) {
            return Err(PoolError::SessionExpired(id));
        }

        // Allow activation from Idle or Suspended
        if !matches!(session.state, SessionState::Active | SessionState::Idle | SessionState::Suspended) {
            return Err(PoolError::InvalidTransition {
                from: session.state,
                to: SessionState::Active,
            });
        }

        session.touch();
        session.state = SessionState::Active;
        self.pool.track_in_state_set(id, SessionState::Active);
        self.cascade_to_registry(session.service_name.as_deref(), SessionState::Active);

        Ok(())
    }

    fn suspend(&self, id: SessionId) -> Result<(), PoolError> {
        let mut sessions = self.pool.sessions.write().unwrap();
        let session = sessions.get_mut(&id).ok_or(PoolError::SessionNotFound(id))?;

        if !matches!(session.state, SessionState::Active | SessionState::Idle) {
            return Err(PoolError::InvalidTransition {
                from: session.state,
                to: SessionState::Suspended,
            });
        }

        session.transition_to(SessionState::Suspended);
        self.pool.track_in_state_set(id, SessionState::Suspended);
        self.cascade_to_registry(session.service_name.as_deref(), SessionState::Suspended);

        Ok(())
    }

    fn expire(&self, id: SessionId) -> Result<(), PoolError> {
        let mut sessions = self.pool.sessions.write().unwrap();
        let session = sessions.get_mut(&id).ok_or(PoolError::SessionNotFound(id))?;

        if session.state == SessionState::Expired {
            return Err(PoolError::InvalidTransition {
                from: session.state,
                to: SessionState::Expired,
            });
        }

        session.transition_to(SessionState::Expired);
        self.pool.track_in_state_set(id, SessionState::Expired);
        self.cascade_to_registry(session.service_name.as_deref(), SessionState::Expired);

        Ok(())
    }

    fn destroy(&self, id: SessionId) -> Result<(), PoolError> {
        let mut sessions = self.pool.sessions.write().unwrap();
        sessions.remove(&id).ok_or(PoolError::SessionNotFound(id))?;
        self.pool.untrack_completely(id);
        Ok(())
    }

    fn get(&self, id: SessionId) -> Option<Session> {
        self.pool.sessions.read().unwrap().get(&id).cloned()
    }

    fn active_count(&self) -> usize {
        self.pool.active_ids.read().unwrap().len()
    }

    fn total_count(&self) -> usize {
        self.pool.sessions.read().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_manager() -> SessionManager {
        SessionManager::new(PoolConfig::default())
    }

    #[test]
    fn test_create_session() {
        let mgr = default_manager();
        let id = mgr.create(Some("mcp-gateway")).unwrap();
        let session = mgr.get(id).unwrap();
        assert_eq!(session.state, SessionState::Active);
        assert_eq!(session.service_name.as_deref(), Some("mcp-gateway"));
        assert_eq!(mgr.active_count(), 1);
        assert_eq!(mgr.total_count(), 1);
    }

    #[test]
    fn test_activate_then_suspend_then_expire() {
        let mgr = default_manager();
        let id = mgr.create(None).unwrap();

        // Freshly created: Active → Suspend
        mgr.suspend(id).unwrap();
        let session = mgr.get(id).unwrap();
        assert_eq!(session.state, SessionState::Suspended);

        // Suspend → Activate
        mgr.activate(id).unwrap();
        let session = mgr.get(id).unwrap();
        assert_eq!(session.state, SessionState::Active);

        // Active → Expire
        mgr.expire(id).unwrap();
        let session = mgr.get(id).unwrap();
        assert_eq!(session.state, SessionState::Expired);

        // Verify can't reactivate expired
        let err = mgr.activate(id).unwrap_err();
        assert_eq!(err, PoolError::SessionExpired(id));
    }

    #[test]
    fn test_active_to_idle_transition() {
        let mgr = default_manager();
        let id = mgr.create(Some("test-service")).unwrap();

        // Simulate idle timeout by setting idle_timeout to 0
        let short_config = PoolConfig {
            idle_timeout: Duration::from_secs(0),
            ..PoolConfig::default()
        };
        let mgr_short = SessionManager::new(short_config);

        let id2 = mgr_short.create(Some("test-service")).unwrap();
        let idled = mgr_short.check_idle_timeouts();
        assert!(idled.contains(&id2), "session should be idled out");

        let session = mgr_short.get(id2).unwrap();
        assert_eq!(session.state, SessionState::Idle);
    }

    #[test]
    fn test_cleanup_expired_sessions() {
        let config = PoolConfig {
            idle_timeout: Duration::from_secs(0),
            session_ttl: Duration::from_secs(3600), // long TTL
            ..PoolConfig::default()
        };
        let mgr = SessionManager::new(config);

        let id = mgr.create(None).unwrap();
        mgr.suspend(id).unwrap();
        let session = mgr.get(id).unwrap();
        assert_eq!(session.state, SessionState::Suspended);

        // Suspend → Idle (so cleanup picks it up)
        // We need to transition it to Idle for idle_timeout to trigger
        let mgr2 = SessionManager::new(PoolConfig {
            idle_timeout: Duration::from_secs(0),
            session_ttl: Duration::from_secs(3600),
            ..PoolConfig::default()
        });
        let id2 = mgr2.create(None).unwrap();
        // It starts Active. The cleanup checks idle sessions. Move to idle first.
        // We can't directly set state, so let's use idle timeout check
        let idled = mgr2.check_idle_timeouts();
        assert!(idled.contains(&id2));

        // Now cleanup should expire it
        let expired = mgr2.cleanup_expired();
        assert!(expired.contains(&id2));

        let session = mgr2.get(id2).unwrap();
        assert_eq!(session.state, SessionState::Expired);
    }

    #[test]
    fn test_ttl_expiry_cleans_up_active_sessions() {
        let config = PoolConfig {
            idle_timeout: Duration::from_secs(3600), // long idle
            session_ttl: Duration::from_secs(0),     // immediate TTL expiry
            ..PoolConfig::default()
        };
        let mgr = SessionManager::new(config);

        let id = mgr.create(None).unwrap();
        let expired = mgr.cleanup_expired();
        assert!(expired.contains(&id), "session should be expired by TTL");

        let session = mgr.get(id).unwrap();
        assert_eq!(session.state, SessionState::Expired);
    }

    #[test]
    fn test_pool_exhaustion() {
        let config = PoolConfig {
            max_sessions: 2,
            ..PoolConfig::default()
        };
        let mgr = SessionManager::new(config);

        mgr.create(None).unwrap();
        mgr.create(None).unwrap();

        let err = mgr.create(None).unwrap_err();
        assert_eq!(err, PoolError::PoolExhausted { max: 2 });
    }

    #[test]
    fn test_destroy_removes_session() {
        let mgr = default_manager();
        let id = mgr.create(None).unwrap();
        assert_eq!(mgr.total_count(), 1);

        mgr.destroy(id).unwrap();
        assert!(mgr.get(id).is_none());
        assert_eq!(mgr.total_count(), 0);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_invalid_transition_suspended_to_expired_then_double_expire() {
        let mgr = default_manager();
        let id = mgr.create(None).unwrap();
        mgr.suspend(id).unwrap();
        mgr.expire(id).unwrap();

        let err = mgr.expire(id).unwrap_err();
        assert_eq!(
            err,
            PoolError::InvalidTransition {
                from: SessionState::Expired,
                to: SessionState::Expired,
            }
        );
    }

    #[test]
    fn test_session_not_found() {
        let mgr = default_manager();
        let phantom_id = SessionId::new();

        let err = mgr.activate(phantom_id).unwrap_err();
        assert_eq!(err, PoolError::SessionNotFound(phantom_id));

        let err = mgr.destroy(phantom_id).unwrap_err();
        assert_eq!(err, PoolError::SessionNotFound(phantom_id));
    }

    #[test]
    fn test_session_touch_updates_last_active() {
        let mgr = default_manager();
        let id = mgr.create(None).unwrap();

        let before = mgr.get(id).unwrap().last_active;
        std::thread::sleep(Duration::from_millis(5));

        mgr.activate(id).unwrap(); // touch via activate
        let after = mgr.get(id).unwrap().last_active;
        assert!(after > before);
    }

    #[test]
    fn test_service_name_stored_and_cascaded() {
        let registry = Arc::new(ServiceRegistry::new());
        let mgr = SessionManager::new(PoolConfig::default())
            .with_registry(Arc::clone(&registry));

        // Register a service in the registry first
        registry.register(crate::registry::ServiceEntry {
            name: "llm-gateway".to_string(),
            port: 8080,
            health_port: 0,
            tier: ActivationTier::Hot,
            runtime: crate::registry::ServiceRuntime::Auto,
            health_path: "/health".to_string(),
            last_health: None,
            failure_count: 0,
            hot_idle_timeout_secs: 1800,
            warm_idle_timeout_secs: 600,
            last_request: None,
            request_timestamps: std::collections::VecDeque::new(),
        });

        let id = mgr.create(Some("llm-gateway")).unwrap();
        let session = mgr.get(id).unwrap();
        assert_eq!(session.service_name.as_deref(), Some("llm-gateway"));

        // Verify registry tier was updated
        let entry = registry.get("llm-gateway").unwrap();
        assert_eq!(entry.tier, ActivationTier::Hot);
    }
}
