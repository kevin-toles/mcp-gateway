// platform-lifecycle — platform service health manager
//
// mcp-gateway is now managed by its own LaunchAgent (com.kevintoles.mcp-gateway)
// with KeepAlive=true, like llm-gateway. This process no longer proxies for it.
//
// Responsibilities:
//   - startup_scan: promote already-healthy services Boot→Cold on startup
//   - shim_idle_monitor: Hot→Warm→Cold demotion for idle warm-tier services
//   - BootColdMonitor: poll Boot-tier services until healthy, then promote Cold
//   - health_failure_monitor: respawn services that fail consecutive health checks
//   - llm_router: :8079 mesh layer between VSCode and llm-gateway

use std::sync::Arc;
use shim_mcp_gateway::{config, lifecycle, llm_router, persistence, platform_services, registry};
use tracing_subscriber::EnvFilter;

fn main() {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("shim_mcp_gateway=info")),
        )
        .init();

    let deployment_mode = config::DeploymentMode::from_env();
    if let Err(e) = config::validate_config(&deployment_mode) {
        tracing::error!(error = %e, "config validation failed");
        std::process::exit(1);
    }
    tracing::info!(deployment_mode = ?deployment_mode, "starting platform-lifecycle");

    let registry = Arc::new(registry::ServiceRegistry::new());
    platform_services::seed_platform_services(&registry);
    tracing::info!(
        service_count = registry.len(),
        "seeded platform services into registry"
    );

    persistence::load_tier_state(&registry);

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
    rt.block_on(async {
        lifecycle::startup_scan(&registry).await;

        let reg_for_idle = Arc::clone(&registry);
        tokio::spawn(async move {
            lifecycle::shim_idle_monitor(reg_for_idle).await;
        });

        lifecycle::BootColdMonitor::new(registry.clone()).spawn_boot_tasks();

        let reg_for_health = Arc::clone(&registry);
        let dm_str = deployment_mode.as_deployment_str().to_string();
        tokio::spawn(async move {
            lifecycle::health_failure_monitor(reg_for_health, dm_str).await;
        });

        let reg_for_llm = Arc::clone(&registry);
        tokio::spawn(async move {
            llm_router::run(reg_for_llm).await;
        });

        let reg_for_persist = Arc::clone(&registry);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                persistence::save_tier_state(&reg_for_persist);
            }
        });

        tracing::info!("all monitors running");

        // Block until launchd sends SIGTERM (KeepAlive manages restart).
        std::future::pending::<()>().await;
    });
}
