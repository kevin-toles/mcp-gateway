// GAP-6: Library crate root for shim-mcp-gateway.
//
// Integration tests in tests/ cannot import from binary-only crates
// (Rust restriction). This lib.rs re-exports all modules, allowing
// tests/spawn_test.rs to use `use shim_mcp_gateway::spawn::spawn_service;`.
//
// main.rs also imports from here (after its mod declarations were moved).

pub mod config;
pub mod lifecycle;
pub mod platform_services;
pub mod registry;
pub mod session;
pub mod spawn;
