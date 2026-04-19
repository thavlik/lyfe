//! Scenario initialization and builders.
//!
//! Provides a high-level API for setting up simulation initial conditions,
//! including species registration, solid geometry, and initial concentrations.

mod acid_base;
mod buffers;
mod builder;
mod catalyst;
mod core;
mod demo;
mod enzyme;
mod helpers;
mod leak;

pub use acid_base::create_acid_base_scenario;
pub use buffers::create_buffers_scenario;
pub use builder::ScenarioBuilder;
pub use catalyst::create_catalyst_scenario;
pub use core::Scenario;
pub use demo::create_demo_scenario;
pub use enzyme::create_enzyme_scenario;
pub use leak::create_leak_scenario;
