//! # fluidsim - GPU-accelerated 2D fluid simulation with species mixing
//!
//! This library provides a high-performance Eulerian grid-based simulation for
//! species transport and diffusion in a 2D fluid domain with solid obstacles.
//!
//! ## Architecture
//!
//! - **Species System**: Polymorphic species identified by string names, interned to
//!   dense indices for GPU efficiency. External API uses `HashMap<SpeciesId, f64>`,
//!   internal storage uses `[species][cell]` dense buffers.
//!
//! - **Grid System**: Uniform 2D grid where each cell is 1x1 logical units.
//!   Solid cells act as impermeable barriers.
//!
//! - **GPU Pipeline**: All simulation runs on GPU compute shaders using ping-pong
//!   buffers for diffusion. CPU only handles setup, inspection, and control.
//!
//! - **Inspection**: Mouse inspection aggregates over coarse cells (default 8x8)
//!   using GPU readback of small regions.
//!
//! - **Kinetics Integration**: Once per second of simulated time, a semantic
//!   snapshot is sent to the `kinetics` crate for semantic evaluation. The
//!   returned update provides coefficients/directives for simulation.
//!
//! ## Data Layout
//!
//! Concentration storage uses `[species][cell]` layout where:
//! - `species_index` is the outer dimension (one buffer slice per species)
//! - `cell_index = y * width + x` is the inner dimension
//!
//! This layout optimizes for per-species diffusion passes where each compute
//! shader invocation processes one species channel across all cells.

pub mod chemistry;
pub mod coarse;
pub mod enzyme;
pub mod gpu;
pub mod grid;
pub mod inspect;
pub mod kinetics_integration;
pub mod leak;
pub mod scenario;
pub mod semantic;
pub mod sim;
pub mod solid;
pub mod species;

pub use chemistry::{ChemicalEvolutionRule, NoOpEvolution};
pub use coarse::{CoarseCellData, CoarseGrid};
pub use enzyme::{EnzymeEntity, EnzymeField};
pub use gpu::{GpuRenderBuffers, SharedGpuContext};
pub use grid::{CellCoord, Grid};
pub use inspect::{CoarseCellCoord, InspectionResult};
pub use kinetics_integration::{KineticsIntegration, SemanticUpdateApplicator};
pub use leak::LeakChannel;
pub use scenario::{
    Scenario, ScenarioBuilder, create_acid_base_scenario, create_catalyst_scenario,
    create_enzyme_scenario,
};
pub use semantic::{SemanticConfig, SemanticSnapshotBuilder};
pub use sim::{RenderState, Simulation, SimulationConfig};
pub use solid::{MaterialId, SolidCellMeta, SolidGeometry};
pub use species::{SpeciesId, SpeciesRegistry};
