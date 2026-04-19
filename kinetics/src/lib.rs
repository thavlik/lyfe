//! # kinetics - Semantic / Thermodynamics / Kinetics Layer
//!
//! This crate provides a low-frequency semantic update engine for the `fluidsim`
//! GPU-based fluid simulation. It wraps a Lean-based logic layer to provide
//! validated update directives that improve simulation accuracy.
//!
//! ## Architecture
//!
//! - `fluidsim` remains the owner of high-frequency fine-grid GPU simulation
//! - `kinetics` is invoked once per second of simulated time
//! - It receives a coarse `SemanticSnapshot` summarizing simulation state
//! - It returns a compact `SemanticUpdate` with parameters/coefficients
//!
//! ## Design Principles
//!
//! - Lean is used for semantic reasoning, NOT per-cell-per-frame computation
//! - The snapshot is a reduced/aggregated view, NOT the entire fine grid
//! - The update is compact coefficients/directives, NOT a replacement state
//! - Lean is the single source of truth for semantic rules
//! - No-op operation is first-class only when rule evaluation is explicitly disabled
//!
//! ## Modules
//!
//! - `snapshot`: Types for the semantic snapshot sent to kinetics
//! - `update`: Types for the semantic update returned by kinetics
//! - `config`: Configuration for the kinetics engine
//! - `engine`: The main `KineticsEngine` API
//! - `noop`: No-op implementation for fallback/testing
//! - `lean_bridge`: Placeholder for Lean FFI integration
//! - `diagnostics`: Diagnostic and logging types

pub mod config;
pub mod diagnostics;
pub mod engine;
pub mod lean_bridge;
pub mod lean_evaluator;
pub mod noop;
pub mod snapshot;
pub mod update;

// Re-export primary types for ergonomic use
pub use config::KineticsConfig;
pub use diagnostics::KineticsDiagnostic;
pub use engine::KineticsEngine;
pub use snapshot::{
    BoundaryFlags, BoundarySummary, MaterialFraction, MaterialsTableSnapshot, SemanticSnapshot,
    SemanticTile, SpeciesAmount, SpeciesTableSnapshot, TileFlags,
};
pub use update::{
    BoundaryTransportMode, BoundaryUpdate, MichaelisMentenKinetics, MiscibilityOverride,
    ReactionDirective, ReactionId, ReactionKineticsModel, ReactionSetId, SemanticUpdate,
    SpeciesScalar, TileUpdate,
};

use thiserror::Error;

/// Errors that can occur during kinetics operations.
#[derive(Debug, Error)]
pub enum KineticsError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Lean bridge error: {0}")]
    LeanError(String),

    #[error("Snapshot validation error: {0}")]
    SnapshotError(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// A stable identifier for a chemical species.
///
/// This mirrors `fluidsim::SpeciesId` but is defined here to avoid
/// a circular dependency. The two types should be compatible via
/// their underlying `u64` representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeciesId(pub u64);

impl SpeciesId {
    /// Create a SpeciesId from a species name (mirrors fluidsim implementation).
    pub fn from_name(name: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = ahash::AHasher::default();
        name.hash(&mut hasher);
        Self(hasher.finish())
    }
}

/// A stable identifier for a material type.
///
/// This mirrors `fluidsim::MaterialId` but is defined here to avoid
/// a circular dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MaterialId(pub u32);

impl MaterialId {
    /// No material (fluid cell).
    pub const NONE: MaterialId = MaterialId(0);

    /// Create a new material ID.
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}
