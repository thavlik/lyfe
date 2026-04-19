//! Semantic snapshot types for kinetics evaluation.
//!
//! The semantic snapshot is a reduced/aggregated view of the fine-grid
//! simulation state. It is designed to be:
//! - Compact enough for CPU-side reasoning
//! - Rich enough for meaningful semantic evaluation
//! - Independent of fine-grid resolution
//!
//! ## Tile Grid
//!
//! The snapshot uses a coarse tile grid overlaid on the fine simulation:
//! - Fine grid: e.g., 1024x1024 cells
//! - Coarse grid: e.g., 32x32 or 64x64 tiles
//! - Each tile summarizes the state of many fine cells
//!
//! The exact resolution is configurable and should balance:
//! - Semantic meaningfulness (tiles shouldn't be too fine)
//! - Spatial resolution (tiles shouldn't be too coarse)

use crate::{MaterialId, SpeciesId};

bitflags::bitflags! {
    /// Flags describing the semantic state of a tile.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TileFlags: u32 {
        /// Tile contains at least some fluid cells
        const HAS_FLUID = 0b0001;
        /// Tile contains at least some solid cells
        const HAS_SOLID = 0b0010;
        /// Tile touches a fluid-solid interface
        const TOUCHES_INTERFACE = 0b0100;
        /// Tile is at the domain boundary
        const IS_BOUNDARY = 0b1000;
        /// Tile temperature exceeds a threshold
        const IS_HOT = 0b0001_0000;
        /// Tile temperature is below a threshold
        const IS_COLD = 0b0010_0000;
        /// Tile has active reactions occurring
        const HAS_REACTIONS = 0b0100_0000;
        /// Tile has concentration gradients above threshold
        const HAS_GRADIENT = 0b1000_0000;
    }
}

bitflags::bitflags! {
    /// Flags describing boundary characteristics.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BoundaryFlags: u32 {
        /// Boundary is between fluid and solid
        const FLUID_SOLID = 0b0001;
        /// Boundary is between different materials
        const MATERIAL_INTERFACE = 0b0010;
        /// Boundary is at domain edge
        const DOMAIN_EDGE = 0b0100;
        /// Boundary is permeable
        const PERMEABLE = 0b1000;
        /// Boundary has temperature discontinuity
        const THERMAL_DISCONTINUITY = 0b0001_0000;
    }
}

/// A quantity of a specific species.
#[derive(Debug, Clone)]
pub struct SpeciesAmount {
    /// The species this amount refers to
    pub species_id: SpeciesId,
    /// The value (molarity, moles, or other depending on context)
    pub value: f64,
}

impl SpeciesAmount {
    pub fn new(species_id: SpeciesId, value: f64) -> Self {
        Self { species_id, value }
    }
}

/// Fraction of a tile occupied by a specific material.
#[derive(Debug, Clone)]
pub struct MaterialFraction {
    /// The material this fraction refers to
    pub material_id: MaterialId,
    /// Fraction of the tile (0.0 to 1.0)
    pub fraction: f32,
}

impl MaterialFraction {
    pub fn new(material_id: MaterialId, fraction: f32) -> Self {
        Self {
            material_id,
            fraction,
        }
    }
}

/// A single coarse tile summarizing a region of the fine grid.
#[derive(Debug, Clone)]
pub struct SemanticTile {
    /// Unique identifier for this tile
    pub tile_id: u32,
    /// X coordinate in coarse grid
    pub x: u32,
    /// Y coordinate in coarse grid
    pub y: u32,

    // --- Composition fractions ---
    /// Fraction of cells that are fluid (0.0 to 1.0)
    pub fluid_fraction: f32,
    /// Fraction of cells that are solid (0.0 to 1.0)
    pub solid_fraction: f32,

    // --- Thermodynamic state ---
    /// Mean temperature across fluid cells in this tile (Kelvin)
    pub mean_temperature_kelvin: f64,
    /// Min temperature in tile (Kelvin)
    pub min_temperature_kelvin: f64,
    /// Max temperature in tile (Kelvin)
    pub max_temperature_kelvin: f64,
    /// Temperature variance (for gradient detection)
    pub temperature_variance: f64,

    // --- Species concentrations ---
    /// Mean molarity for each detected species in this tile
    pub species_mean_molarity: Vec<SpeciesAmount>,
    /// Total moles for each species (for conservation tracking)
    pub species_total_moles: Vec<SpeciesAmount>,
    /// Max concentration gradient magnitude per species (mol/m)
    pub species_max_gradient: Vec<SpeciesAmount>,

    // --- Material composition ---
    /// Fraction of tile occupied by each material type
    pub material_fractions: Vec<MaterialFraction>,

    // --- Semantic flags ---
    /// Semantic state flags for this tile
    pub flags: TileFlags,
}

impl SemanticTile {
    /// Create a new empty tile at the given coordinates.
    pub fn new(tile_id: u32, x: u32, y: u32) -> Self {
        Self {
            tile_id,
            x,
            y,
            fluid_fraction: 0.0,
            solid_fraction: 0.0,
            mean_temperature_kelvin: 293.15, // ~20°C default
            min_temperature_kelvin: 293.15,
            max_temperature_kelvin: 293.15,
            temperature_variance: 0.0,
            species_mean_molarity: Vec::new(),
            species_total_moles: Vec::new(),
            species_max_gradient: Vec::new(),
            material_fractions: Vec::new(),
            flags: TileFlags::empty(),
        }
    }

    /// Check if this tile contains any fluid.
    pub fn has_fluid(&self) -> bool {
        self.flags.contains(TileFlags::HAS_FLUID)
    }

    /// Check if this tile contains any solid.
    pub fn has_solid(&self) -> bool {
        self.flags.contains(TileFlags::HAS_SOLID)
    }

    /// Check if this is a mixed fluid-solid tile.
    pub fn is_interface(&self) -> bool {
        self.flags.contains(TileFlags::TOUCHES_INTERFACE)
    }
}

/// Summary of a boundary between adjacent tiles or regions.
#[derive(Debug, Clone)]
pub struct BoundarySummary {
    /// Unique identifier for this boundary
    pub boundary_id: u32,
    /// ID of first adjacent tile
    pub tile_a: u32,
    /// ID of second adjacent tile (or u32::MAX for domain edge)
    pub tile_b: u32,
    /// Material at the boundary (if solid boundary)
    pub material_between: Option<MaterialId>,
    /// Estimated contact area (simulation units²)
    pub contact_area_estimate: f32,
    /// Temperature difference across boundary (Kelvin)
    pub temperature_delta: f64,
    /// Boundary characteristic flags
    pub flags: BoundaryFlags,
}

impl BoundarySummary {
    /// Create a new boundary summary.
    pub fn new(boundary_id: u32, tile_a: u32, tile_b: u32) -> Self {
        Self {
            boundary_id,
            tile_a,
            tile_b,
            material_between: None,
            contact_area_estimate: 0.0,
            temperature_delta: 0.0,
            flags: BoundaryFlags::empty(),
        }
    }
}

/// Snapshot of the species registry for kinetics context.
#[derive(Debug, Clone, Default)]
pub struct SpeciesTableSnapshot {
    /// Species IDs in dense index order
    pub species_ids: Vec<SpeciesId>,
    /// Species names (parallel to species_ids)
    pub species_names: Vec<String>,
    /// Current diffusion coefficients (parallel to species_ids)
    pub diffusion_coefficients: Vec<f32>,
}

/// Snapshot of the materials registry for kinetics context.
#[derive(Debug, Clone, Default)]
pub struct MaterialsTableSnapshot {
    /// Material IDs in dense index order
    pub material_ids: Vec<MaterialId>,
    /// Material names (parallel to material_ids)
    pub material_names: Vec<String>,
}

/// The complete semantic snapshot of simulation state.
///
/// This is the input to `KineticsEngine::evaluate()`.
/// It represents a reduced, aggregated view of the fine-grid simulation
/// suitable for CPU-side semantic reasoning.
#[derive(Debug, Clone)]
pub struct SemanticSnapshot {
    /// Current simulation time (seconds)
    pub sim_time_seconds: f64,
    /// Time window this snapshot represents (seconds since last snapshot)
    pub dt_window_seconds: f64,

    // --- Coarse grid dimensions ---
    /// Width of the coarse semantic grid
    pub coarse_width: u32,
    /// Height of the coarse semantic grid
    pub coarse_height: u32,
    /// Number of fine cells per coarse tile in each dimension
    pub tile_size: u32,

    // --- Fine grid dimensions (for context) ---
    /// Width of the fine simulation grid
    pub fine_width: u32,
    /// Height of the fine simulation grid
    pub fine_height: u32,

    // --- Tile data ---
    /// Coarse semantic tiles (row-major order)
    pub tiles: Vec<SemanticTile>,

    // --- Boundary data ---
    /// Important boundaries between tiles/regions
    pub boundaries: Vec<BoundarySummary>,

    // --- Registry snapshots ---
    /// Species registry snapshot
    pub species_table: SpeciesTableSnapshot,
    /// Materials registry snapshot
    pub materials_table: MaterialsTableSnapshot,

    // --- Global statistics ---
    /// Total moles of each species across entire domain
    pub global_species_totals: Vec<SpeciesAmount>,
    /// Domain-wide mean temperature (Kelvin)
    pub global_mean_temperature: f64,
    /// Total thermal energy estimate (Joules)
    pub global_thermal_energy: f64,
}

impl SemanticSnapshot {
    /// Create an empty snapshot with the given dimensions.
    pub fn new(
        coarse_width: u32,
        coarse_height: u32,
        tile_size: u32,
        fine_width: u32,
        fine_height: u32,
    ) -> Self {
        let tile_count = (coarse_width * coarse_height) as usize;
        let mut tiles = Vec::with_capacity(tile_count);

        for y in 0..coarse_height {
            for x in 0..coarse_width {
                let tile_id = y * coarse_width + x;
                tiles.push(SemanticTile::new(tile_id, x, y));
            }
        }

        Self {
            sim_time_seconds: 0.0,
            dt_window_seconds: 1.0,
            coarse_width,
            coarse_height,
            tile_size,
            fine_width,
            fine_height,
            tiles,
            boundaries: Vec::new(),
            species_table: SpeciesTableSnapshot::default(),
            materials_table: MaterialsTableSnapshot::default(),
            global_species_totals: Vec::new(),
            global_mean_temperature: 293.15,
            global_thermal_energy: 0.0,
        }
    }

    /// Get a tile by coarse coordinates.
    pub fn tile(&self, x: u32, y: u32) -> Option<&SemanticTile> {
        if x >= self.coarse_width || y >= self.coarse_height {
            return None;
        }
        let idx = (y * self.coarse_width + x) as usize;
        self.tiles.get(idx)
    }

    /// Get a mutable tile by coarse coordinates.
    pub fn tile_mut(&mut self, x: u32, y: u32) -> Option<&mut SemanticTile> {
        if x >= self.coarse_width || y >= self.coarse_height {
            return None;
        }
        let idx = (y * self.coarse_width + x) as usize;
        self.tiles.get_mut(idx)
    }

    /// Get a tile by tile ID.
    pub fn tile_by_id(&self, tile_id: u32) -> Option<&SemanticTile> {
        self.tiles.get(tile_id as usize)
    }

    /// Total number of tiles.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Validate the snapshot for consistency.
    pub fn validate(&self) -> Result<(), String> {
        // Check dimensions
        let expected_tiles = (self.coarse_width * self.coarse_height) as usize;
        if self.tiles.len() != expected_tiles {
            return Err(format!(
                "Tile count mismatch: expected {}, got {}",
                expected_tiles,
                self.tiles.len()
            ));
        }

        // Check tile IDs are sequential
        for (i, tile) in self.tiles.iter().enumerate() {
            if tile.tile_id != i as u32 {
                return Err(format!(
                    "Tile ID mismatch at index {}: expected {}, got {}",
                    i, i, tile.tile_id
                ));
            }
        }

        // Check species table consistency
        let species_count = self.species_table.species_ids.len();
        if self.species_table.species_names.len() != species_count {
            return Err("Species names count doesn't match species IDs count".to_string());
        }
        if self.species_table.diffusion_coefficients.len() != species_count {
            return Err("Diffusion coefficients count doesn't match species count".to_string());
        }

        Ok(())
    }
}
