//! Semantic snapshot generation for kinetics integration.
//!
//! This module provides functionality to generate semantic snapshots from
//! the fine-grid simulation state. These snapshots are sent to the `kinetics`
//! crate for semantic evaluation once per second of simulated time.
//!
//! ## Snapshot Generation
//!
//! The snapshot is built by aggregating fine-grid data into coarse tiles:
//! - Fine grid: e.g., 512x512 cells
//! - Coarse grid: e.g., 16x16 or 32x32 tiles
//! - Each tile summarizes many fine cells
//!
//! ## Performance
//!
//! Snapshot generation can be done:
//! - GPU-side: Via reduction compute shaders (preferred for large grids)
//! - CPU-side: By reading back cached data (acceptable for smaller grids)
//!
//! Currently uses CPU-side aggregation from cached data.

use crate::solid::MaterialRegistry;
use crate::species::SpeciesRegistry;
use kinetics::{
    BoundaryFlags, BoundarySummary, MaterialFraction, MaterialId as KineticsMaterialId,
    MaterialsTableSnapshot, SemanticSnapshot, SemanticTile, SpeciesAmount,
    SpeciesId as KineticsSpeciesId, SpeciesTableSnapshot, TileFlags,
};

pub struct SemanticSnapshotInput<'a> {
    pub fine_width: u32,
    pub fine_height: u32,
    pub sim_time: f64,
    pub dt_window: f64,
    pub concentrations: &'a [Vec<f32>],
    pub solid_mask: &'a [u32],
    pub material_ids: &'a [u32],
    pub temperatures: &'a [f32],
    pub species_registry: &'a SpeciesRegistry,
    pub material_registry: &'a MaterialRegistry,
}

struct TileAggregationInput<'a> {
    fine_width: u32,
    fine_height: u32,
    tile_size: u32,
    concentrations: &'a [Vec<f32>],
    solid_mask: &'a [u32],
    material_ids: &'a [u32],
    temperatures: &'a [f32],
    species_registry: &'a SpeciesRegistry,
}

/// Configuration for semantic snapshot generation.
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    /// Number of fine cells per coarse tile in each dimension
    pub tile_size: u32,
    /// Temperature threshold for "hot" flag (Kelvin)
    pub hot_threshold_kelvin: f64,
    /// Temperature threshold for "cold" flag (Kelvin)
    pub cold_threshold_kelvin: f64,
    /// Concentration gradient threshold for "has gradient" flag
    pub gradient_threshold: f64,
    /// Whether to compute boundary summaries
    pub compute_boundaries: bool,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            tile_size: 32,                 // 32x32 fine cells per tile
            hot_threshold_kelvin: 373.15,  // 100°C
            cold_threshold_kelvin: 273.15, // 0°C
            gradient_threshold: 0.01,
            compute_boundaries: true,
        }
    }
}

/// Builder for generating semantic snapshots from simulation state.
pub struct SemanticSnapshotBuilder {
    config: SemanticConfig,
}

impl SemanticSnapshotBuilder {
    /// Create a new snapshot builder with the given configuration.
    pub fn new(config: SemanticConfig) -> Self {
        Self { config }
    }

    /// Create a builder with default configuration.
    pub fn default_config() -> Self {
        Self::new(SemanticConfig::default())
    }

    /// Build a semantic snapshot from simulation state.
    ///
    /// # Arguments
    /// * `fine_width` - Width of the fine simulation grid
    /// * `fine_height` - Height of the fine simulation grid
    /// * `sim_time` - Current simulation time in seconds
    /// * `dt_window` - Time since last snapshot in seconds
    /// * `concentrations` - Concentration data [species][cell]
    /// * `solid_mask` - Solid mask (1 = solid, 0 = fluid)
    /// * `material_ids` - Material IDs per cell
    /// * `temperatures` - Temperature per cell in Kelvin
    /// * `species_registry` - Species registry
    /// * `material_registry` - Material registry
    pub fn build(&self, input: SemanticSnapshotInput<'_>) -> SemanticSnapshot {
        let tile_size = self.config.tile_size;
        let coarse_width = input.fine_width.div_ceil(tile_size);
        let coarse_height = input.fine_height.div_ceil(tile_size);

        let mut snapshot = SemanticSnapshot::new(
            coarse_width,
            coarse_height,
            tile_size,
            input.fine_width,
            input.fine_height,
        );

        snapshot.sim_time_seconds = input.sim_time;
        snapshot.dt_window_seconds = input.dt_window;

        // Build species table
        snapshot.species_table = self.build_species_table(input.species_registry);

        // Build materials table
        snapshot.materials_table = self.build_materials_table(input.material_registry);

        let tile_input = TileAggregationInput {
            fine_width: input.fine_width,
            fine_height: input.fine_height,
            tile_size,
            concentrations: input.concentrations,
            solid_mask: input.solid_mask,
            material_ids: input.material_ids,
            temperatures: input.temperatures,
            species_registry: input.species_registry,
        };

        // Aggregate tiles
        for tile_y in 0..coarse_height {
            for tile_x in 0..coarse_width {
                let tile_id = tile_y * coarse_width + tile_x;
                let tile = self.aggregate_tile(tile_id, tile_x, tile_y, &tile_input);
                snapshot.tiles[tile_id as usize] = tile;
            }
        }

        // Compute global statistics
        self.compute_global_stats(
            &mut snapshot,
            input.concentrations,
            input.temperatures,
            input.solid_mask,
        );

        // Compute boundaries if enabled
        if self.config.compute_boundaries {
            snapshot.boundaries = self.compute_boundaries(&snapshot);
        }

        snapshot
    }

    /// Build the species table snapshot.
    fn build_species_table(&self, registry: &SpeciesRegistry) -> SpeciesTableSnapshot {
        let species_count = registry.count();
        let mut table = SpeciesTableSnapshot {
            species_ids: Vec::with_capacity(species_count),
            species_names: Vec::with_capacity(species_count),
            diffusion_coefficients: Vec::with_capacity(species_count),
        };

        for info in registry.iter() {
            table.species_ids.push(KineticsSpeciesId(info.id.0));
            table.species_names.push(info.name.to_string());
            table
                .diffusion_coefficients
                .push(info.diffusion_coefficient);
        }

        table
    }

    /// Build the materials table snapshot.
    fn build_materials_table(&self, registry: &MaterialRegistry) -> MaterialsTableSnapshot {
        let mut table = MaterialsTableSnapshot {
            material_ids: Vec::new(),
            material_names: Vec::new(),
        };

        for i in 0..registry.count() {
            if let Some(info) = registry.get(crate::solid::MaterialId::new(i as u32)) {
                table.material_ids.push(KineticsMaterialId(info.id.0));
                table.material_names.push(info.name.to_string());
            }
        }

        table
    }

    /// Aggregate a single tile from fine-grid data.
    fn aggregate_tile(
        &self,
        tile_id: u32,
        tile_x: u32,
        tile_y: u32,
        input: &TileAggregationInput<'_>,
    ) -> SemanticTile {
        let mut tile = SemanticTile::new(tile_id, tile_x, tile_y);

        // Calculate tile bounds
        let start_x = tile_x * input.tile_size;
        let start_y = tile_y * input.tile_size;
        let end_x = (start_x + input.tile_size).min(input.fine_width);
        let end_y = (start_y + input.tile_size).min(input.fine_height);

        // Accumulators
        let mut fluid_count = 0u32;
        let mut solid_count = 0u32;
        let mut temp_sum = 0.0f64;
        let mut temp_min = f64::MAX;
        let mut temp_max = f64::MIN;
        let mut temp_sq_sum = 0.0f64;

        let species_count = input.concentrations.len();
        let mut species_sums: Vec<f64> = vec![0.0; species_count];
        let mut species_maxes: Vec<f32> = vec![0.0; species_count];
        let mut species_mins: Vec<f32> = vec![f32::MAX; species_count];

        // Material counts
        let mut material_counts: ahash::AHashMap<u32, u32> = ahash::AHashMap::new();

        let total_cells = (end_x - start_x) * (end_y - start_y);

        // Aggregate over fine cells
        for fy in start_y..end_y {
            for fx in start_x..end_x {
                let cell_idx = (fy * input.fine_width + fx) as usize;

                // Solid/fluid
                let is_solid = input.solid_mask[cell_idx] != 0;
                if is_solid {
                    solid_count += 1;

                    // Track material
                    let mat_id = input.material_ids[cell_idx];
                    *material_counts.entry(mat_id).or_insert(0) += 1;
                } else {
                    fluid_count += 1;

                    // Temperature (only for fluid cells)
                    let temp = input.temperatures[cell_idx] as f64;
                    temp_sum += temp;
                    temp_min = temp_min.min(temp);
                    temp_max = temp_max.max(temp);
                    temp_sq_sum += temp * temp;

                    // Concentrations (only for fluid cells)
                    for (s, species_conc) in input.concentrations.iter().enumerate() {
                        let conc = species_conc[cell_idx];
                        species_sums[s] += conc as f64;
                        species_maxes[s] = species_maxes[s].max(conc);
                        species_mins[s] = species_mins[s].min(conc);
                    }
                }
            }
        }

        // Compute fractions
        let total = total_cells as f32;
        tile.fluid_fraction = fluid_count as f32 / total;
        tile.solid_fraction = solid_count as f32 / total;

        // Temperature statistics
        if fluid_count > 0 {
            let fluid_count_f = fluid_count as f64;
            tile.mean_temperature_kelvin = temp_sum / fluid_count_f;
            tile.min_temperature_kelvin = temp_min;
            tile.max_temperature_kelvin = temp_max;

            // Variance = E[X²] - E[X]²
            let mean = tile.mean_temperature_kelvin;
            tile.temperature_variance = (temp_sq_sum / fluid_count_f) - (mean * mean);
        } else {
            // All solid - use default temperature
            tile.mean_temperature_kelvin = 293.15;
            tile.min_temperature_kelvin = 293.15;
            tile.max_temperature_kelvin = 293.15;
            tile.temperature_variance = 0.0;
        }

        // Species concentrations
        for (s, info) in input.species_registry.iter().enumerate() {
            if fluid_count > 0 {
                let mean = (species_sums[s] / fluid_count as f64) as f64;
                let species_id = KineticsSpeciesId(info.id.0);

                // Mean molarity
                tile.species_mean_molarity
                    .push(SpeciesAmount::new(species_id, mean));

                // Total moles (approximation: mean * fluid volume)
                // Assuming 1 cell = 1 unit volume
                let total_moles = species_sums[s];
                tile.species_total_moles
                    .push(SpeciesAmount::new(species_id, total_moles));

                // Gradient magnitude (approximation from min/max spread)
                let gradient = (species_maxes[s] - species_mins[s]) as f64;
                tile.species_max_gradient
                    .push(SpeciesAmount::new(species_id, gradient));
            }
        }

        // Material fractions
        for (mat_id, count) in material_counts {
            let fraction = count as f32 / total;
            tile.material_fractions
                .push(MaterialFraction::new(KineticsMaterialId(mat_id), fraction));
        }

        // Set flags
        let mut flags = TileFlags::empty();
        if fluid_count > 0 {
            flags |= TileFlags::HAS_FLUID;
        }
        if solid_count > 0 {
            flags |= TileFlags::HAS_SOLID;
        }
        if fluid_count > 0 && solid_count > 0 {
            flags |= TileFlags::TOUCHES_INTERFACE;
        }
        if tile_x == 0
            || tile_y == 0
            || tile_x >= (input.fine_width / input.tile_size) - 1
            || tile_y >= (input.fine_height / input.tile_size) - 1
        {
            flags |= TileFlags::IS_BOUNDARY;
        }
        if tile.mean_temperature_kelvin > self.config.hot_threshold_kelvin {
            flags |= TileFlags::IS_HOT;
        }
        if tile.mean_temperature_kelvin < self.config.cold_threshold_kelvin {
            flags |= TileFlags::IS_COLD;
        }

        // Check for gradients
        for grad in &tile.species_max_gradient {
            if grad.value > self.config.gradient_threshold {
                flags |= TileFlags::HAS_GRADIENT;
                break;
            }
        }

        tile.flags = flags;
        tile
    }

    /// Compute global statistics for the snapshot.
    fn compute_global_stats(
        &self,
        snapshot: &mut SemanticSnapshot,
        concentrations: &[Vec<f32>],
        temperatures: &[f32],
        solid_mask: &[u32],
    ) {
        let species_count = concentrations.len();
        let mut global_totals: Vec<f64> = vec![0.0; species_count];
        let mut temp_sum = 0.0f64;
        let mut fluid_count = 0u64;

        for (i, &is_solid) in solid_mask.iter().enumerate() {
            if is_solid == 0 {
                fluid_count += 1;
                temp_sum += temperatures[i] as f64;

                for (s, species_conc) in concentrations.iter().enumerate() {
                    global_totals[s] += species_conc[i] as f64;
                }
            }
        }

        // Global species totals
        for (s, total) in global_totals.into_iter().enumerate() {
            if s < snapshot.species_table.species_ids.len() {
                let species_id = snapshot.species_table.species_ids[s];
                snapshot
                    .global_species_totals
                    .push(SpeciesAmount::new(species_id, total));
            }
        }

        // Global mean temperature
        if fluid_count > 0 {
            snapshot.global_mean_temperature = temp_sum / fluid_count as f64;
        }

        // Rough thermal energy estimate (simplified)
        // E = m * c * T, assuming unit mass and c ~ 4186 J/(kg·K) for water
        let heat_capacity = 4186.0; // J/(kg·K)
        snapshot.global_thermal_energy =
            fluid_count as f64 * heat_capacity * snapshot.global_mean_temperature;
    }

    /// Compute boundary summaries between adjacent tiles.
    fn compute_boundaries(&self, snapshot: &SemanticSnapshot) -> Vec<BoundarySummary> {
        let mut boundaries = Vec::new();
        let mut boundary_id = 0u32;

        for tile in &snapshot.tiles {
            let x = tile.x;
            let y = tile.y;

            // Check right neighbor
            if x + 1 < snapshot.coarse_width {
                let neighbor_id = y * snapshot.coarse_width + (x + 1);
                if let Some(neighbor) = snapshot.tiles.get(neighbor_id as usize)
                    && let Some(boundary) = self.create_boundary(boundary_id, tile, neighbor)
                {
                    boundaries.push(boundary);
                    boundary_id += 1;
                }
            }

            // Check bottom neighbor
            if y + 1 < snapshot.coarse_height {
                let neighbor_id = (y + 1) * snapshot.coarse_width + x;
                if let Some(neighbor) = snapshot.tiles.get(neighbor_id as usize)
                    && let Some(boundary) = self.create_boundary(boundary_id, tile, neighbor)
                {
                    boundaries.push(boundary);
                    boundary_id += 1;
                }
            }
        }

        boundaries
    }

    /// Create a boundary summary between two adjacent tiles.
    fn create_boundary(
        &self,
        boundary_id: u32,
        tile_a: &SemanticTile,
        tile_b: &SemanticTile,
    ) -> Option<BoundarySummary> {
        // Only create boundaries at interesting interfaces
        let is_interesting = (tile_a.has_fluid() != tile_b.has_fluid())
            || (tile_a.has_solid() != tile_b.has_solid())
            || (tile_a.mean_temperature_kelvin - tile_b.mean_temperature_kelvin).abs() > 10.0;

        if !is_interesting {
            return None;
        }

        let mut boundary = BoundarySummary::new(boundary_id, tile_a.tile_id, tile_b.tile_id);

        // Temperature delta
        boundary.temperature_delta =
            tile_a.mean_temperature_kelvin - tile_b.mean_temperature_kelvin;

        // Contact area estimate (tile edge length)
        boundary.contact_area_estimate = self.config.tile_size as f32;

        // Set flags
        let mut flags = BoundaryFlags::empty();
        if tile_a.has_fluid() != tile_b.has_fluid() {
            flags |= BoundaryFlags::FLUID_SOLID;
        }
        if boundary.temperature_delta.abs() > 10.0 {
            flags |= BoundaryFlags::THERMAL_DISCONTINUITY;
        }

        boundary.flags = flags;

        Some(boundary)
    }
}
