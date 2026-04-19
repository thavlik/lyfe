//! Scenario initialization and builders.
//!
//! Provides a high-level API for setting up simulation initial conditions,
//! including species registration, solid geometry, and initial concentrations.

use crate::enzyme::{EnzymeEntity, EnzymeField};
use crate::grid::{CellCoord, Grid};
use crate::leak::LeakChannel;
use crate::solid::{MaterialId, MaterialRegistry, SolidGeometry, SolidRect};
use crate::species::{SpeciesConcentrations, SpeciesRegistry};
use ahash::AHashMap;

/// Default diffusion coefficient for aqueous ions (scaled for simulation).
const DEFAULT_DIFFUSION: f32 = 1.0;

/// A complete scenario specification before compilation to GPU buffers.
#[derive(Debug, Clone)]
pub struct Scenario {
    pub grid: Grid,
    pub species_registry: SpeciesRegistry,
    pub material_registry: MaterialRegistry,
    pub solid_geometry: SolidGeometry,
    pub leak_channels: Vec<LeakChannel>,
    pub enzyme_entities: Vec<EnzymeEntity>,
    pub enzyme_field: Option<EnzymeField>,
    /// Initial concentrations per cell. Sparse - only non-zero cells stored.
    pub initial_concentrations: AHashMap<usize, SpeciesConcentrations>,
    /// Initial temperature per cell in Kelvin. Dense array, one f32 per cell.
    pub initial_temperatures: Vec<f32>,
}

impl Scenario {
    /// Compile initial concentrations to dense GPU-ready arrays.
    /// Returns `[species_index][cell_index]` layout.
    pub fn compile_concentrations(&self) -> Vec<Vec<f32>> {
        let cell_count = self.grid.cell_count();
        let species_count = self.species_registry.count();

        // Initialize all to zero
        let mut concentrations = vec![vec![0.0f32; cell_count]; species_count];

        // Fill in non-zero values
        for (&cell_index, cell_conc) in &self.initial_concentrations {
            for (species_id, conc) in cell_conc.iter() {
                if let Some(species_idx) = self.species_registry.index_of(species_id) {
                    concentrations[species_idx][cell_index] = conc as f32;
                }
            }
        }

        concentrations
    }

    /// Compile solid mask to dense array.
    pub fn compile_solid_mask(&self) -> Vec<u32> {
        self.solid_geometry
            .material_ids
            .iter()
            .map(|m| if m.0 != 0 { 1u32 } else { 0u32 })
            .collect()
    }

    /// Compile material IDs to dense array.
    pub fn compile_material_ids(&self) -> Vec<u32> {
        self.solid_geometry
            .material_ids
            .iter()
            .map(|m| m.0)
            .collect()
    }

    /// Compile temperatures to dense array (already dense, just clone).
    pub fn compile_temperatures(&self) -> Vec<f32> {
        self.initial_temperatures.clone()
    }
}

/// Builder for creating scenarios with a fluent API.
pub struct ScenarioBuilder {
    grid: Grid,
    species_registry: SpeciesRegistry,
    material_registry: MaterialRegistry,
    solid_geometry: SolidGeometry,
    leak_channels: Vec<LeakChannel>,
    enzyme_entities: Vec<EnzymeEntity>,
    enzyme_field: Option<EnzymeField>,
    initial_concentrations: AHashMap<usize, SpeciesConcentrations>,
    initial_temperatures: Vec<f32>,
}

impl ScenarioBuilder {
    /// Create a new scenario builder with the given grid dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let grid = Grid::new(width, height);
        Self {
            grid,
            species_registry: SpeciesRegistry::new(),
            material_registry: MaterialRegistry::new(),
            solid_geometry: SolidGeometry::new(grid.cell_count()),
            leak_channels: Vec::new(),
            enzyme_entities: Vec::new(),
            enzyme_field: None,
            initial_concentrations: AHashMap::new(),
            initial_temperatures: Vec::new(),
        }
    }

    /// Register a species with default diffusion coefficient.
    pub fn register_species(mut self, name: &str) -> Self {
        self.species_registry.register(name, DEFAULT_DIFFUSION);
        self
    }

    /// Register a species with custom diffusion coefficient.
    pub fn register_species_with_diffusion(mut self, name: &str, diffusion: f32) -> Self {
        self.species_registry.register(name, diffusion);
        self
    }

    /// Register a material.
    pub fn register_material(mut self, name: &str, color: [f32; 4]) -> (Self, MaterialId) {
        let id = self.material_registry.register(name, color);
        (self, id)
    }

    /// Fill a rectangular region with solid material.
    pub fn fill_solid_rect(
        mut self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        material: MaterialId,
    ) -> Self {
        self.solid_geometry
            .fill_rect(x0, y0, x1, y1, self.grid.width, material);
        self
    }

    /// Fill a hollow rectangle with solid material.
    pub fn fill_hollow_rect(
        mut self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        thickness: u32,
        material: MaterialId,
    ) -> Self {
        self.solid_geometry.fill_hollow_rect(
            SolidRect { x0, y0, x1, y1 },
            thickness,
            self.grid.width,
            material,
        );
        self
    }

    /// Set concentration for a specific cell.
    pub fn set_concentration(
        mut self,
        coord: CellCoord,
        species_name: &str,
        concentration: f64,
    ) -> Self {
        if let Some(species_id) = self.species_registry.id_of(species_name) {
            let index = self.grid.index_of(coord);
            self.initial_concentrations
                .entry(index)
                .or_default()
                .set(species_id, concentration);
        }
        self
    }

    /// Fill a rectangular region with a species concentration.
    pub fn fill_concentration_rect(
        mut self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        species_name: &str,
        concentration: f64,
    ) -> Self {
        if let Some(species_id) = self.species_registry.id_of(species_name) {
            for coord in self.grid.iter_rect(x0, y0, x1, y1) {
                // Skip solid cells
                let index = self.grid.index_of(coord);
                if !self.solid_geometry.is_solid(index) {
                    self.initial_concentrations
                        .entry(index)
                        .or_default()
                        .set(species_id, concentration);
                }
            }
        }
        self
    }

    /// Set temperature for all cells to a uniform value.
    pub fn fill_temperature(mut self, temperature: f32) -> Self {
        let cell_count = self.grid.cell_count();
        self.initial_temperatures = vec![temperature; cell_count];
        self
    }

    /// Set temperature for a rectangular region.
    pub fn fill_temperature_rect(
        mut self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        temperature: f32,
    ) -> Self {
        for coord in self.grid.iter_rect(x0, y0, x1, y1) {
            let index = self.grid.index_of(coord);
            self.initial_temperatures[index] = temperature;
        }
        self
    }

    /// Build the final scenario.
    pub fn build(self) -> Scenario {
        let cell_count = self.grid.cell_count();
        let temperatures = if self.initial_temperatures.is_empty() {
            vec![293.15; cell_count] // Default 20°C
        } else {
            self.initial_temperatures
        };
        Scenario {
            grid: self.grid,
            species_registry: self.species_registry,
            material_registry: self.material_registry,
            solid_geometry: self.solid_geometry,
            leak_channels: self.leak_channels,
            enzyme_entities: self.enzyme_entities,
            enzyme_field: self.enzyme_field,
            initial_concentrations: self.initial_concentrations,
            initial_temperatures: temperatures,
        }
    }
}

fn central_box_bounds(width: u32, height: u32) -> (u32, u32, u32, u32, u32, u32, u32, u32, u32) {
    let wall_thickness = 4u32;
    let inner_size = (width.min(height) / 2).max(64);
    let outer_size = inner_size + 2 * wall_thickness;

    let center_x = width / 2;
    let center_y = height / 2;

    let outer_x0 = center_x - outer_size / 2;
    let outer_y0 = center_y - outer_size / 2;
    let outer_x1 = outer_x0 + outer_size;
    let outer_y1 = outer_y0 + outer_size;

    let inner_x0 = outer_x0 + wall_thickness;
    let inner_y0 = outer_y0 + wall_thickness;
    let inner_x1 = outer_x1 - wall_thickness;
    let inner_y1 = outer_y1 - wall_thickness;

    (
        wall_thickness,
        outer_x0,
        outer_y0,
        outer_x1,
        outer_y1,
        inner_x0,
        inner_y0,
        inner_x1,
        inner_y1,
    )
}

fn lcg_next(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

fn lcg_unit(seed: &mut u32) -> f32 {
    (lcg_next(seed) >> 8) as f32 / ((u32::MAX >> 8) as f32)
}

/// Create the demo scenario as specified in requirements.
pub fn create_demo_scenario(width: u32, height: u32) -> Scenario {
    // Calculate hollow square dimensions
    // Square should be centered, wall thickness = 4
    let wall_thickness = 4u32;
    let inner_size = (width.min(height) / 2).max(64); // At least 64 units inside
    let outer_size = inner_size + 2 * wall_thickness;

    let center_x = width / 2;
    let center_y = height / 2;

    let outer_x0 = center_x - outer_size / 2;
    let outer_y0 = center_y - outer_size / 2;
    let outer_x1 = outer_x0 + outer_size;
    let outer_y1 = outer_y0 + outer_size;

    let inner_x0 = outer_x0 + wall_thickness;
    let inner_y0 = outer_y0 + wall_thickness;
    let inner_x1 = outer_x1 - wall_thickness;
    let inner_y1 = outer_y1 - wall_thickness;

    // Build the scenario
    let builder = ScenarioBuilder::new(width, height)
        // Register species (only the ones we use)
        .register_species("Na+")
        .register_species("K+")
        .register_species("Cl-");

    // Register titanium material
    let (builder, titanium) = builder.register_material("titanium", [0.6, 0.6, 0.65, 1.0]);

    // Create the hollow titanium square
    let builder = builder.fill_hollow_rect(
        outer_x0,
        outer_y0,
        outer_x1,
        outer_y1,
        wall_thickness,
        titanium,
    );

    // Initialize temperatures:
    // - Outside box water: 280.0K
    // - Titanium walls: 280.0K
    // - Top-left inside box: 318.15K
    // - Bottom-right inside box: 288.0K
    let mut builder = builder.fill_temperature(280.0); // Default: outside water + titanium at 280K

    // Inside the hollow square, split diagonally:
    // - Top left (above diagonal from top-right to bottom-left): 1.0 M Na+, 1.0 M Cl-
    // - Bottom right (below diagonal): 1.0 M K+, 1.0 M Cl-
    // Outside the box: pure water (no concentration)

    let na_id = builder.species_registry.id_of("Na+").unwrap();
    let k_id = builder.species_registry.id_of("K+").unwrap();
    let cl_id = builder.species_registry.id_of("Cl-").unwrap();

    let inner_width = (inner_x1 - inner_x0) as f32;
    let inner_height = (inner_y1 - inner_y0) as f32;

    for y in inner_y0..inner_y1 {
        for x in inner_x0..inner_x1 {
            let index = builder.grid.index_of(CellCoord::new(x, y));
            if builder.solid_geometry.is_solid(index) {
                continue;
            }

            // Normalized position within inner region (0..1)
            let nx = (x - inner_x0) as f32 / inner_width;
            let ny = (y - inner_y0) as f32 / inner_height;

            // Diagonal line from top-right (1,0) to bottom-left (0,1)
            // Points where nx + ny < 1 are in the top-left triangle
            if nx + ny < 1.0 {
                // Top-left: Na+ and Cl-
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(na_id, 1.0);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(cl_id, 1.0);
                // Top-left temperature: 318.15K
                builder.initial_temperatures[index] = 318.15;
            } else {
                // Bottom-right: K+ and Cl-
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(k_id, 1.0);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(cl_id, 1.0);
                // Bottom-right temperature: 288.0K
                builder.initial_temperatures[index] = 288.0;
            }
        }
    }

    builder.build()
}

/// Create the acid-base neutralization scenario.
///
/// Top-left region: 1.0 M H⁺ + 0.5 M SO₄²⁻ (acidic solution)
/// Bottom-right region: 1.0 M Na⁺ + 1.0 M OH⁻ (basic solution)
///
/// When they mix, kinetics drives: H⁺ + OH⁻ → H₂O
pub fn create_acid_base_scenario(width: u32, height: u32) -> Scenario {
    let wall_thickness = 4u32;
    let inner_size = (width.min(height) / 2).max(64);
    let outer_size = inner_size + 2 * wall_thickness;

    let center_x = width / 2;
    let center_y = height / 2;

    let outer_x0 = center_x - outer_size / 2;
    let outer_y0 = center_y - outer_size / 2;
    let outer_x1 = outer_x0 + outer_size;
    let outer_y1 = outer_y0 + outer_size;

    let inner_x0 = outer_x0 + wall_thickness;
    let inner_y0 = outer_y0 + wall_thickness;
    let inner_x1 = outer_x1 - wall_thickness;
    let inner_y1 = outer_y1 - wall_thickness;

    let builder = ScenarioBuilder::new(width, height)
        .register_species("H+")
        .register_species("OH-")
        .register_species("Na+")
        .register_species("SO4(2-)");

    let (builder, titanium) = builder.register_material("titanium", [0.6, 0.6, 0.65, 1.0]);

    let builder = builder.fill_hollow_rect(
        outer_x0,
        outer_y0,
        outer_x1,
        outer_y1,
        wall_thickness,
        titanium,
    );

    let mut builder = builder.fill_temperature(278.15); // 5°C uniform

    let h_id = builder.species_registry.id_of("H+").unwrap();
    let oh_id = builder.species_registry.id_of("OH-").unwrap();
    let na_id = builder.species_registry.id_of("Na+").unwrap();
    let so4_id = builder.species_registry.id_of("SO4(2-)").unwrap();

    let inner_width = (inner_x1 - inner_x0) as f32;
    let inner_height = (inner_y1 - inner_y0) as f32;

    for y in inner_y0..inner_y1 {
        for x in inner_x0..inner_x1 {
            let index = builder.grid.index_of(CellCoord::new(x, y));
            if builder.solid_geometry.is_solid(index) {
                continue;
            }

            let nx = (x - inner_x0) as f32 / inner_width;
            let ny = (y - inner_y0) as f32 / inner_height;

            if nx + ny < 1.0 {
                // Top-left: acidic — 1.0 M H⁺, 0.5 M SO₄²⁻
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(h_id, 1.0);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(so4_id, 0.5);
            } else {
                // Bottom-right: basic — 1.0 M Na⁺, 1.0 M OH⁻
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(na_id, 1.0);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(oh_id, 1.0);
            }
        }
    }

    builder.build()
}

/// Create a catalyst-gated phosphorylation scenario.
///
/// Uses the same hollow-box geometry as the basic demo, but fills the entire
/// interior with glucose, ATP, and a low concentration of hexokinase.
pub fn create_catalyst_scenario(width: u32, height: u32) -> Scenario {
    let (
        wall_thickness,
        outer_x0,
        outer_y0,
        outer_x1,
        outer_y1,
        inner_x0,
        inner_y0,
        inner_x1,
        inner_y1,
    ) = central_box_bounds(width, height);

    let builder = ScenarioBuilder::new(width, height)
        .register_species("Glucose")
        .register_species("ATP")
        .register_species("ADP")
        .register_species("G6P")
        .register_species("Hexokinase");

    let (builder, titanium) = builder.register_material("titanium", [0.6, 0.6, 0.65, 1.0]);

    let builder = builder.fill_hollow_rect(
        outer_x0,
        outer_y0,
        outer_x1,
        outer_y1,
        wall_thickness,
        titanium,
    );

    let mut builder = builder.fill_temperature(280.0);

    let glucose_id = builder.species_registry.id_of("Glucose").unwrap();
    let atp_id = builder.species_registry.id_of("ATP").unwrap();
    let hexokinase_id = builder.species_registry.id_of("Hexokinase").unwrap();

    let inner_width = (inner_x1 - inner_x0) as f32;
    let inner_height = (inner_y1 - inner_y0) as f32;

    for y in inner_y0..inner_y1 {
        for x in inner_x0..inner_x1 {
            let index = builder.grid.index_of(CellCoord::new(x, y));
            if builder.solid_geometry.is_solid(index) {
                continue;
            }

            builder
                .initial_concentrations
                .entry(index)
                .or_default()
                .set(glucose_id, 1.0);
            builder
                .initial_concentrations
                .entry(index)
                .or_default()
                .set(atp_id, 1.0);
            builder
                .initial_concentrations
                .entry(index)
                .or_default()
                .set(hexokinase_id, 0.01);

            let nx = (x - inner_x0) as f32 / inner_width;
            let ny = (y - inner_y0) as f32 / inner_height;
            builder.initial_temperatures[index] = if nx + ny < 1.0 { 318.15 } else { 288.0 };
        }
    }

    builder.build()
}

/// Create an enzyme-entity phosphorylation scenario.
///
/// Starts from the catalyst demo chemistry and geometry, but models
/// hexokinase as six drifting entities rather than a dissolved catalyst.
pub fn create_enzyme_scenario(width: u32, height: u32) -> Scenario {
    let (
        wall_thickness,
        outer_x0,
        outer_y0,
        outer_x1,
        outer_y1,
        inner_x0,
        inner_y0,
        inner_x1,
        inner_y1,
    ) = central_box_bounds(width, height);

    let builder = ScenarioBuilder::new(width, height)
        .register_species("Glucose")
        .register_species("ATP")
        .register_species("ADP")
        .register_species("G6P");

    let (builder, titanium) = builder.register_material("titanium", [0.6, 0.6, 0.65, 1.0]);
    let builder = builder.fill_hollow_rect(
        outer_x0,
        outer_y0,
        outer_x1,
        outer_y1,
        wall_thickness,
        titanium,
    );
    let mut builder = builder.fill_temperature(293.15);

    let glucose_id = builder.species_registry.id_of("Glucose").unwrap();
    let atp_id = builder.species_registry.id_of("ATP").unwrap();

    let inner_width = (inner_x1 - inner_x0) as f32;
    let inner_height = (inner_y1 - inner_y0) as f32;
    let cool_temperature = 293.15;
    let hot_temperature = cool_temperature + 15.0;

    for y in inner_y0..inner_y1 {
        for x in inner_x0..inner_x1 {
            let index = builder.grid.index_of(CellCoord::new(x, y));
            if builder.solid_geometry.is_solid(index) {
                continue;
            }

            builder
                .initial_concentrations
                .entry(index)
                .or_default()
                .set(glucose_id, 1.0);
            builder
                .initial_concentrations
                .entry(index)
                .or_default()
                .set(atp_id, 1.0);

            let nx = (x - inner_x0) as f32 / inner_width;
            let ny = (y - inner_y0) as f32 / inner_height;
            builder.initial_temperatures[index] = if nx + ny < 1.0 {
                hot_temperature
            } else {
                cool_temperature
            };
        }
    }

    let entity_margin = 10.0;
    let field = EnzymeField {
        min_x: inner_x0 as f32 + entity_margin,
        min_y: inner_y0 as f32 + entity_margin,
        max_x: inner_x1 as f32 - entity_margin,
        max_y: inner_y1 as f32 - entity_margin,
        hot_temperature,
        cool_temperature,
        circulation_strength: 1.35,
        thermophoretic_strength: 0.55,
        brownian_strength: 0.95,
        rotational_diffusion: 0.75,
    };

    let mut seed =
        width.wrapping_mul(73_856_093) ^ height.wrapping_mul(19_349_663) ^ 0xA5A5_1F1Fu32;
    let mut entities = Vec::with_capacity(6);
    let minimum_spacing = 14.0f32;

    for entity_index in 0..6 {
        let mut candidate = glam::Vec2::new(field.min_x, field.min_y);
        for _ in 0..96 {
            candidate = glam::Vec2::new(
                field.min_x + lcg_unit(&mut seed) * (field.max_x - field.min_x),
                field.min_y + lcg_unit(&mut seed) * (field.max_y - field.min_y),
            );
            if entities.iter().all(|existing: &EnzymeEntity| {
                existing.position().distance(candidate) >= minimum_spacing
            }) {
                break;
            }
        }

        let rotation_radians = lcg_unit(&mut seed) * std::f32::consts::TAU;
        let catalytic_scale = 0.85 + 0.35 * lcg_unit(&mut seed);
        let mobility_scale = 0.8 + 0.45 * lcg_unit(&mut seed);
        let thermal_bias = -1.5 + 3.0 * lcg_unit(&mut seed);
        entities.push(EnzymeEntity::new(
            candidate.x,
            candidate.y,
            rotation_radians,
            catalytic_scale,
            mobility_scale,
            thermal_bias,
            seed ^ (entity_index as u32).wrapping_mul(0x9E37_79B9),
        ));
    }

    builder.enzyme_entities = entities;
    builder.enzyme_field = Some(field);
    builder.build()
}

/// Create a weak-acid buffer scenario.
///
/// Top-left region: 0.35 M sodium acetate + 0.35 M acetic acid.
/// - Na+ = 0.35 M
/// - CH3COO- = 0.35 M
/// - CH3COOH = 0.35 M
/// - H+ = Ka ≈ 1.8e-5 M
///
/// Bottom-right region: 0.2 M NaOH represented as:
/// - Na+ = 0.2 M
/// - OH- = 0.2 M
pub fn create_buffers_scenario(width: u32, height: u32) -> Scenario {
    let wall_thickness = 4u32;
    let inner_size = (width.min(height) / 2).max(64);
    let outer_size = inner_size + 2 * wall_thickness;

    let center_x = width / 2;
    let center_y = height / 2;

    let outer_x0 = center_x - outer_size / 2;
    let outer_y0 = center_y - outer_size / 2;
    let outer_x1 = outer_x0 + outer_size;
    let outer_y1 = outer_y0 + outer_size;

    let inner_x0 = outer_x0 + wall_thickness;
    let inner_y0 = outer_y0 + wall_thickness;
    let inner_x1 = outer_x1 - wall_thickness;
    let inner_y1 = outer_y1 - wall_thickness;

    let builder = ScenarioBuilder::new(width, height)
        .register_species("H+")
        .register_species("OH-")
        .register_species("Na+")
        .register_species("CH3COOH")
        .register_species("CH3COO-");

    let (builder, titanium) = builder.register_material("titanium", [0.6, 0.6, 0.65, 1.0]);

    let builder = builder.fill_hollow_rect(
        outer_x0,
        outer_y0,
        outer_x1,
        outer_y1,
        wall_thickness,
        titanium,
    );

    let mut builder = builder.fill_temperature(278.15);

    let h_id = builder.species_registry.id_of("H+").unwrap();
    let oh_id = builder.species_registry.id_of("OH-").unwrap();
    let na_id = builder.species_registry.id_of("Na+").unwrap();
    let acetic_acid_id = builder.species_registry.id_of("CH3COOH").unwrap();
    let acetate_id = builder.species_registry.id_of("CH3COO-").unwrap();

    let inner_width = (inner_x1 - inner_x0) as f32;
    let inner_height = (inner_y1 - inner_y0) as f32;

    for y in inner_y0..inner_y1 {
        for x in inner_x0..inner_x1 {
            let index = builder.grid.index_of(CellCoord::new(x, y));
            if builder.solid_geometry.is_solid(index) {
                continue;
            }

            let nx = (x - inner_x0) as f32 / inner_width;
            let ny = (y - inner_y0) as f32 / inner_height;

            if nx + ny < 1.0 {
                // Top-left: sodium acetate + acetic acid buffer near pKa.
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(na_id, 0.35);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(h_id, 1.8e-5);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(acetate_id, 0.35);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(acetic_acid_id, 0.35);
            } else {
                // Bottom-right: 0.2 M NaOH.
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(na_id, 0.2);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(oh_id, 0.2);
                builder.initial_temperatures[index] = 288.15;
            }
        }
    }

    builder.build()
}

/// Create a leak-channel membrane scenario.
pub fn create_leak_scenario(width: u32, height: u32) -> Scenario {
    let wall_thickness = 4u32;
    let inner_size = (width.min(height) / 2).max(64);
    let outer_size = inner_size + 2 * wall_thickness;

    let center_x = width / 2;
    let center_y = height / 2;

    let outer_x0 = center_x - outer_size / 2;
    let outer_y0 = center_y - outer_size / 2;
    let outer_x1 = outer_x0 + outer_size;
    let outer_y1 = outer_y0 + outer_size;

    let inner_x0 = outer_x0 + wall_thickness;
    let inner_y0 = outer_y0 + wall_thickness;
    let inner_x1 = outer_x1 - wall_thickness;
    let inner_y1 = outer_y1 - wall_thickness;

    let builder = ScenarioBuilder::new(width, height)
        .register_species("H+")
        .register_species("OH-")
        .register_species("Na+")
        .register_species("K+")
        .register_species("Cl-")
        .register_species("CH3COOH")
        .register_species("CH3COO-");

    let (builder, titanium) = builder.register_material("titanium", [0.6, 0.6, 0.65, 1.0]);

    let builder = builder.fill_hollow_rect(
        outer_x0,
        outer_y0,
        outer_x1,
        outer_y1,
        wall_thickness,
        titanium,
    );

    let mut builder = builder.fill_temperature(278.15);

    let h_id = builder.species_registry.id_of("H+").unwrap();
    let oh_id = builder.species_registry.id_of("OH-").unwrap();
    let na_id = builder.species_registry.id_of("Na+").unwrap();
    let k_id = builder.species_registry.id_of("K+").unwrap();
    let cl_id = builder.species_registry.id_of("Cl-").unwrap();
    let acetic_acid_id = builder.species_registry.id_of("CH3COOH").unwrap();
    let acetate_id = builder.species_registry.id_of("CH3COO-").unwrap();

    for y in 0..height {
        for x in 0..width {
            let index = builder.grid.index_of(CellCoord::new(x, y));
            if builder.solid_geometry.is_solid(index) {
                continue;
            }

            let inside_box = x >= inner_x0 && x < inner_x1 && y >= inner_y0 && y < inner_y1;
            if !inside_box {
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(k_id, 1.0);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(cl_id, 1.0);
            }
        }
    }

    let inner_width = (inner_x1 - inner_x0) as f32;
    let inner_height = (inner_y1 - inner_y0) as f32;

    for y in inner_y0..inner_y1 {
        for x in inner_x0..inner_x1 {
            let index = builder.grid.index_of(CellCoord::new(x, y));
            if builder.solid_geometry.is_solid(index) {
                continue;
            }

            let nx = (x - inner_x0) as f32 / inner_width;
            let ny = (y - inner_y0) as f32 / inner_height;

            if nx + ny < 1.0 {
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(na_id, 0.35);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(h_id, 1.8e-5);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(acetate_id, 0.350018);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(acetic_acid_id, 0.349982);
            } else {
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(na_id, 0.2);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(oh_id, 0.2);
                builder.initial_temperatures[index] = 288.15;
            }
        }
    }

    let channel_count = 6u32;
    for i in 0..channel_count {
        let y = inner_y0 + ((i + 1) * (inner_y1 - inner_y0) / (channel_count + 1));
        builder.leak_channels.push(LeakChannel::new(
            4.5,
            k_id,
            (inner_x0 - 1) as i32,
            y as i32,
            0,
        ));
        builder
            .leak_channels
            .push(LeakChannel::new(4.5, na_id, inner_x1 as i32, y as i32, 0));
    }

    builder.build()
}
