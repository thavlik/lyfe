//! Scenario initialization and builders.
//!
//! Provides a high-level API for setting up simulation initial conditions,
//! including species registration, solid geometry, and initial concentrations.

use crate::grid::{Grid, CellCoord};
use crate::solid::{MaterialId, MaterialRegistry, SolidGeometry};
use crate::species::{SpeciesRegistry, SpeciesConcentrations};
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
    /// Initial concentrations per cell. Sparse - only non-zero cells stored.
    pub initial_concentrations: AHashMap<usize, SpeciesConcentrations>,
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
        self.solid_geometry.material_ids
            .iter()
            .map(|m| if m.0 != 0 { 1u32 } else { 0u32 })
            .collect()
    }

    /// Compile material IDs to dense array.
    pub fn compile_material_ids(&self) -> Vec<u32> {
        self.solid_geometry.material_ids.iter().map(|m| m.0).collect()
    }
}

/// Builder for creating scenarios with a fluent API.
pub struct ScenarioBuilder {
    grid: Grid,
    species_registry: SpeciesRegistry,
    material_registry: MaterialRegistry,
    solid_geometry: SolidGeometry,
    initial_concentrations: AHashMap<usize, SpeciesConcentrations>,
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
            initial_concentrations: AHashMap::new(),
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
        x0: u32, y0: u32,
        x1: u32, y1: u32,
        material: MaterialId,
    ) -> Self {
        self.solid_geometry.fill_rect(x0, y0, x1, y1, self.grid.width, material);
        self
    }

    /// Fill a hollow rectangle with solid material.
    pub fn fill_hollow_rect(
        mut self,
        x0: u32, y0: u32,
        x1: u32, y1: u32,
        thickness: u32,
        material: MaterialId,
    ) -> Self {
        self.solid_geometry.fill_hollow_rect(x0, y0, x1, y1, thickness, self.grid.width, material);
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
        x0: u32, y0: u32,
        x1: u32, y1: u32,
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

    /// Build the final scenario.
    pub fn build(self) -> Scenario {
        Scenario {
            grid: self.grid,
            species_registry: self.species_registry,
            material_registry: self.material_registry,
            solid_geometry: self.solid_geometry,
            initial_concentrations: self.initial_concentrations,
        }
    }
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
    
    let inner_mid_x = (inner_x0 + inner_x1) / 2;
    
    // Build the scenario
    let builder = ScenarioBuilder::new(width, height)
        // Register all species (dissociated ions)
        .register_species("Na+")
        .register_species("K+")
        .register_species("Cl-")
        .register_species("H+")
        .register_species("OH-")
        .register_species("Ti");  // For potential titanium dissolution (not used in this demo)
    
    // Register titanium material
    let (builder, titanium) = builder.register_material("titanium", [0.6, 0.6, 0.65, 1.0]);
    
    // Create the hollow titanium square
    let builder = builder.fill_hollow_rect(outer_x0, outer_y0, outer_x1, outer_y1, wall_thickness, titanium);
    
    // Inside the hollow square:
    // Left half: 0.1 M NaCl -> Na+ = 0.1, Cl- = 0.1
    // Right half: 0.2 M KCl -> K+ = 0.2, Cl- = 0.2
    let builder = builder
        .fill_concentration_rect(inner_x0, inner_y0, inner_mid_x, inner_y1, "Na+", 0.1)
        .fill_concentration_rect(inner_x0, inner_y0, inner_mid_x, inner_y1, "Cl-", 0.1)
        .fill_concentration_rect(inner_mid_x, inner_y0, inner_x1, inner_y1, "K+", 0.2)
        .fill_concentration_rect(inner_mid_x, inner_y0, inner_x1, inner_y1, "Cl-", 0.2);
    
    // Outside the hollow square:
    // Need to fill areas outside the square
    // Left half of screen (x < center): 1.0 M NaOH -> Na+ = 1.0, OH- = 1.0
    // Right half of screen (x >= center): 1.0 M HCl -> H+ = 1.0, Cl- = 1.0
    
    // Fill outer regions (excluding the square area)
    let mut builder = builder;
    
    // Left outer region
    builder = fill_outside_rect(builder, 0, 0, center_x, height, outer_x0, outer_y0, outer_x1, outer_y1, "Na+", 1.0);
    builder = fill_outside_rect(builder, 0, 0, center_x, height, outer_x0, outer_y0, outer_x1, outer_y1, "OH-", 1.0);
    
    // Right outer region
    builder = fill_outside_rect(builder, center_x, 0, width, height, outer_x0, outer_y0, outer_x1, outer_y1, "H+", 1.0);
    builder = fill_outside_rect(builder, center_x, 0, width, height, outer_x0, outer_y0, outer_x1, outer_y1, "Cl-", 1.0);
    
    builder.build()
}

/// Helper to fill concentration outside a rectangular exclusion zone.
fn fill_outside_rect(
    mut builder: ScenarioBuilder,
    fill_x0: u32, fill_y0: u32,
    fill_x1: u32, fill_y1: u32,
    exclude_x0: u32, exclude_y0: u32,
    exclude_x1: u32, exclude_y1: u32,
    species: &str,
    concentration: f64,
) -> ScenarioBuilder {
    let species_id = match builder.species_registry.id_of(species) {
        Some(id) => id,
        None => return builder,
    };
    
    for y in fill_y0..fill_y1 {
        for x in fill_x0..fill_x1 {
            // Skip if inside exclusion zone
            if x >= exclude_x0 && x < exclude_x1 && y >= exclude_y0 && y < exclude_y1 {
                continue;
            }
            
            let index = builder.grid.index_of(CellCoord::new(x, y));
            if !builder.solid_geometry.is_solid(index) {
                builder.initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(species_id, concentration);
            }
        }
    }
    
    builder
}
