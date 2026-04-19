use crate::enzyme::{EnzymeEntity, EnzymeField};
use crate::grid::{CellCoord, Grid};
use crate::leak::LeakChannel;
use crate::solid::{MaterialId, MaterialRegistry, SolidGeometry, SolidRect};
use crate::species::{SpeciesConcentrations, SpeciesRegistry};
use ahash::AHashMap;

use super::Scenario;

/// Default diffusion coefficient for aqueous ions (scaled for simulation).
const DEFAULT_DIFFUSION: f32 = 1.0;

/// Builder for creating scenarios with a fluent API.
pub struct ScenarioBuilder {
    pub(crate) grid: Grid,
    pub(crate) species_registry: SpeciesRegistry,
    pub(crate) material_registry: MaterialRegistry,
    pub(crate) solid_geometry: SolidGeometry,
    pub(crate) leak_channels: Vec<LeakChannel>,
    pub(crate) enzyme_entities: Vec<EnzymeEntity>,
    pub(crate) enzyme_field: Option<EnzymeField>,
    pub(crate) initial_concentrations: AHashMap<usize, SpeciesConcentrations>,
    pub(crate) initial_temperatures: Vec<f32>,
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
        self.initial_temperatures = vec![temperature; self.grid.cell_count()];
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
        let temperatures = if self.initial_temperatures.is_empty() {
            vec![293.15; self.grid.cell_count()]
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
