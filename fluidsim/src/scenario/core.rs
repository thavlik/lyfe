use crate::enzyme::{EnzymeEntity, EnzymeField};
use crate::grid::Grid;
use crate::leak::LeakChannel;
use crate::solid::{MaterialRegistry, SolidGeometry};
use crate::species::{SpeciesConcentrations, SpeciesRegistry};
use ahash::AHashMap;

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

        let mut concentrations = vec![vec![0.0f32; cell_count]; species_count];

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
            .map(|material| if material.0 != 0 { 1u32 } else { 0u32 })
            .collect()
    }

    /// Compile material IDs to dense array.
    pub fn compile_material_ids(&self) -> Vec<u32> {
        self.solid_geometry
            .material_ids
            .iter()
            .map(|material| material.0)
            .collect()
    }

    /// Compile temperatures to dense array (already dense, just clone).
    pub fn compile_temperatures(&self) -> Vec<f32> {
        self.initial_temperatures.clone()
    }
}
