//! Cell inspection and aggregation for mouse hover tooltips.
//!
//! Provides coarse-cell inspection that aggregates fine-cell data
//! for display in tooltips.

use crate::species::{SpeciesId, SpeciesRegistry};
use crate::solid::{MaterialId, MaterialRegistry};
use std::fmt;

/// Coordinate of a coarse cell for inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoarseCellCoord {
    /// X coordinate in coarse grid
    pub x: u32,
    /// Y coordinate in coarse grid
    pub y: u32,
    /// Mip factor (fine cells per coarse cell dimension)
    pub mip: u32,
}

impl CoarseCellCoord {
    pub fn new(x: u32, y: u32, mip: u32) -> Self {
        Self { x, y, mip }
    }

    /// Get the fine-cell bounds covered by this coarse cell.
    /// Returns (x0, y0, x1, y1) where x1/y1 are exclusive.
    pub fn fine_bounds(&self) -> (u32, u32, u32, u32) {
        let x0 = self.x * self.mip;
        let y0 = self.y * self.mip;
        (x0, y0, x0 + self.mip, y0 + self.mip)
    }
}

/// Type of content in an inspected region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionContent {
    /// Only fluid cells
    Fluid,
    /// Only solid cells
    Solid,
    /// Mix of fluid and solid
    Mixed,
}

impl fmt::Display for RegionContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegionContent::Fluid => write!(f, "Fluid"),
            RegionContent::Solid => write!(f, "Solid"),
            RegionContent::Mixed => write!(f, "Mixed"),
        }
    }
}

/// A species concentration entry for inspection display.
#[derive(Debug, Clone)]
pub struct SpeciesEntry {
    pub name: String,
    pub id: SpeciesId,
    /// Average concentration in molarity
    pub concentration: f64,
}

/// A material entry for inspection display.
#[derive(Debug, Clone)]
pub struct MaterialEntry {
    pub name: String,
    pub id: MaterialId,
    /// Fraction of cells with this material (0.0 - 1.0)
    pub fraction: f32,
}

/// Result of inspecting a coarse cell region.
#[derive(Debug, Clone)]
pub struct InspectionResult {
    /// Coarse cell coordinate that was inspected
    pub coord: CoarseCellCoord,
    /// Type of content in this region
    pub content: RegionContent,
    /// Species concentrations, sorted by concentration (highest first)
    pub species: Vec<SpeciesEntry>,
    /// Materials present (for solid cells)
    pub materials: Vec<MaterialEntry>,
    /// Number of fluid cells in region
    pub fluid_cell_count: u32,
    /// Number of solid cells in region
    pub solid_cell_count: u32,
    /// Average temperature across the inspected region (Kelvin)
    pub mean_temperature_kelvin: f64,
}

impl InspectionResult {
    /// Create an empty result (e.g., for out-of-bounds coordinates).
    pub fn empty(coord: CoarseCellCoord) -> Self {
        Self {
            coord,
            content: RegionContent::Fluid,
            species: Vec::new(),
            materials: Vec::new(),
            fluid_cell_count: 0,
            solid_cell_count: 0,
            mean_temperature_kelvin: 293.15,
        }
    }

    /// Format as a tooltip string.
    pub fn format_tooltip(&self) -> String {
        let mut lines = Vec::new();
        
        // Header with region type
        lines.push(format!(
            "Region ({}, {}) [{}x{}]: {}",
            self.coord.x, self.coord.y,
            self.coord.mip, self.coord.mip,
            self.content
        ));
        lines.push(format!(
            "Cells: {} fluid, {} solid",
            self.fluid_cell_count, self.solid_cell_count
        ));
        lines.push(format!(
            "Temperature: {:.2} K ({:.2} C)",
            self.mean_temperature_kelvin,
            self.mean_temperature_kelvin - 273.15,
        ));
        
        // Species concentrations
        if !self.species.is_empty() {
            lines.push(String::new());
            lines.push("Species (M):".to_string());
            for entry in &self.species {
                lines.push(format!("  {}: {:.4}", entry.name, entry.concentration));
            }
        }
        
        // Materials
        if !self.materials.is_empty() {
            lines.push(String::new());
            lines.push("Materials:".to_string());
            for entry in &self.materials {
                lines.push(format!("  {}: {:.0}%", entry.name, entry.fraction * 100.0));
            }
        }
        
        lines.join("\n")
    }
}

/// Inspector for aggregating cell data.
pub struct Inspector {
    /// Default mip factor for inspection
    pub default_mip: u32,
    /// Minimum concentration to include in results
    pub epsilon: f64,
}

impl Default for Inspector {
    fn default() -> Self {
        Self {
            default_mip: 8,
            epsilon: 1e-9,
        }
    }
}

impl Inspector {
    pub fn new(default_mip: u32) -> Self {
        Self {
            default_mip,
            epsilon: 1e-9,
        }
    }

    /// Convert screen/world position to coarse cell coordinate.
    pub fn screen_to_coarse(&self, x: f32, y: f32, mip: u32) -> CoarseCellCoord {
        CoarseCellCoord::new(
            (x / mip as f32).floor() as u32,
            (y / mip as f32).floor() as u32,
            mip,
        )
    }

    /// Aggregate inspection data from concentration and material buffers.
    ///
    /// # Arguments
    /// - `coord`: Coarse cell to inspect
    /// - `concentrations`: `[species][cell]` concentration data
    /// - `solid_mask`: Per-cell solid flag
    /// - `material_ids`: Per-cell material ID
    /// - `grid_width`: Width of the fine grid
    /// - `grid_height`: Height of the fine grid
    /// - `species_registry`: For looking up species names
    /// - `material_registry`: For looking up material names
    pub fn inspect(
        &self,
        coord: CoarseCellCoord,
        concentrations: &[Vec<f32>],
        temperatures: &[f32],
        solid_mask: &[u32],
        material_ids: &[u32],
        grid_width: u32,
        grid_height: u32,
        species_registry: &SpeciesRegistry,
        material_registry: &MaterialRegistry,
    ) -> InspectionResult {
        let (x0, y0, x1, y1) = coord.fine_bounds();
        
        // Clamp to grid bounds
        let x1 = x1.min(grid_width);
        let y1 = y1.min(grid_height);
        
        if x0 >= grid_width || y0 >= grid_height {
            return InspectionResult::empty(coord);
        }
        
        // Count cells and accumulate species
        let mut fluid_count = 0u32;
        let mut solid_count = 0u32;
        let mut species_sums = vec![0.0f64; species_registry.count()];
        let mut material_counts = vec![0u32; material_registry.count()];
        let mut temperature_sum = 0.0f64;
        let mut temperature_samples = 0u32;
        
        for y in y0..y1 {
            for x in x0..x1 {
                let idx = (y * grid_width + x) as usize;
                if let Some(&temperature) = temperatures.get(idx) {
                    temperature_sum += temperature as f64;
                    temperature_samples += 1;
                }
                
                if solid_mask.get(idx).copied().unwrap_or(0) != 0 {
                    solid_count += 1;
                    let mat_id = material_ids.get(idx).copied().unwrap_or(0) as usize;
                    if mat_id < material_counts.len() {
                        material_counts[mat_id] += 1;
                    }
                } else {
                    fluid_count += 1;
                    
                    // Accumulate species concentrations
                    for (species_idx, conc_buffer) in concentrations.iter().enumerate() {
                        if let Some(&conc) = conc_buffer.get(idx) {
                            species_sums[species_idx] += conc as f64;
                        }
                    }
                }
            }
        }
        
        let total_cells = fluid_count + solid_count;
        if total_cells == 0 {
            return InspectionResult::empty(coord);
        }
        
        // Determine content type
        let content = match (fluid_count, solid_count) {
            (_, 0) => RegionContent::Fluid,
            (0, _) => RegionContent::Solid,
            _ => RegionContent::Mixed,
        };
        
        // Build species entries (average over fluid cells)
        let mut species = Vec::new();
        if fluid_count > 0 {
            for info in species_registry.iter() {
                let avg = species_sums[info.index] / fluid_count as f64;
                if avg > self.epsilon {
                    species.push(SpeciesEntry {
                        name: info.name.to_string(),
                        id: info.id,
                        concentration: avg,
                    });
                }
            }
        }
        
        // Sort by concentration, highest first
        species.sort_by(|a, b| b.concentration.partial_cmp(&a.concentration).unwrap());
        
        // Build material entries
        let mut materials = Vec::new();
        for (mat_idx, &count) in material_counts.iter().enumerate() {
            if count > 0 && mat_idx > 0 {  // Skip "none" material at index 0
                if let Some(mat_info) = material_registry.get(MaterialId::new(mat_idx as u32)) {
                    materials.push(MaterialEntry {
                        name: mat_info.name.to_string(),
                        id: mat_info.id,
                        fraction: count as f32 / total_cells as f32,
                    });
                }
            }
        }
        
        // Sort materials by fraction
        materials.sort_by(|a, b| b.fraction.partial_cmp(&a.fraction).unwrap());
        let mean_temperature_kelvin = if temperature_samples > 0 {
            temperature_sum / temperature_samples as f64
        } else {
            293.15
        };
        
        InspectionResult {
            coord,
            content,
            species,
            materials,
            fluid_cell_count: fluid_count,
            solid_cell_count: solid_count,
            mean_temperature_kelvin,
        }
    }
}
