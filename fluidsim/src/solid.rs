//! Solid geometry representation and material metadata.
//!
//! Solid cells act as impermeable barriers in the simulation.
//! Each solid cell can have material metadata attached (e.g., titanium).

use ahash::AHashMap;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// A stable identifier for a material type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Pod, Zeroable)]
#[repr(C)]
pub struct MaterialId(pub u32);

impl MaterialId {
    /// No material (fluid cell).
    pub const NONE: MaterialId = MaterialId(0);

    /// Create a new material ID.
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Information about a registered material.
#[derive(Debug, Clone)]
pub struct MaterialInfo {
    pub id: MaterialId,
    pub name: Arc<str>,
    /// Display color as [R, G, B, A] in 0..1 range.
    pub color: [f32; 4],
    /// Arbitrary metadata for the material.
    pub metadata: AHashMap<String, MaterialProperty>,
}

/// A property value for material metadata.
#[derive(Debug, Clone)]
pub enum MaterialProperty {
    Float(f64),
    String(String),
    Bool(bool),
}

/// Registry for material definitions.
#[derive(Debug, Clone)]
pub struct MaterialRegistry {
    materials: Vec<MaterialInfo>,
    name_to_id: AHashMap<Arc<str>, MaterialId>,
}

impl Default for MaterialRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            materials: Vec::new(),
            name_to_id: AHashMap::new(),
        };
        // Register the "none" material at index 0
        registry.materials.push(MaterialInfo {
            id: MaterialId::NONE,
            name: "none".into(),
            color: [0.0, 0.0, 0.0, 0.0],
            metadata: AHashMap::new(),
        });
        registry
    }

    /// Register a new material.
    pub fn register(&mut self, name: &str, color: [f32; 4]) -> MaterialId {
        let id = MaterialId::new(self.materials.len() as u32);
        let name_arc: Arc<str> = name.into();

        self.materials.push(MaterialInfo {
            id,
            name: name_arc.clone(),
            color,
            metadata: AHashMap::new(),
        });
        self.name_to_id.insert(name_arc, id);

        id
    }

    /// Get material info by ID.
    pub fn get(&self, id: MaterialId) -> Option<&MaterialInfo> {
        self.materials.get(id.0 as usize)
    }

    /// Get material ID by name.
    pub fn id_of(&self, name: &str) -> Option<MaterialId> {
        self.name_to_id.get(name).copied()
    }

    /// Get material colors as a flat array for GPU upload.
    pub fn colors(&self) -> Vec<[f32; 4]> {
        self.materials.iter().map(|m| m.color).collect()
    }

    /// Number of registered materials.
    pub fn count(&self) -> usize {
        self.materials.len()
    }
}

/// Metadata attached to a solid cell.
#[derive(Debug, Clone)]
pub struct SolidCellMeta {
    pub material: MaterialId,
    /// Additional key-value annotations.
    pub annotations: AHashMap<String, String>,
}

impl Default for SolidCellMeta {
    fn default() -> Self {
        Self {
            material: MaterialId::NONE,
            annotations: AHashMap::new(),
        }
    }
}

impl SolidCellMeta {
    pub fn new(material: MaterialId) -> Self {
        Self {
            material,
            annotations: AHashMap::new(),
        }
    }

    pub fn with_annotation(mut self, key: &str, value: &str) -> Self {
        self.annotations.insert(key.to_string(), value.to_string());
        self
    }
}

/// Container for all solid geometry in the simulation.
#[derive(Debug, Clone)]
pub struct SolidGeometry {
    /// Material ID for each cell (MaterialId::NONE for fluid cells).
    /// Size: grid.width * grid.height
    pub material_ids: Vec<MaterialId>,
    /// Sparse metadata for solid cells that have annotations.
    pub cell_metadata: AHashMap<usize, SolidCellMeta>,
}

impl SolidGeometry {
    /// Create empty solid geometry for a grid.
    pub fn new(cell_count: usize) -> Self {
        Self {
            material_ids: vec![MaterialId::NONE; cell_count],
            cell_metadata: AHashMap::new(),
        }
    }

    /// Set a cell as solid with the given material.
    pub fn set_solid(&mut self, index: usize, material: MaterialId) {
        if index < self.material_ids.len() {
            self.material_ids[index] = material;
        }
    }

    /// Set metadata for a solid cell.
    pub fn set_metadata(&mut self, index: usize, meta: SolidCellMeta) {
        self.cell_metadata.insert(index, meta);
    }

    /// Check if a cell is solid.
    #[inline]
    pub fn is_solid(&self, index: usize) -> bool {
        self.material_ids
            .get(index)
            .map(|m| m.0 != 0)
            .unwrap_or(false)
    }

    /// Get the material of a cell.
    #[inline]
    pub fn material(&self, index: usize) -> MaterialId {
        self.material_ids
            .get(index)
            .copied()
            .unwrap_or(MaterialId::NONE)
    }

    /// Get metadata for a cell.
    pub fn metadata(&self, index: usize) -> Option<&SolidCellMeta> {
        self.cell_metadata.get(&index)
    }

    /// Fill a rectangular region with solid material.
    pub fn fill_rect(
        &mut self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        width: u32,
        material: MaterialId,
    ) {
        for y in y0..y1 {
            for x in x0..x1 {
                let index = (y * width + x) as usize;
                self.set_solid(index, material);
            }
        }
    }

    /// Create a hollow rectangle (outline only).
    pub fn fill_hollow_rect(
        &mut self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        thickness: u32,
        width: u32,
        material: MaterialId,
    ) {
        // Top edge
        self.fill_rect(x0, y0, x1, y0 + thickness, width, material);
        // Bottom edge
        self.fill_rect(x0, y1 - thickness, x1, y1, width, material);
        // Left edge
        self.fill_rect(
            x0,
            y0 + thickness,
            x0 + thickness,
            y1 - thickness,
            width,
            material,
        );
        // Right edge
        self.fill_rect(
            x1 - thickness,
            y0 + thickness,
            x1,
            y1 - thickness,
            width,
            material,
        );
    }
}
