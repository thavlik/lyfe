//! Species identification and registry.
//!
//! Species are identified by string names (e.g., "Na+", "Cl-", "H+") and interned
//! to stable numeric IDs for efficient GPU storage and lookup.

use ahash::AHashMap;
use std::sync::Arc;

/// A stable identifier for a chemical species, derived from the species name.
/// 
/// This is a 64-bit hash of the species name string, providing:
/// - Fast equality comparison
/// - Stable identity across serialization
/// - No lifetime dependencies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeciesId(pub u64);

impl SpeciesId {
    /// Create a SpeciesId from a species name.
    pub fn from_name(name: &str) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = ahash::AHasher::default();
        name.hash(&mut hasher);
        Self(hasher.finish())
    }
}

/// A registry mapping species names to IDs and dense indices.
///
/// The registry provides two views:
/// - **External**: `SpeciesId` (hash-based) for stable identification
/// - **Internal**: Dense index (0..N) for GPU buffer addressing
///
/// Species must be registered before simulation starts. Once registered,
/// indices are stable for the lifetime of the simulation.
#[derive(Debug, Clone)]
pub struct SpeciesRegistry {
    /// Map from species ID to dense index
    id_to_index: AHashMap<SpeciesId, usize>,
    /// Map from dense index to species info
    species: Vec<SpeciesInfo>,
    /// Map from name to ID for reverse lookup
    name_to_id: AHashMap<Arc<str>, SpeciesId>,
}

/// Information about a registered species.
#[derive(Debug, Clone)]
pub struct SpeciesInfo {
    /// The stable species ID
    pub id: SpeciesId,
    /// The species name (e.g., "Na+", "Cl-")
    pub name: Arc<str>,
    /// Dense index for GPU buffer addressing
    pub index: usize,
    /// Diffusion coefficient (m²/s, scaled for simulation)
    pub diffusion_coefficient: f32,
    /// Integer ionic charge / valence (e.g. H+ = +1, SO4(2-) = -2)
    pub charge: i32,
}

impl Default for SpeciesRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeciesRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            id_to_index: AHashMap::new(),
            species: Vec::new(),
            name_to_id: AHashMap::new(),
        }
    }

    /// Register a new species with the given name and diffusion coefficient.
    /// Returns the assigned SpeciesId.
    ///
    /// # Panics
    /// Panics if a species with this name is already registered.
    pub fn register(&mut self, name: &str, diffusion_coefficient: f32) -> SpeciesId {
        let id = SpeciesId::from_name(name);
        
        if self.id_to_index.contains_key(&id) {
            panic!("Species '{}' is already registered", name);
        }

        let index = self.species.len();
        let name_arc: Arc<str> = name.into();
        let charge = infer_ionic_charge(name);
        
        self.species.push(SpeciesInfo {
            id,
            name: name_arc.clone(),
            index,
            diffusion_coefficient,
            charge,
        });
        self.id_to_index.insert(id, index);
        self.name_to_id.insert(name_arc, id);

        id
    }

    /// Get the dense index for a species ID.
    pub fn index_of(&self, id: SpeciesId) -> Option<usize> {
        self.id_to_index.get(&id).copied()
    }

    /// Get the dense index for a species by name.
    pub fn index_of_name(&self, name: &str) -> Option<usize> {
        let id = self.name_to_id.get(name)?;
        self.id_to_index.get(id).copied()
    }

    /// Get the species ID for a name.
    pub fn id_of(&self, name: &str) -> Option<SpeciesId> {
        self.name_to_id.get(name).copied()
    }

    /// Get species info by ID.
    pub fn get(&self, id: SpeciesId) -> Option<&SpeciesInfo> {
        self.id_to_index.get(&id).map(|&idx| &self.species[idx])
    }

    /// Get species info by dense index.
    pub fn get_by_index(&self, index: usize) -> Option<&SpeciesInfo> {
        self.species.get(index)
    }

    /// Get the number of registered species.
    pub fn count(&self) -> usize {
        self.species.len()
    }

    /// Iterate over all registered species.
    pub fn iter(&self) -> impl Iterator<Item = &SpeciesInfo> {
        self.species.iter()
    }

    /// Get diffusion coefficients as a dense array for GPU upload.
    pub fn diffusion_coefficients(&self) -> Vec<f32> {
        self.species.iter().map(|s| s.diffusion_coefficient).collect()
    }

    /// Get ionic charges as a dense array for GPU upload.
    pub fn charges(&self) -> Vec<i32> {
        self.species.iter().map(|s| s.charge).collect()
    }

    /// Return the largest per-species diffusion coefficient (for CFL checks).
    pub fn max_diffusion_coefficient(&self) -> f32 {
        self.species.iter()
            .map(|s| s.diffusion_coefficient)
            .fold(0.0_f32, f32::max)
    }
}

fn infer_ionic_charge(name: &str) -> i32 {
    if let Some(charge) = parse_parenthesized_charge(name) {
        return charge;
    }
    if let Some(charge) = parse_suffix_charge(name) {
        return charge;
    }
    0
}

fn parse_parenthesized_charge(name: &str) -> Option<i32> {
    let open = name.rfind('(')?;
    let close = name.rfind(')')?;
    if close <= open + 2 || close != name.len() - 1 {
        return None;
    }
    let inner = &name[open + 1..close];
    parse_charge_token(inner)
}

fn parse_suffix_charge(name: &str) -> Option<i32> {
    let trimmed = name.trim();
    let sign = trimmed.chars().last()?;
    if sign != '+' && sign != '-' {
        return None;
    }

    let prefix = &trimmed[..trimmed.len() - sign.len_utf8()];
    let trailing_digits: String = prefix.chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();

    let magnitude = if trailing_digits.is_empty() {
        1
    } else {
        trailing_digits.parse::<i32>().ok()?
    };
    Some(if sign == '+' { magnitude } else { -magnitude })
}

fn parse_charge_token(token: &str) -> Option<i32> {
    let sign = token.chars().last()?;
    if sign != '+' && sign != '-' {
        return None;
    }
    let magnitude_text = &token[..token.len() - sign.len_utf8()];
    let magnitude = if magnitude_text.is_empty() {
        1
    } else {
        magnitude_text.parse::<i32>().ok()?
    };
    Some(if sign == '+' { magnitude } else { -magnitude })
}

/// A map of species concentrations for a single cell.
/// This is the user-facing API for cell data, not used in hot paths.
#[derive(Debug, Clone, Default)]
pub struct SpeciesConcentrations {
    pub concentrations: AHashMap<SpeciesId, f64>,
}

impl SpeciesConcentrations {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set concentration for a species.
    pub fn set(&mut self, id: SpeciesId, concentration: f64) {
        if concentration > 0.0 {
            self.concentrations.insert(id, concentration);
        } else {
            self.concentrations.remove(&id);
        }
    }

    /// Get concentration for a species (0.0 if not present).
    pub fn get(&self, id: SpeciesId) -> f64 {
        self.concentrations.get(&id).copied().unwrap_or(0.0)
    }

    /// Add to existing concentration.
    pub fn add(&mut self, id: SpeciesId, amount: f64) {
        let current = self.get(id);
        self.set(id, current + amount);
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.concentrations.is_empty()
    }

    /// Iterate over non-zero concentrations.
    pub fn iter(&self) -> impl Iterator<Item = (SpeciesId, f64)> + '_ {
        self.concentrations.iter().map(|(&id, &conc)| (id, conc))
    }
}
