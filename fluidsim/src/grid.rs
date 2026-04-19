//! Grid representation and coordinate utilities.
//!
//! The simulation operates on a uniform 2D grid where each cell is 1x1 logical units.
//! Coordinates are stored as (x, y) pairs with x increasing rightward and y increasing downward.

use bytemuck::{Pod, Zeroable};

/// A coordinate in the fine grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellCoord {
    pub x: u32,
    pub y: u32,
}

impl CellCoord {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    /// Convert to linear index for a grid of given width.
    #[inline]
    pub fn to_index(self, width: u32) -> usize {
        (self.y * width + self.x) as usize
    }

    /// Convert from linear index.
    #[inline]
    pub fn from_index(index: usize, width: u32) -> Self {
        let index = index as u32;
        Self {
            x: index % width,
            y: index / width,
        }
    }
}

/// Grid dimensions and metadata.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    /// Width in fine cells
    pub width: u32,
    /// Height in fine cells
    pub height: u32,
}

impl Grid {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Total number of cells.
    #[inline]
    pub fn cell_count(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Check if a coordinate is within bounds.
    #[inline]
    pub fn contains(&self, coord: CellCoord) -> bool {
        coord.x < self.width && coord.y < self.height
    }

    /// Convert coordinate to linear index.
    #[inline]
    pub fn index_of(&self, coord: CellCoord) -> usize {
        coord.to_index(self.width)
    }

    /// Convert linear index to coordinate.
    #[inline]
    pub fn coord_of(&self, index: usize) -> CellCoord {
        CellCoord::from_index(index, self.width)
    }

    /// Iterate over all coordinates in row-major order.
    pub fn iter_coords(&self) -> impl Iterator<Item = CellCoord> {
        let width = self.width;
        let height = self.height;
        (0..height).flat_map(move |y| (0..width).map(move |x| CellCoord::new(x, y)))
    }

    /// Iterate over coordinates in a rectangular region.
    pub fn iter_rect(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> impl Iterator<Item = CellCoord> {
        let x0 = x0.min(self.width);
        let y0 = y0.min(self.height);
        let x1 = x1.min(self.width);
        let y1 = y1.min(self.height);
        (y0..y1).flat_map(move |y| (x0..x1).map(move |x| CellCoord::new(x, y)))
    }

    /// Get the 4-connected neighbors of a cell.
    pub fn neighbors(&self, coord: CellCoord) -> impl Iterator<Item = CellCoord> {
        let mut neighbors = Vec::with_capacity(4);
        if coord.x > 0 {
            neighbors.push(CellCoord::new(coord.x - 1, coord.y));
        }
        if coord.x + 1 < self.width {
            neighbors.push(CellCoord::new(coord.x + 1, coord.y));
        }
        if coord.y > 0 {
            neighbors.push(CellCoord::new(coord.x, coord.y - 1));
        }
        if coord.y + 1 < self.height {
            neighbors.push(CellCoord::new(coord.x, coord.y + 1));
        }
        neighbors.into_iter()
    }
}

/// GPU-friendly cell data structure for solid/fluid mask.
/// Stored as a u32 bitmask where:
/// - bit 0: is_solid
/// - bits 1-31: reserved for future use
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct CellFlags {
    pub flags: u32,
}

impl CellFlags {
    pub const SOLID_BIT: u32 = 1;

    pub fn fluid() -> Self {
        Self { flags: 0 }
    }

    pub fn solid() -> Self {
        Self {
            flags: Self::SOLID_BIT,
        }
    }

    #[inline]
    pub fn is_solid(self) -> bool {
        (self.flags & Self::SOLID_BIT) != 0
    }

    #[inline]
    pub fn is_fluid(self) -> bool {
        !self.is_solid()
    }
}
