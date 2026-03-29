# Fluid Simulation Demo

A high-performance GPU-accelerated 2D fluid simulation with species mixing, implemented in Rust using Vulkan compute shaders.

## Architecture

### Workspace Structure

- **fluidsim**: Core simulation library
  - Species registry and ID interning  
  - Grid representation
  - Solid geometry handling
  - GPU compute pipeline for diffusion
  - Inspection/aggregation APIs
  
- **renderer**: Vulkan rendering library
  - Swapchain management
  - Visualization pipeline  
  - egui integration for tooltips

- **demo**: Main executable
  - Window creation and event loop
  - Simulation/render coordination
  - Mouse inspection handling

## Data Model

### Species System

Species are identified by string names (e.g., `"Na+"`, `"Cl-"`) and internally interned to stable numeric IDs for GPU efficiency:

- **External API**: Uses `HashMap<SpeciesId, f64>` for conceptual cell-level access
- **Internal Storage**: Dense `[species][cell]` buffer layout where:
  - `species_index` is the outer dimension (one slice per species)
  - `cell_index = y * width + x` is the inner dimension (row-major grid)

This layout optimizes for per-species diffusion passes where each compute shader invocation processes one species channel.

### Buffer Layout

```
Concentration buffer: [species_count][cell_count] of f32
  - Total size: species_count * width * height * 4 bytes
  - Access: buffer[species_index * cell_count + cell_index]

Solid mask: [cell_count] of u32
  - 0 = fluid cell, 1 = solid cell

Material IDs: [cell_count] of u32
  - 0 = none (fluid), >0 = material index
```

### Grid Coordinates

- 2D uniform grid where each cell is 1x1 logical units
- Coordinates: (x, y) with x increasing rightward, y increasing downward
- Linear index: `cell_index = y * width + x`

## Simulation Loop

Each simulation step:

1. **Diffusion Pass** (GPU compute)
   - For each species channel, diffuse concentrations between adjacent fluid cells
   - Zero flux at solid boundaries (no transfer into/out of solid cells)
   - Uses ping-pong buffers to avoid read-write hazards
   
2. **Chemistry Evolution** (stub)
   - Hook for future reaction chemistry
   - Currently a no-op that preserves concentrations

### Diffusion Algorithm

The GPU compute shader implements explicit Euler diffusion:

```glsl
new_c[i] = c[i] + D * dt * sum(c[neighbor] - c[i]) for each fluid neighbor
```

- 4-connected neighbors (left, right, top, bottom)
- Solid cells block diffusion (zero flux boundary)
- Concentrations clamped to prevent negative values

## Inspection System

Mouse hover inspection aggregates over coarse cells:

- Default mip factor: 8 (each inspected region covers 8x8 fine cells)
- Aggregation: Average species concentrations across covered fluid cells
- Display: Sorted by concentration (highest first), filtered by epsilon threshold

### Inspection Result

```rust
InspectionResult {
    coord: CoarseCellCoord,      // Which coarse cell was inspected
    content: RegionContent,       // Fluid, Solid, or Mixed
    species: Vec<SpeciesEntry>,   // Species concentrations
    materials: Vec<MaterialEntry>, // Materials present (if solid)
    fluid_cell_count: u32,
    solid_cell_count: u32,
}
```

## Demo Scenario

The demo initializes:

- **Grid**: 512x512 fine cells
- **Solid geometry**: Hollow titanium square (wall thickness = 4 cells) centered in grid
- **Interior**:
  - Left half: 0.1 M NaCl → Na+ = 0.1, Cl- = 0.1
  - Right half: 0.2 M KCl → K+ = 0.2, Cl- = 0.2
- **Exterior**:
  - Left half: 1.0 M NaOH → Na+ = 1.0, OH- = 1.0
  - Right half: 1.0 M HCl → H+ = 1.0, Cl- = 1.0

Over time:
- Interior species homogenize to a uniform mixture
- Exterior species homogenize to a uniform mixture
- Interior and exterior remain isolated by the titanium wall

## Controls

- **Mouse hover**: Inspect coarse cell under cursor
- **Space**: Toggle pause
- **+/-**: Adjust inspection mip factor
- **Escape**: Exit

## Performance Notes

- All simulation runs on GPU compute shaders
- Minimal CPU-GPU synchronization (only for inspection readback)
- Persistent GPU buffers with ping-pong for diffusion
- Struct-of-arrays layout for cache-friendly GPU access
- Dense indexing (no hash maps in hot path)

## Building

```bash
cargo build --release
cargo run --release -p demo
```

Requires:
- Rust 2024 edition
- Vulkan 1.2 capable GPU
- Linux with X11 or Wayland (currently)
