use crate::grid::CellCoord;

use super::helpers::{add_titanium_hollow_box, central_box_bounds, normalized_position};
use super::{Scenario, ScenarioBuilder};

/// Create a weak-acid buffer scenario.
///
/// Top-left region: 0.35 M sodium acetate + 0.35 M acetic acid.
/// - Na+ = 0.35 M
/// - CH3COO- = 0.35 M
/// - CH3COOH = 0.35 M
/// - H+ = Ka ~= 1.8e-5 M
///
/// Bottom-right region: 0.2 M NaOH represented as:
/// - Na+ = 0.2 M
/// - OH- = 0.2 M
pub fn create_buffers_scenario(width: u32, height: u32) -> Scenario {
    let bounds = central_box_bounds(width, height);

    let builder = ScenarioBuilder::new(width, height)
        .register_species("H+")
        .register_species("OH-")
        .register_species("Na+")
        .register_species("CH3COOH")
        .register_species("CH3COO-");
    let builder = add_titanium_hollow_box(builder, bounds);
    let mut builder = builder.fill_temperature(278.15);

    let h_id = builder.species_registry.id_of("H+").unwrap();
    let oh_id = builder.species_registry.id_of("OH-").unwrap();
    let na_id = builder.species_registry.id_of("Na+").unwrap();
    let acetic_acid_id = builder.species_registry.id_of("CH3COOH").unwrap();
    let acetate_id = builder.species_registry.id_of("CH3COO-").unwrap();

    for y in bounds.inner_y0..bounds.inner_y1 {
        for x in bounds.inner_x0..bounds.inner_x1 {
            let index = builder.grid.index_of(CellCoord::new(x, y));
            if builder.solid_geometry.is_solid(index) {
                continue;
            }

            let (nx, ny) = normalized_position(bounds, x, y);
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
                    .set(acetate_id, 0.35);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(acetic_acid_id, 0.35);
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

    builder.build()
}
