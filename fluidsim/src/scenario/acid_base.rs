use crate::grid::CellCoord;

use super::helpers::{add_titanium_hollow_box, central_box_bounds, normalized_position};
use super::{Scenario, ScenarioBuilder};

/// Create the acid-base neutralization scenario.
///
/// Top-left region: 1.0 M H+ + 0.5 M SO4(2-) (acidic solution)
/// Bottom-right region: 1.0 M Na+ + 1.0 M OH- (basic solution)
///
/// When they mix, kinetics drives: H+ + OH- -> H2O
pub fn create_acid_base_scenario(width: u32, height: u32) -> Scenario {
    let bounds = central_box_bounds(width, height);

    let builder = ScenarioBuilder::new(width, height)
        .register_species("H+")
        .register_species("OH-")
        .register_species("Na+")
        .register_species("SO4(2-)");
    let builder = add_titanium_hollow_box(builder, bounds);
    let mut builder = builder.fill_temperature(278.15);

    let h_id = builder.species_registry.id_of("H+").unwrap();
    let oh_id = builder.species_registry.id_of("OH-").unwrap();
    let na_id = builder.species_registry.id_of("Na+").unwrap();
    let so4_id = builder.species_registry.id_of("SO4(2-)").unwrap();

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
                    .set(h_id, 1.0);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(so4_id, 0.5);
            } else {
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
