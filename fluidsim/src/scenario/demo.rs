use crate::grid::CellCoord;

use super::helpers::{add_titanium_hollow_box, central_box_bounds, normalized_position};
use super::{Scenario, ScenarioBuilder};

/// Create the demo scenario as specified in requirements.
pub fn create_demo_scenario(width: u32, height: u32) -> Scenario {
    let bounds = central_box_bounds(width, height);

    let builder = ScenarioBuilder::new(width, height)
        .register_species("Na+")
        .register_species("K+")
        .register_species("Cl-");
    let builder = add_titanium_hollow_box(builder, bounds);
    let mut builder = builder.fill_temperature(280.0);

    let na_id = builder.species_registry.id_of("Na+").unwrap();
    let k_id = builder.species_registry.id_of("K+").unwrap();
    let cl_id = builder.species_registry.id_of("Cl-").unwrap();

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
                    .set(na_id, 1.0);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(cl_id, 1.0);
                builder.initial_temperatures[index] = 318.15;
            } else {
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(k_id, 1.0);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(cl_id, 1.0);
                builder.initial_temperatures[index] = 288.0;
            }
        }
    }

    builder.build()
}
