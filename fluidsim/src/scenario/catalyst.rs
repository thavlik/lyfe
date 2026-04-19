use crate::grid::CellCoord;

use super::helpers::{add_titanium_hollow_box, central_box_bounds, normalized_position};
use super::{Scenario, ScenarioBuilder};

/// Create a catalyst-gated phosphorylation scenario.
///
/// Uses the same hollow-box geometry as the basic demo, but fills the entire
/// interior with glucose, ATP, and a low concentration of hexokinase.
pub fn create_catalyst_scenario(width: u32, height: u32) -> Scenario {
    let bounds = central_box_bounds(width, height);

    let builder = ScenarioBuilder::new(width, height)
        .register_species("Glucose")
        .register_species("ATP")
        .register_species("ADP")
        .register_species("G6P")
        .register_species("Hexokinase");
    let builder = add_titanium_hollow_box(builder, bounds);
    let mut builder = builder.fill_temperature(280.0);

    let glucose_id = builder.species_registry.id_of("Glucose").unwrap();
    let atp_id = builder.species_registry.id_of("ATP").unwrap();
    let hexokinase_id = builder.species_registry.id_of("Hexokinase").unwrap();

    for y in bounds.inner_y0..bounds.inner_y1 {
        for x in bounds.inner_x0..bounds.inner_x1 {
            let index = builder.grid.index_of(CellCoord::new(x, y));
            if builder.solid_geometry.is_solid(index) {
                continue;
            }

            builder
                .initial_concentrations
                .entry(index)
                .or_default()
                .set(glucose_id, 1.0);
            builder
                .initial_concentrations
                .entry(index)
                .or_default()
                .set(atp_id, 1.0);
            builder
                .initial_concentrations
                .entry(index)
                .or_default()
                .set(hexokinase_id, 0.01);

            let (nx, ny) = normalized_position(bounds, x, y);
            builder.initial_temperatures[index] = if nx + ny < 1.0 { 318.15 } else { 288.0 };
        }
    }

    builder.build()
}
