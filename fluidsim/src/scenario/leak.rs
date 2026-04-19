use crate::grid::CellCoord;
use crate::leak::LeakChannel;

use super::helpers::{add_titanium_hollow_box, central_box_bounds, normalized_position};
use super::{Scenario, ScenarioBuilder};

/// Create a leak-channel membrane scenario.
pub fn create_leak_scenario(width: u32, height: u32) -> Scenario {
    let bounds = central_box_bounds(width, height);

    let builder = ScenarioBuilder::new(width, height)
        .register_species("H+")
        .register_species("OH-")
        .register_species("Na+")
        .register_species("K+")
        .register_species("Cl-")
        .register_species("CH3COOH")
        .register_species("CH3COO-");
    let builder = add_titanium_hollow_box(builder, bounds);
    let mut builder = builder.fill_temperature(278.15);

    let h_id = builder.species_registry.id_of("H+").unwrap();
    let oh_id = builder.species_registry.id_of("OH-").unwrap();
    let na_id = builder.species_registry.id_of("Na+").unwrap();
    let k_id = builder.species_registry.id_of("K+").unwrap();
    let cl_id = builder.species_registry.id_of("Cl-").unwrap();
    let acetic_acid_id = builder.species_registry.id_of("CH3COOH").unwrap();
    let acetate_id = builder.species_registry.id_of("CH3COO-").unwrap();

    for y in 0..height {
        for x in 0..width {
            let index = builder.grid.index_of(CellCoord::new(x, y));
            if builder.solid_geometry.is_solid(index) {
                continue;
            }

            let inside_box = x >= bounds.inner_x0
                && x < bounds.inner_x1
                && y >= bounds.inner_y0
                && y < bounds.inner_y1;
            if !inside_box {
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
            }
        }
    }

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
                    .set(acetate_id, 0.350018);
                builder
                    .initial_concentrations
                    .entry(index)
                    .or_default()
                    .set(acetic_acid_id, 0.349982);
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

    let channel_count = 6u32;
    for channel_index in 0..channel_count {
        let y = bounds.inner_y0
            + ((channel_index + 1) * (bounds.inner_y1 - bounds.inner_y0) / (channel_count + 1));
        builder.leak_channels.push(LeakChannel::new(
            4.5,
            k_id,
            (bounds.inner_x0 - 1) as i32,
            y as i32,
            0,
        ));
        builder.leak_channels.push(LeakChannel::new(
            4.5,
            na_id,
            bounds.inner_x1 as i32,
            y as i32,
            0,
        ));
    }

    builder.build()
}
