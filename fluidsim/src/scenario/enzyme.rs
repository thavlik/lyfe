use crate::enzyme::{EnzymeEntity, EnzymeField};
use crate::grid::CellCoord;

use super::helpers::{add_titanium_hollow_box, central_box_bounds, lcg_unit, normalized_position};
use super::{Scenario, ScenarioBuilder};

/// Create an enzyme-entity phosphorylation scenario.
///
/// Starts from the catalyst demo chemistry and geometry, but models
/// hexokinase as six drifting entities rather than a dissolved catalyst.
pub fn create_enzyme_scenario(width: u32, height: u32) -> Scenario {
    let bounds = central_box_bounds(width, height);

    let builder = ScenarioBuilder::new(width, height)
        .register_species("Glucose")
        .register_species("ATP")
        .register_species("ADP")
        .register_species("G6P");
    let builder = add_titanium_hollow_box(builder, bounds);
    let mut builder = builder.fill_temperature(293.15);

    let glucose_id = builder.species_registry.id_of("Glucose").unwrap();
    let atp_id = builder.species_registry.id_of("ATP").unwrap();

    let cool_temperature = 293.15;
    let hot_temperature = cool_temperature + 15.0;

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

            let (nx, ny) = normalized_position(bounds, x, y);
            builder.initial_temperatures[index] = if nx + ny < 1.0 {
                hot_temperature
            } else {
                cool_temperature
            };
        }
    }

    let entity_margin = 10.0;
    let field = EnzymeField {
        min_x: bounds.inner_x0 as f32 + entity_margin,
        min_y: bounds.inner_y0 as f32 + entity_margin,
        max_x: bounds.inner_x1 as f32 - entity_margin,
        max_y: bounds.inner_y1 as f32 - entity_margin,
        hot_temperature,
        cool_temperature,
        circulation_strength: 1.35,
        thermophoretic_strength: 0.55,
        brownian_strength: 0.95,
        rotational_diffusion: 0.75,
    };

    let mut seed =
        width.wrapping_mul(73_856_093) ^ height.wrapping_mul(19_349_663) ^ 0xA5A5_1F1Fu32;
    let mut entities = Vec::with_capacity(6);
    let minimum_spacing = 14.0f32;

    for entity_index in 0..6 {
        let mut candidate = glam::Vec2::new(field.min_x, field.min_y);
        for _ in 0..96 {
            candidate = glam::Vec2::new(
                field.min_x + lcg_unit(&mut seed) * (field.max_x - field.min_x),
                field.min_y + lcg_unit(&mut seed) * (field.max_y - field.min_y),
            );
            if entities.iter().all(|existing: &EnzymeEntity| {
                existing.position().distance(candidate) >= minimum_spacing
            }) {
                break;
            }
        }

        let rotation_radians = lcg_unit(&mut seed) * std::f32::consts::TAU;
        let catalytic_scale = 0.85 + 0.35 * lcg_unit(&mut seed);
        let mobility_scale = 0.8 + 0.45 * lcg_unit(&mut seed);
        let thermal_bias = -1.5 + 3.0 * lcg_unit(&mut seed);
        entities.push(EnzymeEntity::new(
            candidate.x,
            candidate.y,
            rotation_radians,
            catalytic_scale,
            mobility_scale,
            thermal_bias,
            seed ^ (entity_index as u32).wrapping_mul(0x9E37_79B9),
        ));
    }

    builder.enzyme_entities = entities;
    builder.enzyme_field = Some(field);
    builder.build()
}
