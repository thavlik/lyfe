use std::io::{self, Write};

use fluidsim::{Simulation, SimulationConfig};

fn region_total(
    values: &[f32],
    solid_mask: &[u32],
    width: u32,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
) -> f32 {
    let mut total = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            let index = (y * width + x) as usize;
            if solid_mask[index] == 0 {
                total += values[index];
            }
        }
    }
    total
}

fn main() {
    let config = SimulationConfig {
        width: 128,
        height: 128,
        diffusion_rate: 5.0,
        thermal_diffusion_rate: 3.0,
        charge_correction_strength: 1.0,
        diffusion_substeps: 4,
        inspection_mip: 8,
        time_scale: 20.0,
        reaction_rate_scale: 8.0,
        max_frame_dt: 1.0 / 60.0,
    };

    let width = config.width;
    let height = config.height;
    let mut simulation = Simulation::new_leak(config).expect("leak sim should initialize");

    let k_idx = simulation.species_registry().index_of_name("K+").expect("K+ registered");
    let na_idx = simulation.species_registry().index_of_name("Na+").expect("Na+ registered");

    let wall_thickness = 4u32;
    let inner_size = (width.min(height) / 2).max(64);
    let outer_size = inner_size + 2 * wall_thickness;
    let center_x = width / 2;
    let center_y = height / 2;
    let outer_x0 = center_x - outer_size / 2;
    let outer_y0 = center_y - outer_size / 2;
    let outer_x1 = outer_x0 + outer_size;
    let outer_y1 = outer_y0 + outer_size;
    let inner_x0 = outer_x0 + wall_thickness;
    let inner_y0 = outer_y0 + wall_thickness;
    let inner_x1 = outer_x1 - wall_thickness;
    let inner_y1 = outer_y1 - wall_thickness;

    let initial = simulation.render_state().expect("initial render state");
    let initial_k_inside = region_total(
        &initial.concentrations[k_idx],
        &initial.solid_mask,
        width,
        inner_x0,
        inner_y0,
        inner_x1,
        inner_y1,
    );
    let initial_na_outside = region_total(
        &initial.concentrations[na_idx],
        &initial.solid_mask,
        width,
        0,
        0,
        width,
        outer_y0,
    ) + region_total(
        &initial.concentrations[na_idx],
        &initial.solid_mask,
        width,
        0,
        outer_y1,
        width,
        height,
    ) + region_total(
        &initial.concentrations[na_idx],
        &initial.solid_mask,
        width,
        0,
        outer_y0,
        outer_x0,
        outer_y1,
    ) + region_total(
        &initial.concentrations[na_idx],
        &initial.solid_mask,
        width,
        outer_x1,
        outer_y0,
        width,
        outer_y1,
    );

    for _ in 0..72 {
        simulation.step(1.0 / 60.0).expect("simulation step should succeed");
    }

    let final_state = simulation.render_state().expect("final render state");
    let final_k_inside = region_total(
        &final_state.concentrations[k_idx],
        &final_state.solid_mask,
        width,
        inner_x0,
        inner_y0,
        inner_x1,
        inner_y1,
    );
    let final_na_outside = region_total(
        &final_state.concentrations[na_idx],
        &final_state.solid_mask,
        width,
        0,
        0,
        width,
        outer_y0,
    ) + region_total(
        &final_state.concentrations[na_idx],
        &final_state.solid_mask,
        width,
        0,
        outer_y1,
        width,
        height,
    ) + region_total(
        &final_state.concentrations[na_idx],
        &final_state.solid_mask,
        width,
        0,
        outer_y0,
        outer_x0,
        outer_y1,
    ) + region_total(
        &final_state.concentrations[na_idx],
        &final_state.solid_mask,
        width,
        outer_x1,
        outer_y0,
        width,
        outer_y1,
    );

    println!("initial_k_inside={initial_k_inside}");
    println!("initial_na_outside={initial_na_outside}");
    println!("final_k_inside={final_k_inside}");
    println!("final_na_outside={final_na_outside}");

    io::stdout().flush().expect("probe stdout should flush");
}