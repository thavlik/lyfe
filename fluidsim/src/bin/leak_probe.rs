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

fn fluid_total(values: &[f32], solid_mask: &[u32]) -> f32 {
    values
        .iter()
        .zip(solid_mask.iter())
        .filter_map(|(value, mask)| (*mask == 0).then_some(*value))
        .sum()
}

fn charge_metrics(concentrations: &[Vec<f32>], charges: &[i32], solid_mask: &[u32]) -> (f32, f32) {
    let mut max_abs_charge = 0.0_f32;
    let mut total_abs_charge = 0.0_f32;
    let mut fluid_cells = 0usize;

    for (cell_index, mask) in solid_mask.iter().enumerate() {
        if *mask != 0 {
            continue;
        }

        let mut net_charge = 0.0;
        for (species_index, charge) in charges.iter().enumerate() {
            if *charge == 0 {
                continue;
            }
            net_charge += *charge as f32 * concentrations[species_index][cell_index];
        }

        let abs_charge = net_charge.abs();
        max_abs_charge = max_abs_charge.max(abs_charge);
        total_abs_charge += abs_charge;
        fluid_cells += 1;
    }

    let mean_abs_charge = if fluid_cells > 0 {
        total_abs_charge / fluid_cells as f32
    } else {
        0.0
    };

    (max_abs_charge, mean_abs_charge)
}

fn main() {
    let config = SimulationConfig {
        width: 128,
        height: 128,
        diffusion_rate: 5.0,
        thermal_diffusion_rate: 3.0,
        charge_correction_strength: 0.0,
        diffusion_substeps: 4,
        inspection_mip: 8,
        time_scale: 20.0,
        reaction_rate_scale: 8.0,
        max_frame_dt: 1.0 / 60.0,
    };

    let width = config.width;
    let height = config.height;
    let mut simulation = Simulation::new_leak(config).expect("leak sim should initialize");
    let charges: Vec<i32> = simulation
        .species_registry()
        .iter()
        .map(|species| species.charge)
        .collect();

    let k_idx = simulation.species_registry().index_of_name("K+").expect("K+ registered");
    let na_idx = simulation.species_registry().index_of_name("Na+").expect("Na+ registered");
    let cl_idx = simulation.species_registry().index_of_name("Cl-").expect("Cl- registered");

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
    let initial_k_total = fluid_total(&initial.concentrations[k_idx], &initial.solid_mask);
    let initial_na_total = fluid_total(&initial.concentrations[na_idx], &initial.solid_mask);
    let initial_cl_total = fluid_total(&initial.concentrations[cl_idx], &initial.solid_mask);
    let initial_spectator_charge = initial_k_total + initial_na_total - initial_cl_total;
    let (initial_max_abs_charge, initial_mean_abs_charge) =
        charge_metrics(&initial.concentrations, &charges, &initial.solid_mask);
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
    let final_k_total = fluid_total(&final_state.concentrations[k_idx], &final_state.solid_mask);
    let final_na_total = fluid_total(&final_state.concentrations[na_idx], &final_state.solid_mask);
    let final_cl_total = fluid_total(&final_state.concentrations[cl_idx], &final_state.solid_mask);
    let final_spectator_charge = final_k_total + final_na_total - final_cl_total;
    let (final_max_abs_charge, final_mean_abs_charge) =
        charge_metrics(&final_state.concentrations, &charges, &final_state.solid_mask);
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
    println!("initial_k_total={initial_k_total}");
    println!("initial_na_total={initial_na_total}");
    println!("initial_cl_total={initial_cl_total}");
    println!("initial_spectator_charge={initial_spectator_charge}");
    println!("initial_max_abs_charge={initial_max_abs_charge}");
    println!("initial_mean_abs_charge={initial_mean_abs_charge}");
    println!("final_k_inside={final_k_inside}");
    println!("final_na_outside={final_na_outside}");
    println!("final_k_total={final_k_total}");
    println!("final_na_total={final_na_total}");
    println!("final_cl_total={final_cl_total}");
    println!("final_spectator_charge={final_spectator_charge}");
    println!("final_max_abs_charge={final_max_abs_charge}");
    println!("final_mean_abs_charge={final_mean_abs_charge}");

    io::stdout().flush().expect("probe stdout should flush");
}