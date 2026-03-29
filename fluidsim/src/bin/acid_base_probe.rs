use std::io::{self, Write};

use fluidsim::{Simulation, SimulationConfig};

fn mean_in_center_window(values: &[f32], width: u32, height: u32, radius: i32) -> f32 {
    let center_x = (width / 2) as i32;
    let center_y = (height / 2) as i32;
    let mut sum = 0.0f32;
    let mut count = 0u32;

    for y in (center_y - radius)..=(center_y + radius) {
        for x in (center_x - radius)..=(center_x + radius) {
            let x = x.clamp(0, width as i32 - 1) as usize;
            let y = y.clamp(0, height as i32 - 1) as usize;
            sum += values[y * width as usize + x];
            count += 1;
        }
    }

    sum / count as f32
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

    let mut simulation = Simulation::new_acid_base(config).expect("acid-base sim should initialize");
    let h_idx = simulation.species_registry().index_of_name("H+").expect("H+ registered");
    let oh_idx = simulation.species_registry().index_of_name("OH-").expect("OH- registered");

    let initial = simulation.render_state().expect("initial render state");
    let initial_h_center = mean_in_center_window(&initial.concentrations[h_idx], width, height, 1);
    let initial_oh_center = mean_in_center_window(&initial.concentrations[oh_idx], width, height, 1);
    let initial_temp_center = mean_in_center_window(&initial.temperatures, width, height, 1);

    for _ in 0..36 {
        simulation.step(1.0 / 60.0).expect("simulation step should succeed");
    }

    let final_state = simulation.render_state().expect("final render state");
    let final_h_center = mean_in_center_window(&final_state.concentrations[h_idx], width, height, 1);
    let final_oh_center = mean_in_center_window(&final_state.concentrations[oh_idx], width, height, 1);
    let final_temp_center = mean_in_center_window(&final_state.temperatures, width, height, 1);

    println!("initial_h_center={initial_h_center}");
    println!("initial_oh_center={initial_oh_center}");
    println!("initial_temp_center={initial_temp_center}");
    println!("final_h_center={final_h_center}");
    println!("final_oh_center={final_oh_center}");
    println!("final_temp_center={final_temp_center}");

    io::stdout().flush().expect("probe stdout should flush");

    std::process::exit(0);
}
