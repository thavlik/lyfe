use std::io::{self, Write};

use fluidsim::{Simulation, SimulationConfig};

fn total_in_fluid(values: &[f32], solid_mask: &[u32]) -> f32 {
    values
        .iter()
        .zip(solid_mask.iter())
        .filter(|(_, solid)| **solid == 0)
        .map(|(value, _)| *value)
        .sum()
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

    let mut simulation = Simulation::new_buffers(config).expect("buffer sim should initialize");
    let acid_idx = simulation
        .species_registry()
        .index_of_name("CH3COOH")
        .expect("CH3COOH registered");
    let hydroxide_idx = simulation
        .species_registry()
        .index_of_name("OH-")
        .expect("OH- registered");
    let acetate_idx = simulation
        .species_registry()
        .index_of_name("CH3COO-")
        .expect("CH3COO- registered");

    let initial = simulation.render_state().expect("initial render state");
    let initial_acid_total = total_in_fluid(&initial.concentrations[acid_idx], &initial.solid_mask);
    let initial_hydroxide_total =
        total_in_fluid(&initial.concentrations[hydroxide_idx], &initial.solid_mask);
    let initial_acetate_total =
        total_in_fluid(&initial.concentrations[acetate_idx], &initial.solid_mask);

    for _ in 0..72 {
        simulation
            .step(1.0 / 60.0)
            .expect("simulation step should succeed");
    }

    let final_state = simulation.render_state().expect("final render state");
    let final_acid_total = total_in_fluid(
        &final_state.concentrations[acid_idx],
        &final_state.solid_mask,
    );
    let final_hydroxide_total = total_in_fluid(
        &final_state.concentrations[hydroxide_idx],
        &final_state.solid_mask,
    );
    let final_acetate_total = total_in_fluid(
        &final_state.concentrations[acetate_idx],
        &final_state.solid_mask,
    );

    println!("initial_acid_total={initial_acid_total}");
    println!("initial_hydroxide_total={initial_hydroxide_total}");
    println!("initial_acetate_total={initial_acetate_total}");
    println!("final_acid_total={final_acid_total}");
    println!("final_hydroxide_total={final_hydroxide_total}");
    println!("final_acetate_total={final_acetate_total}");

    io::stdout().flush().expect("probe stdout should flush");
    std::process::exit(0);
}
