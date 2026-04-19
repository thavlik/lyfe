use std::collections::HashMap;
use std::process::Command;

const CONSERVATION_TOLERANCE: f32 = 0.1;

fn parse_probe_output(stdout: &str) -> HashMap<String, f32> {
    stdout
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once('=')?;
            let parsed = value.trim().parse::<f32>().ok()?;
            Some((key.trim().to_string(), parsed))
        })
        .collect()
}

#[test]
fn leak_channels_move_k_inward_and_na_outward() {
    let probe = env!("CARGO_BIN_EXE_leak_probe");
    let output = Command::new(probe)
        .output()
        .expect("probe binary should run");

    assert!(
        output.status.success(),
        "probe failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let metrics = parse_probe_output(&String::from_utf8_lossy(&output.stdout));
    let initial_k_inside = metrics["initial_k_inside"];
    let initial_na_outside = metrics["initial_na_outside"];
    let initial_k_total = metrics["initial_k_total"];
    let initial_na_total = metrics["initial_na_total"];
    let initial_cl_total = metrics["initial_cl_total"];
    let initial_spectator_charge = metrics["initial_spectator_charge"];
    let initial_max_abs_charge = metrics["initial_max_abs_charge"];
    let initial_mean_abs_charge = metrics["initial_mean_abs_charge"];
    let final_k_inside = metrics["final_k_inside"];
    let final_na_outside = metrics["final_na_outside"];
    let final_k_total = metrics["final_k_total"];
    let final_na_total = metrics["final_na_total"];
    let final_cl_total = metrics["final_cl_total"];
    let final_spectator_charge = metrics["final_spectator_charge"];
    let final_max_abs_charge = metrics["final_max_abs_charge"];
    let final_mean_abs_charge = metrics["final_mean_abs_charge"];

    assert!(
        final_k_inside > initial_k_inside,
        "expected inward K+ leak: initial={initial_k_inside}, final={final_k_inside}"
    );
    assert!(
        final_na_outside > initial_na_outside,
        "expected outward Na+ leak: initial={initial_na_outside}, final={final_na_outside}"
    );
    assert!(
        (final_k_total - initial_k_total).abs() < CONSERVATION_TOLERANCE,
        "expected K+ mass conservation: initial={initial_k_total}, final={final_k_total}"
    );
    assert!(
        (final_na_total - initial_na_total).abs() < CONSERVATION_TOLERANCE,
        "expected Na+ mass conservation: initial={initial_na_total}, final={final_na_total}"
    );
    assert!(
        (final_cl_total - initial_cl_total).abs() < CONSERVATION_TOLERANCE,
        "expected Cl- mass conservation: initial={initial_cl_total}, final={final_cl_total}"
    );
    assert!(
        (final_spectator_charge - initial_spectator_charge).abs() < CONSERVATION_TOLERANCE,
        "expected spectator charge conservation: initial={initial_spectator_charge}, final={final_spectator_charge}"
    );
    assert!(
        final_max_abs_charge < 0.08,
        "expected leak transport to remain near electroneutral locally: initial_max={initial_max_abs_charge}, final_max={final_max_abs_charge}"
    );
    assert!(
        final_mean_abs_charge < 0.015,
        "expected low mean local charge after leak transport: initial_mean={initial_mean_abs_charge}, final_mean={final_mean_abs_charge}"
    );
}
