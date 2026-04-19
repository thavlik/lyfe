use std::collections::HashMap;
use std::process::Command;

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
fn acid_base_center_window_heats_and_neutralizes() {
    let probe = env!("CARGO_BIN_EXE_acid_base_probe");
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
    let initial_h = metrics["initial_h_center"];
    let initial_oh = metrics["initial_oh_center"];
    let initial_temp = metrics["initial_temp_center"];
    let final_h = metrics["final_h_center"];
    let final_oh = metrics["final_oh_center"];
    let final_temp = metrics["final_temp_center"];

    assert!(
        final_h < initial_h,
        "expected H+ consumption near center: initial={initial_h}, final={final_h}"
    );
    assert!(
        final_oh < initial_oh,
        "expected OH- consumption near center: initial={initial_oh}, final={final_oh}"
    );
    assert!(
        final_temp > initial_temp,
        "expected exothermic heating near center: initial={initial_temp}, final={final_temp}"
    );
}
