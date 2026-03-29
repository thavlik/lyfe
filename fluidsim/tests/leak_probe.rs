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
    let final_k_inside = metrics["final_k_inside"];
    let final_na_outside = metrics["final_na_outside"];

    assert!(
        final_k_inside > initial_k_inside,
        "expected inward K+ leak: initial={initial_k_inside}, final={final_k_inside}"
    );
    assert!(
        final_na_outside > initial_na_outside,
        "expected outward Na+ leak: initial={initial_na_outside}, final={final_na_outside}"
    );
}