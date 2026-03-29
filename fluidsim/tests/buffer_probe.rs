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
fn buffer_consumes_acid_and_hydroxide_into_acetate() {
    let probe = env!("CARGO_BIN_EXE_buffer_probe");
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
    let initial_acid = metrics["initial_acid_total"];
    let initial_hydroxide = metrics["initial_hydroxide_total"];
    let initial_acetate = metrics["initial_acetate_total"];
    let final_acid = metrics["final_acid_total"];
    let final_hydroxide = metrics["final_hydroxide_total"];
    let final_acetate = metrics["final_acetate_total"];

    assert!(
        final_acid < initial_acid,
        "expected CH3COOH consumption: initial={initial_acid}, final={final_acid}"
    );
    assert!(
        final_hydroxide < initial_hydroxide,
        "expected OH- consumption: initial={initial_hydroxide}, final={final_hydroxide}"
    );
    assert!(
        final_acetate > initial_acetate,
        "expected CH3COO- formation: initial={initial_acetate}, final={final_acetate}"
    );
}