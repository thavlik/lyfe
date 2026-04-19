use std::collections::HashMap;

pub(crate) fn compute_species_colors(registry: &fluidsim::SpeciesRegistry) -> Vec<[f32; 4]> {
    let overrides: HashMap<&str, [f32; 3]> = HashMap::from([
        ("Na+", [1.0, 0.22, 0.08]),
        ("K+", [0.12, 0.92, 0.22]),
        ("Cl-", [1.0, 0.86, 0.08]),
        ("H+", [1.0, 0.14, 0.58]),
        ("OH-", [0.38, 0.18, 1.0]),
        ("Ca2+", [1.0, 0.52, 0.06]),
        ("SO4(2-)", [0.05, 0.82, 0.92]),
        ("CH3COOH", [0.22, 0.95, 0.66]),
        ("CH3COO-", [1.0, 0.62, 0.12]),
    ]);

    registry
        .iter()
        .map(|info| {
            if let Some(&rgb) = overrides.get(info.name.as_ref()) {
                [rgb[0], rgb[1], rgb[2], 1.0]
            } else {
                let mut hue = (info.index as f32 * 0.618_034).fract();
                for _ in 0..8 {
                    let distance = (hue - 0.54).abs().min(1.0 - (hue - 0.54).abs());
                    if distance >= 0.12 {
                        break;
                    }
                    hue = (hue + 0.17).fract();
                }

                let rgb = hue_to_rgb(hue);
                [
                    0.5 + (rgb[0] - 0.5) * 1.1,
                    0.5 + (rgb[1] - 0.5) * 1.1,
                    0.5 + (rgb[2] - 0.5) * 1.1,
                    1.0,
                ]
            }
        })
        .collect()
}

pub(crate) fn leak_channel_color(species_name: &str) -> egui::Color32 {
    match species_name {
        "Na+" => egui::Color32::from_rgb(242, 102, 74),
        "K+" => egui::Color32::from_rgb(92, 214, 110),
        _ => egui::Color32::from_rgb(230, 230, 230),
    }
}

fn hue_to_rgb(hue: f32) -> [f32; 3] {
    let hue = hue * 6.0;
    let x = 1.0 - (hue % 2.0 - 1.0).abs();
    if hue < 1.0 {
        [1.0, x, 0.0]
    } else if hue < 2.0 {
        [x, 1.0, 0.0]
    } else if hue < 3.0 {
        [0.0, 1.0, x]
    } else if hue < 4.0 {
        [0.0, x, 1.0]
    } else if hue < 5.0 {
        [x, 0.0, 1.0]
    } else {
        [1.0, 0.0, x]
    }
}
