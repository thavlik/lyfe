pub(crate) fn format_detail_probe_tooltip(result: &fluidsim::InspectionResult) -> String {
    let mut lines = vec![
        format!("Coarse: ({}, {})", result.coord.x, result.coord.y),
        format!("Region: {}", result.content),
        format!(
            "Cells: {} fluid / {} solid",
            result.fluid_cell_count, result.solid_cell_count
        ),
        format!(
            "Temp: {:.2} K ({:.2} C)",
            result.mean_temperature_kelvin,
            result.mean_temperature_kelvin - 273.15,
        ),
    ];

    if !result.species.is_empty() {
        lines.push("Species:".to_string());
        for entry in result.species.iter().take(4) {
            lines.push(format!(
                "  {:<10} {:>7.3} M",
                entry.name, entry.concentration
            ));
        }
        if result.species.len() > 4 {
            lines.push(format!("  +{} more", result.species.len() - 4));
        }
    }

    if !result.materials.is_empty() {
        lines.push("Materials:".to_string());
        for material in result.materials.iter().take(2) {
            lines.push(format!(
                "  {} {:>5.0}%",
                material.name,
                material.fraction * 100.0
            ));
        }
    }

    lines.join("\n")
}

pub(crate) fn format_async_inspection_tooltip(
    coord: (u32, u32),
    fluid_count: u32,
    solid_count: u32,
    mean_temperature_kelvin: f32,
    concentrations: &[f32],
    species_names: &[String],
) -> String {
    let mut species_rows: Vec<_> = concentrations
        .iter()
        .enumerate()
        .filter(|&(_, concentration)| *concentration > 0.001)
        .map(|(index, concentration)| {
            let name = species_names
                .get(index)
                .map(|value| value.as_str())
                .unwrap_or("?");
            (name.to_string(), *concentration)
        })
        .collect();
    species_rows.sort_by(|left, right| right.1.total_cmp(&left.1));

    let species_lines = species_rows
        .iter()
        .map(|(name, concentration)| format!("  {:<6} {:>8.3} M", name, concentration))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "Coarse Cell ({}, {})\n\
         Fluid: {} / Solid: {}\n\
         Temp: {:.2} K ({:.2} C)\n\
         Species:\n{}",
        coord.0,
        coord.1,
        fluid_count,
        solid_count,
        mean_temperature_kelvin,
        mean_temperature_kelvin - 273.15,
        species_lines,
    )
}
