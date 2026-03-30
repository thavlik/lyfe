#version 450

// Visualization fragment shader for fluid simulation
//
// Coloring scheme:
// - Pure water (zero solute): Light blue
// - Each species gets a pre-computed color from a CPU-side lookup table
// - Lower concentrations are boosted nonlinearly so transient streams remain visible
// - Colors are blended with weighted interpolation biased toward visible streaks

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

// Push constants
layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    uint species_count;
    uint frame_counter;  // For debug animation
    uint thermal_view;   // 1 = show thermal color ramp, 0 = normal
};

// Concentrations: [species][cell] layout
layout(set = 0, binding = 0) readonly buffer Concentrations {
    float concentrations[];
};

// Solid mask: 1 = solid, 0 = fluid
layout(set = 0, binding = 1) readonly buffer SolidMask {
    uint solid_mask[];
};

// Material IDs
layout(set = 0, binding = 2) readonly buffer MaterialIds {
    uint material_ids[];
};

// Pre-computed species colors (vec4 per species, RGB + padding)
layout(set = 0, binding = 3) readonly buffer SpeciesColors {
    vec4 species_colors[];
};

// Temperature per cell in Kelvin
layout(set = 0, binding = 4) readonly buffer Temperatures {
    float temperatures[];
};

// Pure water color (deeper blue so ionic plumes have more contrast)
const vec3 WATER_COLOR = vec3(0.12, 0.2, 0.32);
const float VISIBLE_CONCENTRATION_FLOOR = 0.004;

// Solid titanium color
const vec3 TITANIUM_COLOR = vec3(0.6, 0.6, 0.65);

// Thermal color ramp: shifted down so cryogenic temperatures still span
// the warm half of the palette. 185K should already read as yellow-ish,
// while 278.15K sits well into the hot end.
vec3 thermal_color(float temp_k) {
    const float min_temp_k = 120.0;
    const float max_temp_k = 320.0;
    float normalized = clamp((temp_k - min_temp_k) / (max_temp_k - min_temp_k), 0.0, 1.0);
    float t = pow(normalized, 0.82);
    t = clamp(t, 0.0, 1.0);
    
    // Multi-stop color ramp:
    // t=0.0 (~120K) -> deep blue
    // t~0.18 (~150K) -> cyan
    // t~0.34 (~185K) -> yellow-green / early yellow
    // t~0.58 (~235K) -> yellow
    // t=1.0 (~320K+) -> red
    vec3 color;
    if (t < 0.18) {
        // Blue to cyan
        float f = t / 0.18;
        color = mix(vec3(0.1, 0.15, 0.9), vec3(0.1, 0.7, 0.8), f);
    } else if (t < 0.34) {
        // Cyan to yellow-green
        float f = (t - 0.18) / 0.16;
        color = mix(vec3(0.1, 0.7, 0.8), vec3(0.82, 0.88, 0.2), f);
    } else if (t < 0.58) {
        // Yellow-green to yellow
        float f = (t - 0.34) / 0.24;
        color = mix(vec3(0.82, 0.88, 0.2), vec3(0.97, 0.83, 0.12), f);
    } else {
        // Yellow to red
        float f = (t - 0.58) / 0.42;
        color = mix(vec3(0.97, 0.83, 0.12), vec3(0.95, 0.1, 0.1), f);
    }
    return color;
}

void main() {
    // Convert UV to grid coordinates
    uint x = uint(fragUV.x * float(width));
    uint y = uint(fragUV.y * float(height));
    
    // Clamp to valid range
    x = min(x, width - 1);
    y = min(y, height - 1);
    
    uint cell_index = y * width + x;
    uint cell_count = width * height;
    
    // DEBUG: Add animated border to verify rendering is updating
    float edge_dist = min(min(fragUV.x, 1.0 - fragUV.x), min(fragUV.y, 1.0 - fragUV.y));
    float border_width = 0.01;  // 1% of screen
    if (edge_dist < border_width) {
        float t = float(frame_counter % 120) / 120.0;
        vec3 border_color = vec3(
            0.5 + 0.5 * sin(t * 6.28318),
            0.5 + 0.5 * sin(t * 6.28318 + 2.094),
            0.5 + 0.5 * sin(t * 6.28318 + 4.189)
        );
        outColor = vec4(border_color, 1.0);
        return;
    }
    
    // Thermal view mode: show temperature color ramp for all cells
    if (thermal_view != 0) {
        float temp = temperatures[cell_index];
        outColor = vec4(thermal_color(temp), 1.0);
        return;
    }
    
    // Check if solid
    if (solid_mask[cell_index] != 0) {
        uint mat_id = material_ids[cell_index];
        if (mat_id == 1) {
            outColor = vec4(TITANIUM_COLOR, 1.0);
        } else {
            outColor = vec4(0.3, 0.3, 0.3, 1.0);
        }
        return;
    }
    
    // Fluid cell - blend species colors based on concentration.
    // Use a two-part nonlinear response so faint plumes remain visible while
    // dense regions continue to pull harder toward species color.
    vec3 species_blend = vec3(0.0);
    float total_weight = 0.0;
    float strongest_weight = 0.0;
    vec3 strongest_color = WATER_COLOR;
    
    for (uint s = 0; s < min(species_count, 16u); s++) {
        float conc = concentrations[s * cell_count + cell_index];
        if (conc > VISIBLE_CONCENTRATION_FLOOR) {
            // Lift low concentrations harder out of the background, then add a
            // steeper high-end term so concentrated regions retain contrast.
            float visible_conc = conc - VISIBLE_CONCENTRATION_FLOOR;
            float scaled = visible_conc / 0.09;
            float low_visibility = pow(clamp(scaled, 0.0, 1.0), 0.24) * 2.8;
            float mid_visibility = pow(clamp(visible_conc / 0.28, 0.0, 1.0), 0.72) * 1.9;
            float high_visibility = pow(clamp(visible_conc / 1.0, 0.0, 1.0), 1.9) * 3.2;
            float weight = min(low_visibility + mid_visibility + high_visibility, 6.8);
            species_blend += species_colors[s].rgb * weight;
            total_weight += weight;
            if (weight > strongest_weight) {
                strongest_weight = weight;
                strongest_color = species_colors[s].rgb;
            }
        }
    }
    
    // Blend between pure water and species mixture
    // At 0 total concentration: 100% water color
    // As concentration increases: blend toward species colors
    vec3 final_color;
    if (total_weight > 0.001) {
        vec3 normalized_species = species_blend / total_weight;
        float dominance = strongest_weight / total_weight;
        float dominance_emphasis = smoothstep(0.58, 0.86, dominance);
        vec3 emphasized_species = mix(normalized_species, strongest_color, dominance_emphasis);
        float blend_factor = 1.0 - exp(-total_weight * 1.9);
        blend_factor = pow(clamp(blend_factor, 0.0, 1.0), 0.42);
        final_color = mix(WATER_COLOR, emphasized_species, blend_factor);
        final_color = pow(final_color, vec3(0.82));
        final_color += emphasized_species * max(strongest_weight - 2.5, 0.0) * 0.012;
        final_color = clamp(final_color, 0.0, 1.0);
    } else {
        final_color = WATER_COLOR;
    }
    
    outColor = vec4(final_color, 1.0);
}
