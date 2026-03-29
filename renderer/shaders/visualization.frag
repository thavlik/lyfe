#version 450

// Visualization fragment shader for fluid simulation
//
// Coloring scheme:
// - Pure water (zero solute): Light blue
// - Each species gets a pre-computed color from a CPU-side lookup table
// - 1.0 M concentration = 100% color intensity for that species
// - Colors are blended with weighted linear interpolation

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

// Pure water color (light blue)
const vec3 WATER_COLOR = vec3(0.7, 0.85, 0.95);

// Solid titanium color
const vec3 TITANIUM_COLOR = vec3(0.6, 0.6, 0.65);

// Thermal color ramp: logarithmic scale from 273.15K (blue) to 373.15K (red)
// 280K = green-blue, 288-318K = green to yellow, 373.15K = red
vec3 thermal_color(float temp_k) {
    // Logarithmic mapping: t = log(T - 273.15 + 1) / log(100 + 1)
    float offset = max(temp_k - 273.15, 0.0);
    float t = log(offset + 1.0) / log(101.0);
    t = clamp(t, 0.0, 1.0);
    
    // Multi-stop color ramp:
    // t=0.0 (273.15K freezing) -> deep blue
    // t~0.15 (280K) -> green-blue/cyan
    // t~0.30 (288K) -> green
    // t~0.65 (318K) -> yellow
    // t=1.0 (373.15K boiling) -> red
    vec3 color;
    if (t < 0.15) {
        // Blue to cyan
        float f = t / 0.15;
        color = mix(vec3(0.1, 0.15, 0.9), vec3(0.1, 0.7, 0.8), f);
    } else if (t < 0.30) {
        // Cyan to green
        float f = (t - 0.15) / 0.15;
        color = mix(vec3(0.1, 0.7, 0.8), vec3(0.2, 0.85, 0.2), f);
    } else if (t < 0.65) {
        // Green to yellow
        float f = (t - 0.30) / 0.35;
        color = mix(vec3(0.2, 0.85, 0.2), vec3(0.95, 0.9, 0.15), f);
    } else {
        // Yellow to red
        float f = (t - 0.65) / 0.35;
        color = mix(vec3(0.95, 0.9, 0.15), vec3(0.95, 0.1, 0.1), f);
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
    
    // Fluid cell - blend species colors based on concentration
    // 1.0 M = 100% weight for that species color
    vec3 species_blend = vec3(0.0);
    float total_conc = 0.0;
    
    for (uint s = 0; s < min(species_count, 16u); s++) {
        float conc = concentrations[s * cell_count + cell_index];
        if (conc > 0.0) {
            // Weight is directly the concentration (molar)
            // Clamp to reasonable range for visualization
            float weight = min(conc, 2.0);
            species_blend += species_colors[s].rgb * weight;
            total_conc += weight;
        }
    }
    
    // Blend between pure water and species mixture
    // At 0 total concentration: 100% water color
    // As concentration increases: blend toward species colors
    vec3 final_color;
    if (total_conc > 0.001) {
        // Normalize species blend
        vec3 normalized_species = species_blend / total_conc;
        // Blend factor: how much species color vs water color
        // At 1.0 M total, we want about 50% species color
        float blend_factor = 1.0 - exp(-total_conc * 0.5);
        final_color = mix(WATER_COLOR, normalized_species, blend_factor);
    } else {
        final_color = WATER_COLOR;
    }
    
    outColor = vec4(final_color, 1.0);
}
