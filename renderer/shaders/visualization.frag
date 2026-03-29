#version 450

// Visualization fragment shader for fluid simulation
//
// Coloring scheme:
// - Pure water (zero solute): Light blue
// - Each species gets a distinct color from hash of its index
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

// Pure water color (light blue)
const vec3 WATER_COLOR = vec3(0.7, 0.85, 0.95);

// Solid titanium color
const vec3 TITANIUM_COLOR = vec3(0.6, 0.6, 0.65);

// Water color hue (cyan-blue range, ~0.54 in 0-1 space)
const float WATER_HUE = 0.54;
const float WATER_HUE_TOLERANCE = 0.12;  // Avoid hues within this range of water

// Small lookup table of pseudo-random offsets for fast hue adjustment
const float HUE_OFFSETS[8] = float[8](
    0.17, 0.31, 0.23, 0.41, 0.13, 0.37, 0.29, 0.19
);

// Check if hue is too close to water color
bool is_near_water_hue(float h) {
    float dist = abs(h - WATER_HUE);
    // Handle wrap-around (hue is circular)
    dist = min(dist, 1.0 - dist);
    return dist < WATER_HUE_TOLERANCE;
}

// Convert hue to RGB (saturation=1, value=1)
vec3 hue_to_rgb(float h) {
    float hue = h * 6.0;
    float x = 1.0 - abs(mod(hue, 2.0) - 1.0);
    
    vec3 rgb;
    if (hue < 1.0) rgb = vec3(1.0, x, 0.0);
    else if (hue < 2.0) rgb = vec3(x, 1.0, 0.0);
    else if (hue < 3.0) rgb = vec3(0.0, 1.0, x);
    else if (hue < 4.0) rgb = vec3(0.0, x, 1.0);
    else if (hue < 5.0) rgb = vec3(x, 0.0, 1.0);
    else rgb = vec3(1.0, 0.0, x);
    
    return rgb;
}

// Hash function to generate distinct color from species index
// Ensures color is visually distinct from water color
vec3 species_color(uint species_idx) {
    // Use golden ratio based hash for good color distribution
    float h = fract(float(species_idx) * 0.618033988749895);
    
    // If too close to water color, apply offsets until distinct
    uint offset_idx = species_idx;
    for (int i = 0; i < 8 && is_near_water_hue(h); i++) {
        h = fract(h + HUE_OFFSETS[offset_idx & 7u]);
        offset_idx++;
    }
    
    vec3 rgb = hue_to_rgb(h);
    
    // Slightly desaturate for better aesthetics
    return mix(vec3(0.5), rgb, 0.85);
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
            species_blend += species_color(s) * weight;
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
