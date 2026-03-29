#version 450

// Visualization fragment shader for fluid simulation
//
// Reads concentration data and solid mask to produce colors:
// - Solid cells: Gray/metallic color based on material
// - Fluid cells: Colored by species concentration mix
//
// Species coloring (for the demo):
// - Na+: Blue
// - K+: Purple  
// - Cl-: Green
// - H+: Red
// - OH-: Yellow
// - Ti: Gray (metallic)

layout(location = 0) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

// Push constants
layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    uint species_count;
    uint _pad;
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

// Species colors (hardcoded for demo)
// Index mapping based on registration order in scenario:
// 0: Na+ (blue)
// 1: K+ (purple)  
// 2: Cl- (green)
// 3: H+ (red)
// 4: OH- (yellow)
// 5: Ti (gray)
const vec3 SPECIES_COLORS[6] = vec3[](
    vec3(0.2, 0.4, 1.0),   // Na+ - blue
    vec3(0.7, 0.2, 0.9),   // K+ - purple
    vec3(0.2, 0.8, 0.3),   // Cl- - green
    vec3(1.0, 0.2, 0.2),   // H+ - red
    vec3(1.0, 0.9, 0.2),   // OH- - yellow
    vec3(0.5, 0.5, 0.5)    // Ti - gray
);

// Material colors
const vec3 TITANIUM_COLOR = vec3(0.6, 0.6, 0.65);

void main() {
    // Convert UV to grid coordinates
    uint x = uint(fragUV.x * float(width));
    uint y = uint(fragUV.y * float(height));
    
    // Clamp to valid range
    x = min(x, width - 1);
    y = min(y, height - 1);
    
    uint cell_index = y * width + x;
    uint cell_count = width * height;
    
    // Check if solid
    if (solid_mask[cell_index] != 0) {
        // Solid cell - use material color
        uint mat_id = material_ids[cell_index];
        if (mat_id == 1) {
            // Titanium
            outColor = vec4(TITANIUM_COLOR, 1.0);
        } else {
            outColor = vec4(0.3, 0.3, 0.3, 1.0);
        }
        return;
    }
    
    // Fluid cell - blend colors based on concentrations
    vec3 color = vec3(0.0);
    float total_conc = 0.0;
    
    // Sum up weighted colors
    for (uint s = 0; s < min(species_count, 6); s++) {
        float conc = concentrations[s * cell_count + cell_index];
        if (conc > 0.0) {
            // Use log scale for better visualization
            float weight = log(1.0 + conc * 10.0);
            color += SPECIES_COLORS[s] * weight;
            total_conc += weight;
        }
    }
    
    // Normalize and apply base color
    if (total_conc > 0.0) {
        color /= total_conc;
        // Darken based on total concentration (darker = more concentrated)
        float intensity = 0.3 + 0.7 * (1.0 - exp(-total_conc * 0.5));
        color *= intensity;
    } else {
        // Empty fluid - dark blue water
        color = vec3(0.05, 0.05, 0.15);
    }
    
    outColor = vec4(color, 1.0);
}
