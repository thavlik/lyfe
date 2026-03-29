#version 450

// Fullscreen triangle vertex shader
// Generates a fullscreen triangle from vertex ID (0, 1, 2)
// No vertex buffers needed - positions computed from gl_VertexIndex

layout(location = 0) out vec2 fragUV;

void main() {
    // Generate fullscreen triangle vertices
    // Vertex 0: (-1, -1), UV (0, 0)
    // Vertex 1: (3, -1), UV (2, 0)  
    // Vertex 2: (-1, 3), UV (0, 2)
    
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0)
    );
    
    vec2 uvs[3] = vec2[](
        vec2(0.0, 0.0),
        vec2(2.0, 0.0),
        vec2(0.0, 2.0)
    );
    
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragUV = uvs[gl_VertexIndex];
}
