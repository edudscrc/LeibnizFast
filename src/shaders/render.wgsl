// Vertex + Fragment shader: renders a textured full-screen quad with camera transform.
//
// The vertex shader generates a full-screen quad from 6 hardcoded vertices (2 triangles).
// The fragment shader samples the colored matrix texture, applying the camera's
// UV offset and scale for zoom/pan.

// Binding 0: Colored matrix texture (output from compute shader)
@group(0) @binding(0) var colored_texture: texture_2d<f32>;

// Binding 1: Nearest-neighbor sampler for pixel-perfect cell display at high zoom
@group(0) @binding(1) var texture_sampler: sampler;

// Binding 2: Camera uniform — defines the visible region in UV space
struct CameraUniforms {
    uv_offset: vec2<f32>,
    uv_scale: vec2<f32>,
}
@group(0) @binding(2) var<uniform> camera: CameraUniforms;

// Vertex output: clip-space position and UV coordinates for texture sampling
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Vertex shader: generates a full-screen quad from 6 hardcoded vertices.
// No vertex buffer needed — positions and UVs are computed from vertex_index.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Two triangles forming a quad covering the entire clip space [-1, 1]
    // Triangle 1: (0,1,2) = top-left, bottom-left, top-right
    // Triangle 2: (3,4,5) = top-right, bottom-left, bottom-right
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0,  1.0),  // top-left
        vec2<f32>(-1.0, -1.0),  // bottom-left
        vec2<f32>( 1.0,  1.0),  // top-right
        vec2<f32>( 1.0,  1.0),  // top-right
        vec2<f32>(-1.0, -1.0),  // bottom-left
        vec2<f32>( 1.0, -1.0),  // bottom-right
    );

    // UV coordinates [0,1] — top-left is (0,0), bottom-right is (1,1)
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

// Fragment shader: samples the colored texture with camera transform applied.
//
// The camera's uv_offset and uv_scale define which portion of the matrix is visible.
// This approach avoids re-running the compute shader on every pan/zoom — only the
// fragment shader's UV mapping changes.
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Apply camera transform: map screen UV [0,1] to matrix UV region
    let matrix_uv = camera.uv_offset + input.uv * camera.uv_scale;

    // Discard fragments outside the texture bounds.
    // With tiled rendering each tile draws a full-screen quad; fragments that
    // don't belong to this tile must be discarded so they don't overwrite other
    // tiles' output. The render pass clears to a dark background first.
    if matrix_uv.x < 0.0 || matrix_uv.x > 1.0 || matrix_uv.y < 0.0 || matrix_uv.y > 1.0 {
        discard;
    }

    // Sample the colored matrix texture at the camera-adjusted UV coordinates.
    // Uses textureSampleLevel (explicit LOD=0) instead of textureSample to avoid
    // the WGSL uniform control flow requirement — the early return above makes
    // control flow non-uniform since matrix_uv depends on per-fragment input.uv.
    return textureSampleLevel(colored_texture, texture_sampler, matrix_uv, 0.0);
}
