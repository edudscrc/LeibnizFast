// Vertex + Fragment shader: renders a textured full-screen quad with camera transform.
//
// The vertex shader generates a full-screen quad from 6 hardcoded vertices (2 triangles).
// The fragment shader samples the colored matrix texture, applying the camera's
// UV offset and scale for zoom/pan.

// Binding 0: Colored matrix texture (output from compute shader)
@group(0) @binding(0) var colored_texture: texture_2d<f32>;

// Binding 1: Nearest-neighbor sampler for pixel-perfect cell display at high zoom
@group(0) @binding(1) var texture_sampler: sampler;

// Binding 2: Camera uniform — 32 bytes, 8 × f32
//
// X is kept in full-matrix UV space so the ring offset can be applied once at the
// correct scale before per-tile mapping.  Y is pre-composed to tile-local UV on the
// CPU side (tiles never span multiple rows in the ring-buffer path, and there is no
// ring along Y).
struct CameraUniforms {
    uv_x_offset:   f32,   // full-matrix X camera offset
    uv_y_offset:   f32,   // tile-local Y camera offset  (pre-composed on CPU)
    uv_x_scale:    f32,   // full-matrix X camera scale
    uv_y_scale:    f32,   // tile-local Y camera scale   (pre-composed on CPU)
    // Ring buffer offset in full-matrix UV space: cursor / total_cols.
    // 0.0 = no ring.  Applied to X before tile mapping so the same value is
    // correct regardless of how many tiles there are.
    ring_offset:   f32,
    _pad0:         f32,
    tile_x_offset: f32,   // tile column start in full-matrix UV [0, 1]
    tile_x_size:   f32,   // tile column width in full-matrix UV (0, 1]
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

// Fragment shader: samples the colored tile texture with camera + ring transform.
//
// X pipeline (ring-aware, full-matrix space):
//   1. Compute full-matrix X UV from screen UV.
//   2. Apply ring offset (fract) so the ring cursor position maps to display-left.
//      This must happen in full-matrix space — applying it in tile-local space
//      would produce a different pixel shift for each tile width.
//   3. Map full-matrix X UV to this tile's local [0,1] UV and discard if outside.
//
// Y pipeline (no ring, tile-local space):
//   The CPU pre-composes the camera Y transform with the tile row offset/size, so
//   uv_y_offset/uv_y_scale already produce tile-local Y directly.
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // --- X: full-matrix UV → ring → tile-local UV ---
    var full_x = camera.uv_x_offset + input.uv.x * camera.uv_x_scale;

    if camera.ring_offset != 0.0 {
        // Ring active: wrap full-matrix X so the oldest column (at ring cursor)
        // maps to the left edge of the display.  Result is always in [0, 1).
        full_x = fract(full_x + camera.ring_offset);
    } else {
        // No ring: discard if the camera is panned outside the matrix.
        if full_x < 0.0 || full_x > 1.0 {
            discard;
        }
    }

    // Map full-matrix X to this tile's texture column.
    let tile_x = (full_x - camera.tile_x_offset) / camera.tile_x_size;
    if tile_x < 0.0 || tile_x > 1.0 {
        discard;
    }

    // --- Y: tile-local UV (pre-composed on CPU, no ring) ---
    let tile_y = camera.uv_y_offset + input.uv.y * camera.uv_y_scale;
    if tile_y < 0.0 || tile_y > 1.0 {
        discard;
    }

    // Uses textureSampleLevel (explicit LOD=0) instead of textureSample because
    // control flow above is non-uniform (depends on per-fragment input.uv).
    return textureSampleLevel(colored_texture, texture_sampler,
                              vec2<f32>(tile_x, tile_y), 0.0);
}
