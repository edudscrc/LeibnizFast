// Vertex + Fragment shader: renders a textured full-screen quad with camera transform.
//
// The vertex shader generates a full-screen quad from 6 hardcoded vertices (2 triangles).
// The fragment shader samples the raw float matrix texture, normalizes the value using
// the range uniform, and applies the colormap LUT to produce the final RGBA color.
//
// Bind group 0 (per-tile): raw data texture + camera uniform
// Bind group 1 (shared):   colormap LUT + sampler + range params

// --- Group 0: Per-tile resources ---

// Binding 0: Raw float matrix texture (R32Float, output from compute shader)
// Uses textureLoad (integer coords) because R32Float is unfilterable.
@group(0) @binding(0) var data_texture: texture_2d<f32>;

// Binding 1: Camera uniform — 32 bytes, 8 x f32
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
@group(0) @binding(1) var<uniform> camera: CameraUniforms;

// --- Group 1: Shared colormap resources ---

// Binding 0: Colormap lookup table — 256x1 RGBA texture
@group(1) @binding(0) var colormap_lut: texture_2d<f32>;

// Binding 1: Linear sampler for the colormap LUT
@group(1) @binding(1) var colormap_sampler: sampler;

// Binding 2: Range uniform for normalization
struct RangeParams {
    min_val: f32,
    max_val: f32,
    _pad0:   f32,
    _pad1:   f32,
}
@group(1) @binding(2) var<uniform> range_params: RangeParams;

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

// Fragment shader: samples the raw float tile texture with camera + ring transform,
// then normalizes and applies the colormap LUT.
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
    // --- X: full-matrix UV -> ring -> tile-local UV ---
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

    // --- Read raw float value via textureLoad (R32Float is unfilterable) ---
    let dims = textureDimensions(data_texture);
    let texel = vec2<i32>(
        clamp(i32(tile_x * f32(dims.x)), 0, i32(dims.x) - 1),
        clamp(i32(tile_y * f32(dims.y)), 0, i32(dims.y) - 1),
    );
    let value = textureLoad(data_texture, texel, 0).r;

    // --- Normalize to [0, 1] using range uniform ---
    let range = range_params.max_val - range_params.min_val;
    var normalized: f32;
    if range > 0.0 {
        normalized = clamp((value - range_params.min_val) / range, 0.0, 1.0);
    } else {
        normalized = 0.5;
    }

    // --- Apply colormap LUT ---
    return textureSampleLevel(colormap_lut, colormap_sampler,
                              vec2<f32>(normalized, 0.5), 0.0);
}
